defmodule RecGPT.Decode do
  @moduledoc """
  SPMD-style catalog-aware beam search for next-item prediction.

  Decodes 4 tokens (one RecGPT item) over 4 steps, restricting at each step
  to tokens that are valid prefixes in the catalog trie. Uses trie tensors
  and batch inference on device with a single CPU sync at the end.
  """

  alias RecGPT.Trie

  @neg_inf -1.0e9

  @doc """
  SPMD-style beam search: trie and scoring on device, one sync at end.

  - `trie_tensors`: %{next_state: tensor, item_at_leaf: tensor} from Trie.to_tensors/2 (on device).
  - `item_id_to_tokens`: tensor {num_items, 4} on same backend.
  - `context_item_ids`: 1D tensor of context item IDs (on same backend) or list.
  - `batch_tensor_fn`: (batch_tensor {b, seq_len}, cache) -> {logits {b, vocab_size}, new_cache}.
  - `backend`: backend for tensors (e.g. EXLA); required when context_item_ids is a list.
  - `trie`: optional map trie from Trie.build/1; when given, used to resolve item_id from 4-token
    sequence when tensor item_at_leaf returns -1 (ensures correct item_ids with one path).

  Returns {:ok, [item_id]} or :not_found. Single sync after the 4 steps to get top-k item_ids.
  """
  @spec beam_search_top_k_spmd(
          %{next_state: Nx.Tensor.t(), item_at_leaf: Nx.Tensor.t()},
          Nx.Tensor.t(),
          Nx.Tensor.t() | [non_neg_integer()],
          pos_integer(),
          (Nx.Tensor.t(), term() -> {Nx.Tensor.t(), term()}),
          term(),
          map() | nil
        ) :: {:ok, [non_neg_integer()]} | :not_found
  def beam_search_top_k_spmd(trie_tensors, item_id_to_tokens, context_item_ids, top_k, batch_tensor_fn, backend, trie \\ nil)
      when is_map(trie_tensors) and top_k >= 1 and is_function(batch_tensor_fn, 2) do
    {context_tokens, context_len} =
      if is_list(context_item_ids) and context_item_ids == [] do
        # No context: use single padding token (Nx disallows zero-sized dimensions)
        pad = Nx.tensor([[0]], type: {:s, 32}) |> Nx.backend_transfer(backend)
        {pad, 1}
      else
        context_item_ids =
          if is_list(context_item_ids) do
            Nx.tensor(context_item_ids, type: {:s, 32}) |> Nx.backend_transfer(backend)
          else
            context_item_ids
          end

        ctx = Nx.gather(item_id_to_tokens, context_item_ids) |> Nx.reshape({:auto})
        len = Nx.size(ctx)
        {Nx.reshape(ctx, {1, len}), len}
      end

    {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
    next_state = trie_tensors.next_state
    item_at_leaf = trie_tensors.item_at_leaf
    # Adaptive: cap at 12 to avoid over-beam; use top_k+2 for small top_k exploration
    beam_width = max(4, min(top_k + 2, 12))
    root_state = Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)

    # Step 0: one candidate (state 0), forward context only
    {logits, cache} = batch_tensor_fn.(context_tokens, nil)
    logits = Nx.reshape(logits, {:auto})
    valid = Nx.gather(next_state, root_state) |> Nx.reshape({:auto})
    valid_mask = Nx.greater_equal(valid, 0)
    neg_inf = Nx.tensor(@neg_inf, type: Nx.type(logits)) |> Nx.backend_transfer(backend)
    scores = Nx.select(valid_mask, logits, neg_inf)
    {top_scores, top_indices} = Nx.top_k(scores, k: beam_width)
    top_token_ids = Nx.reshape(top_indices, {:auto}) |> Nx.as_type({:s, 32})
    new_state_ids = gather_2d(next_state, root_state, top_token_ids)
    new_state_ids = Nx.squeeze(new_state_ids, axes: [1])
    prefix_tokens = Nx.new_axis(top_token_ids, 1)
    beam_scores = Nx.as_type(top_scores, {:f, 32})

    # Steps 1, 2, 3 (step 3 uses item_at_leaf for valid mask and returns item_ids)
    {_state_ids, prefix_tokens, beam_scores, _cache, item_ids} =
      Enum.reduce(1..3, {new_state_ids, prefix_tokens, beam_scores, cache, nil}, fn step, {state_ids, prefixes, scores, c, _} ->
        spmd_step(next_state, item_at_leaf, batch_tensor_fn, context_tokens, context_len,
          state_ids, prefixes, scores, step, beam_width, vocab_size, c, backend)
      end)

    # Single sync: transfer item_ids, scores, and prefix_tokens (4 tokens per candidate) to host
    item_ids_list = Nx.to_flat_list(item_ids)
    scores_list = Nx.to_flat_list(beam_scores)
    prefix_tokens =
      prefix_tokens
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
    prefix_tokens_list = Enum.chunk_every(prefix_tokens, 4)

    candidates =
      item_ids_list
      |> Enum.zip(scores_list)
      |> Enum.zip(prefix_tokens_list)
      |> Enum.map(fn {{iid, score}, tokens} ->
        iid_resolved = if iid >= 0, do: iid, else: resolve_item_id(trie, tokens)
        {iid_resolved, score}
      end)
      |> Enum.filter(fn {iid, _} -> iid >= 0 end)
      |> Enum.sort_by(fn {_, s} -> s end, :desc)
      |> Enum.uniq_by(fn {iid, _} -> iid end)
      |> Enum.take(top_k)
      |> Enum.map(fn {iid, _} -> iid end)

    case candidates do
      [] -> :not_found
      list -> {:ok, list}
    end
  end

  defp resolve_item_id(nil, _tokens), do: -1
  defp resolve_item_id(trie, tokens) when length(tokens) == 4 do
    [t0, t1, t2, t3] = Enum.map(tokens, &round/1)
    case Trie.lookup(trie, [t0, t1, t2, t3]) do
      {:ok, id} -> id
      :not_found -> -1
    end
  end
  defp resolve_item_id(_trie, _), do: -1

  defp spmd_step(next_state, item_at_leaf, batch_tensor_fn, context_tokens, context_len,
         state_ids, prefix_tokens, beam_scores, step, beam_width, vocab_size, cache, backend) do
    k = Nx.axis_size(prefix_tokens, 0)
    prefix_len = step
    context_broadcast = Nx.broadcast(context_tokens, {k, context_len})
    prefix_slice = prefix_tokens |> Nx.slice_along_axis(0, prefix_len, axis: 1)
    batch = Nx.concatenate([context_broadcast, prefix_slice], axis: 1)
    {logits, new_cache} = batch_tensor_fn.(batch, cache)

    # Clamp so we never pass -1 as row index (when valid pairs < beam_width we can get -1 state_ids)
    state_ids_safe = Nx.max(state_ids, 0)
    idx_2d = Nx.new_axis(state_ids_safe, -1)
    {valid_rows, transition_tensor} =
      if step == 3 do
        {Nx.gather(item_at_leaf, idx_2d) |> Nx.reshape({beam_width, vocab_size}), item_at_leaf}
      else
        {Nx.gather(next_state, idx_2d) |> Nx.reshape({beam_width, vocab_size}), next_state}
      end

    valid_mask = Nx.greater_equal(valid_rows, 0)
    neg_inf = Nx.tensor(@neg_inf, type: Nx.type(logits)) |> Nx.backend_transfer(backend)
    scores_per_token = Nx.select(valid_mask, logits, neg_inf)
    beam_scores_broadcast = Nx.new_axis(beam_scores, 1)
    scores_per_token = Nx.add(scores_per_token, beam_scores_broadcast)
    flat = Nx.reshape(scores_per_token, {:auto})
    {top_scores, top_flat} = Nx.top_k(flat, k: beam_width)
    vocab_t = Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_transfer(backend)
    top_flat = Nx.as_type(top_flat, {:s, 32}) |> Nx.backend_transfer(backend)
    batch_indices = Nx.quotient(top_flat, vocab_t)
    token_ids = Nx.remainder(top_flat, vocab_t)

    # Use state_ids[batch_indices], not batch_indices, as row into next_state/item_at_leaf.
    # For step 3: item_id = item_at_leaf[current_state_at_beam, token]; same.
    state_at_top = Nx.gather(state_ids, Nx.new_axis(batch_indices, -1)) |> Nx.reshape({:auto})
    state_at_top_safe = Nx.max(state_at_top, 0)
    new_state_ids =
      if step == 3 do
        gather_2d(item_at_leaf, state_at_top_safe, token_ids) |> Nx.squeeze(axes: [1])
      else
        gather_2d(transition_tensor, state_at_top_safe, token_ids) |> Nx.squeeze(axes: [1])
      end

    old_prefixes =
      Nx.gather(prefix_tokens, Nx.new_axis(batch_indices, -1))
      |> Nx.reshape({beam_width, prefix_len})
    new_col = Nx.reshape(token_ids, {beam_width, 1})
    new_prefix_tokens = Nx.concatenate([old_prefixes, new_col], axis: 1)

    item_ids = if step == 3, do: new_state_ids, else: nil
    {new_state_ids, new_prefix_tokens, top_scores, new_cache, item_ids}
  end

  defp gather_2d(tensor, row_indices, col_indices) do
    row_2d = Nx.new_axis(row_indices, -1)
    rows = Nx.gather(tensor, row_2d)
    rows = if Nx.rank(rows) == 1, do: Nx.reshape(rows, {1, :auto}), else: rows
    rows = if Nx.rank(rows) == 3, do: Nx.reshape(rows, {Nx.axis_size(rows, 0), Nx.axis_size(rows, 2)}), else: rows
    k = Nx.axis_size(col_indices, 0)
    {num_rows, vocab_size} = Nx.shape(rows)
    rows = if num_rows == 1 and k > 1, do: Nx.broadcast(rows, {k, vocab_size}), else: rows
    indices = Nx.reshape(col_indices, {k, 1})
    Nx.take_along_axis(rows, indices, axis: 1)
  end
end
