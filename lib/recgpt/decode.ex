defmodule RecGPT.Decode do
  @moduledoc """
  Catalog-aware beam search for next-item prediction.

  Decodes 4 tokens (one RecGPT item) over 4 steps, restricting at each step
  to tokens that are valid prefixes in the catalog trie. Use with
  `RecGPT.Trie` and a get_logits function (e.g. from `RecGPT.Inference.forward/4`).
  """

  alias RecGPT.Trie

  @seq_len 4
  @neg_inf -1.0e9

  @doc """
  Beam search for the next 4-token sequence (one item), then map to item_id via trie.

  - `get_logits_fn`: function that takes the current token sequence (context + predicted so far)
    and returns logits tensor of shape `{1, vocab_size}` (e.g. 15_361). Pass the full sequence
    that would be fed to the model (previous items' tokens concatenated + current step tokens).
  - `trie`: from `RecGPT.Trie.build/1` over the catalog token_id_list.
  - `context_token_ids`: list of token IDs already generated (e.g. previous items' 4 tokens each).
  - `beam_width`: number of candidates to keep at each step (default 4).

  Returns `{:ok, item_id}` for the best candidate, or `:not_found` if no complete catalog match.
  """
  @spec beam_search(
          (list(non_neg_integer()) -> Nx.Tensor.t()),
          map(),
          [non_neg_integer()],
          pos_integer()
        ) :: {:ok, non_neg_integer()} | :not_found
  def beam_search(get_logits_fn, trie, context_token_ids, beam_width \\ 4)
      when is_function(get_logits_fn, 1) and is_map(trie) and beam_width >= 1 do
    # Beam: list of {token_list_for_current_item, log_sum_score}. Start with empty prefix.
    initial = {[], 0.0}
    beam = [initial]

    beam =
      Enum.reduce(0..(@seq_len - 1), beam, fn _step, current_beam ->
        expand_beam(get_logits_fn, trie, context_token_ids, current_beam, beam_width)
      end)

    # Beam now has {[t0,t1,t2,t3], score}. Resolve to item_ids and pick best.
    candidates =
      Enum.flat_map(beam, fn {tokens, score} ->
        case Trie.lookup(trie, tokens) do
          {:ok, item_id} -> [{item_id, score}]
          :not_found -> []
        end
      end)

    case candidates do
      [] ->
        :not_found

      _ ->
        {item_id, _score} = Enum.max_by(candidates, fn {_, s} -> s end)
        {:ok, item_id}
    end
  end

  @doc """
  Like `beam_search/4` but returns up to `top_k` item_ids, sorted by score (best first).
  Deduplicates by item_id (keeps highest-scoring). Uses `beam_width = max(4, top_k)` internally.

  Requires `batch_fn` (2-arity: list of token lists, cache -> {logits, new_cache}) so inference
  always uses the batched path (one forward per step). No unbatched fallback.
  """
  @spec beam_search_top_k(
          (list(non_neg_integer()) -> Nx.Tensor.t()),
          map(),
          [non_neg_integer()],
          pos_integer(),
          ([[non_neg_integer()]], term() -> {Nx.Tensor.t(), term()})
        ) :: {:ok, [non_neg_integer()]} | :not_found
  def beam_search_top_k(_get_logits_fn, _trie, _context_token_ids, _top_k, batch_fn)
      when not is_function(batch_fn, 2) do
    raise ArgumentError,
          "beam_search_top_k/5 requires batch_fn (2-arity: list, cache -> {logits, new_cache}). " <>
            "No unbatched path. Use Serve.recommend/3 or provide a batched inference function."
  end

  def beam_search_top_k(get_logits_fn, trie, context_token_ids, top_k, batch_fn)
      when is_function(get_logits_fn, 1) and is_map(trie) and top_k >= 1 do
    beam_width = max(4, top_k)

    {final_beam, _cache} =
      Enum.reduce(0..(@seq_len - 1), {[{[], 0.0}], nil}, fn _step, {current_beam, cache} ->
        expand_beam_batched(batch_fn, trie, context_token_ids, current_beam, beam_width, cache)
      end)

    beam = final_beam

    candidates =
      beam
      |> Enum.flat_map(fn {tokens, score} ->
        case Trie.lookup(trie, tokens) do
          {:ok, item_id} -> [{item_id, score}]
          :not_found -> []
        end
      end)
      |> Enum.sort_by(fn {_, s} -> s end, :desc)
      |> Enum.uniq_by(fn {item_id, _} -> item_id end)
      |> Enum.take(top_k)
      |> Enum.map(fn {item_id, _} -> item_id end)

    case candidates do
      [] -> :not_found
      list -> {:ok, list}
    end
  end

  @doc """
  Runs beam search for multiple contexts in lockstep, batching forward passes.
  `list_of_context_token_ids` is a list of B context token-ID lists. Returns a list of B results,
  each `{:ok, [item_id, ...]}` (top_k) or `:not_found`. Uses one batched forward per step (batch size
  = sum of beam sizes across B contexts), so typically much faster than B separate recommend calls.
  """
  @spec beam_search_top_k_batched(
          map(),
          [[non_neg_integer()]],
          pos_integer(),
          ([[non_neg_integer()]], term() -> {Nx.Tensor.t(), term()})
        ) :: [{:ok, [non_neg_integer()]} | :not_found]
  def beam_search_top_k_batched(trie, list_of_context_token_ids, top_k, batch_fn)
      when is_map(trie) and is_list(list_of_context_token_ids) and top_k >= 1 and
             is_function(batch_fn, 2) do
    if list_of_context_token_ids == [] do
      []
    else
      beam_width = max(4, top_k)
      initial_beams = list_of_context_token_ids |> Enum.map(fn _ -> [{[], 0.0}] end)

      {final_beams, _} =
        Enum.reduce(0..(@seq_len - 1), {initial_beams, nil}, fn _step, {beams, _cache} ->
          expand_beams_batched(batch_fn, trie, list_of_context_token_ids, beams, beam_width)
        end)

      Enum.map(final_beams, fn beam ->
        candidates =
          beam
          |> Enum.flat_map(fn {tokens, score} ->
            case Trie.lookup(trie, tokens) do
              {:ok, item_id} -> [{item_id, score}]
              :not_found -> []
            end
          end)
          |> Enum.sort_by(fn {_, s} -> s end, :desc)
          |> Enum.uniq_by(fn {item_id, _} -> item_id end)
          |> Enum.take(top_k)
          |> Enum.map(fn {item_id, _} -> item_id end)

        case candidates do
          [] -> :not_found
          list -> {:ok, list}
        end
      end)
    end
  end

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
    context_item_ids =
      if is_list(context_item_ids) do
        Nx.tensor(context_item_ids, type: {:s, 32}) |> Nx.backend_transfer(backend)
      else
        context_item_ids
      end

    {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
    next_state = trie_tensors.next_state
    item_at_leaf = trie_tensors.item_at_leaf
    beam_width = max(4, top_k)
    root_state = Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)

    # Context: gather tokens for each context item, flatten to 1D
    context_tokens = Nx.gather(item_id_to_tokens, context_item_ids) |> Nx.reshape({:auto})
    context_len = Nx.size(context_tokens)
    context_tokens = Nx.reshape(context_tokens, {1, context_len})

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

  defp expand_beam(get_logits_fn, trie, context_token_ids, beam, beam_width) do
    # For each candidate in beam, get logits for next token (context + prefix), filter to valid, take top by score.
    all_candidates =
      Enum.flat_map(beam, fn {prefix, parent_score} ->
        full_prefix = context_token_ids ++ prefix
        logits = get_logits_fn.(full_prefix)
        valid = Trie.valid_next_tokens(trie, prefix)

        if valid == [] do
          []
        else
          # logits shape {1, vocab_size}; pick valid token scores
          logits_1d = logits |> Nx.squeeze(axes: [0])

          for token_id <- valid do
            logit =
              logits_1d
              |> Nx.slice_along_axis(token_id, 1, axis: 0)
              |> Nx.squeeze(axes: [0])
              |> Nx.to_number()

            score = parent_score + logit
            {prefix ++ [token_id], score}
          end
        end
      end)

    all_candidates
    |> Enum.sort_by(fn {_, s} -> s end, :desc)
    |> Enum.take(beam_width)
  end

  defp expand_beam_batched(batch_fn, trie, context_token_ids, beam, beam_width, cache) do
    # One forward pass for all beam candidates (with optional KV-cache); then score and prune.
    full_prefixes = Enum.map(beam, fn {prefix, _} -> context_token_ids ++ prefix end)
    {logits, new_cache} = batch_fn.(full_prefixes, cache)
    # logits: {batch_size, vocab_size}
    {batch_size, vocab_size} = Nx.shape(logits)

    # Build flat list of {batch_idx, token_id, prefix, parent_score} for every (candidate, valid_token).
    entries =
      Enum.flat_map(0..(batch_size - 1), fn i ->
        {prefix, parent_score} = Enum.at(beam, i)
        valid = Trie.valid_next_tokens(trie, prefix)

        Enum.map(valid, fn token_id ->
          {i, token_id, prefix, parent_score}
        end)
      end)

    if entries == [] do
      {beam, new_cache}
    else
      # Per-entry index = batch_idx * vocab_size + token_id (row-major).
      flat_logits = Nx.reshape(logits, {batch_size * vocab_size})
      # Per-index slice + to_number so we never call to_list on EXLA tensor (avoids scalar/wrong-shape).
      scores_list =
        Enum.map(entries, fn {b, t, _, _} ->
          idx = b * vocab_size + t
          flat_logits |> Nx.slice_along_axis(idx, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number()
        end)

      all_candidates =
        Enum.zip(entries, scores_list)
        |> Enum.map(fn {{_b, token_id, prefix, parent_score}, logit} ->
          {prefix ++ [token_id], parent_score + logit}
        end)

      result_beam =
        all_candidates
        |> Enum.sort_by(fn {_, s} -> s end, :desc)
        |> Enum.take(beam_width)

      {result_beam, new_cache}
    end
  end

  defp expand_beams_batched(batch_fn, trie, list_of_context_token_ids, beams, beam_width) do
    full_prefixes =
      beams
      |> Enum.zip(list_of_context_token_ids)
      |> Enum.flat_map(fn {beam, ctx} ->
        Enum.map(beam, fn {prefix, _} -> ctx ++ prefix end)
      end)

    if full_prefixes == [] do
      {beams, nil}
    else
      {logits, _} = batch_fn.(full_prefixes, nil)

      # Offsets into logits per beam: beam i uses rows [offset[i], offset[i] + length(beam[i]))
      lengths = Enum.map(beams, &length/1)
      offsets =
        [0 | Enum.take(Enum.scan(lengths, 0, fn len, acc -> acc + len end), length(lengths) - 1)]

      new_beams =
        Enum.map(Enum.zip([beams, offsets, lengths]), fn {beam, offset, len} ->
          logits_slice = Nx.slice_along_axis(logits, offset, len, axis: 0)

          entries =
            beam
            |> Enum.with_index()
            |> Enum.flat_map(fn {{prefix, parent_score}, cand_idx} ->
              valid = Trie.valid_next_tokens(trie, prefix)

              Enum.map(valid, fn token_id ->
                logit =
                  logits_slice
                  |> Nx.slice_along_axis(cand_idx, 1, axis: 0)
                  |> Nx.slice_along_axis(token_id, 1, axis: 1)
                  |> Nx.squeeze()
                  |> Nx.to_number()

                {prefix ++ [token_id], parent_score + logit}
              end)
            end)

          entries
          |> Enum.sort_by(fn {_, s} -> s end, :desc)
          |> Enum.take(beam_width)
        end)

      {new_beams, nil}
    end
  end
end
