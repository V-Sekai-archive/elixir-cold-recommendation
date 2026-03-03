defmodule RecGPT.Decode do
  @moduledoc """
  SPMD-style catalog-aware beam search for next-item prediction.

  Decodes 4 tokens (one RecGPT item) over 4 steps, restricting at each step
  to tokens that are valid prefixes in the catalog trie. Uses trie tensors
  and batch inference on device with a single CPU sync at the end.

  STATIC-style minimal sync: top-k selection is done on GPU (Nx.top_k + gather);
  only the top-k item_ids, scores, and prefix_tokens are transferred to host.
  """

  alias RecGPT.Trie

  @neg_inf -1.0e9

  @doc """
  SPMD-style beam search: one forward, trie and scoring on device, one sync at end.

  Single-forward decode: get_logits_4_fn runs one model forward, returns logits for the last 4
  positions (1, 4, vocab_size). Beam search runs over those precomputed logits.

  - `get_logits_4_fn`: (context_tokens) -> logits_4 with shape (1, 4, vocab_size)
  - `trie`: optional map trie from Trie.build/1; when given, used to resolve item_id from 4-token
    sequence when tensor item_at_leaf returns -1.

  Returns {:ok, [item_id]} or :not_found. Single sync after the 4 steps to get top-k item_ids.
  """
  @spec beam_search_top_k_spmd(
          %{next_state: Nx.Tensor.t(), item_at_leaf: Nx.Tensor.t()},
          Nx.Tensor.t(),
          Nx.Tensor.t() | [non_neg_integer()],
          pos_integer(),
          (Nx.Tensor.t() -> Nx.Tensor.t()),
          term(),
          map() | nil,
          keyword()
        ) :: {:ok, [non_neg_integer()]} | :not_found
  def beam_search_top_k_spmd(
        trie_tensors,
        item_id_to_tokens,
        context_item_ids,
        top_k,
        get_logits_4_fn,
        backend,
        trie \\ nil,
        opts \\ []
      )
      when is_map(trie_tensors) and top_k >= 1 and is_function(get_logits_4_fn, 1) do
    {context_tokens, _context_len} =
      if is_list(context_item_ids) and context_item_ids == [] do
        pad = Nx.tensor([[0]], type: {:s, 32}) |> Nx.backend_transfer(backend)
        {pad, 1}
      else
        context_item_ids =
          if is_list(context_item_ids) do
            Nx.tensor(context_item_ids, type: {:s, 32}) |> Nx.backend_transfer(backend)
          else
            context_item_ids
          end

        context_item_ids = Nx.new_axis(context_item_ids, -1)
        ctx = Nx.gather(item_id_to_tokens, context_item_ids, axes: [0]) |> Nx.reshape({:auto})
        len = Nx.size(ctx)
        {Nx.reshape(ctx, {1, len}), len}
      end

    {_num_states, vocab_size} = Nx.shape(trie_tensors.next_state)
    next_state = trie_tensors.next_state
    item_at_leaf = trie_tensors.item_at_leaf
    beam_width =
      case Keyword.get(opts, :beam_width_override) || Application.get_env(:recgpt, :beam_width_override) do
        n when is_integer(n) and n >= 1 -> n
        _ -> max(4, min(top_k + 2, 20))
      end
    constants = Keyword.get(opts, :constants)
    root_state = if constants, do: constants.root_state, else: Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)

    {item_ids, beam_scores, prefix_tokens} =
      run_single_forward_beam(
        get_logits_4_fn,
        context_tokens,
        next_state,
        item_at_leaf,
        root_state,
        backend,
        constants,
        beam_width,
        vocab_size,
        opts
      )

    # STATIC: GPU-side top-k selection so we transfer only top_k elements (minimal sync).
    RecGPT.NVTX.range_push("decode_sync")
    k = min(top_k, Nx.axis_size(item_ids, 0))
    {_top_scores, sort_indices} = Nx.top_k(beam_scores, k: k)
    sort_indices = Nx.new_axis(sort_indices, -1)
    item_ids_slice = Nx.gather(item_ids, sort_indices) |> Nx.reshape({:auto})
    scores_slice = Nx.gather(beam_scores, sort_indices) |> Nx.reshape({:auto})
    prefix_tokens_slice = Nx.gather(prefix_tokens, sort_indices)

    # Single host transfer: only the top_k slice (not full beam).
    item_ids_host = Nx.backend_transfer(item_ids_slice, Nx.BinaryBackend)
    item_ids_list = item_ids_host |> Nx.to_flat_list() |> Enum.map(&decode_item_id_to_int/1)
    scores_list = Nx.backend_transfer(scores_slice, Nx.BinaryBackend) |> Nx.to_flat_list()
    prefix_tokens_list =
      prefix_tokens_slice
      |> Nx.backend_transfer(Nx.BinaryBackend)
      |> Nx.to_flat_list()
      |> Enum.chunk_every(4)

    RecGPT.NVTX.range_pop()

    candidates =
      item_ids_list
      |> Enum.zip(scores_list)
      |> Enum.zip(prefix_tokens_list)
      |> Enum.map(fn {{iid, score}, tokens} ->
        iid_int = decode_item_id_to_int(iid)
        iid_resolved = if iid_int >= 0, do: iid_int, else: resolve_item_id(trie, tokens)
        {iid_resolved, score}
      end)
      |> Enum.filter(fn {iid, _} -> iid >= 0 end)
      |> Enum.sort_by(fn {_, s} -> s end, :desc)
      |> Enum.uniq_by(fn {iid, _} -> iid end)
      |> Enum.take(top_k)
      |> Enum.map(fn {iid, _} -> iid end)

    # Final coercion so response never contains Nx.Tensor (e.g. from any code path)
    list = Enum.map(candidates, &decode_item_id_to_int/1)

    case list do
      [] -> :not_found
      ids -> {:ok, ids}
    end
  end

  defp decode_item_id_to_int(x) when is_integer(x), do: x
  defp decode_item_id_to_int(%Nx.Tensor{} = t), do: t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number() |> round()
  defp decode_item_id_to_int(x) when is_number(x), do: round(x)

  defp run_single_forward_beam(
         get_logits_4_fn,
         context_tokens,
         next_state,
         item_at_leaf,
         root_state,
         backend,
         constants,
         beam_width,
         vocab_size,
         _opts
       ) do
    RecGPT.NVTX.range_push("single_forward")
    logits_4 = get_logits_4_fn.(context_tokens)
    RecGPT.NVTX.range_pop()

    RecGPT.NVTX.range_push("beam_search_step_0")
    logits_0 = logits_4 |> Nx.slice_along_axis(0, 1, axis: 1) |> Nx.squeeze(axes: [1])
    logits = Nx.reshape(logits_0, {:auto})
    valid = Nx.gather(next_state, root_state) |> Nx.reshape({:auto})
    valid_mask = Nx.greater_equal(valid, 0)
    neg_inf =
      if constants, do: constants.neg_inf, else: Nx.tensor(@neg_inf, type: Nx.type(logits)) |> Nx.backend_transfer(backend)
    scores = Nx.select(valid_mask, logits, neg_inf)
    {top_scores, top_indices} = Nx.top_k(scores, k: beam_width)
    top_token_ids =
      Nx.reshape(top_indices, {:auto}) |> Nx.as_type({:s, 32}) |> Nx.backend_transfer(backend)
    new_state_ids = gather_2d(next_state, root_state, top_token_ids, backend)
    new_state_ids = Nx.squeeze(new_state_ids, axes: [1])
    prefix_tokens = Nx.new_axis(top_token_ids, 1)
    beam_scores = Nx.as_type(top_scores, {:f, 32})
    RecGPT.NVTX.range_pop()

    {_state_ids, prefix_tokens, beam_scores, item_ids} =
      Enum.reduce(1..3, {new_state_ids, prefix_tokens, beam_scores, nil}, fn step,
                                                                              {state_ids, prefixes, scores, _} ->
        logits_i = logits_4 |> Nx.slice_along_axis(step, 1, axis: 1) |> Nx.squeeze(axes: [1])
        logits_broadcast = Nx.broadcast(logits_i, {beam_width, vocab_size})
        spmd_step_from_logits(
          next_state,
          item_at_leaf,
          logits_broadcast,
          state_ids,
          prefixes,
          scores,
          step,
          beam_width,
          vocab_size,
          backend,
          constants
        )
      end)

    {item_ids, beam_scores, prefix_tokens}
  end

  defp spmd_step_from_logits(
         next_state,
         item_at_leaf,
         logits,
         state_ids,
         prefix_tokens,
         beam_scores,
         step,
         beam_width,
         vocab_size,
         backend,
         constants
       ) do
    RecGPT.NVTX.range_push("beam_search_step_#{step}")
    state_ids_safe = Nx.max(state_ids, 0) |> Nx.backend_transfer(backend)
    idx_2d = Nx.new_axis(state_ids_safe, -1)

    {valid_rows, transition_tensor} =
      if step == 3 do
        {Nx.gather(item_at_leaf, idx_2d) |> Nx.reshape({beam_width, vocab_size}), item_at_leaf}
      else
        {Nx.gather(next_state, idx_2d) |> Nx.reshape({beam_width, vocab_size}), next_state}
      end

    valid_mask = Nx.greater_equal(valid_rows, 0)
    neg_inf =
      if constants, do: constants.neg_inf, else: Nx.tensor(@neg_inf, type: Nx.type(logits)) |> Nx.backend_transfer(backend)
    scores_per_token = Nx.select(valid_mask, logits, neg_inf)
    beam_scores_broadcast = Nx.new_axis(beam_scores, 1)
    scores_per_token = Nx.add(scores_per_token, beam_scores_broadcast)
    flat = Nx.reshape(scores_per_token, {:auto})
    {top_scores, top_flat} = Nx.top_k(flat, k: beam_width)
    vocab_t =
      if constants, do: constants.vocab_t, else: Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_transfer(backend)
    top_flat = Nx.as_type(top_flat, {:s, 32}) |> Nx.backend_transfer(backend)
    batch_indices = Nx.quotient(top_flat, vocab_t)
    token_ids = Nx.remainder(top_flat, vocab_t)

    batch_indices_b = Nx.backend_transfer(batch_indices, backend)
    state_at_top = Nx.gather(state_ids, Nx.new_axis(batch_indices_b, -1)) |> Nx.reshape({:auto})
    state_at_top_safe = Nx.max(state_at_top, 0) |> Nx.backend_transfer(backend)
    token_ids_b = Nx.backend_transfer(token_ids, backend)

    new_state_ids =
      if step == 3 do
        gather_2d(item_at_leaf, state_at_top_safe, token_ids_b, backend) |> Nx.squeeze(axes: [1])
      else
        gather_2d(transition_tensor, state_at_top_safe, token_ids_b, backend) |> Nx.squeeze(axes: [1])
      end

    prefix_len = step
    old_prefixes =
      Nx.gather(prefix_tokens, Nx.new_axis(batch_indices_b, -1))
      |> Nx.reshape({beam_width, prefix_len})

    new_col = Nx.reshape(token_ids, {beam_width, 1})
    new_prefix_tokens = Nx.concatenate([old_prefixes, new_col], axis: 1)
    item_ids = if step == 3, do: new_state_ids, else: nil
    RecGPT.NVTX.range_pop()
    {new_state_ids, new_prefix_tokens, top_scores, item_ids}
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

  defp gather_2d(tensor, row_indices, col_indices, backend) do
    row_2d = Nx.new_axis(row_indices |> Nx.backend_transfer(backend), -1)
    rows = Nx.gather(tensor, row_2d)
    rows = if Nx.rank(rows) == 1, do: Nx.reshape(rows, {1, :auto}), else: rows

    rows =
      if Nx.rank(rows) == 3,
        do: Nx.reshape(rows, {Nx.axis_size(rows, 0), Nx.axis_size(rows, 2)}),
        else: rows

    k = Nx.axis_size(col_indices, 0)
    {num_rows, vocab_size} = Nx.shape(rows)
    rows = if num_rows == 1 and k > 1, do: Nx.broadcast(rows, {k, vocab_size}), else: rows
    indices = Nx.reshape(col_indices |> Nx.backend_transfer(backend), {k, 1})
    Nx.take_along_axis(rows, indices, axis: 1)
  end
end
