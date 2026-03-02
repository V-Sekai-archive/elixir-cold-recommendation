defmodule RecGPT.Decode do
  @moduledoc """
  Catalog-aware beam search for next-item prediction.

  Decodes 4 tokens (one RecGPT item) over 4 steps, restricting at each step
  to tokens that are valid prefixes in the catalog trie. Use with
  `RecGPT.Trie` and a get_logits function (e.g. from `RecGPT.Inference.forward/4`).
  """

  alias RecGPT.Trie

  @seq_len 4

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
      {_batch_size, _vocab_size} = Nx.shape(logits)

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
