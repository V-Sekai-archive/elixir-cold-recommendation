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
    and returns logits tensor of shape `{1, vocab_size}` (e.g. 15361). Pass the full sequence
    that would be fed to the model (previous items' tokens concatenated + current step tokens).
  - `trie`: from `RecGPT.Trie.build/1` over the catalog token_id_list.
  - `context_token_ids`: list of token IDs already generated (e.g. previous items' 4 tokens each).
  - `beam_width`: number of candidates to keep at each step (default 4).

  Returns `{:ok, item_id}` for the best candidate, or `:not_found` if no complete catalog match.
  """
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
  Returns `{:ok, [item_id, ...]}` or `:not_found`.
  """
  def beam_search_top_k(get_logits_fn, trie, context_token_ids, top_k)
      when is_function(get_logits_fn, 1) and is_map(trie) and top_k >= 1 do
    beam_width = max(4, top_k)

    beam =
      Enum.reduce(0..(@seq_len - 1), [{[], 0.0}], fn _step, current_beam ->
        expand_beam(get_logits_fn, trie, context_token_ids, current_beam, beam_width)
      end)

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
          # logits shape {1, vocab_size}; take log_softmax for numerical stability, then pick valid token scores
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
end
