defmodule RecGPT.Eval do
  @moduledoc """
  Next-item evaluation: Hit@k and MRR vs a standard test set.

  Test format (JSON): `{"test_cases": [{"context": [id, ...], "next_item": id}, ...]}`.
  Item IDs are 0-based catalog indices. Use with Steam or other FOSS datasets.
  so numbers are comparable across runs and papers.

  ## Metrics
  - **Hit@k**: fraction of test cases where the ground-truth next item is in the top-k recommendations.
  - **MRR**: mean reciprocal rank of the ground-truth next item (1/rank, 0 if not in list).

  ## Random baseline
  For a catalog of size N, random Hit@1 ≈ 1/N and MRR ≈ 1/N. Comparing to these values
  shows whether the model beats the null hypothesis (no predictive signal).
  """

  @doc """
  Runs evaluation given serve state and a list of test cases.

  Test cases: list of maps with keys `"context"` (list of item_ids) and `"next_item"` (single item_id).
  Keys may be atoms or strings. Optional `:top_k` in opts (default 10) limits recommendation size for MRR.

  Returns a map with:
  - `:n` - number of test cases
  - `:hit_at_1`, `:hit_at_5`, `:hit_at_10` - Hit@k (0.0 .. 1.0)
  - `:mrr` - mean reciprocal rank (0.0 .. 1.0)
  - `:catalog_size` - from state, for random baseline comparison
  - `:random_hit_at_1` - 1/catalog_size (null baseline)
  - `:rejects_null` - true if Hit@1 > random_hit_at_1 (model beats random baseline)
  """
  def evaluate(state, test_cases, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10)
    top_k = min(top_k, 20)

    {h1, h5, h10, rr_sum, n} =
      Enum.reduce(test_cases, {0, 0, 0, 0.0, 0}, fn tc,
                                                    {acc_h1, acc_h5, acc_h10, acc_rr, acc_n} ->
        context = get_tc_context(tc)
        next_item = get_tc_next_item(tc)

        if context == [] or next_item == nil do
          {acc_h1, acc_h5, acc_h10, acc_rr, acc_n}
        else
          case RecGPT.Serve.recommend(state, context, top_k) do
            {:ok, preds} ->
              preds = List.wrap(preds)
              idx = index_of(preds, next_item)
              rr = if idx, do: 1.0 / (idx + 1), else: 0.0
              h1 = if idx != nil and idx < 1, do: 1, else: 0
              h5 = if idx != nil and idx < 5, do: 1, else: 0
              h10 = if idx != nil and idx < 10, do: 1, else: 0
              {acc_h1 + h1, acc_h5 + h5, acc_h10 + h10, acc_rr + rr, acc_n + 1}

            _ ->
              {acc_h1, acc_h5, acc_h10, acc_rr, acc_n}
          end
        end
      end)

    n = max(n, 1)
    catalog_size = state.num_items
    random_hit_at_1 = if catalog_size > 0, do: 1.0 / catalog_size, else: 0.0

    hit_at_1 = h1 / n
    rejects_null = hit_at_1 > random_hit_at_1

    %{
      n: n,
      hit_at_1: hit_at_1,
      hit_at_5: h5 / n,
      hit_at_10: h10 / n,
      mrr: rr_sum / n,
      catalog_size: catalog_size,
      random_hit_at_1: random_hit_at_1,
      rejects_null: rejects_null
    }
  end

  @doc """
  Loads test cases from a JSON file.

  Expected keys: `"test_cases"` (list of `{"context": [id, ...], "next_item": id}`).
  Returns `{:ok, list}` or `{:error, reason}`.
  """
  def load_test_cases(path) do
    if File.regular?(path) do
      raw = File.read!(path) |> Jason.decode!()
      cases = raw["test_cases"] || []
      {:ok, List.wrap(cases)}
    else
      {:error, "file not found: #{path}"}
    end
  end

  defp get_tc_context(tc) when is_map(tc) do
    (tc["context"] || tc[:context] || []) |> List.wrap()
  end

  defp get_tc_next_item(tc) when is_map(tc) do
    tc["next_item"] || tc[:next_item]
  end

  defp index_of(list, value) do
    Enum.find_index(list, &(&1 == value))
  end
end
