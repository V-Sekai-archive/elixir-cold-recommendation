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
  @spec evaluate(RecGPT.Serve.state(), [map()], keyword()) :: map()
  def evaluate(state, test_cases, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10) |> min(20)
    check_interval = Keyword.get(opts, :resource_check_interval, 100)

    check_opts =
      Keyword.get(opts, :resource_check_opts, [])
      |> Keyword.put_new(:start_monotonic_sec, System.monotonic_time(:second))

    recommend_fn = fn ctx, k -> RecGPT.Serve.recommend(state, ctx, k) end

    result =
      Enum.reduce_while(test_cases, {0, 0, 0, 0.0, 0}, fn tc, acc ->
        if run_resource_check?(elem(acc, 4), check_interval) do
          case RecGPT.ResourceCheck.check(check_opts) do
            :ok -> {:cont, update_acc_for_tc(tc, acc, recommend_fn, top_k)}
            {:halt, reason} -> {:halt, {:halted, reason, acc}}
          end
        else
          {:cont, update_acc_for_tc(tc, acc, recommend_fn, top_k)}
        end
      end)

    {h1, h5, h10, rr_sum, n} = final_metrics_tuple(result)
    halted_reason = halted_reason_from_result(result)
    build_metrics(h1, h5, h10, rr_sum, n, state.num_items, halted_reason)
  end

  @doc """
  Keeps only test cases whose context and next_item are in 0..(num_items - 1).
  Use when fixture was built with a limit (e.g. 100) so eval stays within that catalog
  and avoids high memory. Larger limits can be restored later when the stack supports them.
  """
  @spec filter_to_catalog([map()], non_neg_integer()) :: [map()]
  def filter_to_catalog(test_cases, num_items) when num_items > 0 do
    Enum.filter(test_cases, fn tc ->
      context = get_tc_context(tc)
      next_item = get_tc_next_item(tc)

      next_item != nil and next_item >= 0 and next_item < num_items and
        Enum.all?(context, fn id -> is_integer(id) and id >= 0 and id < num_items end)
    end)
  end

  def filter_to_catalog(test_cases, _), do: test_cases

  @doc """
  Loads test cases from a JSON file.

  Expected keys: `"test_cases"` (list of `{"context": [id, ...], "next_item": id}`).
  Returns `{:ok, list}` or `{:error, reason}`.
  """
  @spec load_test_cases(String.t()) :: {:ok, [map()]} | {:error, String.t()}
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

  defp run_resource_check?(acc_n, check_interval) do
    check_interval > 0 and acc_n > 0 and rem(acc_n, check_interval) == 0
  end

  defp update_acc_for_tc(tc, acc, recommend_fn, top_k) do
    context = get_tc_context(tc)
    next_item = get_tc_next_item(tc)

    if context == [] or next_item == nil do
      acc
    else
      case recommend_fn.(context, top_k) do
        {:ok, preds} -> add_hit_metrics(acc, List.wrap(preds), next_item)
        _ -> acc
      end
    end
  end

  defp add_hit_metrics({acc_h1, acc_h5, acc_h10, acc_rr, acc_n}, preds, next_item) do
    idx = index_of(preds, next_item)
    rr = if idx, do: 1.0 / (idx + 1), else: 0.0
    h1 = if idx != nil and idx < 1, do: 1, else: 0
    h5 = if idx != nil and idx < 5, do: 1, else: 0
    h10 = if idx != nil and idx < 10, do: 1, else: 0
    {acc_h1 + h1, acc_h5 + h5, acc_h10 + h10, acc_rr + rr, acc_n + 1}
  end

  defp final_metrics_tuple(result) do
    case result do
      {:halted, _reason, acc_tuple} -> acc_tuple
      acc_tuple -> acc_tuple
    end
  end

  defp halted_reason_from_result(result) do
    case result do
      {:halted, reason, _} -> reason
      _ -> nil
    end
  end

  defp build_metrics(h1, h5, h10, rr_sum, n, catalog_size, halted_reason) do
    n = max(n, 1)
    random_hit_at_1 = if catalog_size > 0, do: 1.0 / catalog_size, else: 0.0
    hit_at_1 = h1 / n

    metrics = %{
      n: n,
      hit_at_1: hit_at_1,
      hit_at_5: h5 / n,
      hit_at_10: h10 / n,
      mrr: rr_sum / n,
      catalog_size: catalog_size,
      random_hit_at_1: random_hit_at_1,
      rejects_null: hit_at_1 > random_hit_at_1
    }

    if halted_reason, do: Map.put(metrics, :halted, halted_reason), else: metrics
  end
end
