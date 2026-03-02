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

  ## Constant memory and wavefront
  Eval is designed for **constant memory**: one test case at a time, so the computation
  **wavefront** stays bounded—no backlog of tensors or data. We keep only running counts
  (h1, h5, h10, rr_sum, n); each case is computed, metrics updated, then released. When
  you pass a **stream** (e.g. `stream_test_cases_from_db/1`), the wavefront advances
  through the stream without accumulating. No full test set or tensor batch in memory.

  **SPMD** is the right design for scaling: same program on every rank, sharded test
  data (and optionally sharded catalog/model), then reduce metrics (sum h1, h5, h10,
  rr_sum, n). See GSPMD and N-D parallelism (e.g.
  [pytorch/torchtitan](https://github.com/pytorch/torchtitan),
  [openxla/shardy](https://github.com/openxla/shardy)). We want SPMD for all tensor
  code here, but we build a rope bridge across the chasm before a road or a busy
  highway: this eval is one rank (single wavefront: one case → recommend → update
  counts → release). Pipelining (multiple wavefronts with bounded queues) belongs
  *within* a rank. Scaling out to more ranks and full SPMD is the intended road and
  highway.
  """

  @doc """
  Runs evaluation given serve state and test cases (list or stream).

  Test cases: enumerable of maps with keys `"context"` (list of item_ids) and `"next_item"` (single item_id).
  Use a stream (e.g. from `stream_test_cases_from_db/2`) for constant memory; pass `:total` in opts for progress.
  Optional `:top_k` in opts (default 10) limits recommendation size for MRR.

  Returns a map with:
  - `:n` - number of test cases
  - `:hit_at_1`, `:hit_at_5`, `:hit_at_10` - Hit@k (0.0 .. 1.0)
  - `:mrr` - mean reciprocal rank (0.0 .. 1.0)
  - `:catalog_size` - from state, for random baseline comparison
  - `:random_hit_at_1` - 1/catalog_size (null baseline)
  - `:rejects_null` - true if Hit@1 > random_hit_at_1 (model beats random baseline)
  """
  @spec evaluate(RecGPT.Serve.state(), Enumerable.t(), keyword()) :: map()
  def evaluate(state, test_cases, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 10) |> min(20)
    check_interval = Keyword.get(opts, :resource_check_interval, 100)
    progress_interval_sec = Keyword.get(opts, :progress_interval_sec)
    progress_fn = Keyword.get(opts, :progress_fn) || default_progress_fn()
    total = Keyword.get(opts, :total) || if is_list(test_cases), do: length(test_cases), else: nil

    check_opts =
      Keyword.get(opts, :resource_check_opts, [])
      |> Keyword.put_new(:start_monotonic_sec, System.monotonic_time(:second))

    recommend_fn =
      Keyword.get(opts, :recommend_fn) ||
        fn ctx, k -> RecGPT.Serve.recommend(state, ctx, k) end

    start_sec = System.monotonic_time(:second)

    # Constant memory / bounded wavefront: one test case at a time; acc = (h1, h5, h10, rr_sum, n, last_progress_sec).
    result =
      Enum.reduce_while(test_cases, {0, 0, 0, 0.0, 0, start_sec}, fn tc, acc ->
        {h1, h5, h10, rr_sum, n, last_progress_sec} = acc
        metrics_5 = update_acc_for_tc(tc, {h1, h5, h10, rr_sum, n}, recommend_fn, top_k)
        {nh1, nh5, nh10, nrr_sum, nn} = metrics_5
        now_sec = System.monotonic_time(:second)

        maybe_report_progress =
          if progress_interval_sec && progress_interval_sec > 0 &&
               now_sec - last_progress_sec >= progress_interval_sec do
            progress_fn.(nn, total, metrics_5)
            now_sec
          else
            last_progress_sec
          end

        next_acc = {nh1, nh5, nh10, nrr_sum, nn, maybe_report_progress}

        if run_resource_check?(nn, check_interval) do
          case RecGPT.ResourceCheck.check(check_opts) do
            :ok -> {:cont, next_acc}
            {:halt, reason} -> {:halt, {:halted, reason, next_acc}}
          end
        else
          {:cont, next_acc}
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
  Streams test cases from the catalog DB (test_cases + test_context).

  Use for constant-memory eval: stream is consumed one test case at a time.
  Optional `num_items` filters to in-catalog only (context and next_item in 0..num_items-1).
  Requires Ecto Repo and RECGPT_SQLITE_PATH (or sync_test_from_json) to have been run.
  """
  @spec stream_test_cases_from_db(non_neg_integer() | nil) :: Enumerable.t()
  def stream_test_cases_from_db(num_items \\ nil) do
    import Ecto.Query
    alias RecGPT.Catalog.{TestCase, TestContext}
    alias RecGPT.Repo

    query =
      from(tc in TestCase,
        left_join: ctx in TestContext,
        on: ctx.case_id == tc.case_id,
        order_by: [asc: tc.case_id, asc: ctx.pos],
        select: {tc.case_id, tc.next_item, ctx.pos, ctx.item_id}
      )

    stream = Repo.stream(query)

    stream
    |> Stream.transform(
      {nil, [], nil},
      fn row, state ->
        {case_id, next_item, pos, item_id} = row
        {prev_cid, prev_ctx, prev_next} = state

        emit =
          if prev_cid != nil and case_id != prev_cid do
            tc = %{"context" => prev_ctx, "next_item" => prev_next}

            in_catalog =
              num_items == nil or
                (prev_next != nil and prev_next >= 0 and prev_next < num_items and
                   Enum.all?(prev_ctx, fn id -> id >= 0 and id < num_items end))

            if in_catalog, do: [tc], else: []
          else
            []
          end

        new_ctx =
          if case_id != prev_cid do
            if pos != nil and item_id != nil, do: [item_id], else: []
          else
            if pos != nil and item_id != nil, do: prev_ctx ++ [item_id], else: prev_ctx
          end

        new_state = {case_id, new_ctx, if(case_id != prev_cid, do: next_item, else: prev_next)}
        {emit, new_state}
      end,
      fn state ->
        {prev_cid, prev_ctx, prev_next} = state

        if prev_cid == nil do
          []
        else
          tc = %{"context" => prev_ctx, "next_item" => prev_next}

          in_catalog =
            num_items == nil or
              (prev_next != nil and prev_next >= 0 and prev_next < num_items and
                 Enum.all?(prev_ctx, fn id -> id >= 0 and id < num_items end))

          if in_catalog, do: [tc], else: []
        end
      end
    )
  end

  @doc """
  Count test cases in the DB (for progress total when streaming).
  """
  @spec count_test_cases_from_db() :: non_neg_integer()
  def count_test_cases_from_db do
    import Ecto.Query
    alias RecGPT.Catalog.TestCase
    alias RecGPT.Repo
    Repo.aggregate(from(t in TestCase), :count, :case_id)
  end

  @doc """
  Stream cold test cases from DB. Same as stream_test_cases_from_db but for cold_test_cases / cold_test_context.
  """
  @spec stream_cold_test_cases_from_db(non_neg_integer() | nil) :: Enumerable.t()
  def stream_cold_test_cases_from_db(num_items \\ nil) do
    import Ecto.Query
    alias RecGPT.Catalog.{ColdTestCase, ColdTestContext}
    alias RecGPT.Repo

    query =
      from(tc in ColdTestCase,
        left_join: ctx in ColdTestContext,
        on: ctx.case_id == tc.case_id,
        order_by: [asc: tc.case_id, asc: ctx.pos],
        select: {tc.case_id, tc.next_item, ctx.pos, ctx.item_id}
      )

    stream = Repo.stream(query)

    stream
    |> Stream.transform(
      {nil, [], nil},
      fn row, state ->
        {case_id, next_item, pos, item_id} = row
        {prev_cid, prev_ctx, prev_next} = state

        emit =
          if prev_cid != nil and case_id != prev_cid do
            tc = %{"context" => prev_ctx, "next_item" => prev_next}

            in_catalog =
              num_items == nil or
                (prev_next != nil and prev_next >= 0 and prev_next < num_items and
                   Enum.all?(prev_ctx, fn id -> id >= 0 and id < num_items end))

            if in_catalog, do: [tc], else: []
          else
            []
          end

        new_ctx =
          if case_id != prev_cid do
            if pos != nil and item_id != nil, do: [item_id], else: []
          else
            if pos != nil and item_id != nil, do: prev_ctx ++ [item_id], else: prev_ctx
          end

        new_state = {case_id, new_ctx, if(case_id != prev_cid, do: next_item, else: prev_next)}
        {emit, new_state}
      end,
      fn state ->
        {prev_cid, prev_ctx, prev_next} = state

        if prev_cid == nil do
          []
        else
          tc = %{"context" => prev_ctx, "next_item" => prev_next}

          in_catalog =
            num_items == nil or
              (prev_next != nil and prev_next >= 0 and prev_next < num_items and
                 Enum.all?(prev_ctx, fn id -> id >= 0 and id < num_items end))

          if in_catalog, do: [tc], else: []
        end
      end
    )
  end

  @spec count_cold_test_cases_from_db() :: non_neg_integer()
  def count_cold_test_cases_from_db do
    import Ecto.Query
    alias RecGPT.Catalog.ColdTestCase
    alias RecGPT.Repo
    Repo.aggregate(from(t in ColdTestCase), :count, :case_id)
  end

  @doc """
  Loads test cases from a JSON file.

  Expected keys: `"test_cases"` (list of `{"context": [id, ...], "next_item": id}`).
  Returns `{:ok, list}` or `{:error, reason}`.
  """
  @spec load_test_cases(String.t()) :: {:ok, [map()]} | {:error, String.t()}
  def load_test_cases(path) do
    if File.regular?(path) do
      raw = File.read!(path) |> Jason.decode!()

      cases =
        raw["test_cases"] ||
          (raw["sequences"] || [])
          |> Enum.map(fn seq ->
            seq = List.wrap(seq)

            if length(seq) >= 1 do
              %{"context" => Enum.drop(seq, -1), "next_item" => List.last(seq)}
            else
              %{"context" => [], "next_item" => 0}
            end
          end)

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
    raw =
      case result do
        {:halted, _reason, acc} -> acc
        acc -> acc
      end

    # Strip progress timestamp (6th element) if present
    case raw do
      {h1, h5, h10, rr_sum, n, _last_sec} -> {h1, h5, h10, rr_sum, n}
      t when tuple_size(t) == 5 -> t
    end
  end

  defp default_progress_fn do
    fn done, total, {h1, _h5, _h10, rr_sum, n} ->
      hit1 = if n > 0, do: Float.round(h1 / n, 4), else: 0.0
      mrr = if n > 0, do: Float.round(rr_sum / n, 4), else: 0.0
      total_str = if total, do: "#{done}/#{total}", else: "#{done}"
      IO.puts("  eval progress: #{total_str}  Hit@1=#{hit1}  MRR=#{mrr}")
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
