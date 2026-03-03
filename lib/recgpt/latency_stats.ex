defmodule RecGPT.LatencyStats do
  @moduledoc """
  Optional sliding window of recommendation latencies for SLO checks.

  When `config :recgpt, :trace_predict` is true, PredictBatchCollector records
  each request's latency here. Use `get_percentiles/0` for recent P50/P95/P99
  and `check_slo/0` to assert RecGPT stays within target_p50_ms / target_p99_ms
  (e.g. from CI or health endpoint).
  """
  use Agent

  @max_samples 500

  @doc "Start the latency stats agent (called from Application)."
  def start_link(opts \\ []) do
    Agent.start_link(fn -> [] end, opts)
  end

  @doc "Record one recommendation latency in microseconds. Call when trace_predict is true."
  def record(us) when is_integer(us) and us >= 0 do
    case Process.whereis(__MODULE__) do
      nil -> :ok
      _ -> Agent.update(__MODULE__, fn list -> [us | list] |> Enum.take(@max_samples) end)
    end
  end

  def record(_), do: :ok

  @doc "Return recent percentiles in μs, or nil if no samples. Keys: p50_us, p95_us, p99_us, n."
  def get_percentiles do
    case Process.whereis(__MODULE__) do
      nil -> nil
      _ ->
        list = Agent.get(__MODULE__, & &1)
        compute_percentiles(list)
    end
  end

  @doc """
  Check whether recent latencies are within SLO (target_p50_ms, target_p99_ms).
  Returns :ok or {:warn, message} for use in health checks or CI.
  """
  def check_slo do
    target_p50 = Application.get_env(:recgpt, :target_p50_ms, 20)
    target_p99 = Application.get_env(:recgpt, :target_p99_ms, 60)

    case get_percentiles() do
      nil -> :ok
      %{n: n} when n < 5 -> :ok
      %{p50_us: p50_us, p99_us: p99_us} ->
        p50_ms = p50_us / 1000
        p99_ms = p99_us / 1000

        cond do
          p50_ms > target_p50 and p99_ms > target_p99 ->
            {:warn, "RecGPT P50=#{Float.round(p50_ms, 1)}ms (target #{target_p50}ms) P99=#{Float.round(p99_ms, 1)}ms (target #{target_p99}ms)"}

          p50_ms > target_p50 ->
            {:warn, "RecGPT P50=#{Float.round(p50_ms, 1)}ms exceeds target #{target_p50}ms"}

          p99_ms > target_p99 ->
            {:warn, "RecGPT P99=#{Float.round(p99_ms, 1)}ms exceeds target #{target_p99}ms"}

          true ->
            :ok
        end
    end
  end

  defp compute_percentiles([]), do: nil

  defp compute_percentiles(list) do
    n = length(list)
    sorted = Enum.sort(list)
    p50 = percentile(sorted, 50)
    p95 = percentile(sorted, 95)
    p99 = percentile(sorted, 99)
    %{p50_us: p50, p95_us: p95, p99_us: p99, n: n}
  end

  defp percentile(sorted, p) do
    idx = max(0, min(length(sorted) - 1, div(length(sorted) * p, 100)))
    Enum.at(sorted, idx)
  end
end
