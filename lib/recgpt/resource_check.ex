defmodule RecGPT.ResourceCheck do
  @moduledoc """
  Resource monitoring and circuit breaking for long-running loops (eval, pretrain).
  Call `check/1` periodically; returns `:ok` or `{:halt, reason}` so the loop can stop
  before exceeding memory or time limits. Configure via opts or env RECGPT_MAX_MEMORY_MB.
  """

  @doc """
  Checks current VM memory (and optionally elapsed time). Returns `:ok` to continue
  or `{:halt, reason}` to stop the loop. Use in eval/pretrain reduce loops.

  Options:
  - `:max_memory_mb` - max VM memory in MB (default: env RECGPT_MAX_MEMORY_MB or 4096)
  - `:max_elapsed_sec` - max wall time in seconds (optional)
  - `:start_monotonic_sec` - value from System.monotonic_time(:second) at loop start (required if using max_elapsed_sec)
  """
  @spec check(keyword()) :: :ok | {:halt, String.t()}
  def check(opts \\ []) do
    max_mb = Keyword.get_lazy(opts, :max_memory_mb, &max_memory_mb_from_env/0)

    if max_mb && max_mb > 0 do
      current = :erlang.memory(:total)
      limit_bytes = max_mb * 1024 * 1024

      if current > limit_bytes do
        {:halt,
         "circuit break: VM memory #{current |> div(1024 * 1024)} MB exceeds limit #{max_mb} MB"}
      else
        check_elapsed(opts)
      end
    else
      check_elapsed(opts)
    end
  end

  defp check_elapsed(opts) do
    case {Keyword.get(opts, :max_elapsed_sec), Keyword.get(opts, :start_monotonic_sec)} do
      {max_sec, start_sec} when is_integer(max_sec) and max_sec > 0 and is_integer(start_sec) ->
        elapsed = System.monotonic_time(:second) - start_sec

        if elapsed >= max_sec do
          {:halt, "circuit break: elapsed #{elapsed}s exceeds limit #{max_sec}s"}
        else
          :ok
        end

      _ ->
        :ok
    end
  end

  defp max_memory_mb_from_env do
    case System.get_env("RECGPT_MAX_MEMORY_MB") do
      nil ->
        4096

      s ->
        case Integer.parse(s) do
          {n, _} when n > 0 -> n
          _ -> 4096
        end
    end
  end
end
