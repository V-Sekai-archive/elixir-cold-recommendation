defmodule RecGPT.NVTX do
  @moduledoc """
  Optional NVTX markers for nsys/Nsight Systems GPU profiling.

  When NVTX (libnvToolsExt) is available, `range_push/1` and `range_pop/0` annotate
  the timeline for correlation with GPU kernels. When not available, calls no-op.

  Example: wrap phases for clearer nsys UI:

      RecGPT.NVTX.range_push("beam_search_step_0")
      # ... work ...
      RecGPT.NVTX.range_pop()

  Run `mix recgpt.ad_hoc_test --profile` or `mix recgpt.trace_predict --profile`
  and open the .nsys-rep file in Nsight Systems GUI.
  """

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:recgpt), "recgpt_nvtx")
    case :erlang.load_nif(path, 0) do
      :ok -> :ok
      {:error, {:load_failed, _}} -> :ok  # NIF not built or unavailable; module still works via fallbacks
      {:error, reason} -> "NVTX NIF load failed: #{inspect(reason)}"
    end
  end

  @doc """
  Push an NVTX range with the given name. Must be paired with `range_pop/0`.
  Name can be a string or binary (e.g. "beam_search_step_0").
  No-op when NVTX is unavailable.
  """
  def range_push(_name) do
    :ok
  end

  @doc """
  Pop the current NVTX range. Must match a previous `range_push/1`.
  No-op when NVTX is unavailable.
  """
  def range_pop do
    :ok
  end
end
