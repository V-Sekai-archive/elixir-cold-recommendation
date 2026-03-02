defmodule Mix.Tasks.Recgpt.CheckGpu do
  @shortdoc "Check if Nx/EXLA is using GPU (CUDA)"
  @moduledoc """
  Verifies EXLA is loaded and that a CUDA client is available.
  Optionally checks that the default Nx backend allocates on GPU.

  Run: mix recgpt.check_gpu

  Requires EXLA with XLA_TARGET=cuda12 (or cuda13) and a CUDA-capable GPU
  when checking for GPU. With XLA_TARGET=cpu, the task reports CPU-only.
  """
  use Mix.Task

  @impl true
  def run(_args) do
    Application.ensure_all_started(:nx)

    unless Code.ensure_loaded?(EXLA) do
      Mix.raise(
        "EXLA not loaded. Add {:exla, \"~> 0.10\"} to deps and set XLA_TARGET (e.g. cuda12 or cpu)."
      )
    end

    case Application.ensure_all_started(:exla) do
      {:ok, _} ->
        do_check()
      {:error, {app, reason}} ->
        IO.puts("Nx default_backend: #{inspect(Nx.default_backend())}")
        IO.puts("EXLA application failed to start: #{inspect(app)} - #{inspect(reason)}")
        IO.puts("")
        IO.puts("Result: EXLA not running (check NIF/CUDA libs, XLA_TARGET, and troubleshooting in EXLA README).")
    end
  end

  defp do_check do
    backend = Nx.default_backend()
    IO.puts("Nx default_backend: #{inspect(backend)}")

    # Try to use EXLA with CUDA client; if it works, GPU is available
    cuda_available =
      try do
        t = Nx.tensor([1.0, 2.0, 3.0])
        _on_cuda = Nx.backend_transfer(t, {EXLA.Backend, client: :cuda})
        true
      rescue
        _ -> false
      catch
        _ -> false
      end

    if cuda_available do
      IO.puts("EXLA CUDA client: available")
    else
      IO.puts("EXLA CUDA client: not available (XLA_TARGET may be cpu or CUDA not installed)")
    end

    # Sample tensor on default backend
    t = Nx.tensor([1.0, 2.0, 3.0])
    backend_str = inspect(t)
    IO.puts("Sample tensor: #{backend_str}")

    on_gpu = String.contains?(String.downcase(backend_str), "cuda") or cuda_available

    cond do
      cuda_available and on_gpu ->
        IO.puts("")
        IO.puts("Result: EXLA GPU (CUDA) available.")
      true ->
        IO.puts("")
        IO.puts("Result: EXLA loaded; running on CPU (no CUDA client). Set XLA_TARGET=cuda12 and ensure GPU drivers for GPU.")
    end
  end
end
