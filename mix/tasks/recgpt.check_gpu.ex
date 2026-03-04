defmodule Mix.Tasks.Recgpt.CheckGpu do
  @shortdoc "Check if Nx/Torchx is loaded and which device is used"
  @moduledoc """
  Verifies Torchx is loaded and reports the default Nx backend and Torchx device.
  Torchx can use :cuda or :cpu. Use config :nx, default_backend: {Torchx.Backend, device: :cuda} for GPU.

  Run: mix recgpt.check_gpu

  Requires {:torchx, "~> 0.11.0"} in deps. LibTorch must be built with CUDA for GPU.
  """
  use Mix.Task

  @impl true
  def run(_args) do
    Application.ensure_all_started(:nx)

    unless Code.ensure_loaded?(Torchx) do
      Mix.raise("Torchx not loaded. Add {:torchx, \"~> 0.11.0\"} to deps.")
    end

    case Application.ensure_all_started(:torchx) do
      {:ok, _} ->
        do_check()

      {:error, {app, reason}} ->
        IO.puts("Nx default_backend: #{inspect(Nx.default_backend())}")
        IO.puts("Torchx application failed to start: #{inspect(app)} - #{inspect(reason)}")
        IO.puts("")
        IO.puts("Result: Torchx not running (check LibTorch install).")
    end
  end

  defp do_check do
    backend = Nx.default_backend()
    IO.puts("Nx default_backend: #{inspect(backend)}")

    cuda_avail = Torchx.device_available?(:cuda)
    IO.puts("Torchx CUDA available: #{cuda_avail}")

    t = Nx.tensor([1.0, 2.0, 3.0])
    backend_str = inspect(t)
    IO.puts("Sample tensor: #{backend_str}")

    on_cuda = String.contains?(String.downcase(backend_str), "cuda") or cuda_avail

    if on_cuda do
      IO.puts("")
      IO.puts("Result: Torchx loaded; CUDA available.")
    else
      IO.puts("")
      IO.puts("Result: Torchx loaded; using CPU (or device not shown in inspect).")
    end
  end
end
