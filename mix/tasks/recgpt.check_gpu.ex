defmodule Mix.Tasks.Recgpt.CheckGpu do
  @shortdoc "Check if Nx/Torchx is using GPU (CUDA)"
  @moduledoc """
  Verifies CUDA availability and that the default Nx backend allocates tensors on GPU.
  Exits with failure if no GPU is available.

  Run: mix recgpt.check_gpu

  Requires Torchx built with LIBTORCH_TARGET (e.g. cu129).
  """
  use Mix.Task

  @impl true
  def run(_args) do
    Application.ensure_all_started(:nx)

    backend = Nx.default_backend()
    IO.puts("Nx default_backend: #{inspect(backend)}")

    cuda_available =
      try do
        Torchx.device_available?(:cuda)
      rescue
        _ -> false
      end

    unless cuda_available do
      Mix.raise(
        "No GPU: Torchx CUDA is not available. Build Torchx with LIBTORCH_TARGET (e.g. cu129) and ensure a CUDA GPU is present."
      )
    end

    IO.puts("Torchx CUDA available: true")

    count =
      try do
        Torchx.device_count(:cuda)
      rescue
        _ -> 0
      end

    IO.puts("CUDA device count: #{count}")

    # Allocate a small tensor and see where it lives
    t = Nx.tensor([1.0, 2.0, 3.0])
    backend_str = inspect(t)
    on_gpu = String.contains?(backend_str, "cuda")
    IO.puts("Sample tensor: #{backend_str}")

    unless on_gpu do
      Mix.raise(
        "No GPU: default backend is set to :cuda but tensors are not on GPU (got: #{backend_str}). Check config and Torchx build."
      )
    end

    IO.puts("")
    IO.puts("Result: Running on GPU.")
  end
end
