defmodule Mix.Tasks.Recgpt.CheckGpu do
  @shortdoc "Check if Nx/EXLA is loaded and which client is used"
  @moduledoc """
  Verifies EXLA is loaded and reports the default Nx backend and EXLA client.
  On Linux, EXLA can use :cuda or :host (CPU). Config sets config :exla, :default_client.

  Run: mix recgpt.check_gpu

  Requires {:exla, "~> 0.10"} in deps. Set EXLA_TARGET (e.g. cuda12) and build
  for GPU; otherwise EXLA uses the host (CPU) client.
  """
  use Mix.Task

  @impl true
  def run(_args) do
    Application.ensure_all_started(:nx)

    unless Code.ensure_loaded?(EXLA) do
      Mix.raise("EXLA not loaded. Add {:exla, \"~> 0.10\"} to deps.")
    end

    case Application.ensure_all_started(:exla) do
      {:ok, _} ->
        do_check()

      {:error, {app, reason}} ->
        IO.puts("Nx default_backend: #{inspect(Nx.default_backend())}")
        IO.puts("EXLA application failed to start: #{inspect(app)} - #{inspect(reason)}")
        IO.puts("")
        IO.puts("Result: EXLA not running (check XLA/EXLA install and EXLA_TARGET).")
    end
  end

  defp do_check do
    backend = Nx.default_backend()
    IO.puts("Nx default_backend: #{inspect(backend)}")

    client = Application.get_env(:exla, :default_client, :host)
    IO.puts("EXLA default_client: #{inspect(client)}")

    t = Nx.tensor([1.0, 2.0, 3.0])
    backend_str = inspect(t)
    IO.puts("Sample tensor: #{backend_str}")

    on_cuda = String.contains?(String.downcase(backend_str), "cuda") or client == :cuda

    if on_cuda do
      IO.puts("")
      IO.puts("Result: EXLA loaded; using CUDA.")
    else
      IO.puts("")
      IO.puts("Result: EXLA loaded; using host (CPU) or device not shown in inspect.")
    end
  end
end
