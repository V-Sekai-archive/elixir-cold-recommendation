defmodule RecGPT.HealthServer do
  @moduledoc """
  Minimal HTTP health endpoint for readiness probes (e.g. K8s).
  Listens on a configurable port; GET / returns 200 when serve_state is loaded, 503 otherwise.
  """
  @spec start_link(non_neg_integer()) :: {:ok, pid()} | {:error, term()}
  def start_link(port) do
    Task.start_link(fn -> accept_loop(port) end)
  end

  defp accept_loop(port) do
    case :gen_tcp.listen(port, [:binary, active: false, reuseaddr: true]) do
      {:ok, listen_socket} -> do_accept(listen_socket)
      {:error, _} = err -> require Logger; Logger.warning("Health server failed to listen on #{port}: #{inspect(err)}")
    end
  end

  defp do_accept(listen_socket) do
    case :gen_tcp.accept(listen_socket) do
      {:ok, socket} -> spawn(fn -> handle(socket) end); do_accept(listen_socket)
      _ -> do_accept(listen_socket)
    end
  end

  defp handle(socket) do
    _ = :gen_tcp.recv(socket, 0, 5000)
    status = if Application.get_env(:recgpt, :serve_state), do: 200, else: 503
    body = if status == 200, do: "OK", else: "Service Unavailable"
    resp = "HTTP/1.1 #{status} #{body}\r\nContent-Length: #{byte_size(body)}\r\nConnection: close\r\n\r\n#{body}"
    :gen_tcp.send(socket, resp)
    :gen_tcp.close(socket)
  end
end
