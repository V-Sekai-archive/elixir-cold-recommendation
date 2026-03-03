defmodule RecGPT.HealthServer do
  @moduledoc """
  Minimal HTTP health endpoint for readiness probes (e.g. K8s).
  Listens on a configurable port.
  GET / returns 200 when serve_state is loaded, 503 otherwise.
  GET /slo returns 200 when recent RecGPT P50/P99 are within target_p50_ms/target_p99_ms, 503 with message otherwise (for CI or alerting).
  """
  @spec child_spec(non_neg_integer() | keyword()) :: map()
  def child_spec(port) when is_integer(port) do
    %{
      id: __MODULE__,
      start: {__MODULE__, :start_link, [port]},
      type: :worker,
      restart: :temporary
    }
  end

  def child_spec(opts) when is_list(opts) do
    port = Keyword.fetch!(opts, :port)
    child_spec(port)
  end

  @spec start_link(non_neg_integer()) :: {:ok, pid()} | {:error, term()}
  def start_link(port) do
    case :gen_tcp.listen(port, [:binary, active: false, reuseaddr: true]) do
      {:ok, listen_socket} ->
        Task.start_link(fn -> do_accept(listen_socket) end)

      {:error, _} = err ->
        require Logger

        Logger.warning(
          "Health server failed to listen on #{port}: #{inspect(err)} (gRPC still available)"
        )

        err
    end
  end

  defp do_accept(listen_socket) do
    case :gen_tcp.accept(listen_socket) do
      {:ok, socket} ->
        spawn(fn -> handle(socket) end)
        do_accept(listen_socket)

      _ ->
        do_accept(listen_socket)
    end
  end

  defp handle(socket) do
    {:ok, data} = :gen_tcp.recv(socket, 0, 5000)
    first_line = data |> String.split("\r\n") |> List.first() || ""

    {status, body} =
      if String.starts_with?(first_line, "GET /slo") do
        case RecGPT.LatencyStats.check_slo() do
          :ok -> {200, "OK"}
          {:warn, msg} -> {503, "SLO exceeded: " <> msg}
        end
      else
        if Application.get_env(:recgpt, :serve_state) do
          {200, "OK"}
        else
          {503, "Service Unavailable"}
        end
      end

    resp =
      "HTTP/1.1 #{status} #{body}\r\nContent-Length: #{byte_size(body)}\r\nConnection: close\r\n\r\n#{body}"

    :gen_tcp.send(socket, resp)
    :gen_tcp.close(socket)
  end
end
