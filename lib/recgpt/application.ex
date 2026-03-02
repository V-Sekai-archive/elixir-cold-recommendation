defmodule RecGPT.Application do
  @moduledoc false
  use Application

  @impl Application
  def start(_type, _args) do
    children =
      if Application.get_env(:recgpt, RecGPT.Repo)[:database] do
        [RecGPT.Repo]
      else
        []
      end

    opts = [strategy: :one_for_one, name: RecGPT.Supervisor]
    Supervisor.start_link(children, opts)
  end
end

defmodule RecGPT.ReleaseTasks do
  @moduledoc """
  Entrypoint for running the gRPC server from a release.
  Usage: bin/recgpt eval "RecGPT.ReleaseTasks.serve()"
  Uses Elixir inference (RecGPT.Serve). Set RECGPT_FIXTURE and RECGPT_CKPT_EXPORT (or RECGPT_CKPT_PATH)
  so state is loaded and stored in :serve_state.
  Optional: RECGPT_GRPC_PORT (default 50051), RECGPT_HEALTH_PORT (default 50052; 0 to disable).
  """
  def serve do
    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    fixture_path = System.get_env("RECGPT_FIXTURE")
    ckpt_dir = System.get_env("RECGPT_CKPT_EXPORT") || System.get_env("RECGPT_CKPT_PATH")

    if fixture_path != "" and fixture_path != nil and ckpt_dir != "" and ckpt_dir != nil do
      case RecGPT.Serve.load_state(fixture_path, ckpt_dir, nil) do
        {:ok, state} ->
          Application.put_env(:recgpt, :serve_state, state)

        {:error, reason} ->
          raise "Failed to load state: #{inspect(reason)}"
      end
    end

    grpc_port = env_port("RECGPT_GRPC_PORT", 50_051)
    health_port = env_port("RECGPT_HEALTH_PORT", 50_052)

    children = [
      {GRPC.Server.Supervisor, endpoint: RecGPT.GRPCEndpoint, port: grpc_port, start_server: true}
    ]

    children =
      if is_integer(health_port) and health_port > 0 do
        [{RecGPT.HealthServer, health_port} | children]
      else
        children
      end

    {:ok, _} = Supervisor.start_link(children, strategy: :one_for_one)
    Process.sleep(:infinity)
  end

  defp env_port(name, default) do
    case System.get_env(name) do
      nil ->
        default

      s ->
        case Integer.parse(s) do
          {n, _} -> n
          :error -> default
        end
    end
  end
end
