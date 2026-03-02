defmodule Mix.Tasks.Recgpt.Serve do
  @shortdoc "Start RecGPT gRPC recommendation server"
  @moduledoc """
  Start the RecGPT gRPC API server (no REST).

  Loads checkpoint and fixture once. Serves:
  - gRPC: recgpt.v1.PredictionService/Predict on port 50051 (default)

  ## Options
    * `--grpc-port` - gRPC port (default: 50051)
    * `--health-port` - HTTP health port for readiness (default: 50052); 0 to disable
    * `--fixture` - Path to fixture JSON (default: data/serve_e2e_fixture.json)
    * `--ckpt` - Path to checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--catalog` - Optional path to catalog JSON (item_id -> text)

  ## Environment
    * RECGPT_FIXTURE - override fixture path
    * RECGPT_CKPT_EXPORT - override checkpoint export dir
    * RECGPT_HEALTH_PORT - health HTTP port (default 50052); set to 0 to disable

  ## Examples
      mix recgpt.serve
      mix recgpt.serve --grpc-port 50052
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          grpc_port: :integer,
          health_port: :integer,
          fixture: :string,
          ckpt: :string,
          catalog: :string
        ]
      )

    grpc_port = opts[:grpc_port] || 50_051

    health_port =
      opts[:health_port] ||
        case System.get_env("RECGPT_HEALTH_PORT") do
          nil ->
            nil

          s ->
            case Integer.parse(s) do
              {n, _} -> n
              :error -> nil
            end
        end

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        resolve_path("data/serve_e2e_fixture.json")

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_EXPORT") ||
        resolve_path(Path.join([File.cwd!(), "thirdparty", "checkpoints", "recgpt"]))

    catalog_path = opts[:catalog]

    Application.ensure_all_started(:recgpt)

    Mix.shell().info("Loading model and fixture...")

    case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
      {:ok, state} ->
        Application.put_env(:recgpt, :serve_state, state)
        Mix.shell().info("gRPC: 0.0.0.0:#{grpc_port} (recgpt.v1.PredictionService/Predict)")

        children = [
          {RecGPT.PredictBatchCollector, []},
          {GRPC.Server.Supervisor,
           endpoint: RecGPT.GRPCEndpoint, port: grpc_port, start_server: true}
        ]

        children =
          if is_integer(health_port) and health_port == 0 do
            children
          else
            port = if is_integer(health_port) and health_port > 0, do: health_port, else: 50_052
            Mix.shell().info("Health: 0.0.0.0:#{port} GET /")
            [{RecGPT.HealthServer, port} | children]
          end

        {:ok, _pid} = Supervisor.start_link(children, strategy: :one_for_one)
        Process.sleep(:infinity)

      {:error, reason} ->
        Mix.raise(
          "Failed to load: #{inspect(reason)}. Ensure fixture at #{fixture_path} and checkpoint at #{ckpt_dir}"
        )
    end
  end

  defp resolve_path(path) do
    if absolute_path?(path),
      do: path,
      else:
        first_existing(Path.join(File.cwd!(), path), Path.join(Path.dirname(File.cwd!()), path))
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/

  defp first_existing(a, b) do
    cond do
      File.regular?(a) or File.dir?(a) -> a
      File.regular?(b) or File.dir?(b) -> b
      true -> a
    end
  end
end
