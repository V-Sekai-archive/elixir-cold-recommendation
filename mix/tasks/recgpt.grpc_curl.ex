defmodule Mix.Tasks.Recgpt.GrpcCurl do
  @shortdoc "Call Predict RPC with grpcurl (server must be running)"
  @moduledoc """
  Runs grpcurl against a running RecGPT gRPC server to test the Predict RPC.

  Start the server first in another terminal:
      mix recgpt.serve --fixture data/steam/fixture.json --ckpt data/fuxi_ckpt_export --catalog data/steam/items.json

  Then run:
      mix recgpt.grpc_curl
      mix recgpt.grpc_curl --port 50051 --context "0,1" --max-results 10

  If the task hangs, the server is likely not running; start it in another terminal first.
  If you see "cannot convert a scalar tensor to a list", restart the serve process so it loads the latest code.

  ## Options
    * `--port` - gRPC port (default: 50051, or RECGPT_GRPC_PORT)
    * `--context` - Comma-separated context item IDs (default: 0). On PowerShell use quotes: "0,1"
    * `--max-results` or `--max-result` - Max recommendations (default: 10)
    * `--format` - grpcurl output format: json or text (default: json)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          port: :integer,
          context: :string,
          max_results: :integer,
          max_result: :integer,
          format: :string
        ]
      )

    port = opts[:port] || port_from_env() || 50_051
    context_str = opts[:context] || "0"
    max_results = opts[:max_results] || opts[:max_result] || 10
    format = opts[:format] || "json"

    context_ids =
      context_str
      |> String.split(",", trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.map(fn s ->
        case Integer.parse(s) do
          {n, _} -> n
          :error -> Mix.raise("Invalid context id: #{inspect(s)}")
        end
      end)

    payload = Jason.encode!(%{context_item_ids: context_ids, max_results: max_results})
    proto_import = Path.expand("priv/proto", File.cwd!())

    args = [
      "-plaintext",
      "-import-path",
      proto_import,
      "-proto",
      "recgpt/v1/recommendation.proto",
      "-d",
      payload,
      "-format",
      format,
      "localhost:#{port}",
      "recgpt.v1.PredictionService/Predict"
    ]

    Mix.shell().info(
      "grpcurl -plaintext -import-path #{proto_import} -proto recgpt/v1/recommendation.proto -d '#{payload}' -format #{format} localhost:#{port} recgpt.v1.PredictionService/Predict"
    )

    Mix.shell().info("")

    case System.cmd("grpcurl", args, stderr_to_stdout: true) do
      {output, 0} ->
        IO.puts(output)

        if format == "json" do
          case Jason.decode(output) do
            {:ok, %{"item_ids" => ids, "items" => items}} when is_list(ids) and is_list(items) ->
              Mix.shell().info("")
              Mix.shell().info("OK: got #{length(ids)} item_ids, #{length(items)} items")

            {:ok, %{"itemIds" => ids, "items" => items}} when is_list(ids) and is_list(items) ->
              Mix.shell().info("")
              Mix.shell().info("OK: got #{length(ids)} item_ids, #{length(items)} items")

            _ ->
              :ok
          end
        end

        :ok

      {output, code} ->
        IO.puts(output)
        Mix.raise("grpcurl exited with #{code}. Is the server running? mix recgpt.serve ...")
    end
  end

  defp port_from_env do
    case System.get_env("RECGPT_GRPC_PORT") do
      nil ->
        nil

      s ->
        case Integer.parse(s) do
          {n, _} -> n
          :error -> nil
        end
    end
  end
end
