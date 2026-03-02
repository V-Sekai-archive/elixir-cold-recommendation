defmodule Mix.Tasks.Recgpt.GrpcCurlUpdateFixture do
  @shortdoc "Run grpcurl, save response to test fixture (updates test expected values)"
  @moduledoc """
  Calls the Predict RPC with grpcurl and writes the response to
  test/fixtures/steam_predict_grpcurl.json. Use this to update the test cases
  when the real catalogue or model changes.

  Start the server first in another terminal:
      mix recgpt.serve --fixture data/steam/fixture.json --ckpt data/recgpt_ckpt_export --catalog data/steam/items.json

  Then run:
      mix recgpt.grpc_curl_update_fixture

  The fixture file is used by the integration test "grpcurl Predict returns
  item_ids and items (real catalogue)" when present: the test asserts the
  server response matches the fixture (item_ids and display_name per item).

  ## Options
    * `--port` - gRPC port (default: 50051, or RECGPT_GRPC_PORT)
    * `--context` - Comma-separated context item IDs (default: 0)
    * `--max-results` - Max recommendations (default: 10)
    * `--out` - Output path (default: test/fixtures/steam_predict_grpcurl.json)
  """
  use Mix.Task

  @default_fixture_path "test/fixtures/steam_predict_grpcurl.json"

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [port: :integer, context: :string, max_results: :integer, out: :string]
      )

    port = opts[:port] || port_from_env() || 50_051
    context_str = opts[:context] || "0"
    max_results = opts[:max_results] || 10
    out_path = opts[:out] || Path.join(File.cwd!(), @default_fixture_path)
    out_path = Path.expand(out_path, File.cwd!())

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
      "json",
      "localhost:#{port}",
      "recgpt.v1.PredictionService/Predict"
    ]

    Mix.shell().info("Calling Predict via grpcurl (port #{port})...")

    case System.cmd("grpcurl", args, stderr_to_stdout: true) do
      {output, 0} ->
        case Jason.decode(output) do
          {:ok, %{"item_ids" => item_ids, "items" => items}}
          when is_list(item_ids) and is_list(items) ->
            items_norm = normalize_items(items)
            write_fixture(out_path, context_ids, max_results, item_ids, items_norm)

          {:ok, %{"itemIds" => item_ids, "items" => items}}
          when is_list(item_ids) and is_list(items) ->
            items_norm = normalize_items(items)
            write_fixture(out_path, context_ids, max_results, item_ids, items_norm)

          {:ok, _} ->
            Mix.raise("grpcurl response missing item_ids/items or wrong shape")

          {:error, _} ->
            Mix.raise("grpcurl output is not valid JSON: #{String.slice(output, 0, 200)}")
        end

      {output, code} ->
        IO.puts(output)
        Mix.raise("grpcurl exited with #{code}. Is the server running? mix recgpt.serve ...")
    end
  end

  defp normalize_items(items) do
    Enum.map(items, fn item ->
      %{
        "item_id" => item["itemId"] || item["item_id"],
        "display_name" => item["displayName"] || item["display_name"] || ""
      }
    end)
  end

  defp write_fixture(out_path, context_ids, max_results, item_ids, items) do
    fixture = %{
      "request" => %{
        "context_item_ids" => context_ids,
        "max_results" => max_results
      },
      "response" => %{
        "item_ids" => item_ids,
        "items" => items
      }
    }

    json = Jason.encode!(fixture, pretty: true)
    File.mkdir_p!(Path.dirname(out_path))
    File.write!(out_path, json)

    Mix.shell().info(
      "Wrote #{length(item_ids)} item_ids and #{length(items)} items to #{out_path}"
    )

    Mix.shell().info("Re-run integration tests to assert against this fixture.")
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
