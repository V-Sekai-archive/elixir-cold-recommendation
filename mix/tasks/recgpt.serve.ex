defmodule Mix.Tasks.Recgpt.Serve do
  @shortdoc "Start RecGPT recommendation HTTP server (port of serve.py)"
  @moduledoc """
  Start the RecGPT REST API server (Google API Design Guide).

  Loads checkpoint and fixture once. Serves only /v1/ endpoints:
  GET /v1/catalog/items, POST /v1/catalog:recommend, GET /v1/health.

  ## Options
    * `--port` - Port (default: 8000)
    * `--fixture` - Path to serve_e2e_fixture.json (default: data/serve_e2e_fixture.json)
    * `--ckpt` - Path to checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--catalog` - Optional path to catalog JSON (item_id -> text)

  ## Environment
    * RECGPT_FIXTURE - override fixture path (e.g. path to fixture from M:\\reflex-logic-other\\data)
    * RECGPT_CKPT_EXPORT - override checkpoint export dir

  ## Examples
      mix recgpt.serve
      mix recgpt.serve --port 8080
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [port: :integer, fixture: :string, ckpt: :string, catalog: :string]
      )

    port = opts[:port] || 8000

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        resolve_path("data/serve_e2e_fixture.json")

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_EXPORT") ||
        resolve_path("data/recgpt_ckpt_export")

    catalog_path = opts[:catalog]

    Application.ensure_all_started(:nx)

    Mix.shell().info("Loading model and fixture...")

    case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
      {:ok, state} ->
        Application.put_env(:recgpt, :serve_state, state)
        Mix.shell().info("REST API: http://0.0.0.0:#{port}/v1/")
        Mix.shell().info("  GET  /v1/catalog/items?q=...&pageSize=20")

        Mix.shell().info(
          "  POST /v1/catalog:recommend  body: {\"context_item_ids\": [0,1], \"max_results\": 5}"
        )

        Mix.shell().info("  GET  /v1/health")

        spec = {Plug.Cowboy, scheme: :http, plug: RecGPT.Serve.Plug, options: [port: port]}
        {:ok, _pid} = Supervisor.start_link([spec], strategy: :one_for_one)
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
