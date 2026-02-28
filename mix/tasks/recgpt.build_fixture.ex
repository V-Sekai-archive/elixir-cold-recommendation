defmodule Mix.Tasks.Recgpt.BuildFixture do
  @shortdoc "Build fixture.json from items.json (Embedding + FSQ → token_id_list)"
  @moduledoc """
  Reads items.json, encodes item text via Embedding, quantizes with FSQ, writes fixture.json.

  Required for eval and serve after Fetch. Pipeline: Fetch → build_fixture → pretrain → eval.

  ## Options
    * `--items` - Path to items.json (default: data/steam/items.json)
    * `--out` - Output fixture path (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--fsq` - FSQ params export dir (required if FSQ not in --ckpt)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [items: :string, out: :string, ckpt: :string, fsq: :string]
      )

    items_path = opts[:items] || resolve("data/steam/items.json")
    out_path = opts[:out] || resolve("data/steam/fixture.json")
    ckpt_dir = opts[:ckpt] || resolve("data/recgpt_ckpt_export")
    fsq_dir = opts[:fsq]

    unless File.regular?(items_path) do
      Mix.raise(
        "items file not found: #{items_path}. Run Fetch first (e.g. mix recgpt.fetch_steam data/steam)."
      )
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise(
        "checkpoint not found: #{ckpt_dir}. Export a checkpoint first (mix recgpt.export_ckpt)."
      )
    end

    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:bumblebee)

    Mix.shell().info("Building fixture from #{items_path}...")
    fixture_opts = if fsq_dir, do: [fsq_dir: resolve(fsq_dir)], else: []
    fixture = RecGPT.FixtureBuild.build(items_path, ckpt_dir, fixture_opts)
    :ok = RecGPT.FixtureBuild.write_fixture(fixture, out_path)
    Mix.shell().info("Wrote #{out_path} (num_items=#{fixture["num_items"]})")
  end

  defp resolve(path) do
    if absolute_path?(path), do: path, else: Path.expand(path, File.cwd!())
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/
end
