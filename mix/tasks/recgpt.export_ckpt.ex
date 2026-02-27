defmodule Mix.Tasks.Recgpt.ExportCkpt do
  @shortdoc "Export checkpoint to manifest.json + .npy (from existing export or from .pt)"
  @moduledoc """
  Writes an export directory (manifest.json + .npy files) for `RecGPT.CheckpointLoader.load_from_export/1`.

  ## From existing export (pure Elixir)
  Re-export or copy an existing export dir to another path (loads with CheckpointLoader, writes with CheckpointExport).

      mix recgpt.export_ckpt --from-export data/recgpt_ckpt_export --out data/recgpt_ckpt_export_copy

  ## From PyTorch .pt (pure Elixir)
  Converts a zip-format .pt file to export dir using Unpickler + Unzip (no Python).

      mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export

  ## Options
    * `--from-export` - Path to existing export dir (manifest + .npy)
    * `--from-pt` - Path to PyTorch .pt checkpoint (zip format, PyTorch 1.6+)
    * `--out` - Output export directory (required)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [from_export: :string, from_pt: :string, out: :string])

    out_dir = opts[:out]
    unless out_dir do
      Mix.raise("--out DIR is required")
    end

    cond do
      export_dir = opts[:from_export] ->
        export_from_export_dir(export_dir, out_dir)

      pt_path = opts[:from_pt] ->
        export_from_pt(pt_path, out_dir)

      true ->
        Mix.raise("Use --from-export DIR or --from-pt PATH to specify the source")
    end
  end

  defp export_from_export_dir(source_dir, out_dir) do
    unless File.dir?(source_dir) do
      Mix.raise("Source export dir not found: #{source_dir}")
    end

    Application.ensure_all_started(:nx)
    Mix.shell().info("Loading checkpoint from #{source_dir}...")
    params = RecGPT.CheckpointLoader.load_from_export(source_dir)
    Mix.shell().info("Writing export to #{out_dir} (#{map_size(params)} tensors)...")
    :ok = RecGPT.CheckpointExport.write_export(params, out_dir)
    Mix.shell().info("Done.")
  end

  defp export_from_pt(pt_path, out_dir) do
    unless File.regular?(pt_path) do
      Mix.raise("PyTorch checkpoint not found: #{pt_path}")
    end

    pt_path = Path.expand(pt_path)
    out_dir = Path.expand(out_dir)
    File.mkdir_p!(out_dir)

    Application.ensure_all_started(:nx)
    Mix.shell().info("Loading .pt from #{pt_path}...")
    params = RecGPT.PtLoader.load!(pt_path)
    Mix.shell().info("Writing export to #{out_dir} (#{map_size(params)} tensors)...")
    :ok = RecGPT.CheckpointExport.write_export(params, out_dir)
    Mix.shell().info("Done.")
  end
end
