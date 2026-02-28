defmodule Mix.Tasks.Recgpt.ExportCkpt do
  @shortdoc "Export checkpoint to manifest.json + .npy (from .pt)"
  @moduledoc """
  Converts a PyTorch .pt file to an export directory (manifest.json + .npy) for
  `RecGPT.CheckpointLoader.load_from_export/1`. Uses Unpickler + Unzip (no Python).

  ## Example

      mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export

  ## Options
    * `--from-pt` - Path to PyTorch .pt checkpoint (zip format, PyTorch 1.6+)
    * `--out` - Output export directory (required)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [from_pt: :string, out: :string])

    out_dir = opts[:out]
    pt_path = opts[:from_pt]

    unless out_dir do
      Mix.raise("--out DIR is required")
    end

    unless pt_path do
      Mix.raise("--from-pt PATH is required")
    end

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
