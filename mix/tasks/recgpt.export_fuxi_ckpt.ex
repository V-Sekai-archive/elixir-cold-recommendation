defmodule Mix.Tasks.Recgpt.ExportFuxiCkpt do
  @shortdoc "Export FuXi-Linear init params to manifest.json + .npy"
  @moduledoc """
  Exports FuXi-Linear init params to an export directory for `mix recgpt.serve`
  or pretraining. Uses FuxiLinearInference.init_full_params/1 (no trained checkpoint required).

  Use for: testing FuXi serve path, training from scratch, or CI.

  ## Example

      mix recgpt.export_fuxi_ckpt --out data/fuxi_ckpt_export

  ## Options

    * `--out` - Output export directory (required)
    * `--n-blocks` - Number of FuXi blocks (default 4)
    * `--max-seq-len` - Max sequence length for positional emb (default 1024)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [out: :string, n_blocks: :integer, max_seq_len: :integer])

    out_dir = opts[:out]
    n_blocks = opts[:n_blocks] || 4
    max_seq_len = opts[:max_seq_len] || 1024

    unless out_dir do
      Mix.raise("--out DIR is required")
    end

    out_dir = Path.expand(out_dir)
    File.mkdir_p!(out_dir)

    Application.ensure_all_started(:nx)
    # Use BinaryBackend for init (no GPU needed; saves to disk)
    prev = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)

    Mix.shell().info("Initializing FuXi-Linear params (n_blocks: #{n_blocks}, max_seq_len: #{max_seq_len})...")
    params = RecGPT.FuxiLinearInference.init_full_params(n_blocks: n_blocks, max_seq_len: max_seq_len)

    Nx.default_backend(prev)

    Mix.shell().info("Writing export to #{out_dir} (#{map_size(params)} tensors)...")
    :ok = RecGPT.CheckpointExport.write_export(params, out_dir)
    Mix.shell().info("Done. Serve with: mix recgpt.serve --fixture <path> --ckpt #{out_dir}")
  end
end
