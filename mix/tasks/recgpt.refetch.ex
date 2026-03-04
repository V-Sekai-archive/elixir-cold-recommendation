defmodule Mix.Tasks.Recgpt.Refetch do
  @shortdoc "Refetch all bulk data: checkpoint, VAE, Steam (recreates data/ + thirdparty/checkpoints)"
  @moduledoc """
  Runs all fetch tasks in order so bulk data is reproducible from a clean clone.
  Does not run build_fixture or eval — use `mix recgpt.first_step` after refetch
  for the full pipeline.

  ## Order
  1. `mix recgpt.fetch_ckpt` — RecGPT .pt to thirdparty/checkpoints/recgpt/
  2. `mix recgpt.export_ckpt` — .pt → manifest.json + *.npy
  3. `mix recgpt.fetch_vae_ckpt` — VAE to thirdparty/checkpoints/vae/
  4. `mix recgpt.fetch_steam` — Steam data to data/steam/

  ## Options
    * `--force` - Remove existing data/steam and thirdparty/checkpoints before refetch
    * `--steam-dir` - Steam output dir (default: data/steam)
    * `--ckpt-dir` - Checkpoint dir (default: thirdparty/checkpoints/recgpt)

  ## Examples
      mix recgpt.refetch
      mix recgpt.refetch --force
      mix recgpt.refetch --steam-dir data/steam --ckpt-dir data/recgpt_ckpt_export
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [force: :boolean, steam_dir: :string, ckpt_dir: :string]
      )

    cwd = File.cwd!()
    ckpt_dir = opts[:ckpt_dir] || Path.join([cwd, "thirdparty", "checkpoints", "recgpt"])
    steam_dir = opts[:steam_dir] || Path.join([cwd, "data", "steam"])
    vae_dir = Path.join([cwd, "thirdparty", "checkpoints", "vae"])

    if opts[:force] do
      Mix.shell().info("--force: removing existing bulk data...")
      for path <- [steam_dir, ckpt_dir, vae_dir] do
        if File.exists?(path) do
          File.rm_rf!(path)
          Mix.shell().info("  removed #{path}")
        end
      end
    end

    force_args = if opts[:force], do: ["--force"], else: []

    Mix.shell().info("")
    Mix.shell().info("Step 1/4: fetch_ckpt")
    Mix.Task.reenable("recgpt.fetch_ckpt")
    Mix.Task.run("recgpt.fetch_ckpt", ["--out", Path.join(ckpt_dir, "recgpt_layer_3_weight.pt") | force_args])

    Mix.shell().info("")
    Mix.shell().info("Step 2/4: export_ckpt")
    pt_path = Path.join(ckpt_dir, "recgpt_layer_3_weight.pt")
    Mix.Task.reenable("recgpt.export_ckpt")
    Mix.Task.run("recgpt.export_ckpt", ["--from-pt", pt_path, "--out", ckpt_dir])

    Mix.shell().info("")
    Mix.shell().info("Step 3/4: fetch_vae_ckpt")
    Mix.Task.reenable("recgpt.fetch_vae_ckpt")
    Mix.Task.run("recgpt.fetch_vae_ckpt", force_args)

    Mix.shell().info("")
    Mix.shell().info("Step 4/4: fetch_steam")
    Mix.Task.reenable("recgpt.fetch_steam")
    Mix.Task.run("recgpt.fetch_steam", [steam_dir])

    Mix.shell().info("")
    Mix.shell().info("Refetch complete. Next: mix recgpt.first_step (or build_fixture + pretrain + eval)")
  end
end
