defmodule Mix.Tasks.Recgpt.Refetch do
  @shortdoc "Refetch all bulk data: FuXi checkpoint, VAE, Steam (recreates data/ + thirdparty/checkpoints)"
  @moduledoc """
  Runs all fetch tasks in order so bulk data is reproducible from a clean clone.
  Does not run build_fixture or eval — use `mix recgpt.first_step` after refetch
  for the full pipeline.

  ## Order
  1. `mix recgpt.export_fuxi_ckpt` — FuXi-Linear init params to ckpt-dir
  2. `mix recgpt.fetch_vae_ckpt` — VAE to thirdparty/checkpoints/vae/
  3. `mix recgpt.fetch_steam` — Steam data to data/steam/

  ## Options
    * `--force` - Remove existing data/steam and checkpoint dir before refetch
    * `--steam-dir` - Steam output dir (default: data/steam)
    * `--ckpt-dir` - Checkpoint output dir (default: data/fuxi_ckpt_export)

  ## Examples
      mix recgpt.refetch
      mix recgpt.refetch --force
      mix recgpt.refetch --steam-dir data/steam --ckpt-dir data/fuxi_ckpt_export
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [force: :boolean, steam_dir: :string, ckpt_dir: :string]
      )

    cwd = File.cwd!()

    ckpt_dir =
      opts[:ckpt_dir] || Path.join([cwd, "data", "fuxi_ckpt_export"])

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
    Mix.shell().info("Step 1/3: export_fuxi_ckpt (FuXi-Linear init)")
    Mix.Task.reenable("recgpt.export_fuxi_ckpt")
    Mix.Task.run("recgpt.export_fuxi_ckpt", ["--out", ckpt_dir])

    Mix.shell().info("")
    Mix.shell().info("Step 2/3: fetch_vae_ckpt")
    Mix.Task.reenable("recgpt.fetch_vae_ckpt")
    Mix.Task.run("recgpt.fetch_vae_ckpt", force_args)

    Mix.shell().info("")
    Mix.shell().info("Step 3/3: fetch_steam")
    Mix.Task.reenable("recgpt.fetch_steam")
    Mix.Task.run("recgpt.fetch_steam", [steam_dir])

    Mix.shell().info("")

    Mix.shell().info(
      "Refetch complete. Next: mix recgpt.first_step (or build_fixture + pretrain + eval)"
    )
  end
end
