defmodule Mix.Tasks.Recgpt.CkptSha256 do
  @shortdoc "Compute SHA256 of checkpoint for integrity verification"
  @moduledoc """
  Computes a deterministic SHA256 hash of the checkpoint export directory
  (manifest.json + all .npy files in sorted order). Use this to get the expected
  hash, then set config :recgpt, :ckpt_expected_sha256 or RECGPT_CKPT_SHA256.

  ## Options
    * `--ckpt` - Checkpoint export directory (default: data/recgpt_ckpt_export)

  ## Examples
      mix recgpt.ckpt_sha256
      mix recgpt.ckpt_sha256 --ckpt data/recgpt_ckpt_export

  ## Config
  Add to config/config.exs (or config/runtime.exs):
      config :recgpt, :ckpt_expected_sha256, "abc123..."

  Or set RECGPT_CKPT_SHA256 env var.
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [ckpt: :string])

    ckpt =
      opts[:ckpt] ||
        Path.expand("data/recgpt_ckpt_export", File.cwd!())

    manifest_path = Path.join(ckpt, "manifest.json")

    unless File.regular?(manifest_path) do
      Mix.raise("manifest.json not found at #{manifest_path}")
    end

    manifest = File.read!(manifest_path) |> Jason.decode!()
    hash = RecGPT.CheckpointLoader.compute_sha256(ckpt, manifest)

    Mix.shell().info("Checkpoint SHA256: #{hash}")
    Mix.shell().info("")
    Mix.shell().info("Add to config:")
    Mix.shell().info("  config :recgpt, :ckpt_expected_sha256, \"#{hash}\"")
  end
end
