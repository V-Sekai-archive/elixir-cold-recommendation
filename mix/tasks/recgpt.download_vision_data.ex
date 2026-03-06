defmodule Mix.Tasks.Recgpt.DownloadVisionData do
  @shortdoc "Download image-caption dataset and compute DINOv2 + MPNet embeddings (.npy)"
  @moduledoc """
  Runs the Python script that downloads a Hugging Face dataset (flickr30k or AniGamePersonaCaps),
  runs DINOv2 on images and MPNet on captions, and saves vision_768.npy and text_768.npy.

  Requires: uv (or python with deps), and datasets/transformers/sentence-transformers.

  After this, run: mix recgpt.train_vision_contrastive --dataset-dir data/vision_contrastive --out priv/vision_proj_ckpt

  ## Options
    * `--dataset` - flickr30k (default) or anigame
    * `--limit` - Max samples (default: 5000). Use 0 for full dataset.
    * `--out` - Output directory (default: data/vision_contrastive)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [dataset: :string, limit: :integer, out: :string]
      )

    dataset = Keyword.get(opts, :dataset, "flickr30k")
    limit = Keyword.get(opts, :limit, 5000)
    out = Keyword.get(opts, :out, "data/vision_contrastive")

    root = File.cwd!()
    script = Path.join(root, "scripts/download_vision_contrastive_data.py")
    unless File.regular?(script) do
      Mix.raise("Script not found: #{script}")
    end

    argv = ["--dataset", dataset, "--limit", to_string(limit), "--out", out]
    Mix.shell().info("Running: uv run python scripts/download_vision_contrastive_data.py #{Enum.join(argv, " ")}")

    {output, status} = System.cmd("uv", ["run", "python", script | argv], cd: root)
    IO.write(output)
    if status != 0 do
      Mix.raise("Download script exited with #{status}")
    end

    Mix.shell().info("Embeddings saved to #{out}. Train with: mix recgpt.train_vision_contrastive --dataset-dir #{out} --out priv/vision_proj_ckpt")
  end
end
