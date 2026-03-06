defmodule Mix.Tasks.Recgpt.TrainVisionContrastive do
  @shortdoc "Train vision projector with contrastive loss (synthetic or dataset dir)"
  @moduledoc """
  Trains RecGPT.VisionProjector with InfoNCE contrastive loss. DINOv2 and MPNet are frozen;
  only the projector is updated.

  By default uses synthetic random 768-d pairs. For real data, run
  `mix recgpt.download_vision_data` first, then pass `--dataset-dir`.

  ## Options
    * `--out` - Output checkpoint directory (required). Saved as manifest.json + .npy files.
    * `--dataset-dir` - Directory with vision_768.npy and text_768.npy (from download script). If set, training uses this data instead of synthetic.
    * `--steps` - Training steps (default: 500).
    * `--batch-size` - Batch size (default: 32).
    * `--lr` - Learning rate (default: 1.0e-4).
    * `--log-every` - Log loss every N steps (default: 50).
    * `--epochs` - Dataset epochs when using --dataset-dir (default: 1).
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          out: :string,
          dataset_dir: :string,
          steps: :integer,
          batch_size: :integer,
          lr: :float,
          log_every: :integer,
          epochs: :integer
        ]
      )

    out_dir = opts[:out]
    unless out_dir && out_dir != "" do
      Mix.raise("--out DIR is required (e.g. priv/vision_proj_ckpt)")
    end

    Application.ensure_all_started(:recgpt)

    steps = Keyword.get(opts, :steps, 500)
    batch_size = Keyword.get(opts, :batch_size, 32)
    lr = Keyword.get(opts, :lr, 1.0e-4)
    log_every = Keyword.get(opts, :log_every, 50)
    epochs = Keyword.get(opts, :epochs, 1)
    dataset_dir = opts[:dataset_dir]

    stream =
      if dataset_dir && dataset_dir != "" do
        Mix.shell().info("Loading batches from #{dataset_dir}")
        RecGPT.VisionContrastive.stream_from_dataset_dir(dataset_dir,
          batch_size: batch_size,
          shuffle: true,
          epochs: epochs
        )
      else
        nil
      end

    proj_params = RecGPT.VisionProjector.init_params()
    data_src = if stream, do: "dataset", else: "synthetic"
    Mix.shell().info("Training vision projector: steps=#{steps} batch=#{batch_size} lr=#{lr} (#{data_src})")

    train_opts = [
      steps: steps,
      batch_size: batch_size,
      learning_rate: lr,
      log_every: log_every
    ]
    train_opts = if stream, do: Keyword.put(train_opts, :stream, stream), else: train_opts

    trained = RecGPT.VisionContrastive.run(proj_params, train_opts)

    File.mkdir_p!(out_dir)
    RecGPT.CheckpointExport.write_export(trained, out_dir)
    Mix.shell().info("Saved projector checkpoint to #{out_dir}")
  end
end
