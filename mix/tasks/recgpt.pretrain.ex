defmodule Mix.Tasks.Recgpt.Pretrain do
  @shortdoc "Pretrain on train_sequences with fixture and checkpoint; write updated export"
  @moduledoc """
  Loads checkpoint, train_sequences.json, fixture (token_id_list), and items;
  runs AxonTrain.stream_batches + AxonTrain.run; always writes updated params to --out.

  ## Options
    * `--ckpt` - Checkpoint export dir (default: data/fuxi_ckpt_export)
    * `--fixture` - Fixture JSON path (default: data/steam/fixture.json)
    * `--train` - train_sequences.json path (default: data/steam/train_sequences.json)
    * `--items` - items.json for embeddings (default: data/steam/items.json)
    * `--out` - Output export dir (required)
    * `--epochs` - Number of full passes (default: 1)
    * `--batch-size` - Batch size (default: 8)
    * `--learning-rate` - Learning rate (default: 1.0e-4)
    * `--log` - Log every N batches (default: 50)
    * `--mtp-loss-weight` - Weight for MTP loss (default: 1.0). Set 0 to use only shifted CE.
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          ckpt: :string,
          fixture: :string,
          train: :string,
          items: :string,
          out: :string,
          epochs: :integer,
          batch_size: :integer,
          learning_rate: :float,
          log: :integer,
          mtp_loss_weight: :float
        ]
      )

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_PATH") ||
        Path.join([File.cwd!(), "data", "fuxi_ckpt_export"])

    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        Path.join([File.cwd!(), "data", "steam", "fixture.json"])

    fixture_path = Path.expand(fixture_path, File.cwd!())

    train_path =
      case opts[:train] do
        nil -> Path.join([File.cwd!(), "data", "steam", "train_sequences.json"])
        s when is_binary(s) -> Path.expand(s, File.cwd!())
      end

    items_path =
      case opts[:items] do
        nil -> Path.join([File.cwd!(), "data", "steam", "items.json"])
        "" -> Path.join([File.cwd!(), "data", "steam", "items.json"])
        s when is_binary(s) -> Path.expand(s, File.cwd!())
      end

    out_dir = Keyword.fetch!(opts, :out)
    out_dir = Path.expand(out_dir, File.cwd!())

    unless File.regular?(fixture_path) do
      Mix.raise("Fixture not found: #{fixture_path}")
    end

    unless File.regular?(train_path) do
      Mix.raise("Train sequences not found: #{train_path}")
    end

    unless File.regular?(items_path) do
      Mix.raise("Items not found: #{items_path}")
    end

    RecGPT.PretrainRunner.run(
      ckpt_dir: ckpt_dir,
      fixture_path: fixture_path,
      train_path: train_path,
      items_path: items_path,
      out_dir: out_dir,
      epochs: opts[:epochs] || 1,
      batch_size: opts[:batch_size] || 8,
      learning_rate: opts[:learning_rate] || 1.0e-4,
      log: opts[:log] || 50,
      mtp_loss_weight: opts[:mtp_loss_weight] || 1.0
    )

    Mix.shell().info("Pretraining complete. Output: #{out_dir}")
  end
end
