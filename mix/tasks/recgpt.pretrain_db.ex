defmodule Mix.Tasks.Recgpt.PretrainDb do
  @shortdoc "Pretrain from DB for 5 epochs (constant memory)"
  @moduledoc """
  Runs pretraining with train sequences and items from the database (SQLite).
  Uses constant memory: streams sequences from DB per batch; supports multiple epochs.

  Equivalent to:
    mix recgpt.pretrain --train db --items db --epochs 5 --out <dir> [options]

  Requires RECGPT_SQLITE_PATH and a fixture (from `mix recgpt.build_fixture` with DB).

  ## Options
    * `--out` - Output export dir (required)
    * `--epochs` - Full passes over training data (default: 5)
    * `--ckpt` - Checkpoint dir (default: data/fuxi_ckpt_export or artifact)
    * `--fixture` - Fixture JSON path (default: data/steam/fixture.json or artifact)
    * `--save-every` - Save checkpoint every N steps (0 = disable)
    * `--eval-test-every` - Compute test loss every N steps (requires --test)
    * `--test` - Path to test_sequences.json
    * `--batch-size` - Batch size (default: 8)
    * `--learning-rate` - Learning rate (default: 1.0e-4)
    * `--log` - Log every N batches (default: 50)
    * `--log-interval-sec` - Log at least every N seconds (default: 20)
    * `--limit` - Max items to use (default: fixture num_items)
    * `--mtp-loss-weight` - MTP loss weight (default: 1.0)

  ## Example (all options)

      mix recgpt.pretrain_db \\
        --out data/pretrain_out \\
        --epochs 5 \\
        --ckpt data/fuxi_ckpt_export \\
        --fixture data/steam/fixture.json \\
        --save-every 500 \\
        --eval-test-every 200 \\
        --test data/steam/test_sequences.json \\
        --batch-size 8 \\
        --learning-rate 1.0e-4 \\
        --log 50 \\
        --log-interval-sec 20 \\
        --limit 10000 \\
        --mtp-loss-weight 1.0
  """
  use Mix.Task

  @impl true
  def run(args) do
    # Inject --train db --items db and default --epochs 5, then delegate to pretrain
    injected = ["--train", "db", "--items", "db", "--epochs", "5" | args]
    Mix.Task.run("recgpt.pretrain", injected)
  end
end
