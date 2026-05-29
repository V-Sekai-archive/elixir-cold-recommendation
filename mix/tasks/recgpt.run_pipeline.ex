defmodule Mix.Tasks.Recgpt.RunPipeline do
  @shortdoc "Convert → build_fixture → pretrain in one command (e.g. MovieLens 20M)"
  @moduledoc """
  Runs the full RecGPT pipeline in one command: convert_trajectories → build_fixture → pretrain.

  Use this for MovieLens 20M (or other supported formats) so you don't have to run three
  separate tasks. Output directory holds items, sequences, fixture, and trained checkpoint.

  ## Usage

      mix recgpt.run_pipeline

  (With no args, uses --from ../recgpt-trajectories/movielens-20m and --out data/movielens if that dir exists.)

      mix recgpt.run_pipeline --from /path/to/movielens-20m --out data/movielens

  With Ecto (SQLite) and full split:

      mix recgpt.run_pipeline --from /path/to/movielens-20m --out data/movielens --sync-to-db --train-limit 0 --test-limit 0

  ## Options

    * `--from` - Input directory (e.g. path to movielens-20m). Default: ../recgpt-trajectories/movielens-20m when present.
    * `--out` - Output/data directory (default: data/movielens). Receives items, sequences, fixture.json, and ckpt/.
    * `--format` - movielens (default), ml1m, kuairand
    * `--train-limit` - Max train sequences (default: 0 = no cap). Set to limit subset.
    * `--test-limit` - Max test cases (default: 0 = no cap). Set to limit subset.
    * `--epochs` - Pretrain epochs (default: 5)
    * `--sync-to-db` - Sync convert output to SQLite; use db for items/train in build_fixture and pretrain.
    * `--ckpt` - Base checkpoint dir for build_fixture and pretrain (default: data/fuxi_ckpt_export)
    * `--seed` - Random seed for train/test split (default: 42)

  ## Steps performed

  0. If `--ckpt` dir has no manifest.json, raises: checkpoint required.
  1. Convert: writes items (and sequences) to `<out>/` (or DB if --sync-to-db).
  2. Build fixture: `<out>/fixture.json` from items (or db).
  3. Pretrain: writes checkpoint to `<out>/ckpt/`.
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          from: :string,
          out: :string,
          format: :string,
          train_limit: :integer,
          test_limit: :integer,
          epochs: :integer,
          sync_to_db: :boolean,
          ckpt: :string,
          seed: :integer
        ]
      )

    from_dir = opts[:from]
    out_dir = opts[:out] || "data/movielens"
    format = opts[:format] || "movielens"
    train_limit = opts[:train_limit] || 0
    test_limit = opts[:test_limit] || 0
    epochs = opts[:epochs] || 5
    sync_to_db = opts[:sync_to_db] || false
    ckpt_dir = opts[:ckpt] || "data/fuxi_ckpt_export"
    seed = opts[:seed] || 42

    # Default --from to sibling recgpt-trajectories/movielens-20m when not set
    from_dir =
      if from_dir && from_dir != "" do
        from_dir
      else
        Path.expand("../recgpt-trajectories/movielens-20m", File.cwd!())
      end

    unless File.dir?(from_dir) do
      hint =
        if from_dir =~ ~r|/path/to| do
          " Replace the placeholder with the real path, e.g. --from /FIRE202602/Experiments/recgpt-trajectories/movielens-20m or --from ../recgpt-trajectories/movielens-20m"
        else
          " Use a path containing ratings.csv and movies.csv (e.g. ../recgpt-trajectories/movielens-20m)."
        end

      Mix.raise("Input directory not found: #{from_dir}.#{hint}")
    end

    out_dir = Path.expand(out_dir, File.cwd!())
    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())
    File.mkdir_p!(out_dir)

    # 0. Ensure checkpoint exists (required for build_fixture and pretrain)
    manifest_path = Path.join(ckpt_dir, "manifest.json")

    unless File.regular?(manifest_path) do
      Mix.shell().info(
        "Step 0/3: Checkpoint missing at #{ckpt_dir}. Exporting FuXi-Linear init..."
      )

      Mix.Task.reenable("recgpt.export_fuxi_ckpt")
      Mix.Task.run("recgpt.export_fuxi_ckpt", ["--out", ckpt_dir])

      unless File.regular?(manifest_path) do
        Mix.raise(
          "Checkpoint export failed: #{manifest_path} not created. " <>
            "Run manually: mix recgpt.export_fuxi_ckpt --out #{ckpt_dir}"
        )
      end

      Mix.shell().info("Checkpoint ready.")
    end

    # 1. Convert
    Mix.shell().info("Step 1/3: Convert trajectories...")

    convert_args = [
      "--from",
      from_dir,
      "--out",
      out_dir,
      "--format",
      format,
      "--train-limit",
      to_string(train_limit),
      "--test-limit",
      to_string(test_limit),
      "--seed",
      to_string(seed)
    ]

    convert_args = if sync_to_db, do: convert_args ++ ["--sync-to-db"], else: convert_args

    Mix.Task.reenable("recgpt.convert_trajectories")
    Mix.Task.run("recgpt.convert_trajectories", convert_args)

    # 2. Build fixture
    Mix.shell().info("Step 2/3: Build fixture...")
    items_arg = if sync_to_db, do: "db", else: Path.join(out_dir, "items.json")
    fixture_path = Path.join(out_dir, "fixture.json")

    build_args = [
      "--items",
      items_arg,
      "--out",
      fixture_path,
      "--ckpt",
      ckpt_dir
    ]

    Mix.Task.reenable("recgpt.build_fixture")
    Mix.Task.run("recgpt.build_fixture", build_args)

    # 3. Pretrain
    Mix.shell().info("Step 3/3: Pretrain...")
    train_arg = if sync_to_db, do: "db", else: Path.join(out_dir, "train_sequences.json")

    pretrain_args = [
      "--ckpt",
      ckpt_dir,
      "--fixture",
      fixture_path,
      "--train",
      train_arg,
      "--items",
      items_arg,
      "--out",
      Path.join(out_dir, "ckpt"),
      "--epochs",
      to_string(epochs)
    ]

    Mix.Task.reenable("recgpt.pretrain")
    Mix.Task.run("recgpt.pretrain", pretrain_args)

    Mix.shell().info("Pipeline done. Checkpoint: #{Path.join(out_dir, "ckpt")}")
  end
end
