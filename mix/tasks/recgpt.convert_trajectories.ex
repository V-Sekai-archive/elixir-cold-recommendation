defmodule Mix.Tasks.Recgpt.ConvertTrajectories do
  @shortdoc "Convert trajectory dataset (MovieLens, etc.) to RecGPT canonical JSON"
  @moduledoc """
  Converts raw trajectory datasets to RecGPT canonical format for pretrain and eval.

  Produces items.json, train_sequences.json, test_sequences.json,
  cold_test_sequences.json, cold_train_sequences.json in the output directory.

  ## Usage

      mix recgpt.convert_trajectories --from /path/to/movielens-20m --out data/training_signal_test
      mix recgpt.convert_trajectories --from /path/to/recgpt-trajectories/movielens-20m --out data/training_signal_test --format movielens

  ## Options
    * `--from` - Input directory (e.g. path to KuaiRand-Pure or movielens-20m). Required.
    * `--out` - Output directory (default: data/training_signal_test)
    * `--format` - kuairand (default), movielens, ml1m
    * `--train-limit` - Max train sequences (default: 0 = no cap). Set to limit subset.
    * `--test-limit` - Max test cases (default: 0 = no cap). Set to limit subset.
    * `--seed` - Random seed for reproducible split (default: 42)
    * `--sync-to-db` - Sync items and sequences to SQLite (ETNF tables). Requires RECGPT_SQLITE_PATH. Run mix ecto.migrate first.

  ## MovieLens-20M
  Expects ratings.csv (userId, movieId, rating, timestamp) and movies.csv (movieId, title, genres)
  in the input directory. Download from https://grouplens.org/datasets/movielens/20m/

  ## MovieLens 1M (ml1m) – title + genres in item descriptions
  Expects .dat files: ratings.dat (UserID::MovieID::Rating::Timestamp), movies.dat (MovieID::Title::Genres).
  Item titles are joined with pipe-separated genres so categories are filled. See https://files.grouplens.org/datasets/movielens/ml-1m-README.txt
  Example: --from /path/to/ml-1m --format ml1m

  ## KuaiRand-Pure (default)
  Expects log_standard_*.csv, log_random_*.csv (user_id, video_id, time_ms) and optionally
  video_features_basic_pure.csv in the --from directory. Example: --from thirdparty/KuaiRand-Pure
  or --from "C:\\Users\\...\\KuaiRand-Pure" (Windows). Download from https://kuairand.com/

  ## Next steps
  After conversion: build_fixture then pretrain (see docs/features/93_pretraining_plan.md).
  Use --items <out>/items.json --train <out>/train_sequences.json for build_fixture and pretrain.
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
          seed: :integer,
          sync_to_db: :boolean
        ]
      )

    from_dir = opts[:from]
    out_dir = opts[:out] || "data/training_signal_test"
    format = parse_format(opts[:format])
    train_limit = opts[:train_limit] || 0
    test_limit = opts[:test_limit] || 0
    seed = opts[:seed] || 42
    sync_to_db = opts[:sync_to_db] || false

    if !from_dir || from_dir == "" do
      Mix.raise("--from DIR is required. Example: --from /path/to/KuaiRand-Pure")
    end

    from_dir = Path.expand(from_dir, File.cwd!())
    out_dir = Path.expand(out_dir, File.cwd!())

    unless File.dir?(from_dir) do
      Mix.raise("Input directory not found: #{from_dir}")
    end

    Mix.shell().info("Converting #{format} from #{from_dir} to #{out_dir}...")

    case RecGPT.Trajectories.Convert.run(from_dir, out_dir,
           format: format,
           train_limit: train_limit,
           test_limit: test_limit,
           seed: seed,
           sync_to_db: sync_to_db
         ) do
      :ok ->
        next =
          if sync_to_db do
            "Done. Next: mix recgpt.fetch_vae_ckpt  then  mix recgpt.build_fixture --items db --out #{Path.join(out_dir, "fixture.json")} --ckpt data/fuxi_ckpt_export --vae-ckpt thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt"
          else
            "Done. Next: mix recgpt.fetch_vae_ckpt  then  mix recgpt.build_fixture --items #{Path.join(out_dir, "items.json")} --out #{Path.join(out_dir, "fixture.json")} --ckpt data/fuxi_ckpt_export --vae-ckpt thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt --no-canonical-texts"
          end

        Mix.shell().info(next)

      {:error, reason} ->
        Mix.raise("Conversion failed: #{inspect(reason)}")
    end
  end

  defp parse_format(nil), do: :kuairand
  defp parse_format("movielens"), do: :movielens
  defp parse_format("ml1m"), do: :ml1m
  defp parse_format("kuairand"), do: :kuairand

  defp parse_format(other),
    do: Mix.raise("Unknown format: #{inspect(other)}. Use movielens, ml1m, or kuairand.")
end
