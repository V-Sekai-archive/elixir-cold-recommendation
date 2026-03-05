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
    * `--from` - Input directory (e.g. movielens-20m/ or path to it). Required.
    * `--out` - Output directory (default: data/training_signal_test)
    * `--format` - movielens, kuairand, jon_becker (default: jon_becker)
    * `--train-limit` - Max train sequences (default: 10000). Use 0 for no cap.
    * `--test-limit` - Max test cases (default: 2000). Use 0 for no cap.
    * `--seed` - Random seed for reproducible split (default: 42)
    * `--sync-to-db` - Sync items and sequences to SQLite (ETNF tables). Requires RECGPT_SQLITE_PATH. Run mix ecto.migrate first.
      Required for Jon-Becker (Phase 1). Skips writing train sequence JSON; always writes items.json and test_sequences.json (for --eval-test-every).
    * `--no-fetch-titles` - Skip Polymarket API title lookup (Jon-Becker only). Use "market {asset_id}" fallback.
    * `--max-api-requests` - Max Polymarket API requests when fetching titles (default: 500).
    * `--api-delay-ms` - Delay between API requests in ms; exponential backoff on 429 (default: 200).

  ## MovieLens-20M
  Expects ratings.csv (userId, movieId, rating, timestamp) and movies.csv (movieId, title, genres)
  in the input directory. Download from https://grouplens.org/datasets/movielens/20m/

  ## KuaiRand-Pure
  Expects log_standard_*.csv, log_random_*.csv (user_id, video_id, time_ms) and optionally
  video_features_basic_pure.csv in thirdparty/KuaiRand-Pure/. Download from https://kuairand.com/

  ## Jon-Becker (Polymarket)
  Expects polymarket/{markets,trades,blocks}/ Parquet from prediction-market-analysis.
  Use the git submodule:
    git submodule update --init thirdparty/prediction-market-analysis
    cd thirdparty/prediction-market-analysis && make setup   # ~36 GiB
    RECGPT_SQLITE_PATH=priv/recgpt.sqlite3 mix recgpt.convert_trajectories --from thirdparty/prediction-market-analysis --out data/polymarket --format jon_becker --sync-to-db
  Fetches item metadata from Polymarket Gamma API; produces stable JCS canonical JSON-LD for embedding text (see docs/features/92_polymarket_semantic_source.md). Use --no-fetch-titles for fallback (canonical placeholder).
  Withholds test sequences (20%% wallets) and test items/markets (15%% never in train).
  Phase 1 requires --sync-to-db. See docs/features/93_pretraining_plan.md.

  ## Next steps
  After conversion (Jon-Becker uses DB-backed pipeline):
    mix recgpt.build_fixture --items db --out <out>/fixture.json --ckpt data/fuxi_ckpt_export
    mix recgpt.pretrain --ckpt data/fuxi_ckpt_export --fixture <out>/fixture.json --train db --items db --out <out>/ckpt_pretrained --epochs 5 --eval-test-every 50 --test <out>/test_sequences.json
    mix recgpt.eval --data-dir <out> --ckpt <out>/ckpt_pretrained
  For MovieLens/KuaiRand (file-based), use --items <out>/items.json and --train <out>/train_sequences.json.
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
          sync_to_db: :boolean,
          no_fetch_titles: :boolean,
          max_api_requests: :integer,
          api_delay_ms: :integer
        ]
      )

    from_dir = opts[:from]
    out_dir = opts[:out] || "data/training_signal_test"
    format = parse_format(opts[:format])
    train_limit = opts[:train_limit] || 10_000
    test_limit = opts[:test_limit] || 2_000
    seed = opts[:seed] || 42
    sync_to_db = opts[:sync_to_db] || false

    if format == :jon_becker and !sync_to_db do
      Mix.raise(
        "Phase 1 (Jon-Becker) requires --sync-to-db. See docs/features/93_pretraining_plan.md."
      )
    end

    fetch_titles_from_api = !opts[:no_fetch_titles]
    max_api_requests = opts[:max_api_requests] || 500
    api_delay_ms = opts[:api_delay_ms] || 200

    if !from_dir || from_dir == "" do
      Mix.raise("--from DIR is required. Example: --from /path/to/movielens-20m")
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
           sync_to_db: sync_to_db,
           fetch_titles_from_api: fetch_titles_from_api,
           max_api_requests: max_api_requests,
           api_delay_ms: api_delay_ms
         ) do
      :ok ->
        next =
          if sync_to_db do
            "Done. Next: mix recgpt.build_fixture --items db --out #{out_dir}/fixture.json"
          else
            "Done. Next: mix recgpt.build_fixture --items #{out_dir}/items.json --out #{out_dir}/fixture.json"
          end

        Mix.shell().info(next)

      {:error, reason} ->
        Mix.raise("Conversion failed: #{inspect(reason)}")
    end
  end

  defp parse_format(nil), do: :jon_becker
  defp parse_format("movielens"), do: :movielens
  defp parse_format("kuairand"), do: :kuairand
  defp parse_format("jon_becker"), do: :jon_becker

  defp parse_format(other),
    do: Mix.raise("Unknown format: #{inspect(other)}. Use movielens, kuairand, or jon_becker.")
end
