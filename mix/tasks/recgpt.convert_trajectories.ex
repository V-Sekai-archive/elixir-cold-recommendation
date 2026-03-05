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
    * `--format` - movielens (default). KuaiRand, merrec planned.
    * `--train-limit` - Max train sequences (default: 10000). Use 0 for no cap.
    * `--test-limit` - Max test cases (default: 2000). Use 0 for no cap.
    * `--seed` - Random seed for reproducible split (default: 42)

  ## MovieLens-20M
  Expects ratings.csv (userId, movieId, rating, timestamp) and movies.csv (movieId, title, genres)
  in the input directory. Download from https://grouplens.org/datasets/movielens/20m/

  ## Next steps
  After conversion, run:
    mix recgpt.build_fixture --items <out>/items.json --out <out>/fixture.json
    mix recgpt.pretrain --ckpt thirdparty/checkpoints/recgpt --fixture <out>/fixture.json --train <out>/train_sequences.json --items <out>/items.json --out <out>/ckpt_pretrained --iterations 500
    mix recgpt.eval --data-dir <out> --ckpt thirdparty/checkpoints/recgpt
    mix recgpt.eval --data-dir <out> --ckpt <out>/ckpt_pretrained
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
          seed: :integer
        ]
      )

    from_dir = opts[:from]
    out_dir = opts[:out] || "data/training_signal_test"
    format = parse_format(opts[:format])
    train_limit = opts[:train_limit] || 10_000
    test_limit = opts[:test_limit] || 2_000
    seed = opts[:seed] || 42

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
           seed: seed
         ) do
      :ok ->
        Mix.shell().info(
          "Done. Next: mix recgpt.build_fixture --items #{out_dir}/items.json --out #{out_dir}/fixture.json"
        )

      {:error, reason} ->
        Mix.raise("Conversion failed: #{inspect(reason)}")
    end
  end

  defp parse_format(nil), do: :movielens
  defp parse_format("movielens"), do: :movielens
  defp parse_format("kuairand"), do: :kuairand
  defp parse_format(other), do: Mix.raise("Unknown format: #{inspect(other)}. Use movielens.")
end
