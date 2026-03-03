defmodule Mix.Tasks.Recgpt.ConvertMovielens do
  @shortdoc "Convert MovieLens 20M CSV to RecGPT canonical JSON"
  @moduledoc """
  Converts MovieLens 20M CSV data into RecGPT canonical JSON artifacts.

  Reads ratings.csv and movies.csv from the source directory, builds sessions
  (per-user, timestamp-ordered), splits 80%% train / 20%% test (last-item-out),
  defines cold items (≤ K sessions in train), and writes:

  - items.json
  - train_sequences.json
  - test_sequences.json
  - cold_test_sequences.json
  - cold_train_sequences.json

  Next: mix recgpt.build_fixture → pretrain → eval.

  ## Usage

      mix recgpt.convert_movielens
      mix recgpt.convert_movielens --src thirdparty/recgpt-trajectories/movielens-20m --out data/movielens-20m
      mix recgpt.convert_movielens --max-items 10000

  ## Options

  * `--src` - Source directory containing ratings.csv and movies.csv (default: thirdparty/recgpt-trajectories/movielens-20m)
  * `--out` - Output directory for JSON artifacts (default: data/movielens-20m)
  * `--max-items` - Cap catalog size for faster iteration (default: no cap)
  * `--cold-k` - Items in ≤ K train sessions are cold (default: 2)
  * `--train-ratio` - Fraction of sessions for train, 0..1 (default: 0.8)
  """
  use Mix.Task

  alias RecGPT.MovieLens.Convert

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          src: :string,
          out: :string,
          max_items: :integer,
          cold_k: :integer,
          train_ratio: :float
        ]
      )

    src = opts[:src] || "thirdparty/recgpt-trajectories/movielens-20m"
    out = opts[:out] || "data/movielens-20m"

    convert_opts =
      []
      |> maybe_put(:max_items, opts[:max_items])
      |> maybe_put(:cold_k, opts[:cold_k])
      |> maybe_put(:train_ratio, opts[:train_ratio])

    case Convert.run(src, out, convert_opts) do
      :ok ->
        Mix.shell().info("")
        Mix.shell().info(
          "Done. Next: mix recgpt.build_fixture --items #{out}/items.json --out #{out}/fixture.json --ckpt data/recgpt_ckpt_export"
        )
        Mix.shell().info("Then: mix recgpt.pretrain and mix recgpt.eval")

      {:error, reason} ->
        Mix.raise("MovieLens convert failed: #{inspect(reason)}")
    end
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, val), do: Keyword.put(opts, key, val)
end
