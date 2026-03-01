defmodule Mix.Tasks.Recgpt.CompareEmbeddings do
  @shortdoc "Compare our embeddings to dataset item_text_embeddings.npy"
  @moduledoc """
  Loads the dataset's item_text_embeddings.npy (downloads from HuggingFace if missing),
  generates our Bumblebee embeddings for the same items from items.json, and reports
  cosine similarity (mean, min, max, std). Optionally reports FSQ token agreement.

  Use to see how bad the embedding mismatch is (e.g. if mean cos_sim < 0.95, that
  likely explains why eval does not beat random with the released checkpoint).

  ## Options
    * `--steam-dir` - Directory with items.json (default: data/steam)
    * `--limit` - Max items to compare (default: 500)
    * `--text-format` - `recgpt_item_text` (default) or `title_only` (plain title; try if reference encoded title only)
    * `--ckpt` - Checkpoint export dir; if set, also report FSQ token agreement
    * `--dump-row` - Row index to dump (e.g. 0) as raw float32 for Python sanity check
    * `--dump-path` - Path for dump (default: item{N}_elixir.raw when --dump-row set)

  ## Examples
      mix recgpt.compare_embeddings
      mix recgpt.compare_embeddings --limit 100 --ckpt data/recgpt_ckpt_export
      mix recgpt.compare_embeddings --text-format title_only --limit 100
      mix recgpt.compare_embeddings --limit 1 --dump-row 0
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          steam_dir: :string,
          limit: :integer,
          text_format: :string,
          ckpt: :string,
          dump_row: :integer,
          dump_path: :string
        ]
      )

    steam_dir = opts[:steam_dir] || Path.expand("data/steam", File.cwd!())
    limit = opts[:limit] || 500
    text_format =
      case opts[:text_format] do
        "title_only" -> :title_only
        _ -> :recgpt_item_text
      end

    compare_opts = [limit: limit, text_format: text_format]
    compare_opts = if ckpt = opts[:ckpt], do: Keyword.put(compare_opts, :ckpt_dir, ckpt), else: compare_opts
    compare_opts = if row = opts[:dump_row], do: Keyword.put(compare_opts, :dump_row, row), else: compare_opts
    compare_opts = if path = opts[:dump_path], do: Keyword.put(compare_opts, :dump_path, path), else: compare_opts

    RecGPT.EmbeddingCompare.run(steam_dir, compare_opts)
  end
end
