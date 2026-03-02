defmodule Mix.Tasks.Recgpt.DumpCanonicalTexts do
  @shortdoc "Dump canonical item texts from pkl to SQLite (RecGPT-official bytes, BLOB)"
  @moduledoc """
  Builds item text strings (Python str(dict).replace('{','').replace('}','')) from
  item_text_dict.pkl and stores them in canonical_item_texts (BLOB). Run after
  mix recgpt.fetch_steam; ensure mix ecto.migrate and RECGPT_SQLITE_PATH (or default DB).

  For byte-exact match with the official pipeline, prefer the Python export so both
  inputs (build_fixture and compare_embeddings) come from the same source:

      uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl --verify

  That writes to the same DB/table; Elixir then reads it (canonical-texts is on by default).

  ## Options

    * `--pkl` - Path to item_text_dict.pkl (default: data/steam/item_text_dict.pkl)

  ## Usage

      mix recgpt.dump_canonical_texts
      mix recgpt.dump_canonical_texts --pkl data/steam/item_text_dict.pkl
  """
  use Mix.Task

  alias RecGPT.Steam.CanonicalItemText
  alias RecGPT.Repo

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args, switches: [pkl: :string])

    pkl_path = opts[:pkl] || Path.expand("data/steam/item_text_dict.pkl", File.cwd!())

    unless File.regular?(pkl_path) do
      Mix.raise("item_text_dict.pkl not found at #{pkl_path}. Run mix recgpt.fetch_steam first.")
    end

    Application.ensure_all_started(:recgpt)

    Mix.shell().info("Building canonical strings from #{pkl_path}...")
    ordered = CanonicalItemText.build_ordered_list(pkl_path)
    n = length(ordered)
    Mix.shell().info("Dumping #{n} items to canonical_item_texts (BLOB)...")
    CanonicalItemText.dump_to_repo(Repo, ordered)
    Mix.shell().info("Done. Use --canonical-texts when running build_fixture or compare_embeddings.")
  end
end
