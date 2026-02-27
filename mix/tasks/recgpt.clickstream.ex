defmodule Mix.Tasks.Recgpt.Clickstream do
  @shortdoc "Fetch UCI Clickstream, run migrations, write items.json + test_sequences.json"
  @moduledoc """
  One-shot PoC: download UCI Clickstream zip, run Ecto migrations, load into SQLite,
  write data/clickstream/items.json and test_sequences.json for eval.

  If you see "table already exists", delete data/clickstream/recgpt.db and run again.
  """
  use Mix.Task

  @impl true
  def run(args) do
    Application.ensure_all_started(:recgpt)
    data_dir = List.first(args) || "data/clickstream"

    case RecGPT.Clickstream.Fetch.run(data_dir) do
      :ok ->
        Mix.shell().info(
          "Done. Next: mix recgpt.build_fixture, then mix recgpt.pretrain, then mix recgpt.eval (requires --cold-test)"
        )

      {:error, reason} ->
        Mix.raise("Clickstream fetch failed: #{inspect(reason)}")
    end
  end
end
