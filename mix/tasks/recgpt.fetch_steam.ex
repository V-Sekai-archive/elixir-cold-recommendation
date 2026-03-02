defmodule Mix.Tasks.Recgpt.FetchSteam do
  @shortdoc "Fetch Steam test data from HuggingFace and write recgpt JSON to data/steam"
  @moduledoc """
  Downloads the Steam test split from [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) (test/steam)
  and writes recgpt JSON: items.json, train_sequences.json, test_sequences.json,
  cold_test_sequences.json, cold_train_sequences.json.

  Uses Unpickler to parse Python pickle files (no Python required).

  ## Usage

      mix recgpt.fetch_steam
      mix recgpt.fetch_steam data/steam

  Output directory defaults to `data/steam`. Next: build fixture, then pretrain, then eval.
  """
  use Mix.Task

  alias RecGPT.Steam.Fetch

  @impl true
  def run(args) do
    Application.ensure_all_started(:recgpt)
    out_dir = List.first(args) || "data/steam"

    case Fetch.run(out_dir) do
      :ok ->
        Mix.shell().info(
          "Done. Next: mix recgpt.build_fixture --items #{out_dir}/items.json --out #{out_dir}/fixture.json --ckpt data/recgpt_ckpt_export, then pretrain and eval."
        )

        Mix.shell().info(
          "Or run the full first-step baseline (fetch + build fixture with dataset embeddings + eval): mix recgpt.first_step (requires checkpoint; see docs/24_first_step_plan.md)."
        )

      {:error, reason} ->
        Mix.raise("Steam fetch failed: #{inspect(reason)}")
    end
  end
end
