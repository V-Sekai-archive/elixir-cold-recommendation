defmodule Mix.Tasks.Recgpt.FirstStep do
  @shortdoc "Run the first step (Steam baseline): fetch → build_fixture → eval"
  @moduledoc """
  Runs the full first step from [docs/24_first_step_plan.md](docs/24_first_step_plan.md):
  1. Fetch Steam data to steam dir (and download item_text_embeddings.npy if missing).
  2. Build fixture with dataset embeddings (Elixir; for catalog/fixture artifacts).
  3. Run eval in **Elixir** (RecGPT.Serve + RecGPT.Eval) and print metrics.

  **Prerequisite:** Checkpoint must exist (e.g. from `mix recgpt.fetch_ckpt` and
  `mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export`).

  ## Options

    * `--steam-dir` - Directory for Steam data and fixture (default: data/steam)
    * `--ckpt` - Checkpoint export dir or .pt path (default: data/recgpt_ckpt_export)
    * `--vae-ckpt` - Path to VAE checkpoint .pt (optional; env RECGPT_VAE_CKPT)
    * `--skip-fetch` - Use existing steam dir; do not run fetch_steam
    * `--skip-build` - Use existing fixture; do not run build_fixture
    * `--limit` - Max items for build_fixture (default: 10000). Omit or set high for full Steam split.

  ## Examples

      mix recgpt.first_step
      mix recgpt.first_step --steam-dir data/steam --ckpt data/recgpt_ckpt_export
      mix recgpt.first_step --skip-fetch --skip-build
  """
  use Mix.Task

  @dataset_npy_url "https://huggingface.co/datasets/hkuds/RecGPT_dataset/resolve/main/test/steam/item_text_embeddings.npy"

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          steam_dir: :string,
          ckpt: :string,
          vae_ckpt: :string,
          skip_fetch: :boolean,
          skip_build: :boolean,
          limit: :integer
        ]
      )

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    steam_dir = opts[:steam_dir] || "data/steam"
    ckpt_dir = opts[:ckpt] || Path.join([File.cwd!(), "thirdparty", "checkpoints", "recgpt"])
    vae_ckpt = opts[:vae_ckpt] || System.get_env("RECGPT_VAE_CKPT")
    steam_dir = Path.expand(steam_dir, File.cwd!())
    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())
    vae_ckpt_path = if vae_ckpt, do: Path.expand(vae_ckpt, File.cwd!()), else: nil

    ensure_checkpoint!(ckpt_dir)

    unless opts[:skip_fetch] do
      run_fetch(steam_dir)
      ensure_npy!(steam_dir)
    end

    unless opts[:skip_build] do
      run_build_fixture(steam_dir, ckpt_dir, opts[:limit], vae_ckpt_path)
    end

    run_eval(steam_dir, ckpt_dir)

    Mix.shell().info("")
    Mix.shell().info("First step complete. Baseline recorded; next steps replanned later.")
  end

  defp ensure_checkpoint!(ckpt_dir) do
    manifest = Path.join(ckpt_dir, "manifest.json")
    unless File.dir?(ckpt_dir) and File.regular?(manifest) do
      Mix.raise("""
      Checkpoint required at #{ckpt_dir}.
      Run: mix recgpt.fetch_ckpt
           mix recgpt.export_ckpt --from-pt #{Path.join(ckpt_dir, "recgpt_layer_3_weight.pt")} --out #{ckpt_dir}
      """)
    end
  end

  defp run_fetch(steam_dir) do
    Mix.shell().info("Step 1: Fetch Steam data to #{steam_dir}...")
    case RecGPT.Steam.Fetch.run(steam_dir) do
      :ok -> :ok
      {:error, reason} -> Mix.raise("Fetch failed: #{inspect(reason)}")
    end
  end

  defp ensure_npy!(steam_dir) do
    npy_path = Path.join(steam_dir, "item_text_embeddings.npy")
    if File.regular?(npy_path) do
      :ok
    else
      Mix.shell().info("Downloading item_text_embeddings.npy (~92 MB) to #{steam_dir}...")
      Application.ensure_all_started(:req)
      case Req.get(@dataset_npy_url) do
        {:ok, %{status: 200, body: body}} ->
          File.mkdir_p!(steam_dir)
          File.write!(npy_path, body)
          Mix.shell().info("Wrote #{npy_path}")

        {:ok, %{status: code}} ->
          Mix.raise("HTTP #{code} for #{@dataset_npy_url}")

        {:error, reason} ->
          Mix.raise("Download failed: #{inspect(reason)}")
      end
    end
  end

  defp run_build_fixture(steam_dir, ckpt_dir, limit, vae_ckpt_path) do
    Mix.shell().info("Step 2: Build fixture (dataset embeddings)...")
    items_path = Path.join(steam_dir, "items.json")
    out_path = Path.join(steam_dir, "fixture.json")
    npy_path = Path.join(steam_dir, "item_text_embeddings.npy")

    unless File.regular?(items_path) do
      Mix.raise("items.json not found at #{items_path}. Run without --skip-fetch first.")
    end

    argv =
      ["--items", items_path, "--out", out_path, "--ckpt", ckpt_dir]
      |> maybe_append("--embeddings-npy", npy_path, File.regular?(npy_path))
      |> maybe_append("--vae-ckpt", vae_ckpt_path, is_binary(vae_ckpt_path) and vae_ckpt_path != "")
      |> maybe_append("--limit", to_string(limit || 10000), true)

    Mix.Task.reenable("recgpt.build_fixture")
    Mix.Task.run("recgpt.build_fixture", argv)

    unless File.regular?(npy_path) do
      Mix.shell().info("  (item_text_embeddings.npy not found; build used Bumblebee encoder.)")
    end
  end

  defp maybe_append(argv, _key, _value, false), do: argv
  defp maybe_append(argv, key, value, true), do: argv ++ [key, value]

  defp run_eval(steam_dir, ckpt_dir) do
    Mix.shell().info("Step 3: Run eval (Elixir)...")
    fixture_path = Path.join(steam_dir, "fixture.json")
    test_path = Path.join(steam_dir, "test_sequences.json")

    unless File.regular?(fixture_path) do
      Mix.raise("fixture.json not found at #{fixture_path}. Run without --skip-build first.")
    end

    unless File.regular?(test_path) do
      Mix.raise("test_sequences.json not found at #{test_path}. Run without --skip-fetch first.")
    end

    case RecGPT.Serve.load_state(fixture_path, ckpt_dir, nil) do
      {:ok, state} ->
        case RecGPT.Eval.load_test_cases(test_path) do
          {:ok, cases} ->
            cases = RecGPT.Eval.filter_to_catalog(cases, state.num_items)
            n = length(cases)
            if n == 0 do
              Mix.shell().info("No test cases in catalog range.")
            else
              metrics = RecGPT.Eval.evaluate(state, cases, [top_k: 10, total: n])
              Mix.shell().info("Evaluation (Elixir RecGPT)")
              Mix.shell().info("  n = #{metrics[:n]}")
              Mix.shell().info("  hit_at_1 = #{Float.round(metrics[:hit_at_1] || 0, 4)}")
              Mix.shell().info("  hit_at_5 = #{Float.round(metrics[:hit_at_5] || 0, 4)}")
              Mix.shell().info("  hit_at_10 = #{Float.round(metrics[:hit_at_10] || 0, 4)}")
              Mix.shell().info("  mrr = #{Float.round(metrics[:mrr] || 0, 4)}")
              Mix.shell().info("  rejects_null = #{metrics[:rejects_null]}")
            end
          {:error, reason} ->
            Mix.raise("Eval failed (load test cases): #{inspect(reason)}")
        end
      {:error, reason} ->
        Mix.raise("Eval failed (load state): #{inspect(reason)}")
    end
  end
end
