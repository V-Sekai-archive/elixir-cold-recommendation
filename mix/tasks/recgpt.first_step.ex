defmodule Mix.Tasks.Recgpt.FirstStep do
  @shortdoc "Run the first step (Steam baseline): fetch → build_fixture → eval"
  @moduledoc """
  Runs the full first step:
  1. Fetch Steam data to steam dir.
  2. Build fixture with canonical texts (Elixir Bumblebee + VAE FSQ; semantic IDs match released model).
  3. Run eval in **Elixir** (RecGPT.Serve + RecGPT.Eval) and print metrics.

  **Prerequisites:** Checkpoint export (manifest + .npy), VAE checkpoint, canonical_item_texts in SQLite.
  Run `mix recgpt.refetch` for FuXi (default) or `mix recgpt.refetch --gpt2` for GPT-2.

  ## Options

    * `--steam-dir` - Directory for Steam data and fixture (default: data/steam)
    * `--ckpt` - Checkpoint export dir (default: data/fuxi_ckpt_export). Use --ckpt for GPT-2.
    * `--vae-ckpt` - Path to VAE checkpoint .pt (optional; env RECGPT_VAE_CKPT)
    * `--skip-fetch` - Use existing steam dir; do not run fetch_steam
    * `--skip-build` - Use existing fixture; do not run build_fixture
    * `--limit` - Max items for build_fixture (default: 10000). Omit or set high for full Steam split.

  ## Examples

      mix recgpt.first_step
      mix recgpt.first_step --steam-dir data/steam --ckpt data/fuxi_ckpt_export
      mix recgpt.first_step --skip-fetch --skip-build
  """
  use Mix.Task

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

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_PATH") ||
        Path.join([File.cwd!(), "data", "fuxi_ckpt_export"])

    vae_ckpt = opts[:vae_ckpt] || System.get_env("RECGPT_VAE_CKPT")
    steam_dir = Path.expand(steam_dir, File.cwd!())
    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())
    vae_ckpt_path = if vae_ckpt, do: Path.expand(vae_ckpt, File.cwd!()), else: nil

    ensure_checkpoint!(ckpt_dir)

    unless opts[:skip_fetch] do
      run_fetch(steam_dir)
    end

    unless opts[:skip_build] do
      ensure_canonical_texts!()
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
      Run: mix recgpt.export_fuxi_ckpt --out #{ckpt_dir}
      Or for GPT-2: mix recgpt.fetch_ckpt
           mix recgpt.export_ckpt --from-pt <pt> --out <dir>
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

  defp ensure_canonical_texts! do
    Application.ensure_all_started(:recgpt)

    case RecGPT.Steam.CanonicalItemText.load_from_repo(RecGPT.Repo) do
      [] ->
        Mix.raise("""
        canonical_item_texts is empty. Run once:
          mix ecto.migrate
          uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl --verify
        (Fetch must have run first so item_text_dict.pkl exists.)
        """)

      _ ->
        :ok
    end
  end

  defp run_build_fixture(steam_dir, ckpt_dir, limit, vae_ckpt_path) do
    Mix.shell().info("Step 2: Build fixture (canonical texts + VAE FSQ)...")
    items_path = Path.join(steam_dir, "items.json")
    out_path = Path.join(steam_dir, "fixture.json")

    unless File.regular?(items_path) do
      Mix.raise("items.json not found at #{items_path}. Run without --skip-fetch first.")
    end

    argv =
      ["--items", items_path, "--out", out_path, "--ckpt", ckpt_dir]
      |> maybe_append(
        "--vae-ckpt",
        vae_ckpt_path,
        is_binary(vae_ckpt_path) and vae_ckpt_path != ""
      )
      |> maybe_append("--limit", to_string(limit || 10_000), true)

    Mix.Task.reenable("recgpt.build_fixture")
    Mix.Task.run("recgpt.build_fixture", argv)
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
              metrics = RecGPT.Eval.evaluate(state, cases, top_k: 10, total: n)
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
