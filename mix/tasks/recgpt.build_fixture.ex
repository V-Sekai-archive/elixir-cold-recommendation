defmodule Mix.Tasks.Recgpt.BuildFixture do
  @shortdoc "Build fixture.json from items.json (Embedding + FSQ → token_id_list)"
  @moduledoc """
  Reads items.json, encodes item text via Embedding, quantizes with FSQ, writes fixture.json.

  Required for eval and serve after Fetch. Pipeline: Fetch → build_fixture → pretrain → eval.

  **Checkpoints:** FSQ (semantic IDs) comes from the VAE checkpoint. Run `mix recgpt.fetch_vae_ckpt`
  once to download it; then use `--vae-ckpt` or set RECGPT_VAE_CKPT. The embedder (sentence-transformers)
  downloads automatically on first use.

  When RECGPT_SQLITE_PATH is set, also flushes items, embeddings, tokens, and train/test
  sequences to SQLite (ETNF tables). Run `mix ecto.migrate` once before using SQLite.

  ## Options
    * `--items` - Path to items.json (default: data/steam/items.json). Use `db` to load from SQLite.
    * `--out` - Output fixture path (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: data/fuxi_ckpt_export).
    * `--vae-ckpt` - Path to VAE .pt (e.g. vae_len4_fsq88865_ep90.pt). FSQ from VAE is required for correct token_id_list. Run `mix recgpt.fetch_vae_ckpt` first. Env: RECGPT_VAE_CKPT. If unset, tries thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt and data/vae_len4_fsq88865_ep90.pt.
    * `--canonical-texts` - Use item texts from canonical_item_texts table (default: on). Run mix recgpt.dump_canonical_texts first. Use `--no-canonical-texts` to use items.json text instead.
    * `--canonical-texts-from` - Use enriched item texts from a JSON file (e.g. from mix recgpt.kuairand_canonical_texts). File must have "by_item_id": [str0, str1, ...]. Overrides --canonical-texts / --no-canonical-texts.
    * `--embeddings-npy` - Use this item_text_embeddings.npy from the dataset instead of encoding with Bumblebee (ensures token_id_list matches the released checkpoint)
    * `--limit` - Max items to process. Default: all items (no limit). Use --limit N to cap.
    * `--ramp` - Slowly increase limit from --ramp-start until all items or failure
    * `--ramp-start` - First limit when using --ramp (default: 100)
    * `--ramp-step` - Linear step size (default: 100). Ignored if --ramp-mult is set.
    * `--ramp-mult` - Multiplier for geometric progression (e.g. 2 → 100, 200, 400, 800, ...). Overrides linear step.
    * `--ramp-max` - Stop ramp at this limit (default: all items in items.json)
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          items: :string,
          out: :string,
          ckpt: :string,
          vae_ckpt: :string,
          canonical_texts: :boolean,
          no_canonical_texts: :boolean,
          canonical_texts_from: :string,
          embeddings_npy: :string,
          limit: :integer,
          ramp: :boolean,
          ramp_start: :integer,
          ramp_step: :integer,
          ramp_max: :integer,
          ramp_mult: :integer
        ]
      )

    if opts[:ramp] do
      run_ramp(opts)
    else
      run_once(opts)
    end
  end

  defp run_ramp(opts) do
    if opts[:items] == "db" do
      Mix.raise("--ramp is not supported with --items db. Use --items db without --ramp.")
    end

    Application.ensure_all_started(:recgpt)

    items_path =
      opts[:items] || RecGPT.Catalog.Artifact.resolve_path("items") ||
        resolve("data/steam/items.json")

    out_path =
      opts[:out] || RecGPT.Catalog.Artifact.resolve_path("fixture") ||
        resolve("data/steam/fixture.json")

    ckpt_dir =
      opts[:ckpt] || RecGPT.Catalog.Artifact.resolve_path("checkpoint") ||
        resolve("data/fuxi_ckpt_export")

    start_limit = opts[:ramp_start] || 100
    step = opts[:ramp_step] || 100
    mult = opts[:ramp_mult]

    unless File.regular?(items_path) do
      Mix.raise("items file not found: #{items_path}. Run Fetch first.")
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("checkpoint not found: #{ckpt_dir}. Export a checkpoint first.")
    end

    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:bumblebee)

    raw = File.read!(items_path) |> Jason.decode!()
    total = raw["num_items"] || length(raw["items"] || [])

    cap =
      if is_integer(opts[:ramp_max]) and opts[:ramp_max] > 0,
        do: min(opts[:ramp_max], total),
        else: total

    limits = ramp_limits(start_limit, step, cap, mult)
    Mix.shell().info("Ramping limit: #{inspect(limits)} toward #{cap} items...")

    task_opts = []
    canonical_texts? = !opts[:no_canonical_texts] and Keyword.get(opts, :canonical_texts, true)

    task_opts =
      if canonical_texts?, do: Keyword.put(task_opts, :canonical_texts, true), else: task_opts

    task_opts =
      case opts[:embeddings_npy] do
        nil -> task_opts
        path -> Keyword.put(task_opts, :embeddings_npy, path)
      end

    task_opts =
      case opts[:vae_ckpt] || System.get_env("RECGPT_VAE_CKPT") do
        nil -> task_opts
        path -> Keyword.put(task_opts, :vae_ckpt, Path.expand(path, File.cwd!()))
      end

    last_ok =
      Enum.reduce_while(limits, nil, fn limit, acc ->
        Mix.shell().info("Trying limit #{limit}...")

        if build_one(items_path, ckpt_dir, out_path, limit, task_opts) do
          {:cont, limit}
        else
          Mix.shell().error("Failed at limit #{limit}.")
          {:halt, acc}
        end
      end)

    if last_ok do
      Mix.shell().info("Done. Last successful limit: #{last_ok}. Fixture has #{last_ok} items.")
    end
  end

  defp ramp_limits(start, _step, cap, mult) when start <= cap and is_integer(mult) and mult > 1 do
    Stream.unfold(start, fn n ->
      if n <= cap do
        next = min(n * mult, cap)
        if next <= n, do: {n, nil}, else: {n, next}
      else
        nil
      end
    end)
    |> Enum.to_list()
  end

  defp ramp_limits(start, step, cap, _) when start <= cap do
    Stream.iterate(start, &(&1 + step)) |> Stream.take_while(&(&1 <= cap)) |> Enum.to_list()
  end

  defp build_one(items_path, ckpt_dir, out_path, limit, task_opts) do
    opts = [limit: limit]

    opts =
      if task_opts[:canonical_texts], do: Keyword.put(opts, :canonical_texts, true), else: opts

    opts =
      if path = task_opts[:embeddings_npy],
        do: Keyword.put(opts, :embeddings_npy, path),
        else: opts

    opts = if path = task_opts[:vae_ckpt], do: Keyword.put(opts, :vae_ckpt, path), else: opts

    opts =
      if System.get_env("RECGPT_SQLITE_PATH"), do: Keyword.put(opts, :sqlite, true), else: opts

    fixture = RecGPT.FixtureBuild.build(items_path, ckpt_dir, opts)
    RecGPT.FixtureBuild.write_fixture(fixture, out_path)
    Mix.shell().info("  Wrote #{out_path} (num_items=#{fixture["num_items"]})")
    true
  rescue
    e ->
      Mix.shell().error("  Error: #{Exception.message(e)}")
      false
  end

  defp run_once(opts) do
    Application.ensure_all_started(:recgpt)

    items_path =
      case opts[:items] do
        "db" -> :db
        nil -> RecGPT.Catalog.Artifact.resolve_path("items") || resolve("data/steam/items.json")
        "" -> RecGPT.Catalog.Artifact.resolve_path("items") || resolve("data/steam/items.json")
        s -> resolve(s)
      end

    out_path =
      opts[:out] || RecGPT.Catalog.Artifact.resolve_path("fixture") ||
        resolve("data/steam/fixture.json")

    ckpt_dir =
      opts[:ckpt] || RecGPT.Catalog.Artifact.resolve_path("checkpoint") ||
        resolve("data/fuxi_ckpt_export")

    # Default: all items (no limit). Use --limit N to cap.
    limit = Keyword.get(opts, :limit)

    unless items_path in [:db, "db"] or File.regular?(items_path) do
      Mix.raise(
        "items file not found: #{items_path}. Run Fetch first or use --items db after convert --sync-to-db."
      )
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise(
        "checkpoint not found: #{ckpt_dir}. Run mix recgpt.refetch or mix recgpt.export_fuxi_ckpt --out #{ckpt_dir}."
      )
    end

    canonical_texts_from =
      opts[:canonical_texts_from] && Path.expand(opts[:canonical_texts_from], File.cwd!())

    canonical_texts? = !opts[:no_canonical_texts] and Keyword.get(opts, :canonical_texts, true)
    Application.ensure_all_started(:nx)
    if canonical_texts? or canonical_texts_from, do: Application.ensure_all_started(:recgpt)
    unless opts[:embeddings_npy], do: Application.ensure_all_started(:bumblebee)

    npy_note = if opts[:embeddings_npy], do: " (using dataset embeddings)", else: ""

    canonical_note =
      if canonical_texts_from,
        do: " (canonical texts from file)",
        else: if(canonical_texts?, do: " (canonical texts from DB)", else: "")

    Mix.shell().info(
      "Building fixture from #{items_path}#{if limit, do: " (limit #{limit})", else: " (all items)"}#{npy_note}#{canonical_note}..."
    )

    build_opts = [limit: limit]

    build_opts =
      if canonical_texts_from,
        do: Keyword.put(build_opts, :canonical_texts_from, canonical_texts_from),
        else: build_opts

    build_opts =
      if canonical_texts? and !canonical_texts_from,
        do: Keyword.put(build_opts, :canonical_texts, true),
        else: build_opts

    build_opts =
      case opts[:embeddings_npy] do
        nil -> build_opts
        path -> Keyword.put(build_opts, :embeddings_npy, path)
      end

    build_opts =
      case opts[:vae_ckpt] || System.get_env("RECGPT_VAE_CKPT") do
        nil -> build_opts
        path -> Keyword.put(build_opts, :vae_ckpt, Path.expand(path, File.cwd!()))
      end

    build_opts =
      if System.get_env("RECGPT_SQLITE_PATH"),
        do: Keyword.put(build_opts, :sqlite, true),
        else: build_opts

    try do
      fixture = RecGPT.FixtureBuild.build(items_path, ckpt_dir, build_opts)
      :ok = RecGPT.FixtureBuild.write_fixture(fixture, out_path)
      Mix.shell().info("Wrote #{out_path} (num_items=#{fixture["num_items"]})")
    rescue
      e -> Mix.raise("Fixture build failed: #{Exception.message(e)}")
    end
  end

  defp resolve(path) do
    if absolute_path?(path), do: path, else: Path.expand(path, File.cwd!())
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/
end
