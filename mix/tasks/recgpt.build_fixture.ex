defmodule Mix.Tasks.Recgpt.BuildFixture do
  @shortdoc "Build fixture.json from items.json (Embedding + FSQ → token_id_list)"
  @moduledoc """
  Reads items.json, encodes item text via Embedding, quantizes with FSQ, writes fixture.json.

  Required for eval and serve after Fetch. Pipeline: Fetch → build_fixture → pretrain → eval.

  When RECGPT_SQLITE_PATH is set, also flushes items, embeddings, tokens, and train/test
  sequences to SQLite (ETNF tables). Run `mix ecto.migrate` once before using SQLite.

  ## Options
    * `--items` - Path to items.json (default: data/steam/items.json)
    * `--out` - Output fixture path (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: thirdparty/checkpoints/recgpt)
    * `--vae-ckpt` - Path to VAE .pt (e.g. vae_len4_fsq88865_ep90.pt). Use so FSQ token_id_list matches the Python pipeline (required for all-Elixir parity). Env: RECGPT_VAE_CKPT.
    * `--embeddings-npy` - Use this item_text_embeddings.npy from the dataset instead of encoding with Bumblebee (ensures token_id_list matches the released checkpoint)
    * `--limit` - Max items to process (default: 100; do not exceed per run to avoid NIF issues)
    * `--ramp` - Slowly increase limit from --ramp-start until all items or failure
    * `--ramp-start` - First limit when using --ramp (default: 100)
    * `--ramp-step` - Linear step size (default: 100). Ignored if --ramp-mult is set.
    * `--ramp-mult` - Multiplier for geometric progression (e.g. 2 → 100, 200, 400, 800, ...). Overrides linear step.
    * `--ramp-max` - Stop ramp at this limit (default: all items in items.json)
  """
  use Mix.Task

  @default_limit 100

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          items: :string,
          out: :string,
          ckpt: :string,
          vae_ckpt: :string,
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
    items_path = opts[:items] || resolve("data/steam/items.json")
    out_path = opts[:out] || resolve("data/steam/fixture.json")
    ckpt_dir = opts[:ckpt] || resolve("thirdparty/checkpoints/recgpt")
    start_limit = opts[:ramp_start] || @default_limit
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
    task_opts = if path = opts[:embeddings_npy], do: Keyword.put(task_opts, :embeddings_npy, path), else: task_opts
    task_opts = if path = opts[:vae_ckpt] || System.get_env("RECGPT_VAE_CKPT"), do: Keyword.put(task_opts, :vae_ckpt, Path.expand(path, File.cwd!())), else: task_opts
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
    opts = if path = task_opts[:embeddings_npy], do: Keyword.put(opts, :embeddings_npy, path), else: opts
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
    items_path = opts[:items] || resolve("data/steam/items.json")
    out_path = opts[:out] || resolve("data/steam/fixture.json")
    ckpt_dir = opts[:ckpt] || resolve("thirdparty/checkpoints/recgpt")
    limit = opts[:limit] || @default_limit

    unless File.regular?(items_path) do
      Mix.raise(
        "items file not found: #{items_path}. Run Fetch first (e.g. mix recgpt.fetch_steam data/steam)."
      )
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise(
        "checkpoint not found: #{ckpt_dir}. Export a checkpoint first (mix recgpt.export_ckpt)."
      )
    end

    Application.ensure_all_started(:nx)
    unless opts[:embeddings_npy], do: Application.ensure_all_started(:bumblebee)

    npy_note = if opts[:embeddings_npy], do: " (using dataset embeddings)", else: ""
    Mix.shell().info(
      "Building fixture from #{items_path}#{if limit, do: " (limit #{limit})", else: ""}#{npy_note}..."
    )

    build_opts = [limit: limit]
    build_opts = if path = opts[:embeddings_npy], do: Keyword.put(build_opts, :embeddings_npy, path), else: build_opts
    build_opts = if path = opts[:vae_ckpt] || System.get_env("RECGPT_VAE_CKPT"), do: Keyword.put(build_opts, :vae_ckpt, Path.expand(path, File.cwd!())), else: build_opts
    build_opts =
      if System.get_env("RECGPT_SQLITE_PATH"), do: Keyword.put(build_opts, :sqlite, true), else: build_opts

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
