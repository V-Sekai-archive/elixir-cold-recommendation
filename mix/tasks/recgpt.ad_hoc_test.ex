defmodule Mix.Tasks.Recgpt.AdHocTest do
  @shortdoc "Run ad-hoc recommendation tests and print results"
  @moduledoc """
  Loads fixture + checkpoint (+ optional catalog), runs recommend for a few
  context sequences, and prints/writes results. Use to verify the recommend
  path end-to-end with real or stub data.

  ## Options
    * `--fixture` - Path to fixture JSON (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--catalog` - Optional path to items JSON for display_name in output
    * `--out` - Optional path to write results JSON (default: print only)
    * `--contexts` - Comma-separated context lists, e.g. "0,1" or "0,1|1,2,3" (default: 0 | 0,1 | 1,2,3)
    * `--top-k` - Max recommendations per context (default: 5)
    * `--stub` - Use stub state (no checkpoint); for quick smoke test only

  ## Examples
      mix recgpt.ad_hoc_test
      mix recgpt.ad_hoc_test --fixture data/steam/fixture.json --ckpt data/recgpt_ckpt_export --catalog data/steam/items.json
      mix recgpt.ad_hoc_test --out data/steam/ad_hoc_results.json
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _, _} =
      OptionParser.parse(args,
        switches: [
          fixture: :string,
          ckpt: :string,
          catalog: :string,
          out: :string,
          contexts: :string,
          top_k: :integer,
          stub: :boolean
        ]
      )

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    fixture_path =
      opts[:fixture] || System.get_env("RECGPT_FIXTURE") ||
        RecGPT.Catalog.Artifact.resolve_path("fixture") ||
        Path.join(File.cwd!(), "data/steam/fixture.json")

    fixture_path = Path.expand(fixture_path, File.cwd!())

    ckpt_dir =
      opts[:ckpt] || System.get_env("RECGPT_CKPT_EXPORT") ||
        RecGPT.Catalog.Artifact.resolve_path("checkpoint") ||
        Path.join(File.cwd!(), "data/recgpt_ckpt_export")

    ckpt_dir = Path.expand(ckpt_dir, File.cwd!())

    catalog_path =
      (opts[:catalog] && Path.expand(opts[:catalog], File.cwd!())) ||
        RecGPT.Catalog.Artifact.resolve_path("items")

    out_path = opts[:out] && Path.expand(opts[:out], File.cwd!())
    top_k = opts[:top_k] || 5

    contexts_str = opts[:contexts] || "0|0,1|1,2,3"
    context_list = parse_contexts(contexts_str)

    state =
      if opts[:stub] do
        Mix.shell().info("Using stub state (--stub)")
        build_stub_state()
      else
        unless File.regular?(fixture_path) do
          Mix.raise("Fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
        end

        unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
          Mix.raise("""
          Checkpoint not found: #{ckpt_dir}
          Run: mix recgpt.fetch_ckpt
               mix recgpt.export_ckpt --from-pt <path_to>.pt --out #{ckpt_dir}
          Or use --stub for a quick smoke test.
          """)
        end

        Mix.shell().info("Loading state (fixture=#{fixture_path}, ckpt=#{ckpt_dir})...")

        case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
          {:ok, s} -> s
          {:error, reason} -> Mix.raise("Failed to load state: #{inspect(reason)}")
        end
      end

    Mix.shell().info("Running recommend for #{length(context_list)} context(s), top_k=#{top_k}")
    results = run_recommendations(state, context_list, top_k)
    print_results(results, state)
    if out_path, do: write_results(out_path, results, state)
    Mix.shell().info("Done.")
  end

  defp parse_contexts(str) do
    str
    |> String.split("|", trim: true)
    |> Enum.map(fn part ->
      part
      |> String.split(",", trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.map(fn s ->
        case Integer.parse(s) do
          {n, _} -> n
          :error -> Mix.raise("Invalid context item id: #{inspect(s)}")
        end
      end)
    end)
    |> Enum.reject(&(&1 == []))
  end

  defp run_recommendations(state, context_list, top_k) do
    Enum.map(context_list, fn context ->
      case RecGPT.Serve.recommend(state, context, top_k) do
        {:ok, item_ids} -> %{context: context, item_ids: item_ids}
        {:error, reason} -> %{context: context, error: reason}
      end
    end)
  end

  defp display_name(state, id) do
    case Map.get(state.item_text, id) do
      t when is_binary(t) -> t
      m when is_map(m) -> m["title"] || m["name"] || to_string(id)
      _ -> to_string(id)
    end
  end

  defp print_results(results, state) do
    Mix.shell().info("")
    Mix.shell().info("=== Ad-hoc recommendation results ===")

    for %{context: ctx, item_ids: item_ids} <- results do
      Mix.shell().info("  context #{inspect(ctx)} -> #{inspect(item_ids)}")

      for id <- item_ids do
        Mix.shell().info("    #{id}: #{display_name(state, id)}")
      end
    end

    for %{context: ctx, error: reason} <- results do
      Mix.shell().info("  context #{inspect(ctx)} -> error: #{reason}")
    end

    Mix.shell().info("")
  end

  defp build_stub_state do
    alias RecGPT.Inference
    alias RecGPT.Serve
    alias RecGPT.Trie
    token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
    trie = Trie.build(token_id_list)
    wte = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    params = %{"wte" => wte, "pred_head.weight" => head_w, "pred_head.bias" => head_b}

    get_logits_fn = fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end

    %Serve{
      params: params,
      trie: trie,
      token_id_list: token_id_list,
      token_id_map: nil,
      item_text: %{},
      num_items: 2,
      get_logits_fn: get_logits_fn
    }
  end

  defp write_results(out_path, results, state) do
    payload =
      Enum.map(results, fn
        %{context: ctx, item_ids: item_ids} ->
          %{
            "context" => ctx,
            "item_ids" => item_ids,
            "items" =>
              Enum.map(item_ids, fn id ->
                %{"item_id" => id, "display_name" => display_name(state, id)}
              end)
          }

        %{context: ctx, error: reason} ->
          %{"context" => ctx, "error" => reason}
      end)

    json = Jason.encode!(payload, pretty: true)
    File.mkdir_p!(Path.dirname(out_path))
    File.write!(out_path, json)
    Mix.shell().info("Wrote results to #{out_path}")
  end
end
