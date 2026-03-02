defmodule Mix.Tasks.Recgpt.TracePredict do
  @shortdoc "Trace one recommendation call; print timing breakdown for optimization"
  @moduledoc """
  Runs a single Predict-style recommendation and prints a timing breakdown so you
  can see where time is spent and what to optimize.

  Loads state (fixture + checkpoint + optional catalog), runs one recommend with
  the given context and top_k, and reports:
  - context_to_tokens_us: building context token sequence from item IDs
  - beam_search_us: total time in beam search (4 steps)
  - inference_us: time inside model forward (get_logits_fn) — usually the bottleneck
  - inference_calls: number of forward passes (beam_width × 4 steps)
  - response_build_us: building item_ids + display_name list
  - total_us

  ## Options
    * `--fixture` - Path to fixture JSON (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--catalog` - Optional path to items JSON
    * `--context` - Comma-separated context item IDs (default: 0)
    * `--top-k` - Max recommendations (default: 10)
    * `--stub` - Use stub state (fast, no real inference)

  ## Examples
      mix recgpt.trace_predict
      mix recgpt.trace_predict --context "0,1" --top-k 10
      mix recgpt.trace_predict --stub
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
          context: :string,
          top_k: :integer,
          stub: :boolean
        ]
      )

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

    fixture_path =
      opts[:fixture] ||
        Path.join(File.cwd!(), "data/steam/fixture.json")
        |> Path.expand(File.cwd!())

    ckpt_dir =
      opts[:ckpt] ||
        Path.join(File.cwd!(), "data/recgpt_ckpt_export")
        |> Path.expand(File.cwd!())

    catalog_path = opts[:catalog] && Path.expand(opts[:catalog], File.cwd!())
    context_str = opts[:context] || "0"
    top_k = opts[:top_k] || 10

    context_ids =
      context_str
      |> String.split(",", trim: true)
      |> Enum.map(&String.trim/1)
      |> Enum.map(fn s ->
        case Integer.parse(s) do
          {n, _} -> n
          :error -> Mix.raise("Invalid context id: #{inspect(s)}")
        end
      end)

    state =
      if opts[:stub] do
        Mix.shell().info("Using stub state (--stub)")
        build_stub_state(10)
      else
        unless File.regular?(fixture_path) do
          Mix.raise("Fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
        end

        unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
          Mix.raise("Checkpoint not found: #{ckpt_dir}. Run mix recgpt.export_ckpt first.")
        end

        Mix.shell().info("Loading state...")

        case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
          {:ok, s} -> s
          {:error, reason} -> Mix.raise("Load failed: #{inspect(reason)}")
        end
      end

    # Accumulator for inference time and call count: {total_us, count}
    {:ok, agent} = Agent.start_link(fn -> {0, 0} end)
    original_fn = state.get_logits_fn

    traced_fn = fn token_list ->
      {us, result} = :timer.tc(fn -> original_fn.(token_list) end)
      Agent.update(agent, fn {total, n} -> {total + us, n + 1} end)
      result
    end

    traced_batch_fn =
      if state.get_logits_batch_fn do
        orig_batch = state.get_logits_batch_fn

        fn list_of_lists, cache ->
          {us, result} = :timer.tc(fn -> orig_batch.(list_of_lists, cache) end)
          Agent.update(agent, fn {total, n} -> {total + us, n + 1} end)
          result
        end
      else
        single = state.get_logits_fn

        fn list_of_lists, _cache ->
          {us, result} =
            :timer.tc(fn ->
              logits =
                list_of_lists
                |> Enum.map(fn seq -> single.(seq) |> Nx.squeeze(axes: [0]) end)
                |> Nx.stack(axis: 0)

              {logits, nil}
            end)

          Agent.update(agent, fn {total, n} -> {total + us, n + 1} end)
          result
        end
      end

    traced_state = %{state | get_logits_fn: traced_fn, get_logits_batch_fn: traced_batch_fn}

    Mix.shell().info("Tracing one recommend: context=#{inspect(context_ids)}, top_k=#{top_k}")
    Mix.shell().info("")

    # Phase 1: context to token IDs
    {context_us, context_token_ids} =
      :timer.tc(fn ->
        RecGPT.Serve.item_ids_to_context_token_ids(context_ids, traced_state)
      end)

    # Phase 2: beam search (batched path when get_logits_batch_fn is set)
    {beam_search_us, result} =
      :timer.tc(fn ->
        RecGPT.Decode.beam_search_top_k(
          traced_state.get_logits_fn,
          traced_state.trie,
          context_token_ids,
          top_k,
          traced_state.get_logits_batch_fn
        )
      end)

    {inference_us, inference_calls} = Agent.get(agent, fn {t, n} -> {t, n} end)
    Agent.stop(agent)

    item_ids =
      case result do
        {:ok, ids} -> ids
        :not_found -> []
      end

    # Phase 3: build response (item_ids + display_name)
    {response_us, _items} =
      :timer.tc(fn ->
        Enum.map(item_ids, fn id ->
          name =
            case Map.get(state.item_text, id) do
              t when is_binary(t) -> t
              m when is_map(m) -> m["title"] || m["name"] || to_string(id)
              _ -> to_string(id)
            end

          {id, name}
        end)
      end)

    total_us = context_us + beam_search_us + response_us

    # Print results
    Mix.shell().info("=== Recommendation result ===")
    Mix.shell().info("  item_ids: #{inspect(item_ids)}")
    Mix.shell().info("")
    Mix.shell().info("=== Timing (one call) ===")
    Mix.shell().info("  context_to_tokens:  #{context_us} μs")
    Mix.shell().info("  beam_search_total:   #{beam_search_us} μs")
    Mix.shell().info("  inference (forward): #{inference_us} μs  (#{inference_calls} calls)")
    Mix.shell().info("  response_build:     #{response_us} μs")

    Mix.shell().info(
      "  total:              #{total_us} μs  (#{Float.round(total_us / 1000, 2)} ms)"
    )

    Mix.shell().info("")

    if inference_calls > 0 do
      avg_inference_us = div(inference_us, inference_calls)
      pct = Float.round(100.0 * inference_us / total_us, 1)
      Mix.shell().info("  inference % of total: #{pct}%  (avg #{avg_inference_us} μs/forward)")

      Mix.shell().info(
        "  → Beam search uses batched path (#{inference_calls} forward passes). Speed up: GPU, KV-cache."
      )
    end

    Mix.shell().info("")
  end

  @padding_id 15_360

  defp build_stub_state(num_items) when num_items >= 1 do
    alias RecGPT.Inference
    alias RecGPT.Serve
    alias RecGPT.Trie

    token_id_list =
      Enum.map(0..(num_items - 1), fn i ->
        [100 + i, 200 + i, 300 + i, 400 + i]
      end)

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

    # One batched forward per step instead of N; ~8–10x faster than fallback (map + stack).
    get_logits_batch_fn = fn list_of_token_lists, _cache ->
      max_len = list_of_token_lists |> Enum.map(&length/1) |> Enum.max()

      padded =
        Enum.map(list_of_token_lists, fn tokens ->
          len = length(tokens)
          padding = List.duplicate(@padding_id, max_len - len)
          padding ++ tokens
        end)

      batch = Nx.tensor(padded, type: {:s, 32})
      {batch_size, seq_len} = Nx.shape(batch)
      batch_aux = Nx.broadcast(0.0, {batch_size, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {batch_size, seq_len, 1}) |> Nx.as_type({:f, 32})
      logits = Inference.forward(batch, batch_aux, embed_mask, params)
      {logits, nil}
    end

    %Serve{
      params: params,
      trie: trie,
      token_id_list: token_id_list,
      token_id_map: nil,
      item_text: %{},
      num_items: num_items,
      get_logits_fn: get_logits_fn,
      get_logits_batch_fn: get_logits_batch_fn
    }
  end
end
