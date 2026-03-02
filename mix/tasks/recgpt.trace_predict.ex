defmodule Mix.Tasks.Recgpt.TracePredict do
  @shortdoc "Trace one recommendation call; print timing breakdown for optimization"
  @moduledoc """
  Runs a single Predict-style recommendation and prints a timing breakdown so you
  can see where time is spent and what to optimize.

  The first run can be slow: EXLA does one-time CUDA/XLA init and JIT-compiles
  kernels (ptxas). Subsequent runs are faster. Most time is usually in inference
  (4 forward passes for beam search).

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

  ## Examples
      mix recgpt.trace_predict
      mix recgpt.trace_predict --context "0,1" --top-k 10
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
          top_k: :integer
        ]
      )

    Application.ensure_all_started(:recgpt)
    Application.ensure_all_started(:nx)

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

    # Resolve paths: opts > artifact catalogue > default
    fixture_path =
      opts[:fixture] ||
        RecGPT.Catalog.Artifact.resolve_path("fixture") ||
        Path.expand(Path.join(File.cwd!(), "data/steam/fixture.json"), File.cwd!())

    ckpt_dir =
      opts[:ckpt] ||
        RecGPT.Catalog.Artifact.resolve_path("checkpoint") ||
        Path.expand(Path.join(File.cwd!(), "data/recgpt_ckpt_export"), File.cwd!())

    catalog_path =
      (opts[:catalog] && Path.expand(opts[:catalog], File.cwd!())) ||
        RecGPT.Catalog.Artifact.resolve_path("items")

    unless File.regular?(fixture_path) do
      Mix.raise("Fixture not found: #{fixture_path}. Run mix recgpt.build_fixture first.")
    end

    unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
      Mix.raise("Checkpoint not found: #{ckpt_dir}. Run mix recgpt.export_ckpt first.")
    end

    Mix.shell().info("Loading state...")

    state =
      case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
        {:ok, s} -> s
        {:error, reason} -> Mix.raise("Load failed: #{inspect(reason)}")
      end

    Mix.shell().info("Tracing one recommend: context=#{inspect(context_ids)}, top_k=#{top_k}")
    Mix.shell().info("")

    {:ok, agent} = Agent.start_link(fn -> {0, 0} end)
    traced_tensor_fn = fn batch_tensor, cache ->
      {us, res} = :timer.tc(fn -> state.get_logits_batch_tensor_fn.(batch_tensor, cache) end)
      Agent.update(agent, fn {t, n} -> {t + us, n + 1} end)
      res
    end

    context_us = 0
    {beam_search_us, result} =
      :timer.tc(fn ->
        RecGPT.Decode.beam_search_top_k_spmd(
          state.trie_tensors,
          state.item_id_to_tokens_tensor,
          context_ids,
          top_k,
          traced_tensor_fn,
          state.inference_backend,
          state.trie
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
end
