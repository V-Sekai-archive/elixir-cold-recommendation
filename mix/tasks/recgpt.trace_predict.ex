defmodule Mix.Tasks.Recgpt.TracePredict do
  @shortdoc "Trace one recommendation call; print timing breakdown for optimization"
  @moduledoc """
  Runs a single Predict-style recommendation and prints a timing breakdown so you
  can see where time is spent and what to optimize.

  Aligns with Replicate COG stages: setup (one-time load + JIT compile) ~20–30s;
  predict (per-request inference) ~300–400ms on 12-layer + RTX 4090. Most time
  is usually in beam search (4 forward passes).

  Loads state (fixture + checkpoint + optional catalog), runs recommend with
  the same opts as Serve, and reports:
  - context_to_tokens_us: building context token sequence from item IDs
  - beam_search_total: total time in beam search
  - inference: time inside get_logits (4 forward passes)
  - response_build_us: building item_ids + display_name list
  - total_us and percentiles over runs

  ## Options
    * `--fixture` - Path to fixture JSON (default: data/steam/fixture.json)
    * `--ckpt` - Checkpoint export dir (default: data/recgpt_ckpt_export)
    * `--catalog` - Optional path to items JSON
    * `--context` - Comma-separated context item IDs (default: 0)
    * `--top-k` - Max recommendations (default: 10)
    * `--runs` - Number of timed runs for stats (default: 20)
    * `--jitter-ms` - Max random ms before each run to desync timers (default: 2)
    * `--dtype` - Inference dtype: float32 or bf16 (overrides config for this run)
    * `--scan-dtypes` - Run trace for each dtype (float32, bf16) in sequence
    * `--profile` - Run under `nsys profile` (NVIDIA Nsight Systems) for GPU profiling; writes .nsys-rep file

  ## Examples
      mix recgpt.trace_predict
      mix recgpt.trace_predict --context "0,1" --top-k 10
      mix recgpt.trace_predict --runs 20 --dtype float32
      mix recgpt.trace_predict --runs 20 --scan-dtypes
      mix recgpt.trace_predict --runs 50 --jitter-ms 3
      mix recgpt.trace_predict --profile --context "0,1"
  """
  use Mix.Task

  @dialyzer {:nowarn_function, run: 1}

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
          runs: :integer,
          jitter_ms: :integer,
          dtype: :string,
          scan_dtypes: :boolean,
          profile: :boolean
        ]
      )

    if opts[:profile] do
      run_with_nsys(args, opts)
    else
      if opts[:scan_dtypes] do
        run_scan_dtypes(opts)
      else
        run_trace(opts)
      end
    end
  end

  defp run_scan_dtypes(opts) do
    dtypes = [{"float32", {:f, 32}}, {"bf16", {:bf, 16}}]

    for {label, dtype_value} <- dtypes do
      Mix.shell().info("")
      Mix.shell().info("========== inference_dtype: #{label} ==========")
      apply_dtype(dtype_value)
      run_trace(opts)
    end

    Mix.shell().info("")
    Mix.shell().info("========== scan complete ==========")
  end

  defp apply_dtype(nil), do: :ok

  defp apply_dtype(dtype_value) do
    Application.put_env(:recgpt, :inference_dtype, dtype_value)
  end

  defp parse_dtype_opt(nil), do: nil
  defp parse_dtype_opt("float32"), do: {:f, 32}
  defp parse_dtype_opt("bf16"), do: {:bf, 16}
  defp parse_dtype_opt(_), do: nil

  defp run_with_nsys(args, _opts) do
    unless System.find_executable("nsys") do
      Mix.raise(
        "nsys (NVIDIA Nsight Systems) not found. Install from https://developer.nvidia.com/nsight-systems/get-started"
      )
    end

    args_no_profile = args |> Enum.reject(&(&1 in ["--profile", "-profile"]))
    out_file = "recgpt_trace_#{System.system_time(:millisecond)}.nsys-rep"
    Mix.shell().info("Profiling with Nsight Systems -> #{out_file}")

    nsys_args = [
      "profile",
      "-o",
      out_file,
      "-t",
      "cuda,nvtx,osrt",
      "--cuda-event-trace=true",
      "--stats=true",
      "--sample=process-tree",
      "mix",
      "recgpt.trace_predict" | args_no_profile
    ]

    {output, exit_code} = System.cmd("nsys", nsys_args, stderr_to_stdout: true)
    IO.write(output)

    exit_code =
      case exit_code do
        n when is_integer(n) -> n
        {:exit_status, n} -> n
        _ -> 1
      end

    profile_created = File.regular?(out_file)

    if exit_code != 0 and not profile_created do
      Mix.shell().error("nsys profile failed (exit #{exit_code}) and no report was generated.")
      exit({:shutdown, exit_code})
    end

    if profile_created do
      Mix.shell().info("Profile written to #{out_file}. Open with nsys-ui or Nsight Systems GUI.")
    end

    if exit_code != 0 and profile_created do
      Mix.shell().info("Note: Child process exited abnormally; profile may still be useful.")
    end
  end

  defp run_trace(opts) do
    runs = opts[:runs] || 20
    jitter_ms = opts[:jitter_ms] || 2

    if dtype_override = parse_dtype_opt(opts[:dtype]) do
      apply_dtype(dtype_override)
    end

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

    Mix.shell().info(
      "Tracing #{runs} recommend call(s): context=#{inspect(context_ids)}, top_k=#{top_k}, jitter=0..#{jitter_ms}ms"
    )

    Mix.shell().info("")

    # Setup: JIT-compile kernels before timed predict (COG setup ~20–30s; predict ~300–400ms on 12-layer + RTX 4090)
    Mix.shell().info("Setup (compile kernels)...")

    case RecGPT.Serve.recommend(state, context_ids, top_k) do
      {:ok, _} -> :ok
      {:error, _} -> :ok
    end

    Mix.shell().info("")

    # Run N timed calls with jitter between runs
    samples =
      run_timed_recommends(state, context_ids, top_k, runs, jitter_ms)

    total_us_list = Enum.map(samples, & &1.total_us)
    stats = compute_stats(total_us_list)

    # Last sample for breakdown
    last = List.last(samples)
    item_ids = last.item_ids

    Mix.shell().info("=== Recommendation result (last run) ===")
    Mix.shell().info("  item_ids: #{inspect(item_ids)}")
    Mix.shell().info("")
    Mix.shell().info("=== Timing breakdown (last run) ===")
    Mix.shell().info("  context_to_tokens:  #{last.context_us} μs")
    Mix.shell().info("  beam_search_total:   #{last.beam_search_us} μs")

    if last.inference_calls > 0 do
      Mix.shell().info(
        "  inference (forward): #{last.inference_us} μs  (#{last.inference_calls} calls)"
      )
    else
      Mix.shell().info("  inference: (time in beam_search_total above)")
    end

    Mix.shell().info("  response_build:     #{last.response_us} μs")
    Mix.shell().info("")

    Mix.shell().info("=== Stats over #{runs} runs (jitter 0..#{jitter_ms} ms) ===")

    Mix.shell().info(
      "  total μs:  mean=#{stats.mean}  std=#{stats.std}  min=#{stats.min}  max=#{stats.max}"
    )

    Mix.shell().info(
      "  total ms:  mean=#{Float.round(stats.mean / 1000, 2)}  std=#{Float.round(stats.std / 1000, 2)}"
    )

    Mix.shell().info(
      "  percentiles:  p50=#{stats.p50} μs  p95=#{stats.p95} μs  p99=#{stats.p99} μs"
    )

    Mix.shell().info("")

    if last.inference_calls > 0 do
      avg_inference_us = div(last.inference_us, last.inference_calls)
      pct = Float.round(100.0 * last.inference_us / last.total_us, 1)
      Mix.shell().info("  inference % of total: #{pct}%  (avg #{avg_inference_us} μs/forward)")

      Mix.shell().info("  → #{last.inference_calls} forward passes. Speed up: GPU, KV-cache.")
    else
      pct = Float.round(100.0 * last.beam_search_us / last.total_us, 1)
      Mix.shell().info("  beam_search % of total: #{pct}%")
      Mix.shell().info("  → Fused path (1 graph). Speed up: GPU, KV-cache.")
    end

    Mix.shell().info("")
  end

  defp run_timed_recommends(state, context_ids, top_k, runs, jitter_ms) do
    {:ok, agent} = Agent.start_link(fn -> {0, 0} end)

    traced_get_logits_4_fn = fn context_tokens ->
      {us, res} = :timer.tc(fn -> state.get_logits_4_fn.(context_tokens) end)
      Agent.update(agent, fn {t, n} -> {t + us, n + 1} end)
      res
    end

    do_run = fn ->
      Agent.update(agent, fn _ -> {0, 0} end)

      context_us = 0

      opts = RecGPT.Serve.decode_opts(state, context_ids)

      {beam_search_us, result} =
        :timer.tc(fn ->
          RecGPT.Decode.beam_search_top_k_spmd(
            state.trie_tensors,
            state.item_id_to_tokens_tensor,
            context_ids,
            top_k,
            traced_get_logits_4_fn,
            state.inference_backend,
            state.trie,
            opts
          )
        end)

      {inference_us, inference_calls} = Agent.get(agent, fn {t, n} -> {t, n} end)

      item_ids =
        case result do
          {:ok, ids} -> ids
          :not_found -> []
        end

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

      %{
        total_us: total_us,
        context_us: context_us,
        beam_search_us: beam_search_us,
        inference_us: inference_us,
        inference_calls: inference_calls,
        response_us: response_us,
        item_ids: item_ids
      }
    end

    samples =
      Enum.map(1..runs, fn i ->
        if i > 1 and jitter_ms > 0 do
          jitter = :rand.uniform(jitter_ms + 1) - 1
          Process.sleep(jitter)
        end

        do_run.()
      end)

    Agent.stop(agent)
    samples
  end

  defp compute_stats([]), do: %{mean: 0, std: 0, min: 0, max: 0, p50: 0, p95: 0, p99: 0}

  defp compute_stats(list) do
    n = length(list)
    sorted = Enum.sort(list)
    sum = Enum.sum(list)
    mean = div(sum, n)

    variance =
      list
      |> Enum.map(fn x -> (x - mean) * (x - mean) end)
      |> Enum.sum()
      |> Kernel./(max(1, n - 1))

    std = :math.sqrt(variance)
    min = List.first(sorted)
    max = List.last(sorted)
    p50 = percentile(sorted, 50)
    p95 = percentile(sorted, 95)
    p99 = percentile(sorted, 99)

    %{
      mean: round(mean),
      std: round(std),
      min: min,
      max: max,
      p50: p50,
      p95: p95,
      p99: p99
    }
  end

  defp percentile(sorted, p) do
    n = length(sorted)
    idx = max(0, min(n - 1, trunc(n * p / 100)))
    Enum.at(sorted, idx)
  end
end
