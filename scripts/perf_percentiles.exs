#!/usr/bin/env mix
# Measure recommendation latency percentiles (uses GPU/EXLA)

defmodule PerfBench do
  def stub_logits_4(_context) do
    # Return fixed logits for 4 positions, vocab_size=15361
    Nx.broadcast(-10.0, {1, 4, 15_361}) |> Nx.as_type({:f, 32})
  end

  def percentile(sorted_list, p) do
    idx = round((p / 100.0) * (length(sorted_list) - 1))
    Enum.at(sorted_list, idx)
  end

  def run do
    IO.puts("\n=== Recommendation Latency Percentiles (GPU) ===\n")

    # Use EXLA backend (GPU)
    backend = EXLA.Backend

    # Build test trie
    token_lists = [
      [100, 200, 300, 400],
      [100, 200, 300, 401],
      [100, 200, 301, 400],
      [101, 202, 303, 404],
      [110, 220, 330, 440]
    ]

    trie = RecGPT.Trie.build(token_lists)
    trie_tensors = RecGPT.Trie.to_tensors(trie, 15_361)
    
    # Transfer to GPU
    trie_tensors = %{
      next_state: Nx.backend_transfer(trie_tensors.next_state, backend),
      item_at_leaf: Nx.backend_transfer(trie_tensors.item_at_leaf, backend),
      num_states: trie_tensors.num_states
    }

    item_id_to_tokens =
      token_lists
      |> Nx.tensor(type: {:s, 32})
      |> Nx.backend_transfer(backend)

    # Warmup
    IO.puts("Warming up (5 runs)...")

    for _i <- 1..5 do
      RecGPT.Decode.beam_search_top_k_spmd(
        trie_tensors,
        item_id_to_tokens,
        [],
        5,
        &stub_logits_4/1,
        backend
      )
    end

    # Measure empty context - 100 runs
    IO.puts("Measuring 100 queries (empty context)...\n")

    latencies =
      Enum.reduce(1..100, [], fn _i, acc ->
        {elapsed_us, _result} = :timer.tc(fn ->
          RecGPT.Decode.beam_search_top_k_spmd(
            trie_tensors,
            item_id_to_tokens,
            [],
            5,
            &stub_logits_4/1,
            backend
          )
        end)
        [elapsed_us / 1000.0 | acc]
      end)
      |> Enum.sort()

    min_lat = Enum.min(latencies)
    max_lat = Enum.max(latencies)
    avg_lat = Enum.sum(latencies) / length(latencies)

    p50 = percentile(latencies, 50)
    p90 = percentile(latencies, 90)
    p99 = percentile(latencies, 99)

    IO.puts("~~ GPU Beam Search (SPMD) - Top-5 Recommendations ~~")
    IO.puts("")
    IO.puts("  Min:  #{Float.round(min_lat, 2)} ms")
    IO.puts("  Avg:  #{Float.round(avg_lat, 2)} ms")
    IO.puts("  Max:  #{Float.round(max_lat, 2)} ms")
    IO.puts("")
    IO.puts("  P50:  #{Float.round(p50, 2)} ms (median)")
    IO.puts("  P90:  #{Float.round(p90, 2)} ms (90th percentile)")
    IO.puts("  P99:  #{Float.round(p99, 2)} ms (99th percentile)")
    IO.puts("")

    # Test with 3-item context
    IO.puts("~~ With 3-item context ~~")
    IO.puts("")

    latencies_ctx =
      Enum.reduce(1..100, [], fn _i, acc ->
        {elapsed_us, _result} = :timer.tc(fn ->
          RecGPT.Decode.beam_search_top_k_spmd(
            trie_tensors,
            item_id_to_tokens,
            [0, 1, 2],
            5,
            &stub_logits_4/1,
            backend
          )
        end)
        [elapsed_us / 1000.0 | acc]
      end)
      |> Enum.sort()

    min_ctx = Enum.min(latencies_ctx)
    max_ctx = Enum.max(latencies_ctx)
    avg_ctx = Enum.sum(latencies_ctx) / length(latencies_ctx)

    p50_ctx = percentile(latencies_ctx, 50)
    p90_ctx = percentile(latencies_ctx, 90)
    p99_ctx = percentile(latencies_ctx, 99)

    IO.puts("  Min:  #{Float.round(min_ctx, 2)} ms")
    IO.puts("  Avg:  #{Float.round(avg_ctx, 2)} ms")
    IO.puts("  Max:  #{Float.round(max_ctx, 2)} ms")
    IO.puts("")
    IO.puts("  P50:  #{Float.round(p50_ctx, 2)} ms")
    IO.puts("  P90:  #{Float.round(p90_ctx, 2)} ms")
    IO.puts("  P99:  #{Float.round(p99_ctx, 2)} ms")
    IO.puts("")
  end
end

PerfBench.run()
