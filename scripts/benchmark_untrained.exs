#!/usr/bin/env mix
# Benchmark untrained FSQ model performance
# Tests recommendation latency with random/untrained weights

defmodule UntrainedBenchmark do
  alias RecGPT.Decode
  alias RecGPT.Trie

  @vocab_size 15_361

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  Untrained FSQ Model Performance Benchmark")
    IO.puts(String.duplicate("=", 60))

    # Use EXLA backend (GPU)
    backend = EXLA.Backend

    # Build test trie with realistic catalog size
    num_items = 1000
    token_lists = for i <- 0..(num_items - 1) do
      t0 = rem(i, 100) + 1
      t1 = rem(i + 1, 100) + 100
      t2 = rem(i + 2, 1000) + 200
      t3 = rem(i + 3, 10000) + 1200
      [t0, t1, t2, t3]
    end

    IO.puts("\n📦 Building trie with #{num_items} items...")
    trie = Trie.build(token_lists)
    trie_tensors = Trie.to_tensors(trie, @vocab_size)

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

    # Simulate untrained model: random logits
    stub_logits_fn = fn _context ->
      # Use uniform random instead of normal (Nx.random_normal is private)
      Nx.broadcast(0.0, {1, 4, @vocab_size})
      |> Nx.as_type({:f, 32})
      |> Nx.backend_transfer(backend)
    end

    # Warmup
    IO.puts("🔥 Warming up (5 runs)...")

    for _i <- 1..5 do
      Decode.beam_search_top_k_spmd(
        trie_tensors,
        item_id_to_tokens,
        [],
        5,
        stub_logits_fn,
        backend
      )
    end

    # Measure latency
    IO.puts("📏 Measuring 100 queries (empty context)...\n")

    latencies =
      Enum.reduce(1..100, [], fn _i, acc ->
        {elapsed_us, _result} = :timer.tc(fn ->
          Decode.beam_search_top_k_spmd(
            trie_tensors,
            item_id_to_tokens,
            [],
            5,
            stub_logits_fn,
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

    IO.puts("~~ Untrained Model - Top-5 Recommendations ~~")
    IO.puts("")
    IO.puts("  Min:  #{Float.round(min_lat, 2)} ms")
    IO.puts("  Avg:  #{Float.round(avg_lat, 2)} ms")
    IO.puts("  Max:  #{Float.round(max_lat, 2)} ms")
    IO.puts("")
    IO.puts("  P50:  #{Float.round(p50, 2)} ms (median)")
    IO.puts("  P90:  #{Float.round(p90, 2)} ms (90th percentile)")
    IO.puts("  P99:  #{Float.round(p99, 2)} ms (99th percentile)")
    IO.puts("")

    # Test with context
    IO.puts("~~ With 3-item context ~~")
    IO.puts("")

    latencies_ctx =
      Enum.reduce(1..100, [], fn _i, acc ->
        {elapsed_us, _result} = :timer.tc(fn ->
          Decode.beam_search_top_k_spmd(
            trie_tensors,
            item_id_to_tokens,
            [0, 1, 2],
            5,
            stub_logits_fn,
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

    # Test different catalog sizes
    IO.puts("~~ Catalog Size Scalability ~~")
    IO.puts("")

    for size <- [100, 500, 2000, 5000] do
      token_lists_small = Enum.take(token_lists, size)
      trie_small = Trie.build(token_lists_small)
      trie_tensors_small = Trie.to_tensors(trie_small, @vocab_size)

      trie_tensors_small = %{
        next_state: Nx.backend_transfer(trie_tensors_small.next_state, backend),
        item_at_leaf: Nx.backend_transfer(trie_tensors_small.item_at_leaf, backend),
        num_states: trie_tensors_small.num_states
      }

      item_id_to_tokens_small =
        token_lists_small
        |> Nx.tensor(type: {:s, 32})
        |> Nx.backend_transfer(backend)

      {elapsed_us, _result} = :timer.tc(fn ->
        Decode.beam_search_top_k_spmd(
          trie_tensors_small,
          item_id_to_tokens_small,
          [],
          5,
          stub_logits_fn,
          backend
        )
      end)

      elapsed_ms = elapsed_us / 1000.0
      IO.puts("  #{size} items: #{Float.round(elapsed_ms, 2)} ms")
    end

    IO.puts("")
  end

  defp percentile(sorted_list, p) do
    idx = round((p / 100.0) * (length(sorted_list) - 1))
    Enum.at(sorted_list, idx)
  end
end

UntrainedBenchmark.run()
