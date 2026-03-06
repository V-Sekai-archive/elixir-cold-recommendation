#!/usr/bin/env mix
# Test MTP (Multi-Token Prediction) codepath

defmodule MTPTest do
  alias RecGPT.Decode
  alias RecGPT.Trie

  @vocab_size 15_361

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  MTP (Multi-Token Prediction) Test")
    IO.puts(String.duplicate("=", 60))

    backend = EXLA.Backend

    # Build test catalog
    num_items = 1000
    token_lists = for i <- 0..(num_items - 1) do
      t0 = rem(i, 100) + 1
      t1 = rem(i + 1, 100) + 100
      t2 = rem(i + 2, 1000) + 200
      t3 = rem(i + 3, 10000) + 1200
      [t0, t1, t2, t3]
    end

    IO.puts("\n📦 Building catalog with #{num_items} items...")
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

    # MTP: single forward pass
    stub_logits_fn = fn _context ->
      Nx.broadcast(0.0, {1, 4, @vocab_size})
      |> Nx.as_type({:f, 32})
      |> Nx.backend_transfer(backend)
    end

    # Warmup
    IO.puts("🔥 Warming up MTP...")

    for _i <- 1..3 do
      Decode.lookahead_top_k(
        item_id_to_tokens,
        [],
        5,
        stub_logits_fn,
        backend
      )
    end

    # Measure MTP latency
    IO.puts("📏 Measuring MTP latency (100 queries)...\n")

    latencies =
      Enum.reduce(1..100, [], fn _i, acc ->
        {elapsed_us, result} = :timer.tc(fn ->
          Decode.lookahead_top_k(
            item_id_to_tokens,
            [],
            5,
            stub_logits_fn,
            backend
          )
        end)

        case result do
          {:ok, items} -> [elapsed_us / 1000.0 | acc]
          :not_found -> acc
        end
      end)
      |> Enum.sort()

    if length(latencies) > 0 do
      min_lat = Enum.min(latencies)
      max_lat = Enum.max(latencies)
      avg_lat = Enum.sum(latencies) / length(latencies)

      p50 = percentile(latencies, 50)
      p90 = percentile(latencies, 90)
      p99 = percentile(latencies, 99)

      IO.puts("~~ MTP Performance (Top-5 Recommendations) ~~")
      IO.puts("")
      IO.puts("  Min:  #{Float.round(min_lat, 2)} ms")
      IO.puts("  Avg:  #{Float.round(avg_lat, 2)} ms")
      IO.puts("  Max:  #{Float.round(max_lat, 2)} ms")
      IO.puts("")
      IO.puts("  P50:  #{Float.round(p50, 2)} ms")
      IO.puts("  P90:  #{Float.round(p90, 2)} ms")
      IO.puts("  P99:  #{Float.round(p99, 2)} ms")
      IO.puts("")
      IO.puts("  Total queries: #{length(latencies)}")
      IO.puts("")

      # Compare with beam search
      IO.puts("~~ Comparing MTP vs Beam Search ~~")
      IO.puts("")

      {beam_us, _beam_result} = :timer.tc(fn ->
        Decode.beam_search_top_k_spmd(
          trie_tensors,
          item_id_to_tokens,
          [],
          5,
          stub_logits_fn,
          backend
        )
      end)

      beam_ms = beam_us / 1000.0

      IO.puts("  MTP (single forward):   #{Float.round(avg_lat, 2)} ms avg")
      IO.puts("  Beam Search (4 steps):  #{Float.round(beam_ms, 2)} ms")
      IO.puts("  Speedup:                #{Float.round(beam_ms / avg_lat, 2)}x faster")
      IO.puts("")
    else
      IO.puts("  ❌ No successful queries")
    end
  end

  defp percentile(sorted_list, p) do
    idx = round((p / 100.0) * (length(sorted_list) - 1))
    Enum.at(sorted_list, idx)
  end
end

MTPTest.run()
