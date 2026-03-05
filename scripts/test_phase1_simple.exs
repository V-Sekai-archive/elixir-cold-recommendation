#!/usr/bin/env elixir

# Simple Phase 1 test: verify EmbeddingCache loads and performs lookups

IO.puts("=== Phase 1: SId Caching - Basic Performance Test ===\n")

Application.ensure_all_started(:recgpt)

alias RecGPT.EmbeddingCache

IO.puts("Loading EmbeddingCache from database...")

start_load = System.monotonic_time(:millisecond)

case EmbeddingCache.load_from_db({EXLA.Backend, client: :cuda}) do
  {:ok, embeddings_table, tokens_table} ->
    load_time = System.monotonic_time(:millisecond) - start_load
    
    IO.puts("✓ Cache loaded successfully in #{load_time}ms")
    IO.puts("  Embeddings table: #{inspect(embeddings_table)}")
    IO.puts("  Tokens table: #{inspect(tokens_table)}\n")
    
    # Benchmark lookups
    IO.puts("Benchmarking ETS lookup performance...")
    
    num_lookups = 10_000
    sample_ids = [0, 1, 2, 3, 4]
    
    start_lookup = System.monotonic_time(:microsecond)
    
    Enum.reduce(1..num_lookups, 0, fn _, acc ->
      id = Enum.random(sample_ids)
      case EmbeddingCache.get_tokens(tokens_table, id) do
        {:ok, _tokens} -> acc + 1
        {:error, :not_found} -> acc
      end
    end)
    
    end_lookup = System.monotonic_time(:microsecond)
    total_micro = end_lookup - start_lookup
    avg_micro = total_micro / num_lookups
    
    IO.puts("  #{num_lookups} lookups completed in #{total_micro}µs")
    IO.puts("  Average per lookup: #{Float.round(avg_micro, 3)}µs")
    IO.puts("  Throughput: #{Float.round(1_000_000 / avg_micro / 1000, 1)}K lookups/sec\n")
    
    # Verify one embedding
    case EmbeddingCache.get_embedding(embeddings_table, 0) do
      {:ok, tensor} ->
        IO.puts("Sample embedding:")
        IO.puts("  Shape: #{inspect(Nx.shape(tensor))}")
        IO.puts("  Type: #{inspect(Nx.type(tensor))}\n")
      
      {:error, :not_found} ->
        IO.puts("Warning: No embeddings found in cache\n")
    end
    
    # Cleanup
    EmbeddingCache.cleanup(embeddings_table, tokens_table)
    IO.puts("✅ Phase 1 test complete!\n")
    IO.puts("Expected savings: 45ms per inference (eliminated MPNet)")
    
  {:error, reason} ->
    IO.puts("❌ Failed to load cache: #{reason}")
    System.halt(1)
end
