#!/usr/bin/env mix
# Test extreme catalog sizes - push RTX 4090 to its limits

defmodule ExtremeCatalogTest do
  alias RecGPT.Decode
  alias RecGPT.Trie

  @vocab_size 15_361

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  Extreme Catalog Test - RTX 4090 (24GB VRAM)")
    IO.puts("  Testing 5M, 10M, 13M, 15M items")
    IO.puts(String.duplicate("=", 60))

    backend = EXLA.Backend

    # Test extreme sizes
    test_sizes = [5_000_000, 10_000_000, 13_000_000, 15_000_000]

    stub_logits_fn = fn _context ->
      Nx.broadcast(0.0, {1, 4, @vocab_size})
      |> Nx.as_type({:f, 32})
      |> Nx.backend_transfer(backend)
    end

    for size <- test_sizes do
      IO.puts("\n📦 Testing #{size} items (#{Float.round(size / 1_000_000, 1)}M)...")

      try do
        # Estimate memory first
        estimated_mem_mb = calculate_tensor_memory_for_size(size)
        IO.puts("  Estimated tensor memory: #{Float.round(estimated_mem_mb, 0)} MB")

        if estimated_mem_mb > 20_000 do
          IO.puts("  ⚠️  Warning: Approaching 24GB VRAM limit")
        end

        # Generate token lists (streaming to save memory)
        IO.puts("  Generating tokens...")
        {gen_us, token_lists} = :timer.tc(fn ->
          for i <- 0..(size - 1) do
            t0 = rem(i, 100) + 1
            t1 = rem(i + 1, 100) + 100
            t2 = rem(i + 2, 1000) + 200
            t3 = rem(i + 3, 10000) + 1200
            [t0, t1, t2, t3]
          end
        end)

        IO.puts("  Token generation: #{Float.round(gen_us / 1_000_000, 2)} s")

        # Build trie
        IO.puts("  Building trie...")
        {trie_build_us, trie} = :timer.tc(fn -> Trie.build(token_lists) end)
        IO.puts("  Trie build: #{Float.round(trie_build_us / 1_000_000, 2)} s")

        # Free token lists
        _ = token_lists
        :erlang.garbage_collect()

        # Convert to tensors
        IO.puts("  Converting to tensors...")
        {tensor_us, trie_tensors} = :timer.tc(fn ->
          Trie.to_tensors(trie, @vocab_size)
        end)

        IO.puts("  Tensor conversion: #{Float.round(tensor_us / 1_000_000, 2)} s")

        # Calculate actual tensor memory
        actual_mem_mb = calculate_tensor_memory(trie_tensors)
        IO.puts("  Actual tensor memory: #{Float.round(actual_mem_mb, 0)} MB")

        # Transfer to GPU
        IO.puts("  Transferring to GPU...")
        {transfer_us, trie_tensors_gpu} = :timer.tc(fn ->
          %{
            next_state: Nx.backend_transfer(trie_tensors.next_state, backend),
            item_at_leaf: Nx.backend_transfer(trie_tensors.item_at_leaf, backend),
            num_states: trie_tensors.num_states
          }
        end)

        IO.puts("  GPU transfer: #{Float.round(transfer_us / 1_000_000, 2)} s")

        # Free CPU tensors
        _ = trie_tensors
        :erlang.garbage_collect()

        # Create item tokens tensor
        IO.puts("  Creating item tokens tensor...")
        item_tokens_mem_mb = (size * 4 * 4) / (1024 * 1024)
        IO.puts("  Item tokens memory: #{Float.round(item_tokens_mem_mb, 0)} MB")

        # Test single recommendation
        IO.puts("  Running recommendation...")
        {elapsed_us, result} = :timer.tc(fn ->
          Decode.beam_search_top_k_spmd(
            trie_tensors_gpu,
            nil,  # Will use trie_tensors directly
            [],
            5,
            stub_logits_fn,
            backend
          )
        end)

        elapsed_ms = elapsed_us / 1000.0

        case result do
          {:ok, items} ->
            IO.puts("  ✅ SUCCESS: #{Float.round(elapsed_ms, 2)} ms")
            IO.puts("     Found #{length(items)} items")

          :not_found ->
            IO.puts("  ⚠️  No items found (but no crash!)")
        end

        # Free GPU memory
        _ = trie_tensors_gpu
        :erlang.garbage_collect()

      rescue
        e ->
          IO.puts("  ❌ FAILED: #{Exception.message(e)}")
          IO.puts("  Stopping tests - GPU memory exhausted")
      end
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  Test Complete")
    IO.puts(String.duplicate("=", 60))
  end

  defp calculate_tensor_memory(trie_tensors) do
    next_state_bytes = Nx.size(trie_tensors.next_state) * 4
    item_at_leaf_bytes = Nx.size(trie_tensors.item_at_leaf) * 4
    (next_state_bytes + item_at_leaf_bytes) / (1024 * 1024)
  end

  defp calculate_tensor_memory_for_size(num_items) do
    num_states = num_items
    bytes = num_states * @vocab_size * 4 * 2
    bytes / (1024 * 1024)
  end
end

ExtremeCatalogTest.run()
