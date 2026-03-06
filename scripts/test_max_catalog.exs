#!/usr/bin/env mix
# Test maximum catalog size limits on this GPU

defmodule MaxCatalogTest do
  alias RecGPT.Decode
  alias RecGPT.Trie

  @vocab_size 15_361

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  Maximum Catalog Size Test - RTX 4090 (24GB VRAM)")
    IO.puts(String.duplicate("=", 60))

    backend = EXLA.Backend

    # Test increasingly large catalogs
    test_sizes = [10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]

    stub_logits_fn = fn _context ->
      Nx.broadcast(0.0, {1, 4, @vocab_size})
      |> Nx.as_type({:f, 32})
      |> Nx.backend_transfer(backend)
    end

    for size <- test_sizes do
      IO.puts("\n📦 Testing #{size} items...")

      try do
        # Generate token lists
        token_lists = for i <- 0..(size - 1) do
          t0 = rem(i, 100) + 1
          t1 = rem(i + 1, 100) + 100
          t2 = rem(i + 2, 1000) + 200
          t3 = rem(i + 3, 10000) + 1200
          [t0, t1, t2, t3]
        end

        # Build trie
        {trie_build_us, trie} = :timer.tc(fn -> Trie.build(token_lists) end)
        IO.puts("  Trie build: #{Float.round(trie_build_us / 1000, 2)} ms")

        # Convert to tensors
        {tensor_us, trie_tensors} = :timer.tc(fn ->
          Trie.to_tensors(trie, @vocab_size)
        end)

        IO.puts("  Tensor conversion: #{Float.round(tensor_us / 1000, 2)} ms")

        # Calculate tensor memory
        tensor_mem_mb = calculate_tensor_memory(trie_tensors)
        IO.puts("  Tensor memory: #{Float.round(tensor_mem_mb, 2)} MB")

        # Transfer to GPU
        {transfer_us, trie_tensors_gpu} = :timer.tc(fn ->
          %{
            next_state: Nx.backend_transfer(trie_tensors.next_state, backend),
            item_at_leaf: Nx.backend_transfer(trie_tensors.item_at_leaf, backend),
            num_states: trie_tensors.num_states
          }
        end)

        IO.puts("  GPU transfer: #{Float.round(transfer_us / 1000, 2)} ms")

        item_id_to_tokens =
          token_lists
          |> Nx.tensor(type: {:s, 32})
          |> Nx.backend_transfer(backend)

        # Test single recommendation
        {elapsed_us, result} = :timer.tc(fn ->
          Decode.beam_search_top_k_spmd(
            trie_tensors_gpu,
            item_id_to_tokens,
            [],
            5,
            stub_logits_fn,
            backend
          )
        end)

        elapsed_ms = elapsed_us / 1000.0
        {:ok, items} = result

        IO.puts("  ✅ First recommendation: #{Float.round(elapsed_ms, 2)} ms")
        IO.puts("     Found #{length(items)} items")

      rescue
        e ->
          IO.puts("  ❌ FAILED: #{Exception.message(e)}")
          IO.puts("  Stopping tests - GPU memory likely exhausted")
      end
    end

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  Summary")
    IO.puts(String.duplicate("=", 60))
    IO.puts("\nRTX 4090 has 24GB VRAM")
    IO.puts("Maximum practical catalog size depends on:")
    IO.puts("  - Trie tensor size: ~#{Float.round(calculate_tensor_memory_for_size(100_000), 2)} MB for 100K items")
    IO.puts("  - Item tokens tensor: ~#{Float.round(calculate_item_tokens_memory(100_000), 2)} MB for 100K items")
    IO.puts("  - Model weights: varies by model size")
    IO.puts("  - Activation memory during inference")
    IO.puts("")
  end

  defp calculate_tensor_memory(trie_tensors) do
    next_state_bytes = Nx.size(trie_tensors.next_state) * 4  # s32 = 4 bytes
    item_at_leaf_bytes = Nx.size(trie_tensors.item_at_leaf) * 4
    (next_state_bytes + item_at_leaf_bytes) / (1024 * 1024)
  end

  defp calculate_tensor_memory_for_size(num_items) do
    # Approximate: num_states ≈ num_items (worst case)
    # next_state: {num_states, vocab_size}
    # item_at_leaf: {num_states, vocab_size}
    num_states = num_items
    bytes = num_states * @vocab_size * 4 * 2  # 2 tensors, s32 = 4 bytes
    bytes / (1024 * 1024)
  end

  defp calculate_item_tokens_memory(num_items) do
    # {num_items, 4} s32 tensor
    bytes = num_items * 4 * 4
    bytes / (1024 * 1024)
  end
end

MaxCatalogTest.run()
