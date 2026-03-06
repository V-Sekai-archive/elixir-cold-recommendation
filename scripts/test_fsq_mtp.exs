#!/usr/bin/env mix
# Benchmark FSQ encoding + MTP decoding (no beam search)

defmodule FSQMTPTest do
  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder
  alias RecGPT.Decode

  @vocab_size 15_361
  @n_embd 768

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  FSQ + MTP Performance Test (no beam search)")
    IO.puts(String.duplicate("=", 60))

    preflight_checks()

    backend = EXLA.Backend

    # --- FSQ params from VAE checkpoint ---
    vae_path =
      System.get_env("RECGPT_VAE_CKPT") ||
        "thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt"

    IO.puts("\nLoading VAE checkpoint: #{vae_path}")
    fsq_params = RecGPT.FSQ.load_params_from_vae_pt(vae_path)
    IO.puts("  project_in kernel: #{inspect(Nx.shape(fsq_params["project_in"]["kernel"]))}")
    IO.puts("  project_out kernel: #{inspect(Nx.shape(fsq_params["project_out"]["kernel"]))}")

    # --- Build synthetic item catalog ---
    num_items = 1000
    IO.puts("\nBuilding catalog: #{num_items} items")

    # Synthetic 768-d embeddings (num_items, 768)
    embeddings =
      Nx.iota({num_items, @n_embd}, type: {:f, 32})
      |> Nx.divide(num_items * @n_embd)
      |> Nx.subtract(0.5)

    # --- Benchmark FSQ encoding ---
    IO.puts("Benchmarking FSQ encoding (10 runs)...")

    fsq_times =
      for _ <- 1..10 do
        {us, _} = :timer.tc(fn ->
          FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)
        end)
        us / 1000.0
      end

    token_lists = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)

    print_stats("FSQ Encoding (#{num_items} items)", fsq_times)

    # --- Build item_id_to_tokens tensor ---
    item_id_to_tokens =
      token_lists
      |> Nx.tensor(type: {:s, 32})
      |> Nx.backend_transfer(backend)

    # --- Stub logits (simulates model forward) ---
    stub_logits_fn = fn _context ->
      Nx.broadcast(0.0, {1, 4, @vocab_size})
      |> Nx.as_type({:f, 32})
      |> Nx.backend_transfer(backend)
    end

    # --- Warmup MTP ---
    IO.puts("\nWarming up MTP (3 runs)...")

    for _ <- 1..3 do
      Decode.lookahead_top_k(item_id_to_tokens, [], 5, stub_logits_fn, backend)
    end

    # --- Benchmark MTP decode ---
    IO.puts("Benchmarking MTP decode (100 runs)...")

    mtp_times =
      for _ <- 1..100 do
        {us, _} = :timer.tc(fn ->
          Decode.lookahead_top_k(item_id_to_tokens, [], 5, stub_logits_fn, backend)
        end)
        us / 1000.0
      end

    print_stats("MTP Decode top-5 (#{num_items}-item catalog)", mtp_times)

    IO.puts("")
  end

  defp preflight_checks do
    IO.puts("\nPreflight checks...")

    # VAE checkpoint
    vae_path =
      System.get_env("RECGPT_VAE_CKPT") ||
        "thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt"

    unless File.regular?(vae_path) do
      IO.puts("  [FAIL] VAE checkpoint not found: #{vae_path}")
      IO.puts("         Run: mix recgpt.fetch_vae_ckpt")
      System.halt(1)
    end

    IO.puts("  [OK]   VAE checkpoint: #{vae_path}")

    # MPNet embedding model (Bumblebee cache)
    mpnet_cache =
      Path.join([Bumblebee.cache_dir(), "huggingface", "sentence-transformers--all-mpnet-base-v2"])

    unless File.dir?(mpnet_cache) do
      IO.puts("  [FAIL] MPNet model not cached: #{mpnet_cache}")
      IO.puts("         Run once to download: mix run -e 'RecGPT.Embedding.serving()'")
      System.halt(1)
    end

    IO.puts("  [OK]   MPNet model cache: #{mpnet_cache}")
  end

  defp print_stats(label, times_ms) do
    sorted = Enum.sort(times_ms)
    n = length(sorted)
    avg = Enum.sum(times_ms) / n
    p50 = Enum.at(sorted, div(n, 2))
    p90 = Enum.at(sorted, round(n * 0.9) - 1)
    p99 = Enum.at(sorted, round(n * 0.99) - 1)

    IO.puts("\n  -- #{label} --")
    IO.puts("  Min:  #{Float.round(hd(sorted), 2)} ms")
    IO.puts("  Avg:  #{Float.round(avg, 2)} ms")
    IO.puts("  P50:  #{Float.round(p50, 2)} ms")
    IO.puts("  P90:  #{Float.round(p90, 2)} ms")
    IO.puts("  P99:  #{Float.round(p99, 2)} ms")
    IO.puts("  Max:  #{Float.round(List.last(sorted), 2)} ms")
    IO.puts("  N:    #{n}")
  end
end

FSQMTPTest.run()
