#!/usr/bin/env elixir

# Baseline test for YOUR ACTUAL DESIGN:
# - FuXi Linear Architecture (not transformer attention)
# - MTP Decode Strategy (not beam search)
# - For catalogue item retrieval

defmodule FuXiMTPBaseline do
  alias RecGPT.Trie
  alias RecGPT.Inference
  alias RecGPT.FuxiLinearInferenceParams
  alias RecGPT.LayerFreeze
  alias RecGPT.Decode

  def run do
    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:jason)

    # Force MTP decode strategy (not beam_search)
    Application.put_env(:recgpt, :decode_strategy, :mtp)

    IO.puts("=== FuXi Linear + MTP Baseline ===")
    IO.puts("Backend: #{inspect(Nx.default_backend())}")
    IO.puts("Decode Strategy: #{Application.get_env(:recgpt, :decode_strategy)}")
    IO.puts("")

    # Build FuXi stub state (4 blocks, linear attention)
    IO.puts("Building FuXi Linear stub state (4 blocks)...")
    {setup_time, state} = :timer.tc(fn ->
      token_id_list = [
        [100, 200, 300, 400],  # Item 0
        [101, 201, 301, 401],  # Item 1
        [102, 202, 302, 402],  # Item 2
        [103, 203, 303, 403],  # Item 3
        [104, 204, 304, 404]   # Item 4
      ]

      params = RecGPT.FuxiLinearInference.init_full_params(
        n_blocks: 4,
        max_seq_len: 1024,
        vocab_size: 15_361,
        embed_dim: 768
      )

      build_fuxi_state(token_id_list, params)
    end)

IO.puts("Setup: #{setup_time / 1000} ms")
IO.puts("")

# Create frozen inputs for recommend
frozen = LayerFreeze.record_from_state(state, [0])

# WARMUP: Run 3 iterations to JIT compile kernels
IO.puts("Warming up (3 iterations for FuXi Linear JIT)...")
for i <- 1..3 do
  {warmup_time, _} = :timer.tc(fn ->
    # Use MTP decode path directly
    Decode.lookahead_top_k(
      state.item_id_to_tokens_tensor,
      [0],
      5,
      state.get_logits_4_fn,
      state.inference_backend
    )
  end)
  IO.puts("  Warmup #{i}: #{warmup_time / 1000} ms")
end
IO.puts("")

# MEASUREMENT: Run 10 iterations
IO.puts("Measuring FuXi Linear + MTP (10 iterations)...")
times = for i <- 1..10 do
  {t, result} = :timer.tc(fn ->
    Decode.lookahead_top_k(
      state.item_id_to_tokens_tensor,
      [0],
      5,
      state.get_logits_4_fn,
      state.inference_backend
    )
  end)
  IO.puts("  Run #{i}: #{t / 1000} ms -> #{inspect(result)}")
  t
end

# Statistics
sorted = Enum.sort(times)
p50 = Enum.at(sorted, 4) / 1000
p90 = Enum.at(sorted, 8) / 1000
min_t = hd(sorted) / 1000
max_t = List.last(sorted) / 1000
avg = Enum.sum(times) / length(times) / 1000

IO.puts("")
IO.puts("=== FuXi Linear + MTP Results ===")
IO.puts("Min: #{:erlang.float_to_binary(min_t, decimals: 2)} ms")
IO.puts("P50: #{:erlang.float_to_binary(p50, decimals: 2)} ms")
IO.puts("P90: #{:erlang.float_to_binary(p90, decimals: 2)} ms")
IO.puts("Max: #{:erlang.float_to_binary(max_t, decimals: 2)} ms")
IO.puts("Avg: #{:erlang.float_to_binary(avg, decimals: 2)} ms")
IO.puts("")
IO.puts("Target P50: 50 ms")
IO.puts("Status: #{if p50 > 50, do: "NEEDS OPTIMIZATION", else: "ON TARGET"}")
IO.puts("")
IO.puts("")
IO.puts("=== Performance Summary ===")
IO.puts("Target P50: 50 ms")
IO.puts("Achieved P50: #{:erlang.float_to_binary(p50, decimals: 2)} ms")
IO.puts("Status: #{if p50 > 50, do: "NEEDS OPTIMIZATION", else: "ON TARGET"}")
  end

  # Helper to build FuXi state
  defp build_fuxi_state(token_id_list, params) do
    alias RecGPT.Trie
    alias RecGPT.FuxiLinearInferenceParams

    backend = Nx.default_backend()

    trie = Trie.build(token_id_list)

    # Transfer params to backend
    params_bin = Map.new(params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

    # Build FuXi-specific defn params
    dtype = Application.get_env(:recgpt, :inference_dtype, {:bf, 16})
    defn_params = FuxiLinearInferenceParams.build_defn_params(params_bin, dtype)
    defn_params = Map.new(defn_params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)

    # JIT compile for FuXi
    jit_single = Nx.Defn.jit(&RecGPT.FuxiLinearInferenceDefn.forward_last_4_logits/4, compiler: EXLA)

    get_logits_4_fn = fn context_tokens ->
      context_tokens = Nx.backend_transfer(context_tokens, backend)
      {batch_size, seq_len} = Nx.shape(context_tokens)

      # FuXi doesn't use aux/mask the same way
      logits = jit_single.(context_tokens, defn_params, Nx.tensor(0), Nx.tensor(0))
      # Take last 4 tokens' logits
      Nx.slice(logits, [0, seq_len - 4, 0], [batch_size, 4, 15_361])
    end

    trie_tensors = Trie.to_tensors(trie, 15_361)
    trie_tensors = %{
      next_state: Nx.backend_transfer(trie_tensors.next_state, backend),
      item_at_leaf: Nx.backend_transfer(trie_tensors.item_at_leaf, backend),
      num_states: Nx.shape(trie_tensors.next_state) |> elem(0)
    }

    item_id_to_tokens_tensor =
      token_id_list
      |> Nx.tensor(type: {:s, 32})
      |> Nx.backend_transfer(backend)

    vocab_size = 15_361
    root_state = Nx.tensor([0], type: {:s, 32}) |> Nx.backend_transfer(backend)
    neg_inf = Nx.tensor(-1.0e9, type: {:f, 32}) |> Nx.backend_transfer(backend)
    vocab_t = Nx.tensor(vocab_size, type: {:s, 32}) |> Nx.backend_transfer(backend)
    decode_constants = %{root_state: root_state, neg_inf: neg_inf, vocab_t: vocab_t}

    %RecGPT.Serve{
      params: params_bin,
      trie: trie,
      trie_tensors: trie_tensors,
      token_id_list: token_id_list,
      token_id_map: nil,
      item_id_to_tokens_tensor: item_id_to_tokens_tensor,
      item_text: %{},
      num_items: length(token_id_list),
      get_logits_4_fn: get_logits_4_fn,
      inference_backend: backend,
      beam_width_override: nil,
      decode_constants: decode_constants
    }
  end
end

# Run the baseline
FuXiMTPBaseline.run()
