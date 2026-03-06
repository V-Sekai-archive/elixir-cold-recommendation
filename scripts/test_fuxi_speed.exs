#!/usr/bin/env mix
# FuXi Linear inference speed benchmark
# Measures: forward pass latency across sequence lengths, then end-to-end MTP decode

defmodule FuxiSpeedTest do
  alias RecGPT.FuxiLinearInference
  alias RecGPT.FuxiLinearInferenceParams
  alias RecGPT.FuxiLinearInferenceDefn
  alias RecGPT.Decode

  @vocab_size 15_361
  @n_embd 768
  @seq_lens [4, 16, 64, 128, 256]
  @num_items 1000
  @warmup 5
  @runs 30

  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  FuXi Linear Inference Speed Benchmark")
    IO.puts(String.duplicate("=", 60))

    backend = EXLA.Backend
    dtype = {:f, 32}

    # Build params once
    IO.puts("\nInitializing FuXi Linear params (4 blocks)...")
    {init_ms, params_raw} = timed_ms(fn -> FuxiLinearInference.init_full_params() end)
    IO.puts("  param init: #{Float.round(init_ms, 1)} ms")

    {build_ms, defn_params} = timed_ms(fn ->
      params_raw
      |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)
      |> FuxiLinearInferenceParams.build_defn_params(dtype)
      |> Map.new(fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)
    end)
    IO.puts("  defn param build + transfer: #{Float.round(build_ms, 1)} ms")

    jit_fwd = Nx.Defn.jit(&FuxiLinearInferenceDefn.forward_last_4_logits/4, compiler: EXLA)

    # Zeros for aux/mask (not under test)
    make_inputs = fn seq_len ->
      tokens   = Nx.broadcast(Nx.tensor(1, type: {:s, 32}), {1, seq_len}) |> Nx.backend_transfer(backend)
      aux      = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {1, seq_len, 192}) |> Nx.backend_transfer(backend)
      mask     = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {1, seq_len, 1}) |> Nx.backend_transfer(backend)
      {tokens, aux, mask}
    end

    # --- Forward pass latency across seq_lens ---
    IO.puts("\n-- Forward pass latency (batch=1, #{@warmup} warmup, #{@runs} runs) --\n")
    IO.puts("  seq_len   min_ms   avg_ms   p50_ms   p90_ms   max_ms")
    IO.puts("  " <> String.duplicate("-", 53))

    for seq_len <- @seq_lens do
      {tokens, aux, mask} = make_inputs.(seq_len)

      # Warmup
      for _ <- 1..@warmup do
        jit_fwd.(tokens, aux, mask, defn_params)
        |> Nx.backend_transfer(Nx.BinaryBackend)
      end

      times =
        for _ <- 1..@runs do
          {ms, _} = timed_ms(fn ->
            jit_fwd.(tokens, aux, mask, defn_params)
            |> Nx.backend_transfer(Nx.BinaryBackend)
          end)
          ms
        end

      {min_ms, avg_ms, p50_ms, p90_ms, max_ms} = stats(times)

      IO.puts(
        "  #{String.pad_leading(to_string(seq_len), 7)}   " <>
          "#{fmt(min_ms)}   #{fmt(avg_ms)}   #{fmt(p50_ms)}   #{fmt(p90_ms)}   #{fmt(max_ms)}"
      )
    end

    # --- End-to-end MTP decode at seq_len=64 ---
    IO.puts("\n-- End-to-end MTP decode (seq_len=64, #{@num_items}-item catalog) --")
    IO.puts("   #{@warmup} warmup, #{@runs} runs\n")

    seq_len = 64
    {tokens, aux, mask} = make_inputs.(seq_len)

    # Build item catalog
    token_lists =
      for i <- 0..(@num_items - 1) do
        [rem(i, 100) + 1, rem(i + 1, 100) + 100, rem(i + 2, 1000) + 200, rem(i + 3, 10000) + 1200]
      end

    item_id_to_tokens =
      token_lists
      |> Nx.tensor(type: {:s, 32})
      |> Nx.backend_transfer(backend)

    get_logits_fn = fn _ctx ->
      jit_fwd.(tokens, aux, mask, defn_params)
    end

    for _ <- 1..@warmup do
      Decode.lookahead_top_k(item_id_to_tokens, [], 5, get_logits_fn, backend)
    end

    mtp_times =
      for _ <- 1..@runs do
        {ms, _} = timed_ms(fn ->
          Decode.lookahead_top_k(item_id_to_tokens, [], 5, get_logits_fn, backend)
        end)
        ms
      end

    {min_ms, avg_ms, p50_ms, p90_ms, max_ms} = stats(mtp_times)
    IO.puts("  min:  #{fmt(min_ms)} ms")
    IO.puts("  avg:  #{fmt(avg_ms)} ms")
    IO.puts("  P50:  #{fmt(p50_ms)} ms")
    IO.puts("  P90:  #{fmt(p90_ms)} ms")
    IO.puts("  max:  #{fmt(max_ms)} ms")
    IO.puts("")
  end

  defp timed_ms(f) do
    {us, result} = :timer.tc(f)
    {us / 1000.0, result}
  end

  defp stats(times) do
    sorted = Enum.sort(times)
    n = length(sorted)
    avg = Enum.sum(times) / n
    p50 = Enum.at(sorted, div(n, 2))
    p90 = Enum.at(sorted, min(round(n * 0.9), n - 1))
    {hd(sorted), avg, p50, p90, List.last(sorted)}
  end

  defp fmt(ms), do: ms |> Float.round(2) |> to_string() |> String.pad_leading(6)
end

FuxiSpeedTest.run()
