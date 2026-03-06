#!/usr/bin/env elixir

# Simple baseline test for FuXi Linear + MTP
# Uses existing FrozenHelpers to create FuXi stub state

Application.ensure_all_started(:nx)
Application.ensure_all_started(:jason)

# Force MTP decode strategy
Application.put_env(:recgpt, :decode_strategy, :mtp)

alias RecGPT.TestSupport.FrozenHelpers
alias RecGPT.Decode

IO.puts("=== FuXi Linear + MTP Baseline (Simple) ===")
IO.puts("Backend: #{inspect(Nx.default_backend())}")
IO.puts("Decode Strategy: #{Application.get_env(:recgpt, :decode_strategy)}")
IO.puts("")

# Create FuXi stub checkpoint
ckpt_dir = Path.expand("data/fuxi_test_ckpt_export", File.cwd!())
IO.puts("Creating FuXi stub checkpoint...")
FrozenHelpers.write_fuxi_stub_ckpt!(ckpt_dir)

# Create minimal fixture
fixture_path = Path.expand("data/fuxi_test_fixture.json", File.cwd!())
File.write!(fixture_path, Jason.encode!(%{
  "token_id_list" => [
    [100, 200, 300, 400],
    [101, 201, 301, 401],
    [102, 202, 302, 402],
    [103, 203, 303, 403],
    [104, 204, 304, 404]
  ],
  "num_items" => 5
}, pretty: true))

IO.puts("Loading FuXi state...")
{setup_time, state} = :timer.tc(fn ->
  case RecGPT.Serve.load_state(fixture_path, ckpt_dir, nil) do
    {:ok, s} -> s
    {:error, reason} -> raise "Load failed: #{inspect(reason)}"
  end
end)

IO.puts("Setup: #{setup_time / 1000} ms")
IO.puts("State type: #{if state.params["wte"], do: "Transformer", else: "FuXi Linear"}")

# WARMUP: Run 3 iterations
IO.puts("")
IO.puts("Warming up (3 iterations)...")
for i <- 1..3 do
  {warmup_time, _} = :timer.tc(fn ->
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

# MEASUREMENT: Run 10 iterations
IO.puts("")
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
  IO.puts("  Run #{i}: #{t / 1000} ms")
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

# Cleanup
File.rm_rf!(ckpt_dir)
File.rm!(fixture_path)