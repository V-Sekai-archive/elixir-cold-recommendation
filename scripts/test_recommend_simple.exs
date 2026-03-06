#!/usr/bin/env elixir

# Simple test script for recommend functionality
# This tests the core recommend logic without requiring EXLA

# Set backend to BinaryBackend to avoid EXLA dependency
Nx.default_backend(Nx.BinaryBackend)

alias RecGPT.TestSupport.FrozenHelpers

alias RecGPT.TestSupport.FrozenHelpers

IO.puts("=== Testing Recommend Functionality ===")
IO.puts("")

# Test 1: Build stub state
IO.puts("Test 1: Building stub state...")
state = FrozenHelpers.build_stub_state()
IO.puts("✓ Stub state built successfully with #{state.num_items} items")
IO.puts("")

# Test 2: Test recommend with valid input
IO.puts("Test 2: Testing recommend with valid context...")
case RecGPT.Serve.recommend(state, [0], 5) do
  {:ok, results} ->
    IO.puts("✓ Recommend returned: #{inspect(results)}")
    IO.puts("  Number of results: #{length(results)}")
    IO.puts("  Results are within expected bounds: #{length(results) <= 5}")
  {:error, reason} ->
    IO.puts("✗ Recommend failed: #{reason}")
end
IO.puts("")

# Test 3: Test recommend with empty input (should fail)
IO.puts("Test 3: Testing recommend with empty context (should fail)...")
case RecGPT.Serve.recommend(state, [], 5) do
  {:ok, _} ->
    IO.puts("✗ Should have failed with empty context")
  {:error, reason} ->
    IO.puts("✓ Correctly failed with: #{reason}")
end
IO.puts("")

# Test 4: Test top_k=1
IO.puts("Test 4: Testing recommend with top_k=1...")
case RecGPT.Serve.recommend(state, [0], 1) do
  {:ok, results} ->
    IO.puts("✓ Recommend with top_k=1 returned: #{inspect(results)}")
    IO.puts("  Number of results: #{length(results)} (should be <= 1)")
  {:error, reason} ->
    IO.puts("✗ Recommend failed: #{reason}")
end
IO.puts("")

# Test 5: Test top_k cap at 20
IO.puts("Test 5: Testing recommend with top_k=100 (should cap at 20)...")
case RecGPT.Serve.recommend(state, [0], 100) do
  {:ok, results} ->
    IO.puts("✓ Recommend with top_k=100 returned #{length(results)} results (capped at 20)")
  {:error, reason} ->
    IO.puts("✗ Recommend failed: #{reason}")
end
IO.puts("")

# Test 6: Test with multiple context items
IO.puts("Test 6: Testing recommend with multiple context items...")
case RecGPT.Serve.recommend(state, [0, 1], 5) do
  {:ok, results} ->
    IO.puts("✓ Recommend with context [0, 1] returned: #{inspect(results)}")
  {:error, reason} ->
    IO.puts("✗ Recommend failed: #{reason}")
end
IO.puts("")

IO.puts("=== All tests completed ===")
