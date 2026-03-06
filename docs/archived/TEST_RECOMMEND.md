# Testing the Recommend Functionality

## Overview

The recommend functionality is implemented in `RecGPT.Serve.recommend/3` and is the core of the gRPC PredictionService. This document explains how to test it.

## Implementation

The `recommend/3` function is located in `/workspaces/elixir-recgpt/lib/recgpt/serve.ex`:

```elixir
def recommend(state, item_ids, top_k \\ 5)
    when is_list(item_ids) and is_integer(top_k) and top_k >= 1 do
  if item_ids == [] do
    {:error, "item_ids must be non-empty"}
  else
    top_k = min(top_k, 20)
    
    result =
      Decode.lookahead_top_k(
        state.item_id_to_tokens_tensor,
        item_ids,
        top_k,
        state.get_logits_4_fn,
        state.inference_backend
      )
    
    case result do
      {:ok, list} -> {:ok, list}
      :not_found -> {:ok, []}
    end
  end
end
```

### Key Features:
- **Input validation**: Requires non-empty `item_ids` list
- **Top-k capping**: Automatically caps `top_k` at 20 (max allowed)
- **Delegates to Decode**: Uses `Decode.lookahead_top_k/5` for actual recommendation
- **Returns**: `{:ok, [item_ids]}` or `{:error, reason}`

## Existing Tests

### Unit Tests (test/recgpt/serve_test.exs)

The following tests cover the recommend functionality:

1. **Empty context test**: Verifies error when item_ids is empty
2. **Top-k results test**: Verifies returns up to top_k recommended item_ids
3. **Top-k=1 test**: Verifies single recommendation works
4. **Top-k cap test**: Verifies top_k is capped at 20 even when requesting more
5. **Frozen vs Serve consistency**: Verifies LayerFreeze.recommend matches Serve.recommend

### Test Code Example:

```elixir
describe "recommend/3" do
  test "returns error when item_ids empty" do
    frozen = FrozenHelpers.build_frozen([0])
    assert LayerFreeze.recommend(frozen, [], 5) == {:error, "item_ids must be non-empty"}
  end

  test "returns up to top_k recommended item_ids (best first) via frozen inputs" do
    frozen = FrozenHelpers.build_frozen([0])
    assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 5)
    assert is_list(list)
    assert length(list) <= 5
    assert length(list) <= 2, "stub catalog has 2 items"
    Enum.each(list, fn id -> assert id in [0, 1] end)
  end

  test "top_k=1 returns at most one item" do
    frozen = FrozenHelpers.build_frozen([0])
    assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 1)
    assert length(list) <= 1
  end

  test "top_k is capped at 20" do
    frozen = FrozenHelpers.build_frozen([0])
    assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 100)
    assert length(list) <= 20
  end
end
```

## How to Run Tests

### Running Unit Tests

```bash
# Run serve tests (includes recommend tests)
MIX_ENV=test mix test test/recgpt/serve_test.exs

# Run only unit tests (exclude integration)
MIX_ENV=test mix test test/recgpt/serve_test.exs --exclude integration

# Run with trace output
MIX_ENV=test mix test test/recgpt/serve_test.exs --trace
```

### Running Prediction Service Tests

The gRPC PredictionService also has tests that use the recommend functionality:

```bash
MIX_ENV=test mix test test/recgpt/v1/prediction_service_test.exs
```

## Manual Testing

### Using the gRPC Server

1. **Start the server**:
```bash
mix recgpt.serve --grpc-port 50051
```

2. **Test with grpcurl**:
```bash
# Using the provided script
./scripts/grpcurl_recommend.sh 0 1 2

# Or directly with grpcurl
grpcurl -plaintext \
  -proto priv/proto/recgpt/v1/recommendation.proto \
  -d '{"context_item_ids": [0, 1], "max_results": 5}' \
  localhost:50051 \
  recgpt.v1.PredictionService/Predict
```

### Using Mix Tasks

```bash
# Ad-hoc testing with custom contexts
mix recgpt.ad_hoc_test --contexts "0,1|1,2,3" --top-k 5

# Trace performance of recommend calls
mix recgpt.trace_predict --runs 10 --context "0,1"
```

## Test Helpers

The `RecGPT.TestSupport.FrozenHelpers` module provides utilities for testing:

- `build_stub_state/0` - Creates minimal stub state with 2 items
- `build_stub_state/1` - Creates stub state with n items
- `build_frozen/1` - Creates frozen layer inputs for isolated testing

Example usage:
```elixir
alias RecGPT.TestSupport.FrozenHelpers

# Build stub state for testing
state = FrozenHelpers.build_stub_state()

# Test recommend
{:ok, results} = RecGPT.Serve.recommend(state, [0], 5)
```

## Environment Requirements

### For Unit Tests
- Nx.BinaryBackend (configured in test.exs)
- No GPU or EXLA required for unit tests

### For Integration Tests
- EXLA with CUDA support
- Real checkpoint and fixture files
- GPU recommended for performance testing

## Known Issues

### EXLA Loading Error

If you encounter EXLA loading errors like:
```
Failed to load NIF library: 'nvshmem_transport_ibrc.so.3: cannot open shared object file'
```

This indicates EXLA is trying to load NVIDIA libraries that aren't available. Solutions:

1. **For unit tests**: Ensure `config/test.exs` sets `Nx.BinaryBackend`
2. **For development**: Install proper CUDA drivers or use CPU-only mode
3. **Temporary workaround**: Comment out EXLA in mix.exs deps for testing only

## Related Files

- **Implementation**: `lib/recgpt/serve.ex`
- **Tests**: `test/recgpt/serve_test.exs`, `test/recgpt/v1/prediction_service_test.exs`
- **Test Helpers**: `lib/recgpt/test_support/frozen_helpers.ex`
- **gRPC Contract**: `priv/proto/recgpt/v1/recommendation.proto`
- **Scripts**: `scripts/grpcurl_recommend.sh`

## API Contract

From `priv/proto/recgpt/v1/recommendation.proto`:

```protobuf
message PredictRequest {
  repeated int32 context_item_ids = 1;  // Required, non-empty
  int32 max_results = 2;                 // Optional, default 5, max 20
  optional int32 rank = 3;               // SPMD routing (optional)
}

message PredictResponse {
  repeated int32 item_ids = 1;           // Ordered recommended IDs
  repeated ItemSummary items = 2;        // Item details with display_name
}
```

## Performance Expectations

Based on `mix recgpt.trace_predict`:
- **Setup (one-time)**: ~20-30s (JIT compilation)
- **Per-request**: ~300-400ms on 12-layer model + RTX 4090
- **Most time**: Spent in beam search (4 forward passes)

## Next Steps

To fully test the recommend functionality:

1. ✅ Review existing unit tests in `test/recgpt/serve_test.exs`
2. ✅ Run unit tests with `MIX_ENV=test mix test test/recgpt/serve_test.exs`
3. ⚠️ Fix EXLA dependency issue for integration tests
4. ✅ Test gRPC endpoint manually with grpcurl
5. ✅ Use `mix recgpt.ad_hoc_test` for quick validation
