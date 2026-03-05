# Plan: Single-Forward Decode (Python RecGPT Style)

## Goal

Replace 4 incremental forwards (with KV cache) with **one full-sequence forward**, then run beam search over the precomputed last-4 logits. Keeps trie and beam search on GPU.

## Design

### Current (4-forward)

1. Forward on context → logits for last position, cache
2. For steps 1–3: extend by beam token, incremental forward with cache → logits
3. Beam search interleaved: trie mask, topk, prune

### New (single-forward)

1. **One** forward on context → hidden `(batch, seq_len, 768)`
2. Apply head to **last 4 positions** → logits `(batch, 4, vocab)`
3. Beam search over those 4 slices: step i uses `logits[:, i, :]`, trie mask, topk, prune (no further model calls)

### Trade-off

- **Speed:** 1 kernel launch instead of 4; less launch overhead.
- **Accuracy:** The 4 logits are conditioned on context only (not on previous beam choices). Python RecGPT uses this; current Elixir does proper autoregressive conditioning. Slight approximation for latency gain.

## Implementation

### 1. InferenceDefn: `forward_last_4_logits/4`

- Same as `forward_with_cache` but:
  - Apply head to last 4 positions of hidden (not just last 1)
  - Return `{logits_4, nil}` where `logits_4` has shape `(batch, 4, vocab_size)`
  - No cache (unused in single-forward path)

### 2. Decode: `beam_search_single_forward_spmd/...`

- New function with signature compatible with `beam_search_top_k_spmd` call site.
- Takes `get_logits_4_fn: (context_tokens) -> logits_4` instead of `batch_tensor_fn`.
- Runs same 4-step beam logic but uses `logits_4[:, i, :]` at step i.
- Reuses trie tensors, `gather_2d`, and final sync/top-k selection.

### 3. Serve: config + wiring

- Config: `config :recgpt, :single_forward_decode, true`
- When true: build `get_logits_4_fn` (JIT of `forward_last_4_logits`), pass to new Decode path
- When false: keep current `get_logits_batch_tensor_fn` + `beam_search_top_k_spmd`

### 4. Context cache

- Single-forward path: cache key = context_ids; value = `logits_4` (no cache tuple)
- On hit: skip forward, use cached logits_4 for beam search

## Files Changed

| File                           | Change                                                                                        |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| `lib/recgpt/inference_defn.ex` | Added `forward_last_4_logits/4`                                                               |
| `lib/recgpt/decode.ex`         | Added `beam_search_single_forward_spmd/8`, `run_single_forward_beam`, `spmd_step_from_logits` |
| `lib/recgpt/serve.ex`          | Config flag, `get_logits_4_fn`, recommend branch                                              |
| `config/config.exs`            | Added `single_forward_decode` (RECGPT_SINGLE_FORWARD=1)                                       |

## Usage

Enable single-forward decode:

```bash
RECGPT_SINGLE_FORWARD=1 mix recgpt.serve --fixture ... --ckpt ...
```

Or in config:

```elixir
config :recgpt, :single_forward_decode, true
```
