# Tombstone: FuXi-Linear incremental decode (cache path)

**Status:** Removed. Decode uses a single full forward for the last 4 positions only.

## What was removed

- `forward_with_cache/4` — full forward returning `{logits, fuxi_cache}`.
- `forward_incremental/5` — one-token forward using saved cache.
- All cache-building and incremental block logic: `forward_hidden_with_cache`, `forward_hidden_incremental`, `fuxi_block_with_cache`, `fuxi_block_incremental`, `channel_p_forward_one_row`, `retention_forward_incremental`, `channel_t_forward_with_state`, `channel_t_zero_state`, `channel_t_forward_incremental`.

## Why it was removed

1. **Unused in production.** The serve path uses `get_logits_4_fn` → `FuxiLinearInferenceDefn.forward_last_4_logits/4`, which does one full forward and returns logits for the last 4 positions. No cache or incremental step is ever used.

2. **No benefit for current 4-token decode.** Using the incremental path (1 full forward + 3 incremental steps) would mean 4 graph launches instead of 1, plus cache bookkeeping and Channel P’s O(seq_len) growing buffer per step. That would be slower, not faster.

3. **Benefits only for long autoregressive generation.** The theory (token-by-token decode with cached recurrent state) would help if we ever did many-token LM/chat generation. FuXi’s Channel P still does O(seq_len) work per step, so decode is not O(1) per token. We don’t do long generation today.

## Theory (tombstoned)

- **Idea:** Step 0: full forward over context → logits + cache (Retention S, Channel T state, Channel P history). Steps 1, 2, …: one new token + cache → next logits + updated cache, without re-running the full sequence.
- **Reality:** Current RecGPT recommend = one item = 4 tokens; one full forward for last 4 positions is simpler and faster. The incremental path added code and test surface without being used or beneficial for this use case.

## References

- [27 Latency and performance](../features/27_latency_and_performance.md) — single forward, no KV-cache in FuXi serve path.
- [FUXI_LINEAR_NANOCHAT_INVESTIGATION](FUXI_LINEAR_NANOCHAT_INVESTIGATION.md) — earlier discussion of decode and O(1) per step.
