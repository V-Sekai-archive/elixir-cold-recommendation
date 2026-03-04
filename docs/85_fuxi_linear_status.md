# FuXi-Linear Implementation Status

Implementation status of `RecGPT.FuxiLinearInference`: what is done and what remains. See [84 FuXi-Linear implementation plan](84_fuxi_linear_implementation_plan.md) for architecture and usage.

---

## Implemented

| Component | Status |
|-----------|--------|
| **Retention** | Causal linear attention with positional decay (gamma per head) |
| **LinearTemporalChannel** | Timestamp-based Q/K from sinusoidal encoding; decay by intervals; seq_len &lt; 2 path |
| **LinearPositionalChannel** | Learned sinusoidal emb; attn = emb @ emb.T / dim with causal mask |
| **Multistage FFN** | lin0 + residual; lin1/silu×lin3 + lin2 (gated) |
| **forward/4** | Same interface as `Inference`: (batch_token_ids, batch_aux, embed_mask, params) → logits (batch, 15_361) |
| **init_full_params/1** | Full model params for unit tests or training from scratch |
| **init_params/1** | Block params only; use with existing RecGPT wte/ae/pred_head |
| **Unit tests** | `test/recgpt/fuxi_linear_inference_test.exs` — forward, param shapes, error cases |

---

## Not Implemented

### 1. Defn / JIT entry for Serve/Decode

No `FuxiLinearInferenceDefn` (or similar) that exposes `forward_last_4_logits` for beam search. Serve uses `InferenceDefn` only; FuXi-Linear cannot drive `mix recgpt.serve` or recommendation yet.

**Work:** Add `FuxiLinearInferenceDefn` module; JIT `forward_last_4_logits/4` with EXLA; wire Serve to choose FuXi when checkpoint format is FuXi.

### 2. Checkpoint loader and format

No loader maps FuXi keys (`fuxi.block.*`, `channel_t.*`, `channel_p.*`, `mffn.*`) into a format Serve or training can use. No PyTorch→Elixir conversion script.

**Work:** Option A: new checkpoint format + loader. Option B: train from scratch, export to our format. Option C: script to convert PyTorch checkpoint from fuxi-linear.

### 3. Training support

No `forward_full_sequence` (full-sequence logits for cross-entropy loss). No `forward_with_cache` or `forward_incremental` (KV-cache for incremental decode). `AxonTrain` and pretrain assume `Inference.forward_full_sequence`.

**Work:** Add `forward_full_sequence/4` for training; optionally add cache variants if incremental decode is needed for FuXi.

### 4. Chunk processing

Chunking for long sequences is not ported. Forward is full O(n²) per block. Acceptable for small n (e.g. typical RecGPT context length).

**Work:** Add optional `chunk_size` to block forward; use when seq_len &gt; threshold.

### 5. Real timestamps

Uses position indices as `all_timestamps`. Real timestamps (e.g. from Tape) are not wired.

**Work:** Add optional `all_timestamps` argument to `forward_hidden` when Tape / time series are available.

---

## See also

- [84 FuXi-Linear implementation plan](84_fuxi_linear_implementation_plan.md) — Architecture, usage, params
- [83 Frontier models EXLA path](83_frontier_models_exla_elixir_path.md) — FuXi, Mamba, RWKV feasibility
