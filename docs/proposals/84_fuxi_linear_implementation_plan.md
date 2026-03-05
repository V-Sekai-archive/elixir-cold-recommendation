# FuXi-Linear Implementation

Full FuXi-Linear model in Nx/Elixir for Scout backbone, using RecGPT semantic ID structure. Reference: [USTC-StarTeam/fuxi-linear](https://github.com/USTC-StarTeam/fuxi-linear).

Related: [83 Frontier models EXLA path](83_frontier_models_exla_elixir_path.md), [80 Prediction market trading](80_prediction_market_trading_system.md).

---

## Status

**Implemented** in `RecGPT.FuxiLinearInference` — full model: Retention + LinearTemporalChannel + LinearPositionalChannel, per reference. Interface matches `RecGPT.Inference`.

---

## Architecture (from reference)

| Component | Role |
|-----------|------|
| **Retention** | Causal linear attention with positional decay (gamma per head). `attn[i,j] = exp(-gamma * (i-j))` for i≥j. |
| **LinearTemporalChannel** | Timestamp-based Q/K from sinusoidal encoding; decay by time intervals. Uses `all_timestamps`. |
| **LinearPositionalChannel** | Learned sinusoidal positional embeddings; `attn = emb @ emb.T / dim` with causal mask. |
| **Multistage FFN** | lin0 + residual; lin1/silu*lin3 + lin2 (gated). |
| **Chunk processing** | Optional `chunk_size` opts when seq_len &gt; chunk_size; reduces peak memory. |

---

## RecGPT Semantic ID Interface

| Reference | Our port |
|-----------|----------|
| Jagged tensors (FBGEMM) | Padded (batch, seq_len, dim) — same as RecGPT |
| `all_timestamps` | Position indices by default; `opts[:all_timestamps]` for real timestamps (Tape/time-series) |
| Chunked forward | `opts[:chunk_size]` when seq_len &gt; chunk_size; reduces peak memory |
| Multiple blocks | 4 blocks (config linear-4b) |

- **Input:** `batch_token_ids` (batch, seq_len), `batch_aux` (batch, seq_len, 192), `embed_mask`
- **Output:** `logits` (batch, 15_361) for last position
- **Params:** Reuse `wte`, `ae.*`, `pred_head` from RecGPT. New: per-block `uvqk`, `retention.gamma`, `channel_t.*`, `channel_p.*`, `mffn.*`.

---

## Usage

```elixir
# Forward (same as Inference)
logits = RecGPT.FuxiLinearInference.forward(token_ids, batch_aux, embed_mask, params)

# With real timestamps (Tape/time-series)
logits = RecGPT.FuxiLinearInference.forward(token_ids, batch_aux, embed_mask, params,
  all_timestamps: real_ts  # (batch, seq_len, 8)

# Chunked for long sequences (reduces peak memory)
logits = RecGPT.FuxiLinearInference.forward(token_ids, batch_aux, embed_mask, params,
  chunk_size: 64)

# Init params for training from scratch
params = RecGPT.FuxiLinearInference.init_params(n_blocks: 4, max_seq_len: 1024)
```

---

## Params / Checkpoint

FuXi-Linear uses different param layout than GPT-2. **No pre-trained checkpoint exists** — see [85 FuXi-Linear status](85_fuxi_linear_status.md#checkpoint).

- **A (implemented):** `FuxiLinearInferenceParams` maps FuXi keys (`fuxi.block.*`) to defn params. `mix recgpt.export_fuxi_ckpt --out DIR` exports **init params** (random). Serve auto-detects FuXi checkpoints.
- **B:** Train from scratch with Axon; export to our format.
- **C:** Convert PyTorch checkpoint from fuxi-linear (script) — upstream does not publish trained weights.
