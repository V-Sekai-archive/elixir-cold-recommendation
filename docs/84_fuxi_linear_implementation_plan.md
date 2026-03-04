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
| **Chunk processing** | Optional chunk_size for long sequences; not yet ported (full O(n²) for small n). |

---

## RecGPT Semantic ID Interface

| Reference | Our port |
|-----------|----------|
| Jagged tensors (FBGEMM) | Padded (batch, seq_len, dim) — same as RecGPT |
| `all_timestamps` | Position indices; add real timestamps when Tape ready |
| Chunked forward | `chunk_size = nil` (full O(n²)); add chunk later |
| Multiple blocks | 4 blocks (config linear-4b) |

- **Input:** `batch_token_ids` (batch, seq_len), `batch_aux` (batch, seq_len, 192), `embed_mask`
- **Output:** `logits` (batch, 15_361) for last position
- **Params:** Reuse `wte`, `ae.*`, `pred_head` from RecGPT. New: per-block `uvqk`, `retention.gamma`, `channel_t.*`, `channel_p.*`, `mffn.*`.

---

## Usage

```elixir
# Forward (same as Inference)
logits = RecGPT.FuxiLinearInference.forward(token_ids, batch_aux, embed_mask, params)

# Init params for training from scratch
params = RecGPT.FuxiLinearInference.init_params(n_blocks: 4, max_seq_len: 1024)
```

---

## Params / Checkpoint

FuXi-Linear uses different param layout than GPT-2. Options:

- **A:** New checkpoint format; loader maps FuXi keys.
- **B:** Train from scratch with Axon; export to our format.
- **C:** Convert PyTorch checkpoint from fuxi-linear (script).

Rope bridge: Use `init_params/1` for random init; validate forward shape. Add checkpoint once training pipeline exists.
