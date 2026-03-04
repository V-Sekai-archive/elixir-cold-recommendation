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
| **forward_full_sequence/4** | Full-sequence logits for training (AxonTrain) |
| **forward_with_cache/4** | Returns {logits, []}; FuXi uses single-forward decode |
| **forward_incremental/5** | Runs full forward on single token; returns {logits, []} |
| **init_full_params/1** | Full model params for unit tests or training from scratch |
| **init_params/1** | Block params only; use with existing RecGPT wte/ae/pred_head |
| **FuxiLinearInferenceDefn** | Defn JIT `forward_last_4_logits/4` for Serve/Decode |
| **FuxiLinearInferenceParams** | Builds atom-keyed defn params from checkpoint string keys |
| **Serve integration** | Serve detects FuXi checkpoint (`fuxi.block.*`), uses FuxiLinearInferenceDefn |
| **AxonTrain integration** | Pretrain uses FuxiLinearInference.forward_full_sequence when FuXi params |
| **Export mix task** | `mix recgpt.export_fuxi_ckpt --out DIR` exports init params |
| **Unit tests** | `test/recgpt/fuxi_linear_inference_test.exs` — forward, param shapes, error cases |
| **Serve test** | `ServeTest` — load_state with FuXi checkpoint, recommend returns valid IDs |
| **all_timestamps opts** | Optional real timestamps (batch, seq_len, channel_t_heads) for LinearTemporalChannel. Use when Tape/time-series data available. |
| **chunk_size opts** | When set and seq_len &gt; chunk_size, processes in chunks to reduce peak memory. Matches upstream chunking for long sequences. |

---

## Not Implemented

None. Chunk processing and real timestamps are implemented as opts.

---

## See also

- [84 FuXi-Linear implementation plan](84_fuxi_linear_implementation_plan.md) — Architecture, usage, params
- [83 Frontier models EXLA path](83_frontier_models_exla_elixir_path.md) — FuXi, Mamba, RWKV feasibility
