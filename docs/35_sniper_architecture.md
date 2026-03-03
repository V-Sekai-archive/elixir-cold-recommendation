# Sniper: Architecture — Scout + Gatekeeper

RecGPT Scout and Qwen3 Gatekeeper roles, latency, and training distinction (pretraining vs LoRA finetuning).

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Architecture

| Component | Role | Latency |
|-----------|-----|---------|
| **RecGPT (Scout)** | Top-1 candidate from context; outputs single best item_id | ~200–280 ms warm |
| **Qwen3 (Gatekeeper)** | Veto or approve; sees Tape + XMP + Scout output; returns PICK_ID or PICK_0 | Low-latency veto |
| **Execution window** | 10 minutes (Catalyst/Combinatorial) | Opportunity binds |

Scout feeds the Gatekeeper exactly one candidate. Gatekeeper enforces Triple-Lock before a trade is placed.

---

## Training Distinction

| Model | Training | Output | Tooling |
|-------|----------|--------|---------|
| **RecGPT (Scout)** | **Pretraining** — next-token prediction on item sequences (train_sequences.json, item embeddings) | LoRA or full checkpoint; `mix recgpt.pretrain` | Elixir/Axon; [33 MovieLens 5-epoch](33_movielens_5_epoch_pretrain.md) |
| **Qwen3 (Gatekeeper)** | **LoRA finetuning** — GRPO on veto/strike scenarios | LoRA adapter over base Qwen3 | [OpenPipe ART](https://github.com/OpenPipe/ART) (Python) |

- **RecGPT pretraining:** Learns item correlations, sequences, catalog intent from historical data. No veto logic; outputs probability distribution over item_ids.
- **Qwen LoRA finetuning:** Learns PICK_ID vs PICK_0 from reward-shaped rollouts. Base model (e.g. Qwen3-7B-Instruct) + LoRA weights; ART trains LoRA only.

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [36 Schema](36_sniper_schema.md) — JSON-LD + XMP-JSON-LD
- [38 Qwen LoRA](38_sniper_qwen_lora.md) — GRPO reward and ART
