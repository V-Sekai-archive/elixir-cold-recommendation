# Parity overview

Sub-proposal of the [documentation index](README.md). At-a-glance status and mapping from the [RecGPT reference](https://github.com/HKUDS/RecGPT) to the **recgpt** Elixir package. Per-layer detail: [10 Parity by layer](10_parity_layers.md).

---

## Problem or limitation

We need to track what is implemented vs. the RecGPT reference, with no Python in this codebase. Without a single parity document, it is unclear which layers are done, how they are validated, and what gaps remain.

---

## Proposed improvement

Maintain **parity progress** docs: at-a-glance table, reference-to-Elixir mapping, and per-layer task lists ([10](10_parity_layers.md)). Parity is validated via unit and integration tests only. Reference: [RecGPT paper](https://arxiv.org/abs/2506.06270), [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model).

---

## At a glance

| Layer                                      | Status | Elixir modules / scripts                                           |
| ------------------------------------------ | ------ | ------------------------------------------------------------------ |
| Text → 768-d                               | ✅     | `RecGPT.Embedding`                                                 |
| FSQ (768→4 tokens)                         | ✅     | `RecGPT.FSQ`, `RecGPT.FSQEncoder`                                  |
| Training data + loss                       | ✅     | `RecGPT.Training`                                                  |
| Checkpoint load                            | ✅     | `RecGPT.CheckpointLoader`                                          |
| Model forward (embed + aux + GPT-2 + head) | ✅     | `RecGPT.Inference` (full backbone when params have gpt2model.h.\*) |
| Decode (trie + beam)                       | ✅     | `RecGPT.Trie`, `RecGPT.Decode`                                     |

**Data pipeline:** item text → embeddings → token_id_list → train batches ✅ — [integration tested](../test/recgpt/pipeline_integration_test.exs). **Checkpoint:** export via `mix recgpt.export_ckpt --from-pt <path> --out DIR` → [CheckpointLoader.load_from_export/1](../lib/recgpt/checkpoint_loader.ex). **Inference in Elixir:** load checkpoint → forward (embed + aux + GPT-2 + head) → beam search → item_id ✅ (`RecGPT.Trie.build/1`, `RecGPT.Decode.beam_search/4`).

---

## Reference to Elixir mapping

| Reference (RecGPT repo)                                 | Elixir (recgpt)                                                            | Notes                                                                                                                       |
| ------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| sentence-transformers / MPNet 768-d                     | `RecGPT.Embedding` (Bumblebee, all-mpnet-base-v2)                          | Same model id; parity via unit tests.                                                                                       |
| Token list (reference)                                  | `RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3`                   | Unit and pipeline integration tests.                                                                                        |
| FSQ (reference: levels, bound, quantize, codes↔indices) | `RecGPT.FSQ`                                                               | Unit tests.                                                                                                                 |
| VAE weights `vae_len4_fsq88865_ep90.pt`                 | Weights from export → `FSQ.load_params/1`                                  | Encoder logic in Elixir; export via `mix recgpt.export_ckpt` or reference.                                                  |
| `GPT2RecBatchTrainAuxData`, batch build, loss           | `RecGPT.Training.build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2` | No model forward in package.                                                                                                |
| Pretrain / Eval                                         | `mix recgpt.pretrain` / `mix recgpt.eval`                                  | Full pipeline in Elixir; see [02](02_pipeline_overview.md), [03](03_pipeline_steps.md), [06](06_evaluation_and_testing.md). |
| Serve (reference: HTTP)                                 | **`RecGPT.Serve` + `mix recgpt.serve`**                                    | gRPC only: recgpt.v1.PredictionService/Predict (see recommendation.proto).                                                  |

---

## Summary

| Area                                           | Parity                      | Blocker / note                                                               |
| ---------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| Embeddings                                     | ✅ Implemented              | Same model id; unit tested.                                                  |
| FSQ + FSQEncoder                               | ✅ Implemented, unit tested | Same logic as reference.                                                     |
| Training data + loss                           | ✅ Implemented              | Same shapes and loss as paper.                                               |
| Checkpoint loader                              | ✅ Implemented              | `CheckpointLoader.load_from_export/1`; export via `mix recgpt.export_ckpt`.  |
| Inference forward (embed + aux + GPT-2 + head) | ✅ Implemented              | `RecGPT.Inference.forward/4`; full backbone when params have gpt2model.h.\*. |
| Decode (trie + beam)                           | ✅ Implemented              | `RecGPT.Trie`, `RecGPT.Decode.beam_search/4`.                                |

**Next steps for full parity:** (Optional) Numerical parity: compare Elixir `Inference.forward` logits with reference (e.g. exported golden logits) on same token sequences.

---

## See also

- [09 Parity by layer](09_parity_layers.md) — Per-layer task lists, validate, gaps.
- [04 RecGPT library](04_recgpt_library.md) — Module reference.
- [08 Checkpoint layout](08_recgpt_checkpoint_layout.md) — Export and loader.
