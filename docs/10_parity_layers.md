# Parity by layer

Sub-proposal of the [documentation index](README.md). Per-layer task lists and validation for RecGPT parity. Overview and mapping: [09 Parity overview](09_parity_overview.md).

## Problem or limitation

Per-layer parity with the RecGPT reference must be tracked and validated. Without task lists and validation steps per layer, gaps and regressions go unnoticed.

---

## Proposed improvement

Document per-layer task lists and validation; summarize how to validate and note gaps and risks. Overview and mapping: [09 Parity overview](09_parity_overview.md).

---

## 1. Text â†’ embeddings (768-d)

| Task                                                 | Status        | Notes                                                                                  |
| ---------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------- |
| Text â†’ 768-d embeddings                            | âœ… Done      | `RecGPT.Embedding`: Bumblebee + sentence-transformers/all-mpnet-base-v2, mean pooling. |
| Match reference MPNet (`normalize_embeddings=False`) | âœ… Validated | Same model id; `embedding_processor: nil` (no L2). Parity via unit tests.              |
| encode_item_text_dict (map â†’ tensor)               | âœ… Done      | Sorted keys, encode_texts, stack.                                                      |

---

## 2. FSQ (4 tokens per item, vocab 15360)

| Task                                                      | Status   | Notes                                                                                               |
| --------------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------- |
| FSQ levels [8,8,8,6,5], basis, bound, round_ste, quantize | âœ… Done | `RecGPT.FSQ` port of RecGPT reference FSQ.                                                          |
| codes_to_indices / indices_to_codes                       | âœ… Done | Round-trip and encode path covered by tests.                                                        |
| Load params (project_in / project_out) from export        | âœ… Done | `load_params/1`; keys `project_in/kernel` or `fsq.project_in.weight`, transpose when shape {5,192}. |
| **FSQ encode: 768-d â†’ 4 token IDs**                     | âœ… Done | `RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3`; matches reference token-list logic.       |
| **Reference vs Elixir FSQ parity**                        | âœ… Done | Parity via unit tests and pipeline integration tests.                                               |

---

## 3. Training data and loss

| Task                                                             | Status       | Notes                                                                                                                                                                                                               |
| ---------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build train batch (seqs â†’ batch_seq, labels, aux_embeds, mask) | âœ… Done     | `RecGPT.Training.build_train_batch/4`; padding_id 15360, label_ignore -100, max_length 256, seq_token_capacity 1024.                                                                                                |
| Encode aux (item ids â†’ 192-d per position, mask)               | âœ… Done     | `encode_aux/3`; gather item_embeddings, reshape to (n\*4, 192), mask for valid/padding.                                                                                                                             |
| Loss: shifted CE over FSQ vocab                                  | âœ… Done     | `loss_shifted_ce/2`; ignore -100, mean over valid positions.                                                                                                                                                        |
| Same batch shape / format as RecGPT reference                    | â†’ Verified | Matches HKUDS/RecGPT `utils/data.py` `GPT2RecBatchTrainAuxData`: max_length=256, padding_id=15360, seq_token_capacity=1024, label_ignore=-100, right-padding; aux (256 items â†’ 1024�192) and mask (1024�1) match. |

**FSQ and batch verification:** Encode path and batch format match reference (unit and pipeline integration tests). Batch constants: max_length=256, padding_id=15360, seq_token_capacity=1024; see [HKUDS/RecGPT utils/data.py](https://github.com/HKUDS/RecGPT/blob/main/utils/data.py) `GPT2RecBatchTrainAuxData`.

---

## 4. Model forward (inference / training)

| Task                                                      | Status   | Notes                                                                                                                                                         |
| --------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Checkpoint loader**                                     | âœ… Done | `RecGPT.CheckpointLoader.load_from_export/1`; reads manifest.json + .npy from export dir (see [08_recgpt_checkpoint_layout](08_recgpt_checkpoint_layout.md)). |
| FSQ token embedding table (15361, 768)                    | âœ… Done | Lookup in `RecGPT.Inference.forward/4`; params `wte` or `gpt2model.wte.weight`. Uses first 15361 rows when checkpoint has GPT-2 vocab (50257).                |
| Auxiliary encoder (aux â†’ 768-d, fuse with token embeds) | âœ… Done | `Inference` applies linear + LayerNorm, masks, adds to token embeds. Accepts `ae.*` or `linear_layer.weight` / `norm_aux.*` (RecGPT export keys).             |
| Prediction head (768 â†’ 15361 logits)                    | âœ… Done | `Inference.apply_head/2`; params `pred_head.weight`, `pred_head.bias`.                                                                                        |
| **Forward: embed + aux + head**                           | âœ… Done | `RecGPT.Inference.forward/4`; returns logits (batch, 15361).                                                                                                  |
| **GPT-2 backbone (full transformer)**                     | âœ… Done | When params include `gpt2model.h.{i}.*` (or `transformer.h.{i}.*`): pre-norm attn + MLP blocks, optional wpe, ln_f. Stub used when no layer params.           |

**Checkpoint key compatibility:** Inference accepts RecGPT export keys: `gpt2model.wte.weight` (sliced to 15361 rows when checkpoint has full GPT-2 vocab), `gpt2model.wpe.weight`, `gpt2model.h.{i}.*`, `linear_layer.weight`/`norm_aux.*` for aux encoder, `pred_head.weight`/`pred_head.bias` for head. See [08_recgpt_checkpoint_layout](08_recgpt_checkpoint_layout.md).

---

## 5. Decoding and catalog

| Task                                         | Status   | Notes                                                                                                                                              |
| -------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trie (4-token sequence â†’ item_id)**      | âœ… Done | `RecGPT.Trie.build/1`, `lookup/2`, `valid_next_tokens/2`; built from token_id_list.                                                                |
| **Beam search (4 steps, valid next tokens)** | âœ… Done | `RecGPT.Decode.beam_search/4`; get_logits_fn + trie + context + beam_width.                                                                        |
| Next-item prediction API                     | âœ… Done | Integration test: load checkpoint â†’ trie from token_id_list â†’ get_logits_fn(forward) â†’ Decode.beam_search; run with `--include integration`. |

---

## 6. Checkpoint and export

| Task                                                 | Status        | Notes                                                                                                              |
| ---------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| Export FSQ weights from RecGPT checkpoint for recgpt | âœ… Done      | FSQ params in checkpoint export (manifest + .npy); `FSQ.load_params/1` from CheckpointLoader.                      |
| Export RecGPT .pt to manifest + .npy for Elixir      | âœ… Done      | `mix recgpt.export_ckpt --from-pt <path> --out DIR`.                                                               |
| Load RecGPT checkpoint in Elixir                     | âœ… Done      | `RecGPT.CheckpointLoader.load_from_export/1`; expects manifest.json + .npy in export dir.                          |
| **Load + forward with real checkpoint**              | âœ… Validated | Integration test loads `data/recgpt_ckpt_export` and runs `Inference.forward/4`; run with `--include integration`. |

---

## 7. End-to-end flows

| Flow                                                                                       | Status   | Notes                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data only:** item_text_dict â†’ embeddings â†’ token_id_list â†’ train batches           | âœ… Done | Elixir only: Embedding + FSQEncoder + Training.                                                                                                                                    |
| **Train E2E (Elixir):** fetch → build_fixture → pretrain → eval                            | â†’ Done | Pipeline in Elixir: `mix recgpt.fetch_steam`, `mix recgpt.build_fixture`, `mix recgpt.pretrain`, `mix recgpt.eval`; see [02](02_pipeline_overview.md), [03](03_pipeline_steps.md). |
| **Inference in Elixir:** load checkpoint â†’ forward â†’ beam â†’ item_id                  | âœ… Done | Loader + Inference.forward + Trie + Decode.beam_search. Real checkpoint load + forward + beam_search covered by integration test; run with `--include integration`.                |
| **Serve E2E (serve/predict flow)**                                                         | âœ… Done | Fixture and tests live in a separate repo; set RECGPT_FIXTURE to use that fixture with `mix recgpt.serve`.                                                                         |
| **Zero-shot eval (Elixir):** load base checkpoint, run `mix recgpt.eval` on test_sequences | â†’ Done | Eval with zero-shot or trained checkpoint; `RecGPT.Eval.evaluate/3`; see [06](06_evaluation_and_testing.md).                                                                       |

---

## How to validate

| What                   | Command                                                                          |
| ---------------------- | -------------------------------------------------------------------------------- |
| Unit tests (default)   | `mix test` (excludes integration)                                                |
| With integration       | `mix test --include integration`                                                 |
| Checkpoint + inference | Export ckpt then `mix test test/recgpt/inference_test.exs --include integration` |
| Trie + Decode          | `mix test test/recgpt/trie_test.exs test/recgpt/decode_test.exs`                 |

---

## Gaps and risks

| Gap                                | Impact                                             | Mitigation                                                                                                                                                    |
| ---------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedding numerical parity         | Elixir vs reference embeddings may differ slightly | Same model id; validate via unit tests.                                                                                                                       |
| ~~Batch format vs reference~~      | Closed                                             | Verified against [HKUDS/RecGPT utils/data.py](https://github.com/HKUDS/RecGPT/blob/main/utils/data.py) `GPT2RecBatchTrainAuxData`: same constants and shapes. |
| ~~Beam + trie with real model~~    | Closed                                             | Integration test: load export â†’ trie â†’ get_logits_fn(forward) â†’ beam_search returns `{:ok, item_id}` (`--include integration`).                         |
| Empty batch / empty seq edge cases | Nx limits (e.g., empty tensor)                     | Tests document RuntimeError; callers avoid empty batches.                                                                                                     |
| Empty item_text_dict               | encode_texts([]) fails (tokenizer)                 | Test asserts Enum.EmptyError; doc: use non-empty dict.                                                                                                        |

---

## Summary

| Area                                           | Parity                       | Blocker / note                                                               |
| ---------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------- |
| Embeddings                                     | â†’ Implemented              | Same model id; unit tested.                                                  |
| FSQ + FSQEncoder                               | â†’ Implemented, unit tested | Same logic as reference.                                                     |
| Training data + loss                           | â†’ Implemented              | Same shapes and loss as paper.                                               |
| Checkpoint loader                              | â†’ Implemented              | `CheckpointLoader.load_from_export/1`; export via `mix recgpt.export_ckpt`.  |
| Inference forward (embed + aux + GPT-2 + head) | â†’ Implemented              | `RecGPT.Inference.forward/4`; full backbone when params have gpt2model.h.\*. |
| Decode (trie + beam)                           | â†’ Implemented              | `RecGPT.Trie`, `RecGPT.Decode.beam_search/4`.                                |

**Next steps for full parity**

| Priority | Step                                                            | Effort | Notes                                                                                                                          |
| -------- | --------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------ |
| 1        | ~~(Optional) Wire beam + trie with loaded checkpoint~~          | �      | âœ… Done: `inference_test.exs` "load checkpoint + trie + beam_search returns next item_id" (run with `--include integration`). |
| 2        | (Optional) Numerical parity: Elixir forward vs reference logits | Low    | Compare `Inference.forward` logits with reference (e.g. exported golden logits) on same token sequences; add test or script.   |

---

## See also

- [09 Parity overview](09_parity_overview.md) � At a glance, mapping, summary.
- [04 RecGPT library](04_recgpt_library.md) � Module reference.
- [08 Checkpoint layout](08_recgpt_checkpoint_layout.md) � Export and loader.
