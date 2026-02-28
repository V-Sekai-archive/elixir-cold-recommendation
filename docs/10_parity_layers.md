# Parity by layer

Sub-proposal of the [documentation index](README.md). Per-layer task lists and validation for Python RecGPT parity. Overview and mapping: [08 Parity overview](08_parity_overview.md).

---

## 1. Text Ã¢â€ â€™ embeddings (768-d)

| Task                                              | Status       | Notes                                                                                  |
| ------------------------------------------------- | ------------ | -------------------------------------------------------------------------------------- |
| Text Ã¢â€ â€™ 768-d embeddings                           | Ã¢Å“â€¦ Done      | `RecGPT.Embedding`: Bumblebee + sentence-transformers/all-mpnet-base-v2, mean pooling. |
| Match Python MPNet (`normalize_embeddings=False`) | Ã¢Å“â€¦ Validated | Same model id; `embedding_processor: nil` (no L2). Parity via unit tests.              |
| encode_item_text_dict (map Ã¢â€ â€™ tensor)              | Ã¢Å“â€¦ Done      | Sorted keys, encode_texts, stack.                                                      |

---

## 2. FSQ (4 tokens per item, vocab 15360)

| Task                                                      | Status  | Notes                                                                                                |
| --------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------- |
| FSQ levels [8,8,8,6,5], basis, bound, round_ste, quantize | Ã¢Å“â€¦ Done | `RecGPT.FSQ` port of RecGPT utils/fsq.py.                                                            |
| codes_to_indices / indices_to_codes                       | Ã¢Å“â€¦ Done | Round-trip and encode path covered by tests.                                                         |
| Load params (project_in / project_out) from export        | Ã¢Å“â€¦ Done | `load_params/1`; keys `project_in/kernel` or `fsq.project_in.weight`, transpose when shape {5,192}.  |
| **FSQ encode: 768-d Ã¢â€ â€™ 4 token IDs**                       | Ã¢Å“â€¦ Done | `RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3`; matches Python `make_token_list.py` logic. |
| **Python vs Elixir FSQ parity**                           | Ã¢Å“â€¦ Done | Parity via unit tests and pipeline integration tests.                                                |

---

## 3. Training data and loss

| Task                                                           | Status      | Notes                                                                                                                                                                                                             |
| -------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Build train batch (seqs Ã¢â€ â€™ batch_seq, labels, aux_embeds, mask) | Ã¢Å“â€¦ Done     | `RecGPT.Training.build_train_batch/4`; padding_id 15360, label_ignore -100, max_length 256, seq_token_capacity 1024.                                                                                              |
| Encode aux (item ids Ã¢â€ â€™ 192-d per position, mask)               | Ã¢Å“â€¦ Done     | `encode_aux/3`; gather item_embeddings, reshape to (n\*4, 192), mask for valid/padding.                                                                                                                           |
| Loss: shifted CE over FSQ vocab                                | Ã¢Å“â€¦ Done     | `loss_shifted_ce/2`; ignore -100, mean over valid positions.                                                                                                                                                      |
| Same batch shape / format as Python RecGPT                     | Ã¢Å“â€¦ Verified | Matches HKUDS/RecGPT `utils/data.py` `GPT2RecBatchTrainAuxData`: max_length=256, padding_id=15360, seq_token_capacity=1024, label_ignore=-100, right-padding; aux (256 items Ã¢â€ â€™ 1024Ãƒâ€”192) and mask (1024Ãƒâ€”1) match. |

**FSQ and batch verification:** Encode path and batch format match Python (unit and pipeline integration tests). Batch constants: max_length=256, padding_id=15360, seq_token_capacity=1024; see [HKUDS/RecGPT utils/data.py](https://github.com/HKUDS/RecGPT/blob/main/utils/data.py) `GPT2RecBatchTrainAuxData`.

---

## 4. Model forward (inference / training)

| Task                                                    | Status  | Notes                                                                                                                                                         |
| ------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Checkpoint loader**                                   | Ã¢Å“â€¦ Done | `RecGPT.CheckpointLoader.load_from_export/1`; reads manifest.json + .npy from export dir (see [08_recgpt_checkpoint_layout](08_recgpt_checkpoint_layout.md)). |
| FSQ token embedding table (15361, 768)                  | Ã¢Å“â€¦ Done | Lookup in `RecGPT.Inference.forward/4`; params `wte` or `gpt2model.wte.weight`. Uses first 15361 rows when checkpoint has GPT-2 vocab (50257).                |
| Auxiliary encoder (aux Ã¢â€ â€™ 768-d, fuse with token embeds) | Ã¢Å“â€¦ Done | `Inference` applies linear + LayerNorm, masks, adds to token embeds. Accepts `ae.*` or `linear_layer.weight` / `norm_aux.*` (RecGPT export keys).             |
| Prediction head (768 Ã¢â€ â€™ 15361 logits)                    | Ã¢Å“â€¦ Done | `Inference.apply_head/2`; params `pred_head.weight`, `pred_head.bias`.                                                                                        |
| **Forward: embed + aux + head**                         | Ã¢Å“â€¦ Done | `RecGPT.Inference.forward/4`; returns logits (batch, 15361).                                                                                                  |
| **GPT-2 backbone (full transformer)**                   | Ã¢Å“â€¦ Done | When params include `gpt2model.h.{i}.*` (or `transformer.h.{i}.*`): pre-norm attn + MLP blocks, optional wpe, ln_f. Stub used when no layer params.           |

**Checkpoint key compatibility:** Inference accepts RecGPT export keys: `gpt2model.wte.weight` (sliced to 15361 rows when checkpoint has full GPT-2 vocab), `gpt2model.wpe.weight`, `gpt2model.h.{i}.*`, `linear_layer.weight`/`norm_aux.*` for aux encoder, `pred_head.weight`/`pred_head.bias` for head. See [08_recgpt_checkpoint_layout](08_recgpt_checkpoint_layout.md).

---

## 5. Decoding and catalog

| Task                                         | Status  | Notes                                                                                                                                        |
| -------------------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Trie (4-token sequence Ã¢â€ â€™ item_id)**        | Ã¢Å“â€¦ Done | `RecGPT.Trie.build/1`, `lookup/2`, `valid_next_tokens/2`; built from token_id_list.                                                          |
| **Beam search (4 steps, valid next tokens)** | Ã¢Å“â€¦ Done | `RecGPT.Decode.beam_search/4`; get_logits_fn + trie + context + beam_width.                                                                  |
| Next-item prediction API                     | Ã¢Å“â€¦ Done | Integration test: load checkpoint Ã¢â€ â€™ trie from token_id_list Ã¢â€ â€™ get_logits_fn(forward) Ã¢â€ â€™ Decode.beam_search; run with `--include integration`. |

---

## 6. Checkpoint and export

| Task                                                 | Status       | Notes                                                                                                              |
| ---------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------ |
| Export FSQ weights from RecGPT checkpoint for recgpt | Ã¢Å“â€¦ Done      | FSQ params in checkpoint export (manifest + .npy); `FSQ.load_params/1` from CheckpointLoader.                      |
| Export RecGPT .pt to manifest + .npy for Elixir      | Ã¢Å“â€¦ Done      | `mix recgpt.export_ckpt --from-pt <path> --out DIR`.                                                               |
| Load RecGPT checkpoint in Elixir                     | Ã¢Å“â€¦ Done      | `RecGPT.CheckpointLoader.load_from_export/1`; expects manifest.json + .npy in export dir.                          |
| **Load + forward with real checkpoint**              | Ã¢Å“â€¦ Validated | Integration test loads `data/recgpt_ckpt_export` and runs `Inference.forward/4`; run with `--include integration`. |

---

## 7. End-to-end flows

| Flow                                                                       | Status      | Notes                                                                                                                                                               |
| -------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data only:** item_text_dict Ã¢â€ â€™ embeddings Ã¢â€ â€™ token_id_list Ã¢â€ â€™ train batches | Ã¢Å“â€¦ Done     | No Python needed: Embedding + FSQEncoder + Training.                                                                                                                |
| **Train (Python):** pre_train.py with our token_id_list + embeddings       | Ã¢Å“â€¦ Possible | Build token_id_list in Elixir; export or point Python at same data.                                                                                                 |
| **Inference in Elixir:** load checkpoint Ã¢â€ â€™ forward Ã¢â€ â€™ beam Ã¢â€ â€™ item_id        | Ã¢Å“â€¦ Done     | Loader + Inference.forward + Trie + Decode.beam_search. Real checkpoint load + forward + beam_search covered by integration test; run with `--include integration`. |
| **Serve E2E (serve/predict flow)**                                         | Ã¢Å“â€¦ Done     | Fixture and tests live in a separate repo; set RECGPT_FIXTURE to use that fixture with `mix recgpt.serve`.                                                          |
| **Zero-shot eval (predict.py) with our data**                              | Ã¢Å“â€¦ Possible | Python script; we can produce compatible pkl/npy from Elixir pipeline.                                                                                              |

---

## How to validate

| What                         | Command |
| ---------------------------- | ------- |
| Unit tests (default)        | `mix test` (excludes integration) |
| With integration            | `mix test --include integration` |
| Checkpoint + inference     | Export ckpt then `mix test test/recgpt/inference_test.exs --include integration` |
| Trie + Decode              | `mix test test/recgpt/trie_test.exs test/recgpt/decode_test.exs` |

---

## Gaps and risks

| Gap                                | Impact                                          | Mitigation                                                                                                                                                    |
| ---------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedding numerical parity         | Elixir vs Python embeddings may differ slightly | Same model id; validate via unit tests.                                                                                                                       |
| ~~Batch format vs Python~~         | Closed                                          | Verified against [HKUDS/RecGPT utils/data.py](https://github.com/HKUDS/RecGPT/blob/main/utils/data.py) `GPT2RecBatchTrainAuxData`: same constants and shapes. |
| ~~Beam + trie with real model~~    | Closed                                          | Integration test: load export Ã¢â€ â€™ trie Ã¢â€ â€™ get_logits_fn(forward) Ã¢â€ â€™ beam_search returns `{:ok, item_id}` (`--include integration`).                               |
| Empty batch / empty seq edge cases | Nx limits (e.g., empty tensor)                  | Tests document RuntimeError; callers avoid empty batches.                                                                                                     |
| Empty item_text_dict               | encode_texts([]) fails (tokenizer)              | Test asserts Enum.EmptyError; doc: use non-empty dict.                                                                                                        |

---

## Summary

| Area                                           | Parity                      | Blocker / note                                                               |
| ---------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------- |
| Embeddings                                     | Ã¢Å“â€¦ Implemented              | Same model id; unit tested.                                                  |
| FSQ + FSQEncoder                               | Ã¢Å“â€¦ Implemented, unit tested | Same logic as Python.                                                        |
| Training data + loss                           | Ã¢Å“â€¦ Implemented              | Same shapes and loss as paper.                                               |
| Checkpoint loader                              | Ã¢Å“â€¦ Implemented              | `CheckpointLoader.load_from_export/1`; export via `mix recgpt.export_ckpt`.  |
| Inference forward (embed + aux + GPT-2 + head) | Ã¢Å“â€¦ Implemented              | `RecGPT.Inference.forward/4`; full backbone when params have gpt2model.h.\*. |
| Decode (trie + beam)                           | Ã¢Å“â€¦ Implemented              | `RecGPT.Trie`, `RecGPT.Decode.beam_search/4`.                                |

**Next steps for full Python parity**

| Priority | Step                                                            | Effort | Notes                                                                                                                         |
| -------- | --------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------- |
| 1        | ~~(Optional) Wire beam + trie with loaded checkpoint~~          | Ã¢â‚¬â€      | Ã¢Å“â€¦ Done: `inference_test.exs` "load checkpoint + trie + beam_search returns next item_id" (run with `--include integration`). |
| 2        | (Optional) Numerical parity vs Python predict.py on same inputs | Low    | Export Python logits for a few sequences; compare with Elixir forward.                                                        |

---

## See also

- [08 Parity overview](08_parity_overview.md) Ã¢â‚¬â€ At a glance, mapping, summary.
- [04 RecGPT library](04_recgpt_library.md) Ã¢â‚¬â€ Module reference.
- [08 Checkpoint layout](08_recgpt_checkpoint_layout.md) Ã¢â‚¬â€ Export and loader.


---

## See also

- [08 Parity overview](08_parity_overview.md) - At a glance, mapping, summary.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
- [08 Checkpoint layout](08_recgpt_checkpoint_layout.md) - Export and loader.
