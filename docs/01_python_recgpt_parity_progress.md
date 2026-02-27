# Progress: Elixir recgpt vs Python RecGPT parity

Task list for how close the **recgpt** Elixir package is to matching the [Python RecGPT](https://github.com/HKUDS/RecGPT) (HKUDS/RecGPT) pipeline and model. Reference: [RecGPT paper](https://arxiv.org/abs/2506.06270), [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model).

---

## At a glance

| Layer | Status | Elixir modules / scripts |
|-------|--------|---------------------------|
| Text → 768-d | ✅ | `RecGPT.Embedding` |
| FSQ (768→4 tokens) | ✅ | `RecGPT.FSQ`, `RecGPT.FSQEncoder` |
| Training data + loss | ✅ | `RecGPT.Training` |
| Checkpoint load | ✅ | `RecGPT.CheckpointLoader` |
| Model forward (embed + aux + GPT-2 + head) | ✅ | `RecGPT.Inference` (full backbone when params have gpt2model.h.*) |
| Decode (trie + beam) | ✅ | `RecGPT.Trie`, `RecGPT.Decode` |

**Data pipeline (no Python):** item text → embeddings → token_id_list → train batches ✅ — [integration tested](../test/recgpt/pipeline_integration_test.exs).  
**Checkpoint:** export via `scripts/inspect_recgpt_checkpoint.py --export DIR` → [CheckpointLoader.load_from_export/1](../lib/recgpt/checkpoint_loader.ex).  
**Inference in Elixir:** load checkpoint → forward (embed + aux + GPT-2 + head) → beam search → item_id ✅ (`RecGPT.Trie.build/1`, `RecGPT.Decode.beam_search/4`).

---

## Python ↔ Elixir mapping

| Python (RecGPT repo) | Elixir (recgpt) | Notes |
|----------------------|-----------------|--------|
| sentence-transformers / MPNet 768-d | `RecGPT.Embedding` (Bumblebee, all-mpnet-base-v2) | Same model id; parity test: `export_mpnet_embeddings.py` + `--include compare_embedding`. |
| `data_processing/make_token_list.py` | `RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3` | Compare test validates. |
| `utils/fsq.py` (levels, bound, quantize, codes↔indices) | `RecGPT.FSQ` | PropCheck + compare test. |
| VAE `vae_len4_fsq88865_ep90.pt` | Weights via `export_recgpt_fsq_weights.py` → `FSQ.load_params/1` | Encoder logic in Elixir; weights from export. |
| `GPT2RecBatchTrainAuxData`, batch build, loss | `RecGPT.Training.build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2` | No model forward in package. |
| `pre_train.py` / `predict.py` | — | Use our token_id_list + embeddings from Elixir pipeline. |
| **`serve.py`** (HTTP server) | **`RecGPT.Serve` + `mix recgpt.serve`** | POST /recommend, GET /search, GET /health; loads fixture + checkpoint once. |

---

## 1. Text → embeddings (768-d)

| Task | Status | Notes |
|------|--------|--------|
| Text → 768-d embeddings | ✅ Done | `RecGPT.Embedding`: Bumblebee + sentence-transformers/all-mpnet-base-v2, mean pooling. |
| Match Python MPNet / normalize_embeddings=False | ✅ Validated | Same model id; `embedding_processor: nil` (no L2). Parity test: `export_mpnet_embeddings.py` → fixtures; run with `--include compare_embedding --include embedding`. |
| encode_item_text_dict (map → tensor) | ✅ Done | Sorted keys, encode_texts, stack. |
| Save/load embeddings (no re-encode) | ✅ Done | Nx.serialize / deserialize. |

---

## 2. FSQ (4 tokens per item, vocab 15360)

| Task | Status | Notes |
|------|--------|--------|
| FSQ levels [8,8,8,6,5], basis, bound, round_ste, quantize | ✅ Done | `RecGPT.FSQ` port of RecGPT utils/fsq.py. |
| codes_to_indices / indices_to_codes | ✅ Done | Round-trip and encode path covered by tests. |
| Load params (project_in / project_out) from export | ✅ Done | `load_params/1`; keys `project_in/kernel` or `fsq.project_in.weight`, transpose when shape {5,192}. |
| **FSQ encode: 768-d → 4 token IDs** | ✅ Done | `RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3`; matches Python `make_token_list.py` logic. |
| **Python vs Elixir FSQ parity test** | ✅ Done | `compare_recgpt_fsq.py` → fixtures; `compare_test.exs` asserts token lists match. Run with real params/embeddings for full parity. |
| Load embeddings from .npy | ✅ Done | `FSQEncoder.load_embeddings_from_npy/1` (npy hex). |
| **Property-based tests (PropCheck)** | ✅ Done | See [Property-based testing](#property-based-testing-propcheck) below. |

---

## 3. Training data and loss

| Task | Status | Notes |
|------|--------|--------|
| Build train batch (seqs → batch_seq, labels, aux_embeds, mask) | ✅ Done | `RecGPT.Training.build_train_batch/4`; padding_id 15360, label_ignore -100, max_length 256, seq_token_capacity 1024. |
| Encode aux (item ids → 192-d per position, mask) | ✅ Done | `encode_aux/3`; gather item_embeddings, reshape to (n*4, 192), mask for valid/padding. |
| Loss: shifted CE over FSQ vocab | ✅ Done | `loss_shifted_ce/2`; ignore -100, mean over valid positions. |
| Same batch shape / format as Python RecGPT | ✅ Verified | Matches HKUDS/RecGPT `utils/data.py` `GPT2RecBatchTrainAuxData`: max_length=256, padding_id=15360, seq_token_capacity=1024, label_ignore=-100, right-padding; aux (256 items → 1024×192) and mask (1024×1) match. |
| **Property-based tests (PropCheck)** | ✅ Done | See [Property-based testing](#property-based-testing-propcheck) below. |

**Python ↔ Elixir FSQ port verification:** The encode path (embeddings → token_id_list) is ported from Python and validated as follows.

| Step | Python (compare_recgpt_fsq.py / export_steam_e2e_fixture.py) | Elixir (RecGPT.FSQ, FSQEncoder) | Check |
|------|--------------------------------------------------------------|----------------------------------|--------|
| Constants | LEVELS [8,8,8,6,5], VOCAB_SIZE 15360, BASIS [1,8,64,512,3072] | `@level_list`, `@vocab_size`, `basis()` | Parity constants test: `basis matches levels cumprod` |
| bound(z) | half_l = (levels-1)*(1-ε)/2; offset = 0.5 if even else 0; tanh(z+tanh(offset/half_l))*half_l - offset | Same formula in `FSQ.bound/2` | Same logic |
| quantize(z) | bound → round → divide by levels/2 | bound → round_ste → divide by levels/2 (round_ste forward = round) | Same numeric output |
| codes_to_indices(codes) | zhat = codes*half_width+half_width; round(sum(zhat*basis)); clip 0..vocab_size-1 | `scale_and_shift` then multiply by basis, sum, round, clip | Same; `scale_and_shift` = zhat |
| project_in | z_4_192 @ kernel (192,5) → (batch,4,5) | Nx.dot(z, [2], kernel, [0]); kernel (192,5) | Same |
| project_out | codes (batch,4,5) @ kernel (5,192) → (batch,4,192) | Nx.dot(codes, [2], kernel, [0]); kernel (5,192) | Same |
| Embeddings → token_id_list | Reshape (N,768)→(N,4,192); encode per batch | `FSQEncoder.encode_embeddings_to_token_id_list`: reshape, `FSQ.encode` per batch | Same |

**Tests that validate the port:** (1) `compare_test.exs` (fixtures from `compare_recgpt_fsq.py`): Elixir token_id_list matches Python expected_tokens. (2) Steam FSQ parity: `steam_e2e_test.exs` in **M:\\reflex-logic-other** (steam_e2e project) — Elixir token_id_list matches Python on `steam_e2e_parity.json`. (3) `parity_constants_test.exs`: basis, vocab_size, levels. Run compare + parity from recgpt; for Steam run from reflex-logic-other.

**Batch format verification:** Elixir `RecGPT.Training.build_train_batch/4` and `encode_aux/3` were compared to [HKUDS/RecGPT `utils/data.py`](https://github.com/HKUDS/RecGPT/blob/main/utils/data.py) class `GPT2RecBatchTrainAuxData`. Same constants: `max_length=256`, `padding_id=15360`, token sequence length `1024`, `label_list` padded with `-100`; right-padding for training; `encode_aux` maps 256 item IDs to embeddings (256, 768) → reshape to (1024, 192) and mask (1024, 1). Parity constants test asserts shapes and padding; no runtime Python comparison required.

---

## 4. Model forward (inference / training)

| Task | Status | Notes |
|------|--------|--------|
| **Checkpoint loader** | ✅ Done | `RecGPT.CheckpointLoader.load_from_export/1`; reads manifest.json + .npy from export dir (see [02_recgpt_checkpoint_layout](02_recgpt_checkpoint_layout.md)). |
| FSQ token embedding table (15361, 768) | ✅ Done | Lookup in `RecGPT.Inference.forward/4`; params `wte` or `gpt2model.wte.weight`. Uses first 15361 rows when checkpoint has GPT-2 vocab (50257). |
| Auxiliary encoder (aux → 768-d, fuse with token embeds) | ✅ Done | `Inference` applies linear + LayerNorm, masks, adds to token embeds. Accepts `ae.*` or `linear_layer.weight` / `norm_aux.*` (RecGPT export keys). |
| Prediction head (768 → 15361 logits) | ✅ Done | `Inference.apply_head/2`; params `pred_head.weight`, `pred_head.bias`. |
| **Forward: embed + aux + head** | ✅ Done | `RecGPT.Inference.forward/4`; returns logits (batch, 15361). |
| **GPT-2 backbone (full transformer)** | ✅ Done | When params include `gpt2model.h.{i}.*` (or `transformer.h.{i}.*`): pre-norm attn + MLP blocks, optional wpe, ln_f. Stub used when no layer params. |

**Checkpoint key compatibility:** Inference accepts RecGPT export keys: `gpt2model.wte.weight` (sliced to 15361 rows when checkpoint has full GPT-2 vocab), `gpt2model.wpe.weight`, `gpt2model.h.{i}.*`, `linear_layer.weight`/`norm_aux.*` for aux encoder, `pred_head.weight`/`pred_head.bias` for head. See [02_recgpt_checkpoint_layout](02_recgpt_checkpoint_layout.md).

---

## 5. Decoding and catalog

| Task | Status | Notes |
|------|--------|--------|
| **Trie (4-token sequence → item_id)** | ✅ Done | `RecGPT.Trie.build/1`, `lookup/2`, `valid_next_tokens/2`; built from token_id_list. |
| **Beam search (4 steps, valid next tokens)** | ✅ Done | `RecGPT.Decode.beam_search/4`; get_logits_fn + trie + context + beam_width. |
| Next-item prediction API | ✅ Done | Integration test: load checkpoint → trie from token_id_list → get_logits_fn(forward) → Decode.beam_search; run with `--include integration`. |

---

## 6. Checkpoint and export

| Task | Status | Notes |
|------|--------|--------|
| Export FSQ weights from RecGPT VAE to format recgpt can load | ✅ Script exists | `scripts/export_recgpt_fsq_weights.py` → .npz; load into map for `FSQ.load_params/1`. |
| Export RecGPT .pt to manifest + .npy for Elixir | ✅ Script exists | `scripts/inspect_recgpt_checkpoint.py --export DIR`; auto-finds `thirdparty/RecGPT_repo/ckpt/recgpt_layer_3_weight.pt` or `RecGPT_model/`. |
| Load RecGPT checkpoint in Elixir | ✅ Done | `RecGPT.CheckpointLoader.load_from_export/1`; expects manifest.json + .npy in export dir. |
| **Load + forward with real checkpoint** | ✅ Validated | Integration test loads `data/recgpt_ckpt_export` and runs `Inference.forward/4`; run with `--include integration`. |

---

## 7. End-to-end flows

| Flow | Status | Notes |
|------|--------|--------|
| **Data only:** item_text_dict → embeddings → token_id_list → train batches | ✅ Done | No Python needed: Embedding + FSQEncoder + Training. |
| **Train (Python):** pre_train.py with our token_id_list + embeddings | ✅ Possible | Build token_id_list in Elixir; export or point Python at same data. |
| **Inference in Elixir:** load checkpoint → forward → beam → item_id | ✅ Done | Loader + Inference.forward + Trie + Decode.beam_search. Real checkpoint load + forward + beam_search covered by integration test; run with `--include integration`. |
| **Steam E2E (serve/predict flow)** | ✅ Done | Lives in **M:\\reflex-logic-other**: steam_e2e project + `scripts/export_steam_e2e_fixture.py`. Test loads Steam fixture, checkpoint (from reflex-logic-market or env), runs beam_search; run from reflex-logic-other with `--include e2e_steam --include integration`. |
| **Zero-shot eval (predict.py) with our data** | ✅ Possible | Python script; we can produce compatible pkl/npy from Elixir pipeline. |

---

## How to validate

From repo root or from `recgpt/`. On **PowerShell** use `;` instead of `&&` to chain commands (e.g. `cd recgpt; mix test ...`).

| What | Command |
|------|--------|
| Unit + PropCheck (no HF model) | `cd recgpt && mix test` (excludes embedding, compare_python, integration by default) |
| All tests (embedding, compare_python, integration) | `cd recgpt && mix test --include embedding --include compare_python --include integration` |
| PropCheck only | `cd recgpt && mix test test/recgpt/propcheck_test.exs` |
| Parity constants (doc/code sync) | `cd recgpt && mix test test/recgpt/parity_constants_test.exs` |
| Loader + Inference | `cd recgpt && mix test test/recgpt/checkpoint_loader_test.exs test/recgpt/inference_test.exs` |
| **Real checkpoint load + forward + beam** | From repo root: `python scripts/inspect_recgpt_checkpoint.py --export data/recgpt_ckpt_export` then `cd recgpt && mix test test/recgpt/inference_test.exs --include integration` (runs load, forward, and beam_search with trie). |
| **Steam E2E (like serve.py)** | From **M:\\reflex-logic-other**: `uv run python scripts/export_steam_e2e_fixture.py --output data/steam_e2e_fixture.json` (uses `steam_data/`), then `cd steam_e2e && mix test test/recgpt/steam_e2e_test.exs --include e2e_steam --include integration`. Checkpoint from reflex-logic-market. |
| **Steam FSQ parity (Python vs Elixir)** | From reflex-logic-other: same export writes `data/steam_e2e_parity.json`. Run from steam_e2e: `mix test test/recgpt/steam_e2e_test.exs --include steam_parity --include integration`. |
| Trie + Decode | `cd recgpt && mix test test/recgpt/trie_test.exs test/recgpt/decode_test.exs` |
| FSQ vs Python (fixtures) | `uv run python scripts/compare_recgpt_fsq.py --output-dir data/recgpt_compare` then `cd recgpt && mix test test/recgpt/compare_test.exs` (run with `--include compare_python`) |
| Embedding (downloads model) | `cd recgpt && mix test --include embedding` |
| MPNet vs Python (normalize_embeddings=False) | From repo root: `uv run python scripts/export_mpnet_embeddings.py --output-dir data/recgpt_embedding` then `cd recgpt && mix test --include compare_embedding --include embedding` |
| Coverage | `cd recgpt && mix test --exclude embedding --cover` |

---

## Property-based testing (PropCheck)

Implemented in **`test/recgpt/propcheck_test.exs`** ([PropCheck](https://github.com/alfert/propcheck)):

- **FSQ:** indices round-trip via 5-d codes; `codes_to_indices` in 0..vocab_size-1; `scale_and_shift` / `scale_and_shift_inverse` round-trip; `bound` finite; `quantize` in [-1.1, 1.1]; `encode` indices in range.
- **Training:** `build_train_batch` returns correct tensor shapes; `loss_shifted_ce` non-negative and 0 when all labels -100; `encode_aux` output shapes (n×4, 192) and (n×4, 1).
- **FSQEncoder:** `encode_embeddings_to_token_id_list` length = num_items; each token list 4 elements in 0..vocab_size-1; determinism (same input → same output).

Run: `mix test test/recgpt/propcheck_test.exs` (exclude embedding if no model).

---

## Parity constants test

**`test/recgpt/parity_constants_test.exs`** asserts that code constants match the parity doc (§1–§3): FSQ vocab_size 15360, padding_id 15360, seq_len 4, dim 192; Embedding size 768; Training batch format (seq_token_capacity 1024, max_length 256, padding_id, label_ignore -100); FSQEncoder output (4 tokens per item, 0..vocab_size-1). Run: `mix test test/recgpt/parity_constants_test.exs`.

---

## Gaps and risks

| Gap | Impact | Mitigation |
|-----|--------|------------|
| Embedding numerical parity | Elixir vs Python embeddings may differ slightly | Run `export_mpnet_embeddings.py` + `--include compare_embedding --include embedding` to assert cosine ≥ 0.99. |
| ~~Batch format vs Python~~ | Closed | Verified against [HKUDS/RecGPT utils/data.py](https://github.com/HKUDS/RecGPT/blob/main/utils/data.py) `GPT2RecBatchTrainAuxData`: same constants and shapes. |
| ~~Beam + trie with real model~~ | Closed | Integration test: load export → trie → get_logits_fn(forward) → beam_search returns `{:ok, item_id}` (`--include integration`). |
| Empty batch / empty seq edge cases | Nx limits (e.g. empty tensor) | Tests document RuntimeError; callers avoid empty batches. |
| Empty item_text_dict | encode_texts([]) fails (tokenizer) | Test asserts Enum.EmptyError; doc: use non-empty dict. |

---

## Summary

| Area | Parity | Blocker / note |
|------|--------|------------------|
| Embeddings | ✅ Implemented | Compare test: `export_mpnet_embeddings.py` + `--include compare_embedding --include embedding`. |
| FSQ + FSQEncoder | ✅ Implemented, tested vs Python + PropCheck | Compare test with fixtures; PropCheck for invariants. |
| Training data + loss | ✅ Implemented, PropCheck | Same shapes and loss as paper; properties for shapes and loss. |
| Checkpoint loader | ✅ Implemented | `CheckpointLoader.load_from_export/1`; export via inspect_recgpt_checkpoint.py. |
| Inference forward (embed + aux + GPT-2 + head) | ✅ Implemented | `RecGPT.Inference.forward/4`; full backbone when params have gpt2model.h.*. |
| Decode (trie + beam) | ✅ Implemented | `RecGPT.Trie`, `RecGPT.Decode.beam_search/4`. |

**Next steps for full Python parity**

| Priority | Step | Effort | Notes |
|----------|------|--------|--------|
| 1 | ~~(Optional) Wire beam + trie with loaded checkpoint~~ | — | ✅ Done: `inference_test.exs` "load checkpoint + trie + beam_search returns next item_id" (run with `--include integration`). |
| 2 | (Optional) Numerical parity vs Python predict.py on same inputs | Low | Export Python logits for a few sequences; compare with Elixir forward. |

---

## Links

| Doc | Description |
|-----|--------------|
| [00 RecGPT library](00_recgpt_library.md) | Modules, deps, tests, training flow. |
| [02 RecGPT checkpoint layout](02_recgpt_checkpoint_layout.md) | state_dict layout, inspect_recgpt_checkpoint.py, loader usage. |
| [PropCheck property tests](../test/recgpt/propcheck_test.exs) | FSQ, Training, FSQEncoder properties. |
| [Parity constants test](../test/recgpt/parity_constants_test.exs) | Doc/code sync for §1–§3 constants. |
| [CheckpointLoader](../lib/recgpt/checkpoint_loader.ex) · [Inference](../lib/recgpt/inference.ex) | Load export dir; forward (embed + aux + head). |
| [Trie](../lib/recgpt/trie.ex) · [Decode](../lib/recgpt/decode.ex) | Catalog trie; beam search for next-item. |
| [RecGPT Bumblebee port estimate](../../polymarket/docs/36_recgpt_bumblebee_port_estimate.md) | Inference path, difficulty, what’s done.
| [Fine-tuning RecGPT](../../polymarket/docs/25_recgpt_finetuning.md) | token_id_list, checkpoint, pre_train.py. |
| [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) · [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model) | Python repo and HuggingFace model. |
