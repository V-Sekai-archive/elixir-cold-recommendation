# RecGPT Elixir library - documentation

This folder documents the **recgpt** package: FSQ, embeddings (MPNet/Bumblebee), and training data pipeline for RecGPT-style recommendation. No GenServer; use from any app (e.g. polymarket).

---

## Modules

| Module | Purpose |
|--------|---------|
| **RecGPT.FSQ** | FSQ quantizer: levels [8,8,8,6,5], 4 tokens per item, vocab 15360 + padding. `load_params/1`, `encode/2`, `codes_to_indices/1`, `indices_to_codes/2`. |
| **RecGPT.FSQEncoder** | `encode_embeddings_to_token_id_list/3`: (num_items, 768) embeddings + FSQ params -> list of 4-token lists. `load_embeddings_from_npy/1` (npy hex package). |
| **RecGPT.Embedding** | Text -> 768-d via Bumblebee (sentence-transformers/all-mpnet-base-v2). `serving/0`, `encode_texts/1`, `encode_item_text_dict/1`, `save_embeddings/2`, `load_embeddings/1`. |
| **RecGPT.Training** | `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2` for training data and loss. Model forward (GPT-2 + embed + head) not in this package. |

---

## Dependencies

Nx, Axon, Bumblebee (GitHub `main` for MPNet), Jason, Npy (for `.npy` loading). See [mix.exs](../mix.exs).

---

## Tests

From `recgpt/`:

```bash
mix test --exclude embedding
```

Embedding tests load the HuggingFace model; run with `--include embedding` and long timeout if needed. Compare test uses fixtures from repo root `data/recgpt_compare/` (generate with `uv run python scripts/compare_recgpt_fsq.py --output-dir data/recgpt_compare`). **Property-based tests** ([PropCheck](https://github.com/alfert/propcheck)): `mix test test/recgpt/propcheck_test.exs`. **Parity constants** (doc/code sync): `mix test test/recgpt/parity_constants_test.exs`.

---

## Python comparison

- **Generate fixtures** (repo root): `uv run python scripts/compare_recgpt_fsq.py --output-dir data/recgpt_compare`
- **Run compare test**: `cd recgpt && mix test test/recgpt/compare_test.exs`
- **Compare test** (FSQ parity): `mix test test/recgpt/compare_test.exs --include compare_python` (fixtures from Python script in parent repo).

---

## Evaluation

Evaluation (zero-shot vs trained), **train/eval data split** (held-out eval only), null-hypothesis rejection, and test plan are in [05_evaluation_and_testing](05_evaluation_and_testing.md).

---

## Training flow and zero-shot (summary)

- **Training flow:** item_text_dict -> Embedding -> FSQ params -> FSQEncoder -> token_id_list -> Training.build_train_batch. See [05_evaluation_and_testing](05_evaluation_and_testing.md).
- **Zero-shot:** Pretrained checkpoint + fixture from item text only (no gradient updates). See [05_evaluation_and_testing](05_evaluation_and_testing.md).

---

## Documentation in this repo

| Doc | Content |
|-----|---------|
| [Python RecGPT parity progress](01_python_recgpt_parity_progress.md) | Task list: how close Elixir recgpt matches Python RecGPT (embeddings, FSQ, training data, model, decode). |
| [Checkpoint layout](02_recgpt_checkpoint_layout.md) | state_dict, export script, loader. |
| [Evaluation and testing](05_evaluation_and_testing.md) | Zero-shot vs trained, train/eval split, null-hypothesis rejection, test plan. |

---

## Links

- [README](../README.md) - quick start and module list.
- [RecGPT paper](https://arxiv.org/abs/2506.06270) | [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) | [hkuds/RecGPT_model](https://huggingface.co/hkuds/RecGPT_model)
