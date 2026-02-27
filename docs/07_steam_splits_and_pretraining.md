# Steam dataset splits and pretraining for best quality

Data layout and evaluation strategy aligned with the RecGPT Steam dataset (e.g. [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset/tree/main/test/steam): train/test, optional cold_train/cold_test). **For highest quality, pretrain on the train split then evaluate**; zero-shot is a baseline only.

See also: [05 Evaluation and testing](05_evaluation_and_testing.md), [06 Eval data shapes](06_eval_data_shapes.md).

---

## 1. Regular vs cold splits (Steam layout)

| Split | Purpose | Content |
|-------|---------|--------|
| **Regular** | Train model, then evaluate on seen items | `train` (sequences for pretraining), `test` (held-out last-item-out for eval). |
| **Cold** | Evaluate on unseen items | `cold_train` (optional), `cold_test` (last-item-out where target item is cold). |

- **Regular train:** Full sequences (list of item_ids per session) used for pretraining. No overlap with test.
- **Regular test:** Last-item-out cases: `context` = all but last click, `next_item` = last. Same item set as train.
- **Cold test:** Same shape as test but target items are "cold" (e.g. not in train, or very few interactions). Measures recommendation for new items.

Canonical source layout (Steam): `train.pkl`, `test.pkl`, `item_text_dict.pkl`, optional `cold_train.pkl`, `cold_test.pkl` (see [RecGPT_dataset/test/steam](https://huggingface.co/datasets/hkuds/RecGPT_dataset/tree/main/test/steam)). SQLite/JSON export: `train_sequences`, `test_sequences`, `cold_train_sequences`, `cold_test_sequences`, `item_text`.

---

## 2. Pretraining vs zero-shot (quality)

| Approach | When to use | Quality |
|----------|-------------|--------|
| **Pretrain then eval** | You have a train split and can run training (e.g. Python `pre_train.py` or Elixir `RecGPT.AxonTrain.run/3`). | **Best.** Model adapts to catalog and sequences. |
| **Zero-shot** | No training; pretrained checkpoint + fixture only. | Baseline; often below random on small catalogs. |

**Recommendation:** Generate data with **train + test (and optionally cold)**. Pretrain on the train split, then run eval on test and cold_test. Use zero-shot only as a sanity check or when training is not available.

---

## 3. Artifacts in this repo (Steam layout)

After pipeline run (e.g. `RecGPT.Clickstream.Fetch.run/1`), target layout:

| File | Shape | Use |
|------|--------|-----|
| `items.json` | `{ "items": [{"id", "title"}], "num_items" }` | Catalog; build fixture (Embedding + FSQ → token_id_list). |
| `train_sequences.json` | `{ "sequences": [[id, ...], ...], "num_items" }` | **Pretraining:** input to `Training.build_train_batch` (after token_id_list). |
| `test_sequences.json` | `{ "test_cases": [{"context", "next_item"}], "num_items" }` | **Eval:** regular held-out; `Eval.load_test_cases/1`. |
| `cold_test_sequences.json` | Same as test_sequences | **Eval:** cold-start; optional if cold items are defined. |

Train/test split: e.g. 80% of sessions → train sequences, 20% → test (last-item-out). Cold split requires a definition of cold items (e.g. hold out item set or by frequency); then cold_test = test cases whose `next_item` is cold.

---

## 4. Pipeline order (high quality)

1. **Generate data** with train + test (and cold if applicable).
2. **Build fixture** from items (Embedding + FSQ) → `token_id_list`, `num_items`.
3. **Pretrain** on train_sequences:
   - **Elixir:** Load checkpoint with `RecGPT.CheckpointLoader.load_from_export(export_dir)`, build batches with `RecGPT.AxonTrain.stream_batches/4`, then `RecGPT.AxonTrain.run(stream, params, iterations: N, ...)`. Same batch format and loss as `Training.build_train_batch/4` and `Training.loss_shifted_ce/2`; forward mirrors `Inference` (embed, aux, GPT-2 blocks, head) via `Inference.forward_full_sequence/4`.
   - **Python:** `pre_train.py` with token_id_list + sequences.
4. **Eval** with the fine-tuned checkpoint on `test_sequences.json` (and `cold_test_sequences.json` if present).
5. Compare to zero-shot (pretrained ckpt, same fixture) to confirm pretraining improves metrics.

---

## 5. References

- **Steam test dataset:** [hkuds/RecGPT_dataset — test/steam](https://huggingface.co/datasets/hkuds/RecGPT_dataset/tree/main/test/steam) (`train.pkl`, `test.pkl`, `cold_train.pkl`, `cold_test.pkl`, `item_text_dict.pkl`, `item_text_embeddings.npy`).
- RecGPT training: Python `pre_train.py`; Elixir `RecGPT.Training.build_train_batch/4` for batch format; `RecGPT.AxonTrain` for training loop (load checkpoint, same batch + loss, Polaris optimizer).
