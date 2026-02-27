# Steam splits and pretraining

Data layout and evaluation strategy aligned with the RecGPT Steam dataset ([hkuds/RecGPT_dataset/test/steam](https://huggingface.co/datasets/hkuds/RecGPT_dataset/tree/main/test/steam)): train/test and cold splits. **For best quality, pretrain on the train split then evaluate;** zero-shot is a baseline only.

---

## Regular vs cold splits

| Split | Purpose | Content |
|-------|---------|---------|
| **Regular** | Train and evaluate on seen items | `train` (sequences for pretraining), `test` (held-out last-item-out for eval). |
| **Cold** | Evaluate on rare/unseen items | `cold_train` (train sequences that contain at least one cold item), `cold_test` (last-item-out where target item is cold). |

- **Regular train:** Full sequences (list of item_ids per session) for pretraining. No overlap with test.
- **Regular test:** Last-item-out: `context` = all but last click, `next_item` = last. Same item set as train.
- **Cold test:** Same shape as test; target items are “cold” (e.g. appear in ≤ K sessions in train). Measures recommendation for new or rare items.

Canonical Steam layout: `train.pkl`, `test.pkl`, `item_text_dict.pkl`, optional `cold_train.pkl`, `cold_test.pkl`. In this repo: JSON export as below.

---

## Pretraining vs zero-shot

| Approach | When to use | Quality |
|----------|-------------|---------|
| **Pretrain then eval** | You have a train split and run training (e.g. `mix recgpt.pretrain` or Python `pre_train.py`). | Best; model adapts to catalog and sequences. |
| **Zero-shot** | No training; pretrained checkpoint + fixture only. | Baseline; often below random on small catalogs. |

Recommendation: Generate data with train, test, and cold. Pretrain on the train split, then run eval on test and cold_test. Use zero-shot only as a sanity check or when training is not available.

---

## Artifacts (this repo)

After `RecGPT.Clickstream.Fetch.run/2` (or `mix recgpt.clickstream`):

| File | Shape | Use |
|------|--------|-----|
| `items.json` | `{"items": [{"id", "title"}], "num_items"}` | Catalog; input to build_fixture. |
| `train_sequences.json` | `{"sequences": [[id, ...], ...], "num_items"}` | Pretraining; input to `Training.build_train_batch` (with token_id_list). |
| `test_sequences.json` | `{"test_cases": [{"context", "next_item"}], "num_items"}` | Eval; regular held-out. `Eval.load_test_cases/1`. |
| `cold_test_sequences.json` | Same as test_sequences | Eval; cold-start. **Required** for `mix recgpt.eval`. Produced by Fetch. |
| `cold_train_sequences.json` | Same as train_sequences | Train sequences containing at least one cold item. |

**Train/test split:** e.g. 80% of sessions → train, 20% → test (last-item-out). **Cold split:** Items that appear in ≤ K sessions in train are “cold” (default K=2). Cold_test = test cases whose `next_item` is cold; cold_train = train sequences that contain at least one cold item. Fetch always writes both cold files. Override K via `run/2` opts: `:max_train_sessions_for_cold`.

---

## Pipeline order

**Order:** Fetch → build_fixture → pretrain → eval (with both `--test` and `--cold-test`).

For full commands, options, and file layout, see [08 Pipeline reference](08_pipeline_reference.md).

---

## References

- [hkuds/RecGPT_dataset — test/steam](https://huggingface.co/datasets/hkuds/RecGPT_dataset/tree/main/test/steam) — `train.pkl`, `test.pkl`, `cold_train.pkl`, `cold_test.pkl`, `item_text_dict.pkl`, `item_text_embeddings.npy`.
- RecGPT training: Python `pre_train.py`; Elixir `RecGPT.Training.build_train_batch/4`, `RecGPT.AxonTrain`.

---

## See also

- [05 Evaluation and testing](05_evaluation_and_testing.md) — Zero-shot vs trained, null hypothesis.
- [06 Eval data shapes](06_eval_data_shapes.md) — JSON shapes for these artifacts.
- [08 Pipeline reference](08_pipeline_reference.md) — End-to-end commands and layout.
