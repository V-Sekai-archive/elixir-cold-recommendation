# Investigation: recgpt-trajectories Dataset for Pretrain and Eval

Goal: Find a dataset in `thirdparty/recgpt-trajectories` suitable for RecGPT pretraining and test evaluation to measure performance improvement.

---

## Status

**thirdparty/recgpt-trajectories** exists and contains 4 raw datasets (KuaiRand-Pure, MerRec, Open e-commerce, and others). Write a converter script to produce `items.json`, `train_sequences.json`, and `test_sequences.json` in the canonical shapes below.

---

## Expected Dataset Format (RecGPT Pipeline)

From [05_eval_data_shapes](05_eval_data_shapes.md) and [07_steam_splits_and_pretraining](07_steam_splits_and_pretraining.md):

| File                        | Shape                                                          | Purpose                                   |
| --------------------------- | -------------------------------------------------------------- | ----------------------------------------- |
| `items.json`                | `{"num_items", "items": [{"id", "title"}, ...]}`               | Catalog; input to build_fixture           |
| `train_sequences.json`      | `{"num_items", "sequences": [[id, ...], ...]}`                 | Pretraining input                         |
| `test_sequences.json`       | `{"num_items", "test_cases": [{"context", "next_item"}, ...]}` | Eval (held-out)                           |
| `cold_test_sequences.json`  | Same as test_sequences                                         | Cold-start eval (required for eval)       |
| `cold_train_sequences.json` | Same as train_sequences                                        | Optional; train sequences with cold items |

Optional for embedding parity with released checkpoint:

| File                       | Purpose                                           |
| -------------------------- | ------------------------------------------------- |
| `item_text_dict.pkl`       | Canonical item text (Python pickle)               |
| `item_text_embeddings.npy` | Precomputed 768-d embeddings (for fixture parity) |

---

## What to Look For in recgpt-trajectories

1. **JSON artifacts** – `items.json`, `train_sequences.json`, `test_sequences.json` in the shapes above.
2. **PKL/NPY** – `item_text_dict.pkl`, `item_text_embeddings.npy` if you want embedding parity.
3. **Sequence format** – Each sequence is a list of item_ids (0-based catalog indices). Train sequences are full sessions; test cases are `{context: [id1, id2, ...], next_item: id}`.
4. **Catalog** – `num_items` and `items` with unique `id` (0..num_items-1) and `title`.

---

## When You Have Other Data: Pretrain and Eval Flow

1. **Copy or symlink** into `data/recgpt-trajectories/` (or another dir).

2. **Build fixture**

   ```bash
   mix recgpt.build_fixture --items data/recgpt-trajectories/items.json --out data/recgpt-trajectories/fixture.json
   ```

   Use `--embeddings-npy` and `--vae-ckpt` if you have them for parity with released checkpoint.

3. **Pretrain**

   ```bash
   mix recgpt.pretrain \
     --ckpt data/recgpt_ckpt_export \
     --fixture data/recgpt-trajectories/fixture.json \
     --train data/recgpt-trajectories/train_sequences.json \
     --items data/recgpt-trajectories/items.json \
     --out data/recgpt-trajectories/ckpt_pretrained
   ```

4. **Eval (baseline = zero-shot)**

   ```bash
   mix recgpt.eval \
     --data-dir data/recgpt-trajectories \
     --ckpt data/recgpt_ckpt_export \
     --fixture data/recgpt-trajectories/fixture.json \
     --test data/recgpt-trajectories/test_sequences.json \
     --cold-test data/recgpt-trajectories/cold_test_sequences.json
   ```

5. **Eval (after pretrain)**

   ```bash
   mix recgpt.eval \
     --data-dir data/recgpt-trajectories \
     --ckpt data/recgpt-trajectories/ckpt_pretrained \
     --fixture data/recgpt-trajectories/fixture.json \
     --test data/recgpt-trajectories/test_sequences.json \
     --cold-test data/recgpt-trajectories/cold_test_sequences.json
   ```

6. **Compare** Hit@1, Hit@5, Hit@10, MRR between baseline and pretrained. If pretrained is better, performance improved.

---

## If Data Format Differs

If `recgpt-trajectories` has a different layout (e.g. raw trajectories, different keys, CSV):

- **Converter script** – Write a small Elixir or Python script to produce `items.json`, `train_sequences.json`, `test_sequences.json` in the canonical shapes.
- **Split** – 80% sessions → train, 20% → test (last-item-out). Cold: items in ≤ K train sessions.
- **Reference** – [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) test/steam layout.

---

## See also

- [05 Eval data shapes](05_eval_data_shapes.md)
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md)
- [03 Pipeline steps](03_pipeline_steps.md)
- [53 Mix tasks](53_mix_tasks.md) — mix recgpt.pretrain, mix recgpt.eval
