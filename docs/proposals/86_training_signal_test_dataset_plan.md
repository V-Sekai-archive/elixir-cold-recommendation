# Training Signal Test Dataset — Plan

Goal: Define **one small dataset** from the recgpt-trajectories bundle (KuaiRand-Pure, merrec, movielens-20m, open-ecommerce) to validate that pretraining produces measurable training signal: **loss decreases** and **Hit@k improves** vs zero-shot, in a fast, reproducible run.

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| **Pretrain time** | \< 10 min (single epoch) on typical dev GPU |
| **Reproducibility** | Same data → same loss curve; fixed seed |
| **Signal** | Loss decreases; Hit@1 (pretrained) ≥ Hit@1 (zero-shot) |
| **Null rejection** | Pretrained rejects random baseline |

---

## Dataset Candidates (from recgpt-trajectories)

| Dataset | Notes | Typical size | Suitability |
|---------|-------|--------------|-------------|
| **movielens-20m** | Large; ML-20M is ~20M ratings. Too big for fast signal test. | ~20M rows | Use **subset** only |
| **KuaiRand-Pure** | Video recommendations; used in FuXi-Linear paper (KuaiRand-27K). | ~27K+ items | Good if we subset |
| **merrec** | Likely MERRec (Music). | Unknown | Inspect format first |
| **open-ecommerce** | E-commerce sessions. | Unknown | Inspect format first |

---

## Recommended: Single Subset for Training Signal Test

**Strategy:** Pick the **smallest dataset** (or smallest reasonable subset) that has clear session/item structure, then produce canonical files. Prefer **MovieLens-20M subset** or **KuaiRand-Pure subset** because they are well-documented and commonly used in RecSys benchmarks.

**Subset size (proposed):**

- **5,000–10,000 train sequences** — enough for loss to converge in 1–2 epochs
- **1,000–2,000 test cases** — stable Hit@k estimates
- **Catalog:** All items that appear in train+test (typically 2k–20k items)

**Alternative:** If one dataset is already small (e.g. &lt; 50k sequences), use it whole.

---

## Canonical Output Layout

After conversion, place in `data/training_signal_test/`:

| File | Purpose |
|------|---------|
| `items.json` | `{"num_items", "items": [{"id", "title"}, ...]}` |
| `train_sequences.json` | `{"num_items", "sequences": [[id, ...], ...]}` |
| `test_sequences.json` | `{"num_items", "test_cases": [{"context", "next_item"}, ...]}` |
| `cold_test_sequences.json` | Same shape; cold items (required; can be subset of test) |
| `cold_train_sequences.json` | Same shape; optional |

Shapes: [05 Eval data shapes](05_eval_data_shapes.md).

---

## Converter (implemented)

Use `mix recgpt.convert_trajectories` to convert raw datasets to canonical JSON:

```bash
# MovieLens-20M (recommended for training signal test)
mix recgpt.convert_trajectories --from /path/to/movielens-20m --out data/training_signal_test
mix recgpt.convert_trajectories --from /path/to/recgpt-trajectories/movielens-20m --out data/training_signal_test \
  --train-limit 10000 --test-limit 2000 --seed 42
```

**Options:** `--from` (required), `--out` (default: data/training_signal_test), `--format movielens`, `--train-limit 10000`, `--test-limit 2000`, `--seed 42`.

**MovieLens-20M:** Expects `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres). Download from https://grouplens.org/datasets/movielens/20m/

**Chosen dataset:** MovieLens-20M subset — well-documented format, deterministic subset via `--seed`.

---

## Pipeline After Conversion

```bash
# 1. Build fixture
mix recgpt.build_fixture --items data/training_signal_test/items.json \
  --out data/training_signal_test/fixture.json

# 2. Pretrain (few iterations for signal test)
mix recgpt.pretrain \
  --ckpt thirdparty/checkpoints/recgpt \
  --fixture data/training_signal_test/fixture.json \
  --train data/training_signal_test/train_sequences.json \
  --items data/training_signal_test/items.json \
  --out data/training_signal_test/ckpt_pretrained \
  --iterations 500

# 3. Eval zero-shot (baseline)
mix recgpt.eval --data-dir data/training_signal_test \
  --ckpt thirdparty/checkpoints/recgpt \
  --fixture data/training_signal_test/fixture.json

# 4. Eval pretrained (signal)
mix recgpt.eval --data-dir data/training_signal_test \
  --ckpt data/training_signal_test/ckpt_pretrained \
  --fixture data/training_signal_test/fixture.json
```

Compare Hit@1, Hit@5, MRR: pretrained should be ≥ zero-shot.

**One-shot:** `mix recgpt.training_signal_test --convert-from /path/to/movielens-20m` — runs convert → build_fixture → pretrain → eval (zero-shot + pretrained + cold when `cold_test_sequences.json` exists) → print comparison.

| Option | Purpose |
|--------|---------|
| `--train-limit 0` | Max train sequences (0 = no cap, full splits) |
| `--test-limit 0` | Max test cases (0 = no cap) |
| `--regime` | `single` (default), `10min`, `5epochs`, or `compare` (10 min vs 5 epochs) |
| `--fuxi` | Use FuXi-Linear init instead of GPT-2; saves to `ckpt_fuxi_*` |
| `--epochs` | Pretrain epochs (overrides `--iterations` when set) |
| `--data-dir`, `--ckpt`, `--iterations`, `--fixture-limit` | Paths and limits |
| `--skip-convert`, `--skip-build`, `--skip-pretrain` | Skip steps (data already present) |

---

## Next Actions

1. **Run full pipeline** — Convert → build_fixture → pretrain → eval (zero-shot and pretrained).
2. **Record metrics** — Compare Hit@1, Hit@5, MRR; confirm pretrained ≥ zero-shot.
3. **Add formats** — KuaiRand, MerRec converters (see plan in Known Dataset Formats).

---

## See also

- [53 Mix tasks](53_mix_tasks.md) — recgpt.convert_trajectories, recgpt.build_fixture, recgpt.pretrain, recgpt.eval
- [64 Investigation: recgpt-trajectories dataset](64_investigation_recgpt_trajectories_dataset.md) — Pipeline and format overview
- [05 Eval data shapes](05_eval_data_shapes.md) — Canonical JSON shapes
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Cold vs regular
