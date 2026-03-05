# Training Domain Recommendation

**Recommended domain for training signal:** **MovieLens** (movies).

Train as much as possible with strict withheld test set and withheld catalogue items — no comparisons to other domains (e.g. Steam, games).

---

## MovieLens Domain

| Factor | Value |
|--------|-------|
| **Scale** | Large; train on full train split |
| **Test withheld** | Yes — test sequences never used for training, only for evaluation |
| **Catalogue withheld** | Yes — cold items (≤k appearances in train) evaluated via `cold_test_sequences.json` |
| **Domain** | Movie recommendations |
| **Pipeline** | `mix recgpt.convert_trajectories --format movielens` |
| **Training** | Train as much as possible; no artificial caps |

**Strategy:** Use MovieLens to maximise training signal. Test sequences and catalogue items (cold items) are withheld from training and used only for evaluation. Loss and Hit@k validate pretraining.

---

## Don't cheat

| What | Rule |
|------|------|
| **Test sequences** | Never in train_sequences; split before conversion. |
| **Withheld catalogue items** | Cold items (≤k in train) must not appear in the **training fixture**. Build fixture from warm items only; pretrain only on sequences whose items are all warm. Add cold items to fixture only at eval time. *(Requires pipeline support: warm_items.json, filtered train, dual fixture.)* |

---

## Data needed for signal

| Source | N train | Result |
|--------|---------|--------|
| MovieLens-20M | 10k (subset) | Loss ✓; Hit@k near random (domain mismatch with RecGPT). |
| MovieLens-20M | 0 = full | Run with `--train-limit 0 --test-limit 0` to maximise signal. |

**Proposed experiment:** Convert MovieLens → subsample N train sequences (or full with 0) → pretrain → eval. Plot Hit@1 vs N to find the knee.

---

## Signal comparison: 10 min vs 5 epochs

| Regime | Cmd | Est. time | Batches (10k seqs, batch 8) |
|--------|-----|-----------|-----------------------------|
| **10 min** | `--iterations 1200` (tune to hit ~10 min) | ~10 min | ~1,200 ≈ 1 epoch |
| **5 epochs** | `--epochs 5` | ~50 min | 5 × 1,250 = 6,250 |

**Rough estimate (10k train, batch-size 8):** Doc 86 targets \< 10 min for 1 epoch. So 1 epoch ≈ 1,250 batches ≈ 10 min → ~2 batches/sec. For 10 min fixed: ~1,250 batches ≈ 1 epoch. For 5 epochs: 6,250 batches ≈ 50 min.

**Compare:** Run both, eval on withheld test + cold. Expect 5 epochs ≥ 10 min on Hit@k if more training helps; if 10 min ≈ 5 epochs in signal, time-budget is sufficient.

```bash
# 10 min regime (tune --iterations to hit ~10 min; start with ~1200 for 10k seqs)
mix recgpt.pretrain ... --iterations 1200 --out data/training_signal_test/ckpt_10min

# 5 epochs regime
mix recgpt.pretrain ... --epochs 5 --out data/training_signal_test/ckpt_5epochs
```

---

## Recommendation

**For training signal (movies, maximise training):** Use **MovieLens**.

**One-shot pipeline:** `mix recgpt.training_signal_test` runs convert → build_fixture → pretrain → eval (zero-shot + pretrained, with cold when `cold_test_sequences.json` exists) → print comparison.

```bash
# One-shot: full pipeline (0 = no cap; uses data/recgpt_ckpt_export)
mix recgpt.training_signal_test --convert-from tmp/ml-20m --train-limit 0 --test-limit 0

# Compare 10 min vs 5 epochs
mix recgpt.training_signal_test --convert-from tmp/ml-20m --regime compare

# FuXi-Linear instead of GPT-2 (saves to ckpt_fuxi_*)
mix recgpt.training_signal_test --convert-from tmp/ml-20m --fuxi --iterations 500
```

**Manual steps** (if not using the one-shot):

```bash
# Convert MovieLens → canonical items + train/test (test withheld)
mix recgpt.convert_trajectories --from tmp/ml-20m --out data/training_signal_test \
  --format movielens --train-limit 0 --test-limit 0   # 0 = no cap; use full splits
mix recgpt.build_fixture --items data/training_signal_test/items.json \
  --out data/training_signal_test/fixture.json --no-canonical-texts --limit 5000
mix recgpt.pretrain --ckpt data/recgpt_ckpt_export \
  --fixture data/training_signal_test/fixture.json \
  --train data/training_signal_test/train_sequences.json \
  --items data/training_signal_test/items.json \
  --out data/training_signal_test/ckpt_pretrained --iterations 2000
mix recgpt.eval --data-dir data/training_signal_test --ckpt data/recgpt_ckpt_export
mix recgpt.eval --data-dir data/training_signal_test --ckpt data/training_signal_test/ckpt_pretrained
mix recgpt.eval --data-dir data/training_signal_test --ckpt data/recgpt_ckpt_export --cold   # withheld catalogue items
mix recgpt.eval --data-dir data/training_signal_test --ckpt data/training_signal_test/ckpt_pretrained --cold
```

**Train as much as possible:** Use `--train-limit 0` and `--test-limit 0` so the converter keeps all train/test data (0 = no cap). Train sequences are used for pretraining; test sequences are **never** used for training. For true withheld catalogue (don't cheat), build fixture from warm items only and filter train — *pipeline support planned*.

---

## See also

- [86 Training signal test dataset plan](86_training_signal_test_dataset_plan.md) — Converter, MovieLens format
- [87 MovieLens training signal log](87_movielens_training_signal_log.md) — Loss vs Hit@k result
