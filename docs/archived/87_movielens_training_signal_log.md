# MovieLens Training Signal — Log

Run: 2026-03-04. Dataset: MovieLens-20M subset (10k train sequences, 2k test). Model: RecGPT (GPT-2 base). Fixture: 5k items (Bumblebee + VAE FSQ).

---

## Pipeline

**Recommended:** Use `mix recgpt.training_signal_test` for the full pipeline:

```bash
# 1. Download MovieLens-20M
curl -L -o tmp/ml-20m.zip https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip -o tmp/ml-20m.zip -d tmp/

# 2. One-shot: convert → build_fixture → pretrain → eval (zero-shot + pretrained + cold)
mix recgpt.training_signal_test --convert-from tmp/ml-20m \
  --train-limit 10000 --test-limit 2000 --iterations 200
```

**Manual steps** (equivalent):

```bash
# 2. Convert to canonical JSON
mix recgpt.convert_trajectories --from tmp/ml-20m --out data/training_signal_test \
  --train-limit 10000 --test-limit 2000

# 3. Build fixture (5k items)
mix recgpt.build_fixture --items data/training_signal_test/items.json \
  --out data/training_signal_test/fixture.json --ckpt data/recgpt_ckpt_export \
  --no-canonical-texts --limit 5000

# 4. Pretrain
mix recgpt.pretrain --ckpt data/recgpt_ckpt_export \
  --fixture data/training_signal_test/fixture.json \
  --train data/training_signal_test/train_sequences.json \
  --items data/training_signal_test/items.json \
  --out data/training_signal_test/ckpt_pretrained \
  --iterations 200 --batch-size 4

# 5. Eval (zero-shot and pretrained)
mix recgpt.eval --data-dir data/training_signal_test --ckpt data/recgpt_ckpt_export ...
mix recgpt.eval --data-dir data/training_signal_test --ckpt data/training_signal_test/ckpt_pretrained ...
```

**Note:** Pretrained checkpoint has different SHA256. The task sets `RECGPT_CKPT_SHA256=` (empty) automatically when evaluating custom/pretrained checkpoints.

---

## Training Signal

| Step | Loss |
|------|------|
| Batch 0 | 8.87 |
| Batch 20 | 3.38 |
| Batch 40 | 2.74 |
| Batch 60 | 3.39 |
| Batch 80 | 2.00 |
| Batch 100 | 1.40 |
| Batch 120 | 0.91 |
| Batch 140 | 0.47 |
| Batch 160 | 1.27 |
| Batch 180 | 0.53 |

**Conclusion:** Loss decreases from ~9 to ~0.5. Clear training signal.

**Acceptable loss:** From this run, loss ~0.5 at ~200 iterations (10k seqs). For `mix recgpt.training_signal_test`, default `--iterations 500` (single regime) trains until acceptable loss; `--epochs 5` or `--regime compare` (10 min vs 5 epochs) for longer runs. See [88 Training domain recommendation](88_training_domain_recommendation.md).

**Test loss loop:** To find the best test loss and track progress, use `mix recgpt.pretrain` with `--eval-test-every N --test <test_path>`. Prints `train_loss`, `test_loss`, and `best_test` every N steps.

---

## Eval (pending)

Zero-shot and pretrained evals were started. After JIT warmup, expect Hit@1, Hit@5, MRR. Pretrained should be ≥ zero-shot if signal generalizes.

Test cases after filtering to 5k-item catalog: 949.
