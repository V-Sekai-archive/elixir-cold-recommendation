# MovieLens 20M: Full 5-Epoch Pretrain (Paper Parity)

Guide for running RecGPT pretraining for **5 epochs** on MovieLens 20M, matching the setup in the [RecGPT paper](https://arxiv.org/abs/2506.06270) (EMNLP 2025).

---

## Paper Setup

From the [HKUDS/RecGPT](https://github.com/HKUDS/RecGPT) repository:

```bash
accelerate launch pre_train.py --batch_size 40 --epoch 5 --tf_layer 3
```

The paper uses **5 epochs** of pre-training on 11 datasets. For a single dataset (e.g. MovieLens), we use the same epoch count.

---

## Prerequisites

1. **Convert MovieLens** to RecGPT JSON:

   ```bash
   mix recgpt.convert_movielens --max-items 5000 --out data/movielens-20m
   ```

2. **Build fixture**:

   ```bash
   mix recgpt.build_fixture --items data/movielens-20m/items.json \
     --out data/movielens-20m/fixture.json --ckpt data/recgpt_ckpt_export \
     --limit 5000 --no-canonical-texts
   ```

3. **Checkpoint** at `data/recgpt_ckpt_export` (from `mix recgpt.fetch_ckpt` + `mix recgpt.export_ckpt`).

---

## Run 5-Epoch Pretrain

```bash
mix recgpt.pretrain \
  --ckpt data/recgpt_ckpt_export \
  --fixture data/movielens-20m/fixture.json \
  --train data/movielens-20m/train_sequences.json \
  --items data/movielens-20m/items.json \
  --out data/movielens-20m/ckpt_5epoch \
  --epochs 5 \
  --batch-size 16 \
  --save-every 3236
```

**Options:**

- `--epochs 5` — 5 full passes over the training data (overrides `--iterations`).
- `--batch-size 16` — Larger batches speed up training; use 8 if GPU memory is tight.
- `--save-every 3236` — Save checkpoint every 3236 steps (~1 per epoch). Writes to `--out/step_003236/`, `step_006472/`, etc.

---

## Expected Duration

With ~51,764 train sequences and batch size 16:

- **Steps per epoch:** ~3,236
- **Total steps (5 epochs):** ~16,180
- **Estimated time:** 8–12 hours (depends on GPU)

For faster iteration, use fewer epochs (e.g. `--epochs 1` for ~2 hours) or a smaller `--max-items` in the converter.

---

## After Pretrain: Eval

```bash
# Get checkpoint SHA256 (required when using pretrained ckpt)
mix recgpt.ckpt_sha256 --ckpt data/movielens-20m/ckpt_5epoch

# Eval with the new SHA (replace HASH with output)
RECGPT_CKPT_SHA256=HASH mix recgpt.eval \
  --fixture data/movielens-20m/fixture.json \
  --ckpt data/movielens-20m/ckpt_5epoch \
  --test data/movielens-20m/test_sequences.json \
  --batch-size 32
```

Compare Hit@1, Hit@5, Hit@10, MRR with the baseline (zero-shot) and with the shorter 100-iteration run.

---

## Loss and Checkpoint Selection

The paper states:

> "The goal of evaluation is to select the model weights from the numerous checkpoints saved during the pre-training phase that have the **lowest loss** on the evaluation set."

Use `--save-every N` to save periodic checkpoints during training (e.g. `--save-every 3236` for one per epoch). Checkpoints are written to `--out/step_003236/`, `step_006472/`, etc. The final checkpoint is always written to `--out/`.

**Select the best checkpoint:** Run eval on each saved step and compare metrics:

```bash
# Get SHA for each step dir (run mix recgpt.ckpt_sha256 --ckpt <path> to get hash)
# Then eval each:
RECGPT_CKPT_SHA256=<hash> mix recgpt.eval --fixture data/movielens-20m/fixture.json \
  --ckpt data/movielens-20m/ckpt_5epoch/step_003236 \
  --test data/movielens-20m/test_sequences.json --batch-size 32
```

Repeat for each `step_*` directory. The checkpoint with the best Hit@1 or MRR is the one to use for serving.

---

## See Also

- [RecGPT paper](https://arxiv.org/abs/2506.06270)
- [HKUDS/RecGPT GitHub](https://github.com/HKUDS/RecGPT)
- [64 Investigation: recgpt-trajectories dataset](64_investigation_recgpt_trajectories_dataset.md)
- [53 Mix tasks](53_mix_tasks.md)
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md)
