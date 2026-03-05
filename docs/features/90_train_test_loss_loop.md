# Train–Test Loss Loop

Monitor generalization during pretraining by evaluating loss on a held-out test set at regular intervals.

## Usage

```bash
mix recgpt.pretrain \
  --ckpt data/fuxi_ckpt_export \
  --fixture data/training_signal_test/fixture.json \
  --train data/training_signal_test/train_sequences.json \
  --items data/training_signal_test/items.json \
  --out data/ckpt_pretrained \
  --iterations 500 \
  --eval-test-every 50 \
  --test data/training_signal_test/test_sequences.json
```

## Options

| Option | Purpose |
|--------|---------|
| `--eval-test-every N` | Compute test loss every N training steps |
| `--test PATH` | Path to `test_sequences.json` (required when `--eval-test-every` is set) |

## Output

Every `--eval-test-every` steps, prints:

```
Step 50 train_loss=1.234 test_loss=1.456 best_test=1.456
Step 100 train_loss=0.987 test_loss=1.123 best_test=1.123
...
```

- **train_loss** — Current batch training loss
- **test_loss** — Mean cross-entropy loss on the full test set (no gradients)
- **best_test** — Best (lowest) test loss seen so far

Use `best_test` to track whether the model is generalizing or overfitting. If `test_loss` starts rising while `train_loss` keeps falling, consider early stopping or reducing learning rate.

## Implementation

- `RecGPT.TestLoss.compute/5` — Loads test cases, converts to sequences (context + next_item), builds batches, runs forward + `loss_shifted_ce`, returns mean loss
- `AxonTrain.run/3` — Accepts `:eval_test_every` and `:eval_test_fn`; calls the fn every N steps and logs

## See also

- [87 MovieLens training signal log](87_movielens_training_signal_log.md) — Loss curves, acceptable loss
- [88 Training domain recommendation](88_training_domain_recommendation.md) — MovieLens pipeline
