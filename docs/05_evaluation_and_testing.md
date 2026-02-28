# Evaluation and testing

How to evaluate RecGPT (zero-shot and trained), restrict evaluation to **held-out data only**, and reject the null hypothesis that the model has no predictive signal.

For best quality, pretrain on the train split then evaluate (see [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md), [08 Pipeline reference](08_pipeline_reference.md)). Zero-shot is a baseline only.

---

## Zero-shot vs trained

| Mode          | Checkpoint                                                                       | Fixture                                                     | Training on catalog?               |
| ------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- | ---------------------------------- |
| **Zero-shot** | Pretrained (e.g., hkuds/RecGPT_model export)                                      | Built from item text only (Embedding → FSQ → token_id_list) | No.                                |
| **Trained**   | Fine-tuned on this catalog (e.g., `mix recgpt.pretrain` or Python `pre_train.py`) | Same fixture, same catalog                                  | Yes; only checkpoint path differs. |

Run `mix recgpt.eval` twice: once with the pretrained checkpoint (zero-shot), once with the fine-tuned checkpoint (trained). Compare Hit@1, Hit@5, Hit@10, and MRR.

---

## Null hypothesis

- **Null (H0):** The model has no predictive signal — Hit@1 ≈ 1/N, MRR ≈ 1/N (N = catalog size).
- **Reject H0** if **Hit@1 > random_hit_at_1** (where `random_hit_at_1 = 1/N`). You may also require MRR > 1/N.
- The eval task and `RecGPT.Eval.evaluate/3` report `random_hit_at_1` and print “Reject null (Hit@1 > random): yes/no”.

---

## Held-out eval

Eval must use **data that was not used for training**.

- **Train:** e.g., full sessions or context-only sequences. Use only this for training.
- **Test:** e.g., last-item-out per session — one test case per session: `context` = all but last click, `next_item` = last. Those last-item labels are never used as input during training.

In this repo, `test_sequences.json` and `cold_test_sequences.json` are held-out. `RecGPT.Steam.Fetch` (or the dataset source) produces them; training uses only the train split.

---

## Commands

- **Eval (zero-shot or trained):**  
  `mix recgpt.eval --fixture <path> --ckpt <export_dir> --test <test_sequences.json> --cold-test <cold_test_sequences.json>`

Both `--test` and `--cold-test` are required. Default paths: `data/steam/fixture.json`, `data/steam/test_sequences.json`, `data/steam/cold_test_sequences.json`. Override with `--fixture`, `--ckpt`, `--test`, `--cold-test`, or env vars `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT`.

---

## Automated test

`test/recgpt/eval_test.exs` loads fixture, checkpoint, and test files; runs `RecGPT.Eval.evaluate/3`; and asserts **reject null** (Hit@1 > random_hit_at_1). Requires fixture, checkpoint export dir, and test files; skipped when missing (tags: integration, eval).

```bash
mix test test/recgpt/eval_test.exs --include eval --include integration
```

For zero-shot vs trained in CI: run once with `RECGPT_CKPT_EXPORT` pointing to the pretrained export, then again with the fine-tuned export.

---

## See also

- [00 RecGPT library](00_recgpt_library.md) — Module reference.
- [06 Eval data shapes](06_eval_data_shapes.md) — JSON format for test files.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Train/test/cold splits.
- [08 Pipeline reference](08_pipeline_reference.md) — Eval step in the pipeline.
