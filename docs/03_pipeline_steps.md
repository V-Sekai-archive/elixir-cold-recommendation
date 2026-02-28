# Pipeline steps

Sub-proposal of the [documentation index](README.md). Steps 2-4, serve, checkpoint setup, and file layout. Overview and Step 1: [02 Pipeline overview](02_pipeline_overview.md).

---

## Step 2: Build fixture

**Goal:** Turn `items.json` into `fixture.json` (num_items, token_id_list) via Embedding + FSQ.

**Command:**

```bash
mix recgpt.build_fixture --items data/steam/items.json --out data/steam/fixture.json --ckpt data/recgpt_ckpt_export
```

**Output:** `fixture.json` with `num_items` and `token_id_list`. Same format as expected by Serve and the eval task.

---

## Step 3: Pretrain

**Goal:** Train on `train_sequences.json` using fixture and checkpoint; write updated params to an export dir.

**Command:**

```bash
mix recgpt.pretrain --ckpt data/recgpt_ckpt_export --fixture data/steam/fixture.json --train data/steam/train_sequences.json --items data/steam/items.json --out data/ckpt_after_pretrain --iterations 100 --batch-size 8 --log 50
```

**Output:** New export dir (e.g., `data/ckpt_after_pretrain`) with `manifest.json` and `.npy` files. Use this dir as `--ckpt` for eval and serve.

---

## Step 4: Eval

**Goal:** Run next-item evaluation on both standard test and cold-test sets; print Hit@k, MRR, and null rejection.

**Command:**

```bash
mix recgpt.eval --fixture data/steam/fixture.json --ckpt data/ckpt_after_pretrain --test data/steam/test_sequences.json --cold-test data/steam/cold_test_sequences.json
```

**Requirements:** Fixture and checkpoint must exist and load. **Both** `--test` and `--cold-test` are required; if `cold_test_sequences.json` is missing, the task fails and prompts you to run Fetch first.

**Output:** Two blocks of metrics: "Evaluation (standard test set)" and "Cold test".

---

## Optional: Serve

After pretrain (and optionally eval):

```bash
mix recgpt.serve --fixture data/steam/fixture.json --ckpt data/ckpt_after_pretrain [--grpc-port 50051]
```

gRPC only: **recgpt.v1.PredictionService/Predict**. Contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto). See [01 gRPC API](01_grpc_api.md).

---

## Checkpoint setup (before pipeline)

If you do not have an export dir yet:

```bash
mix recgpt.fetch_ckpt
mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export
```

Use `data/recgpt_ckpt_export` as `--ckpt` for build_fixture and pretrain.

---

## File layout summary

```
data/
├── recgpt_ckpt_export/          # From fetch_ckpt + export_ckpt
│   ├── manifest.json
│   â””── *.npy
├── recgpt_layer_3_weight.pt     # Optional; from fetch_ckpt
├── steam/
│   ├── items.json
│   ├── train_sequences.json
│   ├── test_sequences.json
│   ├── cold_test_sequences.json
│   ├── cold_train_sequences.json
│   â””── fixture.json             # From build_fixture
â””── ckpt_after_pretrain/         # From pretrain --out
    ├── manifest.json
    â””── *.npy
```

---

## Environment variables

| Variable             | Used by     | Purpose                         |
| -------------------- | ----------- | ------------------------------- |
| `RECGPT_FIXTURE`     | eval, serve | Override fixture path.          |
| `RECGPT_CKPT_EXPORT` | eval, serve | Override checkpoint export dir. |

Command-line options override these.

---

## See also

- [02 Pipeline overview](02_pipeline_overview.md) - Overview and Step 1.
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) - Splits and artifact semantics.
- [06 Evaluation and testing](06_evaluation_and_testing.md) - Eval metrics and null hypothesis.
- [05 Eval data shapes](05_eval_data_shapes.md) - JSON shapes.
- [01 gRPC API](01_grpc_api.md) - gRPC contract and serve.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.