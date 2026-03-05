# Video-domain trajectory test (design)

Design for a **video-domain analogue** of a prediction-market “trade test”: prove next-item recommendation signal using **user watch trajectories** (e.g. KuaiRand-Pure). Same pipeline and canonical shapes as Phase 1 pretraining; the “test” is next-video prediction eval on held-out sequences.

---

## Purpose

- **Prediction-market analogue:** In a trade test, each sequence is one actor’s trade legs up to **market closing**; we train on leader sequences until close and evaluate next-trade or outcome. The trajectory has a natural endpoint (market resolution).
- **Video analogue:** Each sequence is one **user’s watch trajectory** (ordered list of video_ids, optionally with `time_ms`). There is **no market close** — no natural endpoint. We define trajectory end by session boundaries or max length (see [Session vs user aggregation](#session-vs-user-aggregation)). We train on user sequences and evaluate **next-video prediction** (Hit@k, MRR, test loss). No leader/retail distinction; success = model generalizes to held-out next-video.

This doc specifies the data model, pipeline, and design choices for the video-domain trajectory test.

---

## Data model

| Concept        | Prediction market (reference) | Video domain                          |
| -------------- | ------------------------------ | ------------------------------------- |
| **Item**       | (market_id, outcome_id)        | video_id (catalogue index)            |
| **Sequence**   | One wallet’s trade legs        | One user’s watch history (ordered)    |
| **Trajectory** | [item_id, …] + [t1, t2, …]     | [video_id, …] + optional [time_ms, …] |
| **Test**       | Next trade / outcome           | Next video (held-out)                 |

- **Items:** `items.json` — one row per video (id, title); titles from KuaiRand `video_features_basic_pure.csv` or placeholder.
- **Train sequences:** One row per user (or per session); sequence = ordered list of item_ids; optionally `timestamps` = `time_ms` from KuaiRand logs.
- **Test cases:** Held-out (user, prefix) → next_item; same shape as [05 Eval data shapes](05_eval_data_shapes.md): `{"context": [...], "next_item": id}`.
- **Cold items:** Optional: hold out some videos from training; measure cold-item Hit@k (see [07 Steam splits](07_steam_splits_and_pretraining.md) for cold semantics). KuaiRand converter can withhold a fraction of items as cold.

---

## Pipeline

Same as [93 Pretraining plan](93_pretraining_plan.md), with video data:

1. **Convert** — `mix recgpt.convert_trajectories --from /path/to/KuaiRand-Pure --out data/kuairand --format kuairand`  
   Produces `items.json`, `train_sequences.json`, `test_sequences.json`, `cold_test_sequences.json`, `cold_train_sequences.json`.

2. **Build fixture** — `mix recgpt.build_fixture --items data/kuairand/items.json --out data/kuairand/fixture.json --ckpt data/fuxi_ckpt_export`

3. **Pretrain** — `mix recgpt.pretrain ... --train data/kuairand/train_sequences.json --items data/kuairand/items.json --out data/kuairand/ckpt_pretrained --eval-test-every N --test data/kuairand/test_sequences.json`

4. **Eval** — `mix recgpt.eval --data-dir data/kuairand --ckpt data/kuairand/ckpt_pretrained --fixture data/kuairand/fixture.json --test data/kuairand/test_sequences.json`  
   Reports Hit@k, MRR, and optionally cold-item metrics.

The **trajectory test** is: after pretrain, run eval on `test_sequences.json`; success = test loss down, Hit@1 (or Hit@k) above random baseline. Optionally run eval on `cold_test_sequences.json` to measure generalization to never-seen-in-train videos.

---

## Design choices

### Session vs user aggregation

- **User-level (default):** One sequence per user = full watch history (or truncated by length). Matches KuaiRand’s `user_id` grouping. No natural close.
- **Session-level (future):** Split by session (e.g. gap threshold in `time_ms`). Each sequence then has a **natural close** (session end), analogous to market closing — see [Domains with a natural close](#domains-with-a-natural-close). Yields more, shorter sequences; may improve next-video signal in short-horizon eval. Not required for initial design; converter can stay user-level.

### Timestamps

- KuaiRand provides `time_ms` per (user_id, video_id). FuXi-Linear can consume per-position timestamps ([91 FuXi real timestamps](91_fuxi_linear_real_timestamps.md)).
- **Current:** KuaiRand converter may not yet emit `timestamps` in train/test output; pipeline still proves loss signal with position-only or zero timestamps.
- **Target:** When converter emits `timestamps` (e.g. `time_ms` or cumulative ms from sequence start), pretrain and eval use them for FuXi temporal channel; trajectory test then includes timing signal (watch order + real time gaps).

### Cold videos

- Withhold a subset of items from training (e.g. 15%); test cases whose `next_item` is in that set form `cold_test_sequences.json`. Same as cold-item design in [07](07_steam_splits_and_pretraining.md).
- Video trajectory test can report: (1) overall Hit@k on test_sequences, (2) cold Hit@k on cold_test_sequences.

### Metrics

- **Loss:** Train loss and test loss (when `--eval-test-every` and `--test` are set); use test loss for early stopping / model selection.
- **Hit@k / MRR:** From `mix recgpt.eval`; reject null (Hit@1 > random) per [06 Evaluation](06_evaluation_and_testing.md).

---

## Contrast with prediction-market trade test

| Aspect           | Prediction market (reference)       | Video trajectory test              |
| ---------------- | ----------------------------------- | ---------------------------------- |
| **Sequence**     | Leader wallet trade legs            | User watch history                 |
| **Horizon**      | **Until market closing** (natural endpoint) | No market close; session or length truncation |
| **Item**         | Market/outcome token                | Video                              |
| **Role**         | Leader vs retail (optional)         | No role; all users                  |
| **Timestamps**   | Trade time (critical for strategy)  | Watch time (optional at first)     |
| **Eval**         | Next trade / outcome                | Next video                         |
| **Success**      | Scout generalizes to leaders        | Model generalizes to holdout       |

The prediction-market analogue **trains (and defines each sequence) until closing of the market**; the video-domain trajectory test has no such natural close and instead relies on session boundaries or fixed length. The rest of the pipeline and data shapes are shared; video only needs user–video–time logs and a train/test/cold split.

---

## Domains with a natural close

We can **define a domain** (or a variant of the video domain) that has a **natural close** analogous to market closing, so that each trajectory has a well-defined endpoint and we train “until close.”

| Domain / variant        | What “close” means                    | Sequence definition                          | Data requirement                          |
| ----------------------- | -------------------------------------- | -------------------------------------------- | ----------------------------------------- |
| **Session-based video** | End of session                         | One session = one trajectory; close = session end | `time_ms` per event; session rule (e.g. gap > τ) |
| **Course / curriculum** | Course completion (last module)       | One learner’s path through a fixed course    | Items tied to course modules; completion events |
| **Conversion (e.g. e‑commerce)** | Purchase (or other conversion) | Actions in a session until conversion       | Session + conversion label or event        |

### Do we have this data?

- **Session-based video: Yes.** KuaiRand-Pure logs include `user_id`, `video_id`, and **`time_ms`** per row. We have what we need to split by gap (e.g. gap > 30 min → new session). The converter today **parses** `time_ms` but does not yet emit timestamps in the output or perform session splitting; that would be an extension (session-level aggregation + optional `timestamps` in train/test JSON).
- **Course / curriculum: No.** KuaiRand has no course structure, modules, or completion events — only watch logs and video metadata (e.g. tag, video_type).
- **Conversion: No.** KuaiRand has no purchase, signup, or other conversion events — only watch/click-style engagement.

So the only “natural close” variant we can support with **current data** is **session-based video**, once the converter is extended to split by `time_ms` gap and (optionally) emit timestamps.

- **Session-based video:** Treat each **session** as one trajectory. “Close” = **session end** (e.g. gap between consecutive `time_ms` > threshold, or explicit session-end event). Same items (videos) and pipeline; only aggregation changes: one sequence per session instead of per user. Converter can split by gap threshold (e.g. 30 min) so that each sequence has a natural close. This is the most direct analogue of “train until market closing” in the video setting.
- **Course completion:** If items are course modules or lessons, a trajectory is one learner’s progression; “close” = **course end** (final module completed). Requires course structure in the data (e.g. ordered modules, completion flags).
- **Conversion:** In e‑commerce or signup flows, a trajectory is a session of actions; “close” = **conversion event** (purchase, signup). Sequence = [item₁, …, itemₙ] up to conversion; no events after close. Requires conversion labels or timestamps.

Choosing **session-based video** gives a natural close (session end) with existing KuaiRand-style logs: define sessions by a gap threshold in `time_ms`, emit one sequence per session, and train/evaluate until that session’s end — analogous to training until market closing.

---

## Implementation status

- **Implemented:** Convert (KuaiRand), build_fixture, pretrain, eval; train/test/cold splits; `--eval-test-every` for test loss. See [43](43_pretraining_plan.md).
- **Optional / future:** KuaiRand converter emitting `timestamps` for FuXi; session-level aggregation; explicit cold-item reporting in eval CLI.

---

## See also

- [93 Pretraining plan](93_pretraining_plan.md) — Phase 1 pipeline (KuaiRand)
- [05 Eval data shapes](05_eval_data_shapes.md) — Canonical JSON shapes
- [06 Evaluation and testing](06_evaluation_and_testing.md) — Metrics, null hypothesis
- [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md) — Cold split semantics
- [91 FuXi-Linear real timestamps](91_fuxi_linear_real_timestamps.md) — Timestamps in converter and training
