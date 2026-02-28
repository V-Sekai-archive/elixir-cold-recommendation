# Quality assurance

Sub-proposal of the [documentation index](README.md). How to run the quality-assurance checklist so the codebase stays lint-clean, type-checked, and tested.

---

## Problem or limitation

Before merging or releasing, there must be a single reference for which checks to run and what pass means. Without it, reviewers and releasers run inconsistent checks.

---

## Proposed improvement

Run a fixed sequence of checks before merge or release. All must pass. This doc is the single reference for what to run and what pass means.

**Validation backwards from here:** Running this QA checklist is how you validate that the implementation works. Steps 1–5 confirm the codebase builds and passes tests; step 6 (with pipeline and RECGPT\_\* env) confirms eval with real weights. This doc is the single pass/fail gate for use.

## QA checklist (run in order)

| Step           | Command                                                                                     | Pass condition                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| 1. Compile     | mix compile --warnings-as-errors                                                            | No warnings; build succeeds.                                                                         |
| 2. Format      | mix format --check-formatted                                                                | No unformatted files.                                                                                |
| 3. Credo       | mix credo                                                                                   | No issues.                                                                                           |
| 4. Unit tests  | mix test --no-start --exclude integration --exclude eval                                    | All tests and properties pass.                                                                       |
| 5. Dialyzer    | mix dialyzer --format short                                                                 | No type errors.                                                                                      |
| 6. Steam top-k | RECGPT\_\* env set; mix test test/recgpt/eval_test.exs --include eval --include integration | Pretrained (on catalogue) rejects null and Hit@1 >= zero-shot. Required; our only real weights test. |

**Steam top-k correctness (required):** This is the only real weights test. Options are **zero-shot** (base checkpoint, no training on this catalogue) vs **pretrained (with the new catalogue)** (checkpoint after pretrain). After the full pipeline (fetch_steam, build_fixture, pretrain), run the eval integration test. Two tests run: (1) pretrained rejects null (Hit@1 > random); (2) pretrained Hit@1 >= zero-shot Hit@1 so we fail when zero-shot equals baseline and pretrain does not improve. Set env and run:

```bash
export RECGPT_FIXTURE=data/steam/fixture.json
export RECGPT_CKPT_ZEROSHOT=data/recgpt_ckpt_export
export RECGPT_CKPT_EXPORT=data/ckpt_after_pretrain
export RECGPT_TEST_SEQUENCES=data/steam/test_sequences.json
mix test test/recgpt/eval_test.exs --include eval --include integration --no-start
```

Pass: both **"rejects null"** and **"pretrained (on catalogue) does not regress vs zero-shot"** pass. CI runs this after pretrain; see [.github/workflows/ci.yml](../.github/workflows/ci.yml).

Step 6 requires the pipeline to have been run (fetch_steam, build_fixture, pretrain) so fixture and both checkpoints exist. CI runs the full pipeline and step 6. Other: `mix test --include integration`; full pipeline per [02](02_pipeline_overview.md), [03](03_pipeline_steps.md).

## CI

The same steps run in [GitHub Actions](../.github/workflows/ci.yml) on push/PR. See the workflow for the full sequence (including integration and pipeline). CI uses Elixir 1.18.4 and OTP 27.

## One-shot QA (local)

Steps 1–5 (no weights):

mix compile --warnings-as-errors && mix format --check-formatted && mix credo && mix test --no-start --exclude integration --exclude eval && mix dialyzer --format short

Full QA including step 6 (Steam top-k, real weights): run the pipeline first, set RECGPT\_\* env, then run the same test command as step 6 above.

---

## See also

- [06 Evaluation and testing](06_evaluation_and_testing.md)
- [15 Layers overview](15_layers_overview.md), [16](16_layer_artifacts.md)-[21](21_layer_application.md) layer docs.
- [22 Top-tier recommendations](22_top_tier_recommendations.md)
