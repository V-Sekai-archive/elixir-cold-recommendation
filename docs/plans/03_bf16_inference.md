# Plan: BF16 inference

**Status:** Done | **Profit:** 1.5 | **Effort:** Medium | **Gain:** High

Profile after change: `mix recgpt.trace_predict --runs 50 --jitter-ms 3`

---

## Goal

1.3–2× inference via Tensor Cores. Prefer BF16 over FP16: same exponent range as FP32 (better numerical stability), same 16-bit memory footprint.

---

## Changes

- lib/recgpt/inference_params.ex: Add `dtype`; cast params via `Nx.as_type(t, {:bf, 16})`
- lib/recgpt/inference_defn.ex: Use dtype for aux, mask, scale, constants
- lib/recgpt/serve.ex: Read `config :recgpt, :inference_dtype, {:f, 32}` and pass to InferenceParams
- config/config.exs: `config :recgpt, :inference_dtype, {:bf, 16}` to enable

---

## Profile

BF16 vs FP32 mean/p50 and inference μs/forward.

To compare: set `config :recgpt, :inference_dtype, {:f, 32}` for FP32, then run
`mix recgpt.trace_predict --fixture data/steam/fixture.json --ckpt data/recgpt_ckpt_export --runs 10 --jitter-ms 3`.
