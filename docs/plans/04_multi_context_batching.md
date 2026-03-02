# Plan: True multi-context batching

**Status:** Todo | **Est. gain:** 2–5× throughput under load (no single-request latency) | **Profit:** 1.2 | **Effort:** Medium–high | **Gain:** High

Profile after change: `mix recgpt.trace_predict --runs 50 --jitter-ms 3`

---

## Goal

One batched SPMD run for B contexts instead of B sequential `recommend` calls. Expect 2–5× throughput under load.

---

## Changes

- lib/recgpt/decode.ex: Add `beam_search_batch_spmd` taking `list_of_context_ids`; stack context tokens to `{B, context_len}`; run steps with batch `B * beam_width`; split results per context
- lib/recgpt/serve.ex: `recommend_batch` calls `beam_search_batch_spmd` when `length(list) > 1`
- lib/recgpt/predict_batch_collector.ex: Group by top_k; call `recommend_batch` once per group; map replies to callers

---

## Profile

Throughput (req/s) at `predict_batch_size: 8` vs sequential.
