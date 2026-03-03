# Plan: Adaptive beam width

**Status:** Done | **Est. gain:** ~15–30% when top_k < beam_width | **Profit:** 1.5 | **Effort:** Low | **Gain:** Low–medium

Profile after change: `mix recgpt.trace_predict --runs 50 --jitter-ms 3`

---

## Goal

Avoid over-beam when `top_k` is small. Currently `beam_width = max(4, top_k)`; when top_k=5 this may over-expand.

---

## Changes

- lib/recgpt/decode.ex line 59: `beam_width = max(4, min(top_k, 12))` or cap at `top_k + 2` instead of `max(4, top_k)`
- Document trade-off: slightly fewer candidates explored; acceptable quality for small top_k

---

## Profile

Compare latency at top_k=5 vs top_k=10.
