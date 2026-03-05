# FuXi-Linear Latency Log

Latency measurements for FuXi-Linear (default) vs GPT-2 RecGPT on the same hardware and fixture.

**Setup:** `mix recgpt.trace_predict --fixture data/steam/fixture.json --ckpt <ckpt> --runs 20 --jitter-ms 3`  
**Hardware:** RTX 4090, CUDA 12.9  
**Fixture:** data/steam/fixture.json (~30k items)  
**Context:** `[0]`, top_k=10

---

## Run: 2026-03-05

| Model | Mean (ms) | P50 (ms) | P95 (ms) | Inference/forward (μs) |
|-------|-----------|----------|----------|------------------------|
| **FuXi-Linear** (data/fuxi_ckpt_export) | 179.15 | 169 | 255 | ~13,372 |
| **GPT-2 RecGPT** (data/recgpt_ckpt_export) | 188.89 | 175 | 273 | ~20,060 |

**Summary:**

- FuXi is **~5% faster** end-to-end (mean 179 vs 189 ms).
- FuXi inference forward **~33% faster** than GPT-2 (13.4 ms vs 20 ms per call).
- Both use single-forward decode for beam search; most time is in `beam_search_total` (~163–165 ms).
- FuXi has no trained signal yet (init params); latency reflects architecture, not quality.

**How to reproduce:**

```bash
# FuXi (default)
mix recgpt.export_fuxi_ckpt --out data/fuxi_ckpt_export
mix recgpt.trace_predict --fixture data/steam/fixture.json --ckpt data/fuxi_ckpt_export --runs 20 --jitter-ms 3

# GPT-2
mix recgpt.trace_predict --fixture data/steam/fixture.json --ckpt data/recgpt_ckpt_export --runs 20 --jitter-ms 3
```

---

## See also

- [42 Latency and performance](42_latency_and_performance.md) — targets, batching, KV-cache
- [85 FuXi-Linear status](85_fuxi_linear_status.md) — implementation status
- [66 Nsight Systems tracing](66_nsys_tracing.md) — GPU profiling
