# Performance and scaling plans

One plan per markdown. Ranked by profit (gain/effort) for **faster recommendations**.

| #   | Plan                                                   | Status  | Profit | Effort      | Gain             |
| --- | ------------------------------------------------------ | ------- | ------ | ----------- | ---------------- |
| 1   | [EXLA JIT disk cache](43_exla_jit_cache.md)            | **Done** | 3.0    | Low         | High (setup)     |
| 2   | [Adaptive beam width](44_adaptive_beam_width.md)       | **Done** | 1.5    | Low         | Low–med (~15–30%) |
| 3   | [BF16 inference](45_bf16_inference.md)                 | **Done** | 1.5    | Medium      | High (1.3–2×)   |
| 4   | [Multi-context batching](49_multi_context_batching.md) | Todo    | 1.2    | Medium–high | Throughput 2–5×  |
| 5   | [Sparse trie](47_sparse_trie.md)                       | Todo    | 1.3    | High        | Critical for 13M |
| 6   | [Trie partitioning](48_trie_partitioning.md)           | Todo    | 1.3    | High        | Enables 13M      |

## Implementation order (by profit, for single-request latency)

1. ~~[43_exla_jit_cache.md](43_exla_jit_cache.md)~~ — **Done** (disk cache + padded KV canonicalization)
2. ~~[45_bf16_inference.md](45_bf16_inference.md)~~ — **Done** (config opt-in; FP32 default after profiling)
3. ~~[44_adaptive_beam_width.md](44_adaptive_beam_width.md)~~ — **Done** (`max(4, min(top_k + 2, 12))`)
4. [49_multi_context_batching.md](49_multi_context_batching.md) — **Throughput 2–5×** under load (no single-req gain)
5. [47_sparse_trie.md](47_sparse_trie.md) — Required for 13M; latency impact TBD (possibly neutral)
6. [48_trie_partitioning.md](48_trie_partitioning.md) — If sparse trie still too large; may add partition-load latency

## Estimated performance gains (not yet implemented)

| Plan | Est. single-request latency | Notes |
| ---- | --------------------------- | ----- |
| 4 Multi-context | **0%** | Throughput only; one recommend = one beam search |
| 5 Sparse trie | **TBD / ±0%** | Memory for 13M; CSR lookup may match or slightly exceed dense gather |
| 6 Partitioning | **0 to slight +** | Partition load can add latency; mainly for 13M scaling |

## 13M catalog

- **Mandatory:** Plan 5 (sparse trie)
- **Optional:** Plan 6 (partitioning) if single sparse trie exceeds memory
- Plans 1–4 improve latency/throughput regardless of catalog size
