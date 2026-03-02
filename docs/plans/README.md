# Performance and scaling plans

One plan per markdown. Ranked by profit (gain/effort).

| #   | Plan                                                   | Profit | Effort      | Gain             |
| --- | ------------------------------------------------------ | ------ | ----------- | ---------------- |
| 1   | [EXLA JIT disk cache](01_exla_jit_cache.md)            | 3.0    | Low         | High (setup)     |
| 2   | [Adaptive beam width](02_adaptive_beam_width.md)       | 1.5    | Low         | Low–med          |
| 3   | [BF16 inference](03_bf16_inference.md)                 | 1.5    | Medium      | High             |
| 4   | [Multi-context batching](04_multi_context_batching.md) | 1.2    | Medium–high | High             |
| 5   | [Sparse trie](05_sparse_trie.md)                       | 1.3    | High        | Critical for 13M |
| 6   | [Trie partitioning](06_trie_partitioning.md)           | 1.3    | High        | Enables 13M      |

## Implementation order (by profit)

1. [01_exla_jit_cache.md](01_exla_jit_cache.md) — highest profit
2. [03_bf16_inference.md](03_bf16_inference.md) — inference speed
3. [02_adaptive_beam_width.md](02_adaptive_beam_width.md) — low effort tweak
4. [04_multi_context_batching.md](04_multi_context_batching.md) — throughput
5. [05_sparse_trie.md](05_sparse_trie.md) — required for 13M
6. [06_trie_partitioning.md](06_trie_partitioning.md) — if sparse trie still too large

## 13M catalog

- **Mandatory:** Plan 5 (sparse trie)
- **Optional:** Plan 6 (partitioning) if single sparse trie exceeds memory
- Plans 1–4 improve latency/throughput regardless of catalog size
