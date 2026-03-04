# Why RecGPT uses EXLA over Torchx

RecGPT uses **EXLA** (XLA/Nx backend) for inference, not **Torchx** (LibTorch). This document records the rationale.

---

## Summary

| Backend   | Runtime     | JIT / compile       | Batched inference        |
| --------- | ----------- | ------------------- | ------------------------ |
| **Torchx** | LibTorch C++ | None (eager)        | Mature kernels, no graph fusion |
| **EXLA**  | XLA         | First run ~20–30 s  | XLA fuses ops; optimized for batched |

We chose EXLA for: ecosystem alignment, batched-KV-cache fit, XLA optimization, and simpler deployment (one backend).

---

## Reasoning

### 1. Nx / Elixir ecosystem alignment

EXLA is the primary GPU backend in the Nx ecosystem. Bumblebee, Axon, and most Nx examples target EXLA. Config (`config :nx, default_backend: EXLA.Backend`) and tooling (`mix recgpt.check_gpu`, `EXLA_TARGET=cuda12`) are well-established. Using EXLA keeps us on the main path for documentation, CI, and future Nx features.

### 2. Batched KV-cache and stable JIT keys

Our decode uses a **padded KV cache** (`batch × n_head × max_cache_len × head_dim`) so JIT cache keys stay stable across steps. EXLA compiles once per shape family (full forward + incremental with fixed `past_len`). Torchx has no JIT, so it doesn’t gain from this; both backends would run the same ops, but EXLA’s graph-level optimization applies to our whole forward, including the batched beam candidates.

### 3. XLA graph optimization

XLA compiles the full tensor graph and can fuse ops, reduce memory traffic, and generate tuned CUDA kernels. For batched inference (beam_width candidates per forward), this often outperforms eager execution. Our 4-forward flow (1 full + 3 incremental) benefits from having each graph compiled and optimized once.

### 4. Long-lived server amortizes cold cost

`mix recgpt.serve` runs a long-lived process. The first request pays JIT cost (~20–30 s); subsequent warm requests see ~200–400 ms on 12-layer + RTX 4090. Torchx has no JIT cold but also no graph fusion. For a serving workload, warm latency dominates; EXLA’s warm performance is acceptable and the cold cost is one-time per process (or per shape change).

### 5. Single backend simplifies config and testing

One backend (EXLA) avoids branching config, CI targets, and deployment variants. We don’t need to maintain both Torchx and EXLA paths or document when to use which.

### 6. SPMD decode and single sync

We keep trie and scoring on device and do a single sync at the end. EXLA handles the tensor graph; both backends could implement this, but EXLA’s compiled graph keeps the batched ops efficient and coherent.

---

## Trade-offs we accepted

| Trade-off          | Mitigation                                             |
| ------------------ | ------------------------------------------------------ |
| JIT cold ~20–30 s  | Long-lived server; cold only on first request or restart |
| Per-request XLA overhead | Batched beam search (4 forwards) reduces relative overhead |
| EXLA build (XLA deps) | Dev container + `EXLA_TARGET=cuda12`; documented in [56](56_dev_container.md) |

---

## When Torchx might be preferable

- **Very short-lived processes** — e.g. serverless where each invocation is a new process; Torchx avoids JIT cold.
- **Tiny batches** — LibTorch may be faster for single-element or very small batches; we target beam_width ≥ 4.
- **Windows + LibTorch** — Torchx supports Windows with LibTorch; EXLA’s CUDA path is Linux-focused (dev container is Linux).

For RecGPT’s design (long-lived serve, batched beam, padded KV cache), EXLA is the better fit.

---

## See also

- [55 Dependencies](55_dependencies.md) — Nx, EXLA, Bumblebee.
- [42 Latency and performance](42_latency_and_performance.md) — Warm/cold, batched inference.
- [65 Latency flow](65_latency_flow.md) — End-to-end flow, EXLA JIT.
- [63 Investigation: RecGPT old vs current](63_investigation_recgpt_old_vs_current.md) — Torchx vs EXLA comparison.
