# Frontier Models: Path from EXLA/Elixir to Implementation

We use Nx + EXLA + Axon. Can we implement the frontier models (FuXi, Mamba, etc.) in this stack? This doc maps the implementation path and gaps.

Related: [82 Zero-shot semantic id reuse](82_zeroshot_semantic_id_reuse.md), [79 EXLA over Torchx](79_exla_over_torchx.md), [62 Ablation tensor graph](62_ablation_tensor_graph.md).

---

## Stack Summary

| Component | Purpose |
|-----------|---------|
| **Nx** | Tensors, `defn` numerical functions |
| **EXLA** | JIT compiler for Nx; compiles `defn` to XLA → CPU/CUDA |
| **Axon** | Training (loss, optimizers, streaming) |
| **RecGPT** | `Inference.forward/4`, `InferenceDefn`, `Decode.beam_search_*` |

**Constraints:** `defn` disallows recursion; must use `while`. EXLA compiles the full graph. We keep: input token_ids, output logits (15_361), hidden 768, trie/decode path.

---

## Nx/EXLA Capabilities for Frontier Models

| Capability | Nx/EXLA | Notes |
|------------|---------|-------|
| **Attention (GPT-2 style)** | ✅ | Matmul, softmax, mask. Current `Inference` uses it. |
| **`while` loops** | ✅ | `Nx.Defn.Kernel.while/4`. Can implement recurrence. |
| **Linear algebra** | ✅ | `Nx.dot`, `Nx.multiply`, `Nx.add`, etc. |
| **Cumulative/prefix ops** | ⚠️ | No native `cumsum`. Use `Nx.window_sum` with stride 1, or `while` over sequence. Linear attention needs prefix sums. |
| **Custom CUDA kernels** | ❌ | EXLA generates from graph; no raw CUDA. Would need Rustler NIF. |
| **Gradient through `while`** | ✅ | Axon/Nx support; XLA has `while` gradient. |

---

## Frontier Model → EXLA Path

### 1. FuXi-Linear (Linear Attention for Sequential Rec)

**What it does:** Temporal Retention + Linear Positional Channel. Linear O(L) complexity. PyTorch code: [USTC-StarTeam/fuxi-linear](https://github.com/USTC-StarTeam/fuxi-linear).

**Status: Fully wired** in `RecGPT.FuxiLinearInference` — full model (Retention + LinearTemporalChannel + LinearPositionalChannel) using RecGPT semantic ID structure. See [84 FuXi-Linear implementation](84_fuxi_linear_implementation_plan.md), [85 FuXi-Linear status](85_fuxi_linear_status.md).

**EXLA path:**
- Linear attention = no softmax over full QK^T. Often: `attn = Q @ (K^T @ V)` with causal masking via cumulative sums.
- `Nx.dot`, `Nx.multiply`, `Nx.add` — all supported.
- Cumulative state: `while` over sequence positions, or `Nx.window_*` tricks.
- **Feasibility: High.** No custom CUDA. Port the math to `defn`.

**Implemented:** `FuxiLinearInferenceDefn` (JIT `forward_last_4_logits/4`), `FuxiLinearInferenceParams` (checkpoint adapter), Serve integration, AxonTrain, `mix recgpt.export_fuxi_ckpt`.

---

### 2. FuXi-γ (Exponential-Power Temporal, Diagonal-Sparse)

**What it does:** Exponential-power temporal encoder, diagonal-sparse positional. Still attention-based but sparsified. arXiv 2512.12740.

**EXLA path:**
- Sparse/diagonal attention: mask or structured matmul. Nx supports masks.
- Exponential-power: `Nx.pow`, elementwise ops.
- **Feasibility: High.** Standard ops; sparsity = mask or sparse matmul (Nx has `Nx.put_slice`, gather for structured sparsity).

---

### 3. Mamba / Mamba-2 (Selective State Space)

**What it does:** Selective scan over sequence. Core op: `selective_scan(u, delta, A, B, C)` — recurrence with input-dependent gating. Official impl uses custom CUDA kernels.

**EXLA path:**
- **Option A: `while` over sequence** — Implement selective scan as `while {i, state}`, body updates state per position. Works in `defn`, compiles to XLA. Risk: sequential loop may be slow; XLA may not parallelize well.
- **Option B: Rustler NIF** — Wrap [mamba-mini](https://github.com/MzeroMiko/mamba-mini) or official CUDA selective_scan in Rustler. Call from `deftransform` when we need the op. Loses EXLA JIT for that op; data round-trips to NIF.
- **Option C: Associative scan** — Mamba-2’s SSD has structure that allows parallel scan in some cases. If we can express as prefix-style op, Nx `window_*` or custom `while` with parallel-friendly formulation might work.

**Feasibility: Medium.** Option A is pure Elixir but may be slow. Option B works but adds NIF + round-trip. Option C depends on Mamba-2 formulation.

---

### 4. Mamba-2-Hybrid / SAMBA (Mamba + Attention)

**What it does:** Mix Mamba layers with attention layers. E.g. 43% Mamba-2 + 7% attention + 50% MLP.

**EXLA path:**
- Attention layers: same as current `Inference` (we have them).
- Mamba layers: per above (while or NIF).
- **Feasibility: Medium.** Same as Mamba; we implement/layer both.

---

### 5. RWKV (Linear RNN-style)

**What it does:** Recurrent with matrix-valued states; no quadratic attention. RWKV-7: constant memory per token.

**EXLA path:**
- Recurrence: `while` over sequence, state update per token.
- Math: linear in sequence for inference. All `Nx.dot`, `Nx.add`, etc.
- **Feasibility: High.** No custom kernels in the reference; recurrence maps to `while`.

---

### 6. MaTrRec (Mamba + Transformer for Rec)

**What it does:** Hybrid for sequential recommendation. Mamba + Transformer blocks.

**EXLA path:**
- Same as Mamba-2-Hybrid: Mamba blocks (while or NIF) + our existing attention blocks.
- **Feasibility: Medium.**

---

## Recommended Order of Implementation

| Priority | Model | Path | Risk |
|----------|-------|------|------|
| 1 | **FuXi-Linear** | Pure `defn`; linear attention math | Low |
| 2 | **FuXi-γ** | Pure `defn`; sparse attention | Low |
| 3 | **RWKV** | `while` recurrence | Low |
| 4 | **Mamba** | `while` first; NIF if too slow | Medium |
| 5 | **MaTrRec / Mamba-Hybrid** | After Mamba works | Medium |

---

## Gaps and Mitigations

| Gap | Mitigation |
|-----|------------|
| No `cumsum` in Nx | Use `Nx.window_sum` with padding + stride 1; or `while` with `Nx.put_slice` into accumulator. |
| Mamba selective_scan slow in `while` | Profile first; if bottleneck, add Rustler NIF wrapping mamba-mini or official CUDA. |
| Checkpoint format differs | New `CheckpointLoader` adapter per model; same output contract (params map for our forward). |
| Training (Axon) | Axon composes with `defn`; `grad` works through `while`. Ensure loss uses our forward. |

---

## Interface Contract (Unchanged)

Any new backbone must satisfy:

- **Input:** `batch_token_ids` (batch, seq_len), `batch_aux`, `embed_mask`
- **Output:** `logits` (batch, 15_361) — or hidden (batch, seq_len, 768) for last position before head
- **Head:** Same `pred_head` linear(768, 15361) — we can keep or retrain

Fixture, trie, decode stay the same. We swap the body between embed and head.
