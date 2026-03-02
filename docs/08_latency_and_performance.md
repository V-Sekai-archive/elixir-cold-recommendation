# Latency and performance vs industry

Terminology aligns with [Replicate COG](https://replicate.com/docs/guides/build/push-a-model): **setup** = one-time load + JIT compile; **predict** = per-request inference. Cold (setup) ~20–30s; warm (predict) ~300–400ms on 12-layer + RTX 4090.

## What we did wrong (and what we fixed)

Industry-grade recommendation APIs typically target **single-digit to low double-digit ms** per request (e.g. P50 &lt; 50 ms). Our initial implementation was **~150–200+ ms** per request. Main causes:

### 1. **No batched inference in beam search** (fixed)

- **Was:** We called the model once per beam candidate per step (31 forward passes for top_k=10). That path is no longer allowed: `beam_search_top_k/5` requires a 2-arity `batch_fn`.
- **Industry:** One forward pass **per step** with **batch size = beam_width**. Same 4 steps → **4 forward passes** with batch size 10. GPU/NX is much more efficient on batched ops.
- **Fix:** Batched beam search: in each step we form one batch of all beam candidates, run `Inference.forward(batch, ...)` once, then compute scores and prune to top-k in memory. See `Decode.expand_beam_batched` and `Serve.get_logits_batch_fn`.

### 2. **Full-sequence forward every time (no KV-cache)** (fixed)

- **Was:** Every forward pass ran the full transformer over the **entire** sequence (context + prefix).
- **Fix:** In-memory KV-cache: `Inference.forward_with_cache/4` and `forward_incremental/5`; `Serve.get_logits_batch_fn` uses them so step 0 runs one full forward and captures cache, steps 1–3 run incremental (one new token + past). Cache is replicated to beam width when going from 1 sequence (context) to beam_width candidates. See `Decode.expand_beam_batched(..., cache)` and `Inference.gpt2_attn_incremental/5`.

### 3. **Pulling scores to CPU in a tight loop**

- **Wrong:** In `expand_beam` we did `Nx.slice_along_axis(...) |> Nx.to_number()` inside a loop over valid tokens. Each `to_number()` can force a device sync.
- **Better:** With batched inference we do one forward; then we only slice/gather from the batched logits tensor (still in tensor space) and at most do one bulk transfer when we need the top scores. We avoid many small syncs.

### 4. **Backend and device**

- **Config:** We use `EXLA.Backend` with `client: :cuda` when configured. If CUDA is not available, Nx uses the host (CPU) client.
- **Industry:** GPU for inference is standard; EXLA uses XLA (CPU or CUDA). Ensure the default Nx backend and EXLA client are set (e.g. `mix recgpt.check_gpu`).

## Summary

| Gap                         | Impact        | Status                          |
|----------------------------|---------------|---------------------------------|
| No batched beam inference  | ~8× too many forwards | **Fixed** (batched expand_beam) |
| No KV-cache                | Extra recompute per step | **Fixed** (forward_with_cache + incremental) |
| Per-token Nx.to_number sync | Extra latency | Mitigated by batching           |
| Backend / GPU              | 10×+ if on CPU | Config + check_gpu              |

After batching, expect **roughly 4–8× lower latency** per request (e.g. 4 forwards instead of 31). For further gains, add KV-cache and ensure GPU/EXLA is used when available.
