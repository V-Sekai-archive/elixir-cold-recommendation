# Investigation: RecGPT-old vs Current — Why Would Old Be Faster?

Comparison of `thirdparty/RecGPT-old` (Elixir port + Python reference) vs the current RecGPT implementation to understand when and why the old code might be faster.

---

## Architecture Comparison

| Aspect | RecGPT-old (Elixir) | RecGPT-old (Python) | Current (Elixir) |
|-------|---------------------|---------------------|------------------|
| **Backend** | Torchx (LibTorch) | PyTorch | EXLA (XLA/CUDA) |
| **Layers** | From checkpoint (often 3 in reference) | **3 layers** (predict.py default) | From checkpoint (**3** in data/recgpt_ckpt_export; **12** in COG) |
| **Beam search** | Per-candidate forwards, no batching | Batched predict_aux, beam per seq | **Batched** (beam_width candidates per forward) |
| **Forward count** | **1 + beam + beam + beam** (e.g. 13 for beam=4) | Varies by batch | **4** (1 full + 3 incremental) |
| **KV cache** | **No** — full sequence every forward | No (in reference) | **Yes** — incremental for steps 1–3 |
| **Per-token sync** | **Nx.to_number() per valid token** (many syncs) | No (stays on GPU) | Single sync at end (SPMD) |
| **Trie** | CPU (Elixir map), no tensors | Python trie | **Trie tensors** on device (SPMD) |

---

## Why RecGPT-old Elixir Could Be Faster (in Some Cases)

### 1. Smaller Model (Most Likely)

The Python reference uses **3 transformer layers** (`predict.py`: `args.tf_layer=3`, `GPT2Config(n_layer=3)`). The current codebase typically uses a **12-layer** checkpoint (e.g. COG, Steam).

**4× fewer layers ⇒ ~4× less compute per forward.** Even with 13 forwards and no KV cache, 3-layer × 13 could be faster than 12-layer × 4.

**Check:** Ensure comparison uses the **same checkpoint** (same layer count). If old uses a 3-layer ckpt and current uses 12-layer, that explains the gap.

### 2. Torchx vs EXLA

- **Torchx:** LibTorch (PyTorch C++). Mature kernels, little or no JIT. No compile step.
- **EXLA:** XLA compilation. First run pays JIT cost (~20–30 s cold). Per-request overhead from XLA scheduling.

If RecGPT-old runs with Torchx on GPU (or CPU with small model), it can have:
- No JIT cold cost
- Potentially faster for very small batches (LibTorch optimized for common shapes)

### 3. Sequence Length

RecGPT-old `item_ids_to_context_token_ids` pads to `@seq_token_capacity 1024` and uses `@max_length 255` items. So it can run very long sequences. But that would make it **slower**, not faster. If the comparison uses **short** contexts (e.g. 1–4 items = 4–16 tokens), the old full-sequence forwards are cheap. Current uses the same token layout; the win is incremental (steps 1–3 only need 1 new token).

### 4. Stub / Zero-Layer Checkpoint

RecGPT-old `Inference.forward` has a stub path: when `gpt2_n_layers(params) == 0`, it skips all transformer blocks and uses "last position as hidden". If the old checkpoint is a stub, it would be nearly instant.

---

## Why Current Should Be Faster (Like-for-Like)

For the **same model size** and **same context length**:

| Factor | Old | Current | Effect |
|--------|-----|---------|--------|
| Forward count | 13 (unbatched) | 4 (batched) | **~3× fewer** |
| KV cache | No | Yes | **Steps 1–3 ~4× less work** (1 token vs full prefix) |
| Per-step sync | Many `Nx.to_number` | Single sync at end | **Large** (GPU↔CPU round-trips) |
| Trie on device | CPU only | SPMD trie tensors | Less sync, better occupancy |

The current implementation is built to minimize latency; RecGPT-old was a port with fewer optimizations.

---

## Recommended Checks

1. **Same checkpoint?** Compare layer count: `grep n_layer` or inspect `manifest.json` / checkpoint keys. If old=3 layers and current=12, that alone explains ~4×.
2. **Same context?** Use identical `context_item_ids` (e.g. `[11]` or `[0,1,2]`) and `top_k`.
3. **Warm vs cold?** EXLA has ~20–30 s JIT cold; Torchx has none. Compare **warm** timings: run several requests before measuring.
4. **Backend?** Ensure RecGPT-old actually uses GPU (Torchx with CUDA) if comparing to current EXLA+CUDA.
5. **Beam width / top_k?** Old uses `beam_width = max(4, top_k)`. Current uses similar. Keep them equal.

---

## Conclusion

**Most plausible reason RecGPT-old is faster:** It uses a **smaller model** (e.g. 3 layers) or a **stub checkpoint**, while current uses a 12-layer model. A 3-layer model with 13 full forwards can still be faster than 12-layer with 4 batched KV-cache forwards.

**If models are identical:** Current should be faster. If not, verify checkpoint, warm vs cold, and backend parity before concluding.

**Action:** Run both with the **same checkpoint** (e.g. `data/recgpt_ckpt_export` which has 3 layers). Ensure warm runs (EXLA JIT already compiled) and identical context/top_k. If RecGPT-old is still faster with the same 3-layer ckpt, the remaining causes are: Torchx vs EXLA backend behavior, or fewer syncs (old may run on CPU where `to_number` is cheap).
