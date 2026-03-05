# Review: thirdparty/bs-p — tricks we can borrow

[bs-p](https://github.com/lubluniky/bs-p) (polymarket-kernel) is a low-latency C/Rust kernel for prediction-market quoting and analytics. It emphasizes **zero hot-path allocation**, **SoA batch layout**, **lock-free SPSC ring buffer**, and **fast math**. Below is what RecGPT can reuse or adapt.

---

## 1. Zero allocations in the hot path

**bs-p:** All buffers are caller-allocated; kernel APIs take `*mut` output pointers and write into them. No `malloc` or heap in the quote/analytics path.

**RecGPT:** We already moved in this direction:

- Pre-alloc aux/mask on device (full + incremental shapes).
- Decode constants (`root_state`, `neg_inf`, `vocab_t`) built once at load and reused.
- Config read at load, not per request.

**Borrow:** Keep the rule “no new allocation in the request path.” When adding features, prefer pre-allocated state or caller-provided buffers (e.g. pass a pre-allocated response struct or reuse a pool of buffers for batch responses).

---

## 2. SoA (Structure of Arrays) for batch

**bs-p:** Separate contiguous arrays for `x_t`, `q_t`, `sigma_b`, etc. so SIMD can stride over each field without gathers.

**RecGPT:** Inference is already SoA-like: one Nx tensor per quantity (e.g. `batch_tensor`, `cache`), and we batch over the leading dimension. No change needed for the GPU path.

**Borrow:** If we ever add a CPU scoring path or a small “pre-filter” step, keep SoA layout (e.g. list of `{context_tokens_list}` vs list of structs) for cache and potential vectorization.

---

## 3. Lock-free SPSC ring buffer for handoff

**bs-p:** Single-producer single-consumer ring buffer with cache-line padding for `head`/`tail` to avoid false sharing. Used for passing market updates from one thread to another at high throughput (~29 M msgs/sec in their bench).

**RecGPT:** `PredictBatchCollector` uses a GenServer and a list as queue; each request is a `GenServer.call`, so the caller blocks until that request’s `recommend` finishes. There is no separate “ingress thread” and “inference thread” with a lock-free queue between them.

**Borrow (optional):**

- If we split “gRPC acceptor” from “inference worker,” we could put a **bounded queue** between them (e.g. ETS-based queue or a small NIF wrapping an SPSC ring) so the acceptor never blocks on inference and we control back-pressure. Elixir’s `:queue` or a single GenStage producer could approximate this without a C NIF.
- For true SPSC NIF: we could mirror bs-p’s ring (caller pre-allocates ring + slots; init/push/pop only touch indices and slot memory). That’s a larger change; only worth it if we see acceptor latency or contention in profiling.

---

## 4. Cache-line padding for shared indices

**bs-p:** `head` and `tail` are in different 64-byte cache lines so producer and consumer don’t false-share.

**RecGPT:** Inference state lives in one process (Serve state). No multi-process shared counters in the hot path. If we later add shared ETS tables or multiple workers touching the same stats (e.g. latency counters), we could align and pad hot fields to cache lines. Low priority until we have that design.

---

## 5. Caller-owned output buffers in batch API

**bs-p:** Batch functions take input slices + output slices; they never return owned buffers. Caller reuses the same `bid_p`/`ask_p` (or similar) across calls.

**RecGPT:** `get_logits_batch_tensor_fn.(batch_tensor, cache)` returns `{logits, new_cache}`; we don’t currently pass a “write logits into this tensor” buffer. EXLA/Nx manage device memory. For our setup this is fine; the important part is “reuse and avoid allocating in the loop,” which we do with cache and pre-alloc aux/mask.

**Borrow:** If we add a CPU-side batch helper (e.g. for trie or post-processing), use the same pattern: `fn(inputs, output_buffer) -> :ok` instead of `fn(inputs) -> new_output`.

---

## 6. Fast math approximations (sigmoid / log1p)

**bs-p:** Hot path uses a fast Pade-like sigmoid and a custom AVX-512 `log1p` so the inner loop stays vectorized and avoids scalar `exp`/`log`.

**RecGPT:** Inference is in Nx/EXLA on GPU; activations and logits are in Defn. We don’t control the exact sigmoid/softmax implementation there. For a future CPU scoring path (e.g. lightweight ranker), we could add a fast-sigmoid or fast-softmax option and gate it by config (similar to `KERNEL_USE_FAST_SIGMOID`).

**Borrow:** Document “fast math” as an option for any CPU-bound scoring we add; consider a config flag to switch exact vs approximate for tests/prod.

---

## 7. Benchmark discipline: warm-up and black_box

**bs-p:** Bench does ~2k warm-up iterations, then timed loop with `black_box` on inputs/outputs so the compiler doesn’t optimize away the work.

**RecGPT:** `mix recgpt.trace_predict` has a single “setup” run (JIT compile + one recommend) then N timed runs. First timed run can still be cold (kernel launch, etc.).

**Borrow:**

- Add an optional warm-up count (e.g. `--warm-up 5`) so the first _timed_ run is clearly warm, and report “drop first K runs” in docs.
- When we have micro-benchmarks (e.g. for a single function), pass inputs through a function that the compiler can’t elide (e.g. store in a module attribute or use a no-op that depends on the value) to avoid over-optimization.

---

## 8. Compile-time / config toggles for fast vs exact

**bs-p:** `KERNEL_USE_FAST_SIGMOID` toggles fast vs exact sigmoid. Lets them keep one code path and switch behavior for prod vs tests.

**RecGPT:** We use runtime config for `inference_dtype` (BF16 vs FP32) and removed `skip_aux_encoder`. We could add more “fast path” toggles (e.g. “skip trie fallback when all item_ids ≥ 0” is already conditional).

**Borrow:** Keep a single “fast path” vs “full/debug path” mindset: document which config options trade off latency vs fidelity and use them consistently (e.g. one config to force full sync or extra checks in dev).

---

## Summary table

| Trick               | bs-p usage              | RecGPT relevance       | Action                                              |
| ------------------- | ----------------------- | ---------------------- | --------------------------------------------------- |
| Zero hot-path alloc | Caller-owned buffers    | Already started        | Keep rule; extend to new code                       |
| SoA batch           | SIMD-friendly layout    | Already tensor batches | No change                                           |
| SPSC ring           | Market-data handoff     | Single-process today   | Optional: bounded queue if we split acceptor/worker |
| Cache-line padding  | head/tail               | No shared counters yet | Later if we add shared stats                        |
| Caller-owned outs   | Batch write into slices | Nx manages device      | Use pattern for any CPU batch helpers               |
| Fast sigmoid/log1p  | AVX-512 hot path        | GPU does its own       | Option for future CPU scoring                       |
| Warm-up + black_box | Bench accuracy          | trace_predict          | Add warm-up option; document                        |
| Fast vs exact flag  | KERNEL_USE_FAST_SIGMOID | Config toggles         | Document and keep one “fast path” config story      |

---

## References

- [bs-p (GitHub)](https://github.com/lubluniky/bs-p) — polymarket-kernel repo
- [README.md](https://github.com/lubluniky/bs-p/blob/main/README.md) — overview, features, benchmark snapshot
- [DOCS.md](https://github.com/lubluniky/bs-p/blob/main/DOCS.md) — math (logit space, Avellaneda–Stoikov, analytics)
- `thirdparty/bs-p/src/ring_buffer.rs` — SPSC API; `c_src/ring_buffer.c` — lock-free impl.
- `thirdparty/bs-p/c_src/kernel.c` — AVX-512 batch sigmoid, log1p, quote loop.
- `thirdparty/bs-p/examples/bench.rs` — warm-up, black_box, quote + SPSC throughput.
