# Latency and performance vs industry

Terminology aligns with [Replicate COG](https://replicate.com/docs/guides/build/push-a-model): **setup** = one-time load + JIT compile; **predict** = per-request inference. Cold (setup) ~20–30s; warm (predict) ~300–400ms on 12-layer + RTX 4090.

**Note:** `mix recgpt.trace_predict` starts a fresh process each run, so it never benefits from in-memory JIT. For warm 200–400ms, run `mix recgpt.serve` and call the gRPC Predict API repeatedly.

**Config:** `predict_timeout_ms` (default 120_000) limits how long a single Predict request may run.

**Stable EXLA cache:** Incremental forward uses a padded KV cache (fixed shape `batch × n_head × max_cache_len × head_dim`) so the JIT cache key stays the same across steps. Configure `config :recgpt, :max_cache_len, 128` (default).

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

---

## SLO: RecGPT target P50 = 20 ms

RecGPT runs in a **combination system** with M:\reflex-logic-market and M:\bs-p. End-to-end P99 must stay under the profitable ceiling:

```text
max_profitable_P99 = min(0.5 × T_real, ~0.8 × competitor_P99, economic_breakeven_latency)
```

RecGPT's share of the latency budget:

- **Target P50:** **20 ms** (primary round-number target for the RecGPT component; reflex-logic-market + bs-p add &lt;0.1 ms).
- **Target P99:** configurable (e.g. 60 ms with buffer under E2E ceiling). Formula: `RecGPT_target_P99 = max_profitable_P99 − reflex_logic_market_P99 − bs_p_P99 − buffer_ms`.

**Config:** `config :recgpt, :target_p50_ms, 20` and `config :recgpt, :target_p99_ms, 60` (or env `RECGPT_TARGET_P50_MS` / `RECGPT_TARGET_P99_MS`). See [65 Latency flow](65_latency_flow.md) for the end-to-end flow diagram and per-stage optimization table. **Strategy given latency ceiling:** [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md) maps RecGPT's ~200–280 ms to the constraint framework (Binary/Bundle vs Catalyst/Combinatorial) and defines when to use or bypass RecGPT.

**Monitoring:** With `config :recgpt, :trace_predict, true`, per-request latencies are recorded in `RecGPT.LatencyStats`. Use `RecGPT.LatencyStats.get_percentiles/0` for recent P50/P95/P99 and `RecGPT.LatencyStats.check_slo/0` to assert targets. The health server exposes **GET /slo** (e.g. `curl http://localhost:50052/slo`): 200 when within SLO, 503 with message when P50 or P99 exceed target (for CI or alerting).

**External benchmark (hyperfine + grpcurl):** With `mix recgpt.serve` running, measure client-side latency and percentiles:

```bash
hyperfine --warmup 3 --runs 50 --export-json /tmp/hf.json -- "grpcurl -plaintext -import-path /workspaces/elixir-recgpt/priv/proto -proto recgpt/v1/recommendation.proto -d '{\"max_results\":5,\"context_item_ids\":[1,2,3,4]}' -format json localhost:50051 recgpt.v1.PredictionService/Predict" && python3 -c "
import json
with open('/tmp/hf.json') as f:
    data = json.load(f)
times = sorted(data['results'][0]['times'])
n = len(times)
p50 = times[int((n-1) * 50 / 100)] * 1000
p90 = times[int((n-1) * 90 / 100)] * 1000
p99 = times[int((n-1) * 99 / 100)] * 1000
print(f'p50: {p50:.2f} ms')
print(f'p90: {p90:.2f} ms')
print(f'p99: {p99:.2f} ms')
"
```
