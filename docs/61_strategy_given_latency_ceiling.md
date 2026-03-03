# Strategy Given Latency Ceiling

RecGPT inference runs **~200–280 ms** warm (BF16, RTX 4090, 4 forward passes). With the current architecture we **cannot go faster** without major changes (STATIC, fused beam, FP8, etc.). This doc maps that reality to the constraint framework in [Which Constraint Binds](../thirdparty/Which_20Constraint_20Binds.md) and defines a deployment strategy.

---

## Constraint Framework (Summary)

| Strategy          | Binding constraint                     | Max P99 |
| ----------------- | -------------------------------------- | ------- |
| **Binary**        | Competition (sub-100 ms bots dominate) | <80 ms  |
| **Bundle**        | Competition                            | <150 ms |
| **Catalyst**      | Opportunity (edge lasts seconds)       | <1 s    |
| **Combinatorial** | Opportunity (edge lasts minutes)       | <10 s   |

---

## RecGPT’s Place

| Metric             | RecGPT (measured) | Binary cap | Bundle cap | Catalyst cap | Combinatorial cap |
| ------------------ | ----------------- | ---------- | ---------- | ------------ | ----------------- |
| Warm P50           | ~230–280 ms       | 80 ms      | 150 ms     | 1 s          | 10 s              |
| Warm P99           | ~250–450 ms       | 80 ms      | 150 ms     | 1 s          | 10 s              |
| **Fits strategy?** | —                 | No         | No         | Yes          | Yes               |

**Conclusion:** RecGPT does **not** meet the latency caps for Binary or Bundle. It **does** fit Catalyst and Combinatorial, where opportunity lifetime binds and competition latency is less critical.

---

## Strategy for What We Have

### 1. Use RecGPT Where the Cap Is Opportunity

- **Catalyst** (2–10 s edge): RecGPT at ~250 ms uses only ~5–25% of the window. Plenty of headroom.
- **Combinatorial** (~1 h window): RecGPT latency is negligible relative to the opportunity.

→ Focus RecGPT on Catalyst and Combinatorial flows. Do **not** rely on RecGPT for latency-sensitive Binary/Bundle.

### 2. Combination System Positioning

RecGPT lives in a stack with reflex-logic-market and bs-p:

```
max_profitable_P99 = min(0.5 × T, ~0.8 × competitor_speed, economic)
```

- **Binary/Bundle:** Competition binds. The whole pipeline (reflex-logic-market, bs-p, RecGPT) must be <80–150 ms. RecGPT alone is 200–280 ms. RecGPT must be **disabled** or **bypassed** in these flows.
- **Catalyst/Combinatorial:** Opportunity binds. RecGPT’s ~250 ms is acceptable. Use RecGPT for richer recommendations when the edge lasts seconds or longer.

### 3. Operational Choices

| Use case                   | RecGPT role   | Reason                                        |
| -------------------------- | ------------- | --------------------------------------------- |
| Binary arbitrage           | Off or bypass | Total latency >80 ms target; RecGPT dominates |
| Bundle arbitrage           | Off or bypass | Total latency >150 ms target                  |
| Catalyst events            | On, full path | Edge ~2–10 s; RecGPT adds ~250 ms             |
| Combinatorial bid grouping | On, full path | Edge ~30 min–1 h; RecGPT negligible           |

### 4. How to Bypass

- **Config / feature flag:** Disable RecGPT for Binary/Bundle strategies so the pipeline uses reflex-logic-market and bs-p only.
- **Fallback:** If E2E latency budget is exceeded, skip RecGPT and return a simpler result (e.g. from bs-p or a cache).

### 5. What “Cannot Go Faster” Means Here

Current ceiling (~200–280 ms) assumes:

- Batched beam search (4 forwards)
- KV cache
- BF16
- No fused beam (unfused)
- No STATIC (CPU trie)

Further gains would need changes we have not implemented:

- STATIC (vectorized trie) to cut sync/trie overhead
- Fused beam to reduce kernel launch overhead
- FP8 (blocked by XLA autotuner on this stack)
- CUDA graphs for fixed-shape paths

Until then, assume this latency ceiling and choose strategies accordingly.

---

## Summary

| Constraint binds | Strategies              | RecGPT?         |
| ---------------- | ----------------------- | --------------- |
| Competition      | Binary, Bundle          | No (bypass)     |
| Opportunity      | Catalyst, Combinatorial | Yes (full path) |

**Action:** Use RecGPT for Catalyst and Combinatorial. Bypass or disable RecGPT for Binary and Bundle. Document this split in routing/config so the combination system applies RecGPT only where latency allows.
