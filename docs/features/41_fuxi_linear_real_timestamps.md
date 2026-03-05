# FuXi-Linear Real Timestamps Implementation Plan

FuXi-Linear uses **real interaction timestamps** in the Temporal Retention Channel, not position indices. Our implementation currently falls back to `[0,1,2,...]` when `all_timestamps` is nil. To match the paper, we must pass real timestamps through the pipeline.

**Primary use case: Prediction market trading** ([80 Prediction market trading system](80_prediction_market_trading_system.md)). Scout is trained on leader wallet sequences; leader context → Scout inference → top-k item_ids → catalog. Timestamps are critical for trade timing and trajectory shape. See [§Prediction Market Rationale](#prediction-market-rationale) below.

---

## What FuXi-Linear Does (paper §4.4)

- **Input:** Per-position timestamps \( t_1, t_2, \ldots, t_n \) (e.g. KuaiRand `time_ms`)
- **Temporal channel formulas:**
  - Sinusoidal Q/K from \( \cos(\theta \cdot t_i) \), \( \sin(\theta \cdot t_i) \) (absolute timestamps)
  - Decay by interval: \( r^{t_n - t_i} \), \( r^{t_n - t_{n-1}} \) (real time gaps)
- **Preprocessing:** Their KuaiRand script keeps `sequence_timestamps` = raw `time_ms` values

---

## Current State (Implemented)

| Layer | Status |
|-------|--------|
| **Convert (Jon-Becker)** | Emits `%{"sequence" => seq, "timestamps" => ts}`; sync_to_db persists timestamps |
| **DB (train_sequence_rows)** | Migration adds `time_ms`; Sync persists when syncing from list/JSON |
| **Sync** | `sync_sequences_from_list` accepts timestamped format; `parse_sequence_rows*` include `time_ms` |
| **Training.build_train_batch** | Optional timestamps; builds `all_timestamps` (batch, seq_len, 8) cumulative ms from sequence start |
| **AxonTrain** | Passes `all_timestamps` to `FuxiLinearInference.forward_full_sequence` when available |
| **FuxiLinearInference** | Uses `all_timestamps` when provided; else `position_timestamps(batch, seq_len)` |

**Deferred:** KuaiRand/MovieLens converters still drop timestamps; Jon-Becker (Polymarket) full pipeline implemented.

---

## Required Changes

### 1. Converter: Preserve Timestamps

- **KuaiRand:** Change `build_sequences_kuairand` to emit `[{item_id, time_ms}, ...]` (or `{seq, timestamps}`) instead of `[item_id, ...]`
- **MovieLens:** Same—keep timestamp from ratings
- **Output format:** Sequences become `[[item_id, item_id, ...], ...]` with a separate `timestamps` field per sequence, or `[{item_id, ts}, ...]` per sequence
- **Polymarket (future):** Leader ingest: wallet → trade legs with `(item_id, time_ms)` per leg. Same format; enables Scout training on prediction-market data per rope bridge

### 2. DB Schema: Add Timestamp Column

- Migration: add `time_ms` (or `timestamp`) to `train_sequence_rows`, `cold_train_sequence_rows`, `test_context`, `cold_test_context`
- Sync: persist timestamps when syncing from list/JSON

### 3. JSON Format (when not using DB)

- `train_sequences.json`: add `"timestamps"` array per sequence, or `"sequence_timestamps"` at top level
- FuXi-Linear upstream uses `sequence_timestamps` in their SASRec-format CSV

### 4. Training Pipeline

- **PretrainRunner.load_train_sequences:** Return `{item_ids, timestamps}` when available
- **Training.build_train_batch:** Accept optional timestamps; build `all_timestamps` tensor `(batch, seq_len, channel_t_heads)` — replicate per head since we have one value per position
- **AxonTrain:** Pass `all_timestamps` to `FuxiLinearInference.forward_full_sequence` when available

### 5. Timestamp Normalization

- Raw `time_ms` can be large (ms since epoch). Paper uses modulus for sinusoidal precision.
- **Numerical precision:** Unix ms (~1.7e12) exceed float32 precision and can cause instability. Use per-sequence relative time.
- Options: (a) use raw values and rely on existing `remainder(ts, intervals)`; (b) normalize per-sequence to `[0, Δt₁, Δt₁+Δt₂, ...]` (cumulative ms from sequence start); (c) scale e.g. `(t - t_min) / 1000` for seconds.
- **Recommendation:** Use (b) — cumulative ms from sequence start. Keeps deltas intact (decay \( r^{t_n - t_i} \) unchanged), reduces magnitude, avoids float32 overflow. Same format for Polymarket, KuaiRand, MovieLens.
- **Do not use exchange-specific epochs** (e.g. Polymarket CLOB launch): they desync from other data and complicate cross-exchange training. Per-sequence relative time is sufficient and portable.

### 6. Serve / Eval

- When recommending from context, timestamps must be provided if the model was trained with them. Serve and eval need to pass `all_timestamps` for the context sequence.

---

## Prediction Market Rationale

Scout must be trained on leader/prediction-market data; off-domain RecGPT has zero signal for Polymarket outcomes. For prediction markets, real timestamps matter **more** than for general recommendation:

| Signal | Why timestamps matter |
|--------|------------------------|
| **Trade recency** | A trade 2 days before resolution vs 2 hours before encodes different intent; recent behavior predicts next action |
| **Market lifecycle** | When the trade occurred relative to market creation, liquidity peaks, resolution — temporal patterns distinguish leaders |
| **Trajectory shape** | [69 Sniper longitudinal leaders](69_sniper_longitudinal_leaders.md) clusters by **timing/size curves**; same items in different temporal patterns = different strategy families |
| **Trap-avoidance** | [68 Wallet longitudinal](68_wallet_longitudinal_adversary_research.md) — veto-like behavior (hesitation, size-down) is encoded in order flow over time |
| **Rope bridge** | [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — minimal Scout training on one market/cohort; if we drop timestamps, we lose the temporal edge that differentiates leaders from retail |

**Leader sequence format:** Each sequence = one leader wallet's trade legs: `[item_id, item_id, ...]` with `timestamps: [t1, t2, ...]` where `item_id` maps to `(market_id, outcome_id)` via catalog. Timestamps = when each trade was placed (Polymarket CLOB / trade event time). The converter for Polymarket must emit this format; KuaiRand and MovieLens are training surrogates until we have leader data.

---

## Implementation Order (Rope Bridge Aligned)

1. **Converter**: Emit timestamps alongside sequences (KuaiRand, MovieLens; later Polymarket leader ingest)
2. **DB migration + sync**: Add `time_ms` to sequence tables; persist when syncing
3. **Training pipeline**: `build_train_batch` + PretrainRunner → build and pass `all_timestamps`
4. **Eval/Serve**: Pass timestamps for test context (format in test_sequences)
5. **Polymarket ingest** (later): Leader wallet → trade legs with timestamps; same sequence format; enables Scout training on real prediction-market data

---

## References

- FuXi-Linear paper: arXiv:2602.23671, §3 (problem), §4.4 (Temporal Retention Channel)
- Upstream preprocess: `preprocess_kuairand27k_data.py` — `sequence_timestamps` in output
- Our `FuxiLinearInference`: `all_timestamps` opts already supported; LinearTemporalChannel consumes them
- Prediction market: Jon-Becker Polymarket subset (Phase 1); see [93 Pretraining plan](93_pretraining_plan.md)
