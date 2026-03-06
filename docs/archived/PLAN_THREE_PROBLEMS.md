# RecGPT Development Plan

## Overview

Three core problems to solve for production-ready sequential recommendation:

1. **Latency**: Reduce inference time from 180ms → 50ms
2. **Accuracy**: Validate timestamp sequence → item matching relevance
3. **Dataset**: Source and validate the right training data (settled markets)

---

## Problem 1: Latency Optimization (180ms → 50ms)

### Current State
- **Current**: ~180ms per Predict call
- **Target**: 50ms P50
- **Hardware**: RTX 4090 (already available)
- **Previous Achievement**: Had 50ms working but reverted

### Investigation Needed
- [ ] Review git history for the 50ms implementation
- [ ] Identify what changes caused regression to 180ms
- [ ] Check `mix recgpt.trace_predict` breakdown:
  - `context_to_tokens_us`
  - `beam_search_total`
  - `inference` (get_logits forward passes)
  - `response_build_us`

### Potential Optimizations

#### A. Model Architecture
- [ ] **FuXi Linear replacement**: Replace transformer with linear layers
  - Eliminates attention overhead
  - Should dramatically reduce 4 forward passes in beam search
  - Document in `docs/features/35_fuxi_linear.md`

#### B. Decode Strategy
- [ ] **MTP (Multi-Token Prediction)**: Predict K tokens at once
  - Already configured via `RECGPT_DECODE_STRATEGY=mtp`
  - Reduces beam search iterations
  - Test with `mix recgpt.trace_predict --decode-strategy mtp`

#### C. Inference Backend
- [ ] **EXLA CUDA backend**: Ensure `:cuda` client is active
  - Config: `config :exla, :default_client, :cuda`
  - Verify CUDA kernels are compiling and caching
  - First-request JIT should be ~20-30s, subsequent ~50ms

#### D. Beam Search Optimization
- [ ] **Reduce beam width**: Default may be too wide
  - Test beam_width=5, 10, 20
  - Measure accuracy vs latency tradeoff
- [ ] **Early stopping**: Stop beam search when confidence high
- [ ] **Trie tensor optimization**: Ensure on-device, single CPU sync

#### E. Batching
- [ ] **Request batching**: Already configured but verify
  - `predict_batch_size` (default 1)
  - `predict_batch_timeout_ms` (default 0)
  - Test with batch_size=4, timeout=10ms

### Validation
```bash
# Baseline measurement
mix recgpt.trace_predict --runs 20 --context "0,1"

# Compare strategies
mix recgpt.trace_predict --decode-strategy mtp --runs 20
mix recgpt.trace_predict --decode-strategy beam_search --runs 20

# Profile function-level timing
mix profile.fprof scripts/profile_predict.exs
```

### Success Criteria
- [ ] P50 < 50ms (after warmup)
- [ ] P99 < 100ms
- [ ] Throughput > 20 req/s on RTX 4090

---

## Problem 2: Accuracy (Timestamp Sequence → Item Matching)

### Current State
- **Unproven**: No validation that timestamp sequences correlate with item relevance
- **Risk**: Garbage in, garbage out
- **Challenge**: No ground truth labels available

### Investigation Plan

#### A. Synthetic Validation
- [ ] Create synthetic sequences with known patterns
- [ ] Train on synthetic data
- [ ] Verify model learns patterns
- [ ] Establish baseline accuracy metrics

#### B. Sequence Quality Metrics
- [ ] **Temporal coherence**: Do nearby timestamps have related items?
- [ ] **User consistency**: Does same user show preference patterns?
- [ ] **Market efficiency**: Do settled prices reflect information flow?

#### C. Alternative Representations
- [ ] **Time delta encoding**: Relative vs absolute timestamps
- [ ] **Event-based**: Only include significant price movements
- [ ] **Aggregated**: Bucket timestamps (e.g., per-minute, per-hour)

#### D. Evaluation Metrics
Since ground truth unavailable, use proxy metrics:
- [ ] **Reconstruction loss**: Can model predict next item better than random?
- [ ] **Temporal consistency**: Similar sequences → similar recommendations
- [ ] **Settlement correlation**: Do recommendations align with eventual settlements?

### Validation Approach
```elixir
# Load historical sequences
sequences = RecGPT.Figgie.DataFetcher.load_market_history()

# Split by time (not random!)
{train, valid, test} = RecGPT.Figgie.TimeSplit.split(sequences, ratio: [0.7, 0.2, 0.1])

# Train on train, validate on valid
# Measure: Hit@10, MRR, NDCG on valid set
# If metrics poor → sequence representation problem
```

### Success Criteria
- [ ] Hit@10 > 0.3 on validation set (better than random)
- [ ] MRR > 0.2
- [ ] Temporal consistency score > 0.7

---

## Problem 3: Dataset Selection

### Current State
- **Option A**: Standard datasets (MovieLens, Steam) - proven but wrong domain
- **Option B**: Real market data - right domain but unproven answers
- **Scale**: Potentially 15M items per market

### Dataset Options

#### A. Standard Benchmarks (Quick Win)
**Pros**: Known ground truth, comparable to literature, fast iteration
**Cons**: Wrong domain, may not transfer to markets

- [ ] **MovieLens**: 25M ratings, 62K items
- [ ] **Steam**: Game playtime, 50K+ items
- [ ] **Amazon Reviews**: Multiple categories

**Action**: 
```bash
mix recgpt.fetch_steam data/steam
mix recgpt.build_fixture --data-dir data/steam
mix recgpt.pretrain --data-dir data/steam
mix recgpt.eval --data-dir data/steam
```

#### B. Real Market Data (Production)
**Pros**: Right domain, settled markets have ground truth
**Cons**: Noisy, complex, requires validation

- [ ] **Settled markets only**: Clean signal
- [ ] **Order book depth**: Additional context
- [ ] **Trade history**: Actual transactions

**Data Requirements**:
```elixir
%{
  market_id: "...",
  items: [%{id: 1, outcome: "YES", settled_price: 0.0 or 1.0}],
  sequences: [
    %{
      user_id: "...",
      timestamp: 1234567890,
      item_id: 1,
      action: :buy | :sell,
      price: 0.65,
      quantity: 10
    }
  ]
}
```

#### C. Hybrid Approach (Recommended)
1. **Phase 1**: Pretrain on standard datasets (prove architecture)
2. **Phase 2**: Fine-tune on settled market data (domain adaptation)
3. **Phase 3**: Continual learning on live markets (production)

### Catalogue Size Considerations

**15M items per market** requires:
- [ ] **Efficient trie**: Sparse representation, pruning
- [ ] **Memory mapping**: Don't load all items at once
- [ ] **Hierarchical**: Market → Category → Item
- [ ] **Caching**: Hot items in memory, cold on disk

**Memory Estimate**:
```
15M items × 4 tokens/item × 4 bytes = 240MB (raw)
Trie overhead: ~2-3x = 720MB
Embeddings: 15M × 768 × 4 bytes = 46GB ❌

Solution:
- On-demand embedding loading
- Quantized embeddings (int8): 15M × 768 × 1 byte = 11.5GB ✓
- Or: Don't embed all items, embed only active sequences
```

### Validation Strategy

Since "answers are not available" for live markets:

#### A. Settled Markets (Proxy Ground Truth)
- [ ] Only markets with resolved outcomes
- [ ] Compare recommendations to eventual settlement
- [ ] Metric: Did model recommend winning outcomes more often?

#### B. Order Book Reconstruction
- [ ] Hide last N trades in sequence
- [ ] Predict next trade
- [ ] Compare to actual order book movement

#### C. Expert Review
- [ ] Sample recommendations manually
- [ ] Domain expert rates relevance
- [ ] Qualitative feedback loop

### Success Criteria
- [ ] Standard datasets: Match published RecGPT metrics
- [ ] Settled markets: >60% accuracy on winning outcomes
- [ ] Catalogue: Support 1M+ items with <100ms latency

---

## Integrated Timeline

### Week 1-2: Latency (Problem 1)
- [ ] Restore 50ms implementation
- [ ] Profile and identify bottlenecks
- [ ] Implement FuXi Linear if needed
- [ ] Document in `docs/features/35_fuxi_linear.md`

### Week 3-4: Dataset Pipeline (Problem 3)
- [ ] Build data fetcher for settled markets
- [ ] Create fixture builder for large catalogues
- [ ] Validate on MovieLens/Steam first
- [ ] Document in `docs/features/36_market_dataset.md`

### Week 5-6: Accuracy Validation (Problem 2)
- [ ] Run eval on standard datasets
- [ ] Establish baseline metrics
- [ ] Test sequence representations
- [ ] Document in `docs/features/37_accuracy_validation.md`

### Week 7-8: Integration
- [ ] End-to-end pipeline: fetch → build → pretrain → eval
- [ ] Performance optimization
- [ ] Production readiness checklist

---

## Immediate Next Steps

1. **Today**: 
   - [ ] Review git history for 50ms implementation
   - [ ] Run `mix recgpt.trace_predict` for baseline
   - [ ] Document current architecture

2. **This Week**:
   - [ ] Fix latency regression (restore 50ms)
   - [ ] Set up Steam dataset for validation
   - [ ] Create evaluation harness

3. **Questions to Answer**:
   - What changed between 50ms and 180ms?
   - Is beam search necessary or can we use direct scoring?
   - What's the minimum viable catalogue size for testing?
   - Which settled markets have cleanest data?

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Can't restore 50ms | High | Profile thoroughly, consider FuXi Linear |
| Timestamp sequences irrelevant | High | Test synthetic data first, validate early |
| 15M items too slow | Medium | Hierarchical trie, quantization, caching |
| No ground truth for accuracy | High | Use settled markets, proxy metrics |
| EXLA/CUDA issues | Medium | Fallback to BinaryBackend for testing |

---

## References

- [RecGPT Paper](https://arxiv.org/abs/2506.06270)
- [01 gRPC API](docs/features/01_grpc_api.md)
- [15 Layers overview](docs/features/15_layers_overview.md)
- [20 Layer Recommendation](docs/features/20_layer_recommendation.md)
- [42 Latency and Performance](docs/features/42_latency_and_performance.md)

---

## Notes from Discussion

- **Pretraining analogy**: Market history pretraining ≈ RecGPT pretraining on user sequences
- **Settled markets only**: Filter for clean signal, ignore open markets
- **Scale**: 15M items per market is achievable with proper data structures
- **Ground truth limitation**: Only settled trades + order book snapshots available
- **Focus strategy**: May need to focus on handful of contracts initially
