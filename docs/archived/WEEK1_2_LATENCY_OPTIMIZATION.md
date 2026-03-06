# Week 1-2: Latency Optimization (678ms → 50ms)

## Baseline Established ✅

**Hardware**: RTX 4090, CUDA 12.9, cuDNN 9.19
**Current P50**: 678.53 ms
**Target P50**: 50 ms
**Gap**: 628 ms (13x improvement needed)

Test command:
```bash
mix run scripts/baseline_minimal.exs
```

---

## Optimization Strategies (in order of impact)

### 1. FuXi Linear Architecture Replacement 🔥 HIGH IMPACT
**Potential**: 300-400ms → 50-100ms (3-6x improvement)

Replace transformer attention with linear layers:
- Eliminates O(n²) attention computation
- Reduces 4 forward passes in beam search
- Simpler model = faster inference

**Action Items**:
- [ ] Review `RecGPT.FuxiLinearInference` module
- [ ] Create FuXi stub checkpoint: `FrozenHelpers.write_fuxi_stub_ckpt!`
- [ ] Run baseline with FuXi: `mix run scripts/baseline_fuxi.exs`
- [ ] Compare latency vs transformer

**Reference**: `docs/features/35_fuxi_linear.md`

---

### 2. Decode Strategy: MTP (Multi-Token Prediction) 🔥 HIGH IMPACT
**Potential**: 30-40% reduction from parallel scoring

Predict K tokens at once instead of sequential beam search:
- `RECGPT_DECODE_STRATEGY=mtp`
- Single forward pass, parallel scoring
- Already implemented, need to test

**Action Items**:
- [ ] Run with MTP strategy: `mix run scripts/baseline_mtp.exs`
- [ ] Compare beam_search vs mtp latency
- [ ] Check accuracy tradeoff

**Reference**: `config/config.exs` - `decode_strategy` config

---

### 3. Beam Width Optimization 🔥 MEDIUM IMPACT
**Potential**: 20-40% reduction

Current beam width may be too wide for speed/accuracy tradeoff:
- Test beam_width: 5, 10, 20 (default?)
- Measure accuracy vs latency
- Find sweet spot for production

**Action Items**:
- [ ] Add beam_width parameter to baseline test
- [ ] Run sweep: beam_width 5, 10, 20, 50
- [ ] Plot accuracy vs latency curve

---

### 4. BF16 vs Float32 Precision 🔥 MEDIUM IMPACT
**Potential**: 10-20% reduction

Test inference dtype:
- Current: `{:bf, 16}` (config says BF16)
- Compare with `{:f, 32}`
- Tensor Cores benefit from BF16

**Action Items**:
- [ ] Run with float32: `scripts/baseline_float32.exs`
- [ ] Compare P50 latency
- [ ] Check numerical stability

---

### 5. Batching Requests 🔥 MEDIUM IMPACT
**Potential**: Better throughput, similar latency

Current: `predict_batch_size=1, predict_batch_timeout_ms=0` (no batching)
- Test batch_size: 4, timeout: 10ms
- Amortize JIT cost across requests
- Good for high-throughput scenarios

**Action Items**:
- [ ] Test with batching enabled
- [ ] Measure per-request latency vs throughput
- [ ] Find optimal batch size for RTX 4090

---

### 6. Trie/Decode Optimization 🔥 LOW-MEDIUM IMPACT
**Potential**: 5-15% reduction

Current decode path:
- `Trie.to_tensors` for device tensors
- `Decode.lookahead_top_k` for beam search
- CPU sync points

**Action Items**:
- [ ] Profile trie tensor creation time
- [ ] Check on-device vs CPU transfer overhead
- [ ] Optimize `item_id_to_tokens_tensor` lookup

---

### 7. Model Quantization 🔥 FUTURE
**Potential**: 2-4x reduction

INT8 or INT4 quantization:
- Smaller memory footprint
- Faster inference on Tensor Cores
- Requires model retraining/finetuning

**Status**: Out of scope for Week 1-2, document for future

---

## Testing Plan

### Create Baseline Scripts

1. `scripts/baseline_transformer.exs` - Current (baseline)
2. `scripts/baseline_fuxi.exs` - FuXi Linear
3. `scripts/baseline_mtp.exs` - MTP decode
4. `scripts/baseline_beam_sweep.exs` - Beam width sweep
5. `scripts/baseline_precision.exs` - BF16 vs F32

### Run Matrix

| Configuration | Expected P50 | Status |
|--------------|--------------|--------|
| Transformer + Beam Search (current) | 678 ms | ✅ Baseline |
| FuXi Linear + Beam Search | 300-400 ms | 🔄 To test |
| Transformer + MTP | 400-500 ms | 🔄 To test |
| FuXi Linear + MTP | 50-100 ms | 🔄 To test |
| FuXi + MTP + BF16 | 50-80 ms | 🔄 To test |

---

## Success Criteria

- [ ] P50 < 50 ms (target)
- [ ] P99 < 100 ms
- [ ] Accuracy drop < 5% (vs baseline)
- [ ] Throughput > 20 req/s
- [ ] Document final config in `config/prod.exs`

---

## Daily Checklist

### Day 1 (Today)
- [x] Establish baseline (678 ms) ✅
- [ ] Create FuXi baseline script
- [ ] Test FuXi Linear vs Transformer

### Day 2-3
- [ ] Test MTP decode strategy
- [ ] Beam width sweep
- [ ] Document findings

### Day 4-5
- [ ] Combine best configs (FuXi + MTP + optimal beam)
- [ ] Precision comparison (BF16 vs F32)
- [ ] Batching tests

### Day 6-7
- [ ] Final optimization pass
- [ ] Accuracy validation
- [ ] Production config

### Day 8-14 (Week 2)
- [ ] Integration testing
- [ ] Load testing
- [ ] Documentation

---

## Notes

### From `mix recgpt.trace_predict` docs:
> Aligns with Replicate COG stages: setup (one-time load + JIT compile) ~20–30s; predict (per-request inference) ~300–400ms on 12-layer + RTX 4090. Most time is usually in beam search (4 forward passes).

**Key insight**: "300-400ms" was the expected target with 12-layer model. Our 678ms suggests either:
1. Model is larger than 12-layer
2. Beam search is more expensive than expected
3. Decode strategy needs optimization

### From `FrozenHelpers`:
- Stub state has minimal params (wte, pred_head)
- Real checkpoint would have transformer layers
- FuXi stub available: `write_fuxi_stub_ckpt!`

---

## Quick Commands

```bash
# Baseline
mix run scripts/baseline_minimal.exs

# With profiling
mix recgpt.trace_predict --runs 20 --profile

# Compare strategies
mix recgpt.trace_predict --decode-strategy mtp --runs 20
mix recgpt.trace_predict --decode-strategy beam_search --runs 20
```
