# FuXi vs RecGPT: Paper Performance Review

Review of FuXi-family paper performance (Table 3) compared to RecGPT. Caveats: different datasets and metrics.

## Summary

| Aspect | FuXi table | RecGPT paper |
|--------|------------|--------------|
| **Datasets** | Kuairand 27K, MovieLens-20M, KuaiRec | Amazon (Baby, Games, Office), Yelp, Washington, Steam |
| **Overlap** | None | — |
| **Metrics** | NG@10, NG@50, HR@10, HR@50, MRR | Hit@3, Hit@5, NDCG@3, NDCG@5 |
| **Setting** | In-domain training | Zero-shot (no target-domain training) |

**No direct numeric comparison is possible** — different datasets, different metrics, different evaluation regimes.

## FuXi Table (user-provided)

Best model: **FuXi-Linear**. Sample results:

| Dataset | FuXi-Linear NG@10 | FuXi-Linear HR@10 |
|---------|-------------------|-------------------|
| Kuairand 27K | 0.0609 | 0.1124 |
| MovieLens-20M | 0.2131 | 0.3592 |
| KuaiRec | 0.1830 | 0.2242 |

## RecGPT Paper (arxiv 2506.06270)

RecGPT is evaluated in **zero-shot** mode: no target-domain training. Baselines use 10% of target-domain data. Sample RecGPT results (Hit@5, NDCG@5):

| Dataset | RecGPT Hit@5 | RecGPT NDCG@5 |
|---------|--------------|---------------|
| Baby (Amazon) | 0.0283 | 0.0279 |
| Games (Amazon) | 0.0376 | 0.0371 |
| Office (Amazon) | 0.0299 | 0.0290 |
| Yelp | 0.0166 | 0.0163 |
| Washington | 0.0130 | 0.0127 |
| **Steam** | **0.1253** | **0.1245** |

Steam is RecGPT’s strongest domain; Hit@5/NDCG@5 ~0.12. FuXi-Linear on MovieLens: NG@10 0.213, HR@10 0.359 — but MovieLens vs Steam, NG@10 vs NDCG@5, in-domain vs zero-shot, so not comparable.

## Takeaways

1. **Different roles:** RecGPT stresses zero-shot and cross-domain; FuXi tables are in-domain. RecGPT wins when there is no or little target-domain data.
2. **Complementary architectures:** `RecGPT.FuxiLinearInference` uses FuXi-Linear as backbone with RecGPT semantic ID. Best of both.
3. **Metric / dataset alignment:** To compare fairly, run both on the same split with the same metrics (e.g. NG@10, HR@10, MRR on Kuairand/MovieLens/KuaiRec).
4. **RecGPT ceiling (~9%):** The ~9% in docs 72/74 refers to **prediction-market trade win rate** when trading Scout top-1 blindly, not to paper recommendation metrics. Recommendation NG@10 / HR@10 are separate.

## Recommendation

For paper benchmarks: run RecGPT and FuXi-Linear on shared datasets (Kuairand, MovieLens, KuaiRec) with shared metrics (NG@10, HR@10, MRR) before drawing strong conclusions. For our use case (copytrade, Scout + Gatekeeper): RecGPT’s recommendation quality matters for candidate generation; the Gatekeeper and survivorship rules drive the ~9% → 75% gap, not raw RecGPT metrics alone.
