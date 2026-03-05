# RL vs Inference vs Pre-training Scaling: Constraining Diff

**Source:** [How Well Does RL Scale?](https://www.tobyord.com/writing/how-well-does-rl-scale) — Toby Ord, October 2025.

Summary of scaling-law constraints for RL-trained reasoning LLMs. Key claim: RL-scaling requires **twice as many orders of magnitude** as inference-scaling for the same capability gain.

---

## Problem or limitation

Reasoning models (o1, o3, GPT-5, Sonnet 3.7, Grok 4) improve via two distinct scaling axes: **(1) RL training compute** and **(2) inference compute** (chain-of-thought length). Pre-training scaling has largely stalled; RL unlocked new gains but with very different cost dynamics. Understanding the **constraining diff** between these regimes matters for deployment costs, AI governance, and whether more compute can still buy more intelligence.

---

## Proposed improvement

Use a **constraining-diff view** of the three scaling regimes. Below, each row is a dimension; the diff shows what is required to achieve a comparable capability boost.

### Constraining diff: compute required for equivalent capability gain

| Target gain                      | Inference-scaling                      | RL-scaling         | Pre-training (reference) |
| -------------------------------- | -------------------------------------- | ------------------ | ------------------------ |
| 20% → 80% on AIME                | **100×** inference                     | **10,000×** RL     | —                        |
| o1 → o3 level boost              | **3×** tokens                          | **10×** RL         | —                        |
| o3 → GPT-5 level boost           | **~3×** tokens                         | **~10×** RL        | —                        |
| One GPT-level jump (GPT-1→2→3→4) | **~1,000×** inference (Jones, EpochAI) | **~1,000,000×** RL | **~100×** pre-training   |

**Rule of thumb:** For the same benefit, RL-scaling needs **~2× more orders of magnitude** than inference-scaling.

---

### Constraining diff: cost structure

| Regime            | Cost type             | Constraint                                                                                  |
| ----------------- | --------------------- | ------------------------------------------------------------------------------------------- |
| Pre-training      | One-off training      | Scaling stalled; ~100× per GPT jump                                                         |
| RL-scaling        | One-off training      | Cheap only when RL ≪ pre-training; once RL ≈ pre-training, 10× RL ≈ 10× total training cost |
| Inference-scaling | **Ongoing per-query** | 30× longer CoT ⇒ 30× deployment cost per use; cannot amortize                               |

**Inflection:** When RL compute reaches pre-training compute (e.g. Grok 4), further RL-scaling becomes effectively infeasible — next 1,000,000× RL would require ~1,000,000× total training cost. Ord estimates one such RL-scaling step would need ~5 datacenters of Colossus scale and ~5 years of world electricity.

---

### Constraining diff: what each regime optimizes

| Regime            | Mechanism                                      | Limits                                                                                    |
| ----------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Pre-training      | Next-token prediction (imitation)              | Stalled; no burst past human-level on novel methods                                       |
| RL-scaling        | Learn better reasoning from verifiable rewards | Poor scaling; RL receives <1/10,000 as much info per FLOP vs pre-training                 |
| Inference-scaling | More tokens / longer chain of thought          | Improves by _more time_, not more intelligence; scales deployment cost, not training cost |

---

### Constraining diff: slope equivalence (log scale)

On log–log AIME charts (o1, o3):

- **Inference-scaling slope:** ~2× steeper than RL-scaling slope.
- **RL-scaling slope:** ~half of inference slope ⇒ requires **2× orders of magnitude** for same Δ on y-axis.
- **Empirical fit:** 4 orders of magnitude RL (o1→o3) ≈ 26%→88% AIME; matches ~10,000× RL for 20%→80%.

---

## Implication for RecGPT / recommendation models

RecGPT-style models use **supervised / contrastive / next-token-style training**, not RL-from-rewards. The constraining diff above applies to **RL-trained reasoning LLMs**. For RecGPT:

- Scaling is dominated by **pre-training** (and possibly supervised / contrastive fine-tuning), not RL.
- **Inference-scaling** (longer sequences, beam search) increases per-query cost; our latency ceiling docs ([42](42_latency_and_performance.md), [61](61_strategy_given_latency_ceiling.md)) address this.
- The Ord analysis does **not** directly constrain RecGPT, but it informs why frontier labs are hitting RL limits and shifting emphasis toward inference-scaling and deployment economics.

---

## See also

- [42 Latency and performance](42_latency_and_performance.md)
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md)
- [71 Polymarket dataset scale](71_polymarket_dataset_scale.md) — Relates dataset size (~40 GiB, ~100k resolved markets) to RL scaling: fixed reward budget ⇒ data-limited before compute-limited.
- Source: [How Well Does RL Scale?](https://www.tobyord.com/writing/how-well-does-rl-scale) — Toby Ord
- Related: [Evidence that Recent AI Gains are Mostly from Inference-Scaling](https://www.tobyord.com/writing/mostly-inference-scaling)
