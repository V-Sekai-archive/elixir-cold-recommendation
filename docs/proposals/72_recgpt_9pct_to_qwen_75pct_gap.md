# RecGPT 9% Ceiling → Qwen 75% Wins: Bridging the Gap

RecGPT Scout alone caps at ~9% effective performance; Qwen Gatekeeper filters to a subsample where we target ~75% win rate on trades we make. This doc relates the gap and how the LLM bridges it.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Problem or limitation

**RecGPT ceiling ~9%:** RecGPT is a sequential recommender trained on next-token prediction. It learns item correlations, sequences, and catalog intent—not trap detection, gamed-ness, or veto logic. When we trade Scout's top-1 **blindly** (no Gatekeeper):

- We hit **traps** (IsGamed=True) frequently → large negative expectancy
- We trade marginal or wrong outcomes when not gamed
- Net effective performance (Veto-Adjusted Expectancy, or win rate on all Scout outputs) stays low—on the order of **~9%** or similar, depending on trap density and market mix

RecGPT cannot cross this ceiling by pretraining alone: it has no notion of "don't trade this." More data or compute on RecGPT improves recommendation quality within its paradigm but does not add veto/gate logic.

**Target ~75%:** To reach the profitable tier (POLYMARKET_PROFITABLE_PCT ~12.7%), we need a **high win rate on the trades we actually make**. That implies ~55–75%+ on the filtered subsample, depending on payoff structure, fees, and Kelly sizing. We set **75%** as the target win rate for trades that pass the Gatekeeper.

---

## Proposed improvement: Qwen bridges the gap

The gap is not closed by improving RecGPT's recommendation accuracy. It is closed by **filtering**: only trade when the Gatekeeper says PICK_ID. Qwen learns PICK_ID vs PICK_0 from GRPO on veto/strike scenarios.

### Constraining diff: Scout vs Gatekeeper

| Dimension               | RecGPT Scout                       | Qwen Gatekeeper                           |
| ----------------------- | ---------------------------------- | ----------------------------------------- |
| **Role**                | Narrow candidate set (top-1)       | Filter: trade or don't                    |
| **Training**            | Pretrain (next-token on sequences) | LoRA finetune (GRPO on PICK_ID/PICK_0)    |
| **Output**              | item_id                            | PICK_ID or PICK_0                         |
| **Knows traps?**        | No                                 | Yes (from XMP IsGamed)                    |
| **Performance ceiling** | ~9% on all outputs                 | N/A (filter, not predictor)               |
| **Effect**              | Candidate generation               | Subsample selection → 75% win rate target |

### How Qwen crosses the gap

1. **Trap veto (PICK_0 when IsGamed):** Avoids catastrophic -5. Trap Escape Rate = % of gamed markets where we abstain. High escape rate → we never trade traps.
2. **Organic strike (PICK_ID when not gamed and Scout right):** Only approve when safe. Organic Strike Rate = % of non-gamed where we pick and win. This is the subsample we trade.
3. **Ambiguous abstain:** When uncertain, PICK_0 gives +0.1 vs -2 for wrong pick. Prefer abstain over guessing.

**Result:** We do **not** trade every Scout output. We trade only when Gatekeeper says PICK_ID. The **win rate of the subsample** (trades we make) can reach ~75% if:

- Trap Escape Rate is high (we skip gamed markets)
- Organic Strike Rate is high (we pick and win when it's safe)
- We abstain on ambiguous cases

RecGPT's ~9% applies to _all_ Scout outputs. Qwen's job is to **reject most of them** and approve only the high-edge subset. The approved subset can have 75%+ win rate.

---

## Constraining diff: 9% → 75%

| Stage                  | What we measure                   | Expected value                      |
| ---------------------- | --------------------------------- | ----------------------------------- |
| **Scout output (all)** | Win rate if we traded every top-1 | ~9%                                 |
| **Gatekeeper filter**  | PICK_ID vs PICK_0                 | Rejects ~85–90% (traps + ambiguous) |
| **Trades we make**     | Win rate on PICK_ID subsample     | **~75%** target                     |

The math: if Scout outputs N candidates, and 90% are rejected (traps + abstain), we trade 0.1N. If that 0.1N has 75% win rate, our _overall_ hit rate on Scout output is 0.1 × 0.75 = 7.5%—but the **trades we make** are profitable because we only trade high-edge situations.

---

## Relation to RL scaling and dataset

[70 RL scaling](70_rl_scaling_constraining_diff.md) and [71 Polymarket dataset scale](71_polymarket_dataset_scale.md): RL (GRPO) scales poorly and we are reward-data-limited (~100k resolved markets). We cannot RL our way from 9% to 75% by scaling GRPO compute—the dataset caps reward signal.

**Qwen's role is different:** It learns a **binary filter** (PICK_ID vs PICK_0), not a generative recommender. The filter is trained on scenarios with Resolved_Win and optional IsGamed (or profit-based reward if IsGamed is unobservable). That needs fewer diverse outcomes than a full recommender. The Gatekeeper's job is to **separate** high-edge from traps/ambiguous; with ~100k resolved markets we have enough signal.

---

## See also

- [73 Gatekeeper data scale and CLOB](73_gatekeeper_data_scale_and_clob.md) — Scaling performance vs data; enough? live CLOB needs
- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md)
- [35 Architecture](35_sniper_architecture.md) — Scout + Gatekeeper
- [38 Qwen LoRA](38_sniper_qwen_lora.md) — GRPO reward
- [37 Gamed-ness + Metrics](37_sniper_gamedness_metrics.md) — Trap Escape, Organic Strike
- [70 RL scaling](70_rl_scaling_constraining_diff.md) — Why we can't scale GRPO to close the gap
- [71 Polymarket dataset scale](71_polymarket_dataset_scale.md) — Reward budget
