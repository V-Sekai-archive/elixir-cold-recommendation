# Sniper: Pipeline + Continuum

Catalog → RecGPT → Cluster → Qwen RL. Head → Mid-Tail → Long Tail.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Pipeline: N → N+3

Extended flow for pretraining and discovery. Goal: find opportunities at the **beginning of the long tail** (early, less crowded).

| Step | Action |
|------|--------|
| **N** | Find and catalog — markets, outcomes, wallets, Tape |
| **N+1** | **RecGPT pretraining / inference** → distribution of probabilities (scores over candidates) |
| **N+2** | Cluster data that passes initial guardrails. Cluster by: |
| | • Specific market |
| | • Winning wallets |
| | • Historical momentary spreads and book depth that greenlit winning wallets to arb |
| **N+3** | **Qwen LoRA finetuning** (GRPO) → additional insights and guardrails for Gatekeeper |

---

## Continuum: Head → Mid-Tail → Long Tail

| Zone | Characteristics | Strategy |
|------|------------------|----------|
| **Fat Head** | High volume, crowded, sub-100ms competition | RecGPT bypassed (Binary/Bundle) |
| **Mid-Tail** | Catalyst/Combinatorial; 2–10 s edge | RecGPT + Gatekeeper |
| **Long Tail (Thin Tail)** | Niche, early; edge at emergence before crowding | Target for N→N+3 pipeline |

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [35 Architecture](35_sniper_architecture.md) — RecGPT vs Qwen training
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md) — Catalyst/Combinatorial fits
