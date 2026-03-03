# Sniper Mode: Moneyball Strategy

RecGPT Scout + Qwen3 Gatekeeper for a 10-minute execution window. Unified JSON-LD / XMP-JSON-LD schema, GRPO training, veto-adjusted metrics, and butterfly arbitrage under continuous gamed-ness. Path to `POLYMARKET_PROFITABLE_PCT` (the share of Polymarket users who are profitable).

Related: [60 Rope bridge](60_rope_bridge_market_analytics_plan.md), [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md).

---

## Parts

| Doc | Content |
|-----|---------|
| [35 Architecture](35_sniper_architecture.md) | Scout + Gatekeeper; RecGPT pretraining vs Qwen LoRA finetuning |
| [36 Schema](36_sniper_schema.md) | JSON-LD (Tape) + XMP-JSON-LD (guardrails) |
| [37 Gamed-ness + Metrics](37_sniper_gamedness_metrics.md) | Butterfly under gamed-ness; Moneyball metrics |
| [38 Qwen LoRA](38_sniper_qwen_lora.md) | GRPO reward, ART rollout, training loop |
| [39 Triple-Lock + Execution](39_sniper_triple_lock_execution.md) | Gatekeeper criteria; Zero-Reserve flow |
| [40 Pipeline + Continuum](40_sniper_pipeline_continuum.md) | N→N+3; Head / Mid-Tail / Long Tail |
| [41 Path to Profitable](41_sniper_path_profitable.md) | POLYMARKET_PROFITABLE_PCT dimensions |

---

## Summary

| Item | Decision |
|------|----------|
| Scout | RecGPT Top-1 only |
| Gatekeeper | Qwen3; PICK_ID or PICK_0 |
| Schema | JSON-LD (Tape) + XMP/XMP-JSON-LD (guardrails) |
| Gamed-ness | Spectrum; butterfly always calculable; size down as risk rises |
| Metrics | Veto-Adjusted Expectancy, Trap Escape Rate |
| Reward | Asymmetric (-5 trap hit, +2 trap veto) |
| RecGPT | Pretraining (next-token on sequences); `mix recgpt.pretrain` |
| Qwen Gatekeeper | LoRA finetuning (GRPO); OpenPipe ART |
| Pipeline | N: catalog → N+1: RecGPT pretrain/inference → N+2: cluster → N+3: Qwen LoRA finetune |
| Continuum | Fat Head (bypass) → Mid-Tail (Scout+Gatekeeper) → Long Tail (early edge) |
| Path | POLYMARKET_PROFITABLE_PCT via edge filter + survivorship + catalyst focus + discipline |

---

## See Also

- [OpenPipe ART](https://github.com/OpenPipe/ART) — Qwen LoRA finetuning via GRPO
- [33 MovieLens 5-epoch pretrain](33_movielens_5_epoch_pretrain.md) — RecGPT pretraining (`mix recgpt.pretrain`)
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — Profit calc, Scout→butterfly, Kelly
- [61 Strategy given latency ceiling](61_strategy_given_latency_ceiling.md) — RecGPT fits Catalyst/Combinatorial
- [67 Thirdparty bs-p](67_thirdparty_bs_p_review.md) — Kelly, Greeks, shock from bs-p
