# Sniper: Unified Schema — JSON-LD + XMP-JSON-LD

Trade legs (The Tape) and meta-guardrails. Same structure for RecGPT pretraining (historical resolved) and Qwen LoRA finetuning (hypothetical trajectories).

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## JSON-LD (Semantic Context)

Trade legs are the primary news signal.

```json
{
  "@context": "https://schema.org/",
  "@type": "FinancialProduct",
  "@id": "pm:outcome:8892",
  "name": "Market Outcome Identifier",
  "description": "The static rule-set text from Polymarket.",
  "subjectOf": {
    "@type": "Dataset",
    "name": "TheTape",
    "description": "Real-time trade legs used as the primary news signal.",
    "hasPart": [
      {
        "@type": "TradeAction",
        "price": "0.65",
        "volume": "1500",
        "agent": "trader:001",
        "timestamp": "2026-03-03T09:00:01Z"
      }
    ]
  },
  "potentialAction": {
    "@type": "TradeAction",
    "actionStatus": "ProposedActionStatus",
    "target": "PICK_ID_8892_OR_PICK_0"
  }
}
```

---

## XMP / XMP-JSON-LD (Meta-Guardrails)

XMP is also **xmpjsonld**. Embed in training file headers to prevent the Gatekeeper from "cheating" (e.g. seeing future price data during Qwen LoRA finetuning).

```json
{
  "@context": "https://schema.org/",
  "@type": "xmp:MarketState",
  "xmp:FinalStatus": "Resolved_Win",
  "xmp:LiquidityScore": 0.88,
  "xmp:IsGamed": false,
  "xmp:ExecutionWindow": "10_MIN"
}
```

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [35 Architecture](35_sniper_architecture.md) — Scout + Gatekeeper
- [38 Qwen LoRA](38_sniper_qwen_lora.md) — GRPO scenarios
