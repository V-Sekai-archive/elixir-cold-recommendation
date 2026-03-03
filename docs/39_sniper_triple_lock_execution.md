# Sniper: Triple-Lock + Zero-Reserve Execution

Gatekeeper approval criteria and end-to-end execution flow.

Part of [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md).

---

## Triple-Lock (Conjunctive)

Gatekeeper approves only when **all three** hold:

1. **Organic Tape** — No structural signs of gaming in JSON-LD
2. **Rule Alignment** — Tape matches Polymarket rule-set
3. **Tape as Signal** — Tape (trade legs, leader activity) is our primary signal; sufficient and internally coherent. We do not use external news (too slow—market leads news; see [50 Virix review](50_virix_polymarket_strategy_review.md)).

If any fails → PICK_0.

---

## Zero-Reserve Execution Flow

1. Scout: `recommend(context_item_ids, top_k=1)` → single best item_id
2. Expand: item_id → (market_id, outcome_id) via catalog
3. Fetch Tape + XMP for the candidate
4. Gatekeeper: analyze Tape + XMP; output PICK_ID or PICK_0
5. If PICK_ID: compute butterfly profit; if profit > 0 and gamed-ness below threshold, size with Kelly and trade
6. If PICK_0: abstain (no trade)

---

## See Also

- [34 Sniper Mode Moneyball](34_sniper_mode_moneyball_strategy.md) — Overview
- [37 Gamed-ness + Metrics](37_sniper_gamedness_metrics.md) — Kelly sizing by gamed-ness
- [60 Rope bridge](60_rope_bridge_market_analytics_plan.md) — Scout→butterfly
