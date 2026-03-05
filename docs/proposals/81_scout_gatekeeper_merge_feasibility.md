# Scout vs Gatekeeper: Merge Feasibility

Analysis: can we merge Scout (RecGPT) and Gatekeeper (Qwen) into one model?

Related: [35 Sniper Architecture](35_sniper_architecture.md), [72 RecGPT 9% → Qwen 75% gap](72_recgpt_9pct_to_qwen_75pct_gap.md), [82 Zero-shot semantic id reuse](82_zeroshot_semantic_id_reuse.md).

---

## Current Split

| Model              | Role                          | Inputs                        | Output      | Training                    |
| ------------------ | ----------------------------- | ----------------------------- | ----------- | --------------------------- |
| **RecGPT (Scout)** | Top-1 candidate from context  | Item sequence (token_ids)     | item_id     | Pretrain on sequences       |
| **Qwen3 (Gatekeeper)** | Veto or approve            | Tape (orderbook) + XMP + Scout pick | PICK_ID / PICK_0 | LoRA GRPO on veto scenarios |

Scout feeds the Gatekeeper exactly one candidate. Gatekeeper decides whether to trade.

---

## Why Merge Is Hard

| Dimension            | Scout (RecGPT)                    | Gatekeeper (Qwen)                      | Merge blocker                                                                 |
| -------------------- | --------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| **Architecture**     | Small transformer (~3.5k params)   | LLM (7B params)                        | Different model families; different runtime/stacks (Elixir/Nx vs Python)      |
| **Input**            | Item sequence only (token_ids)    | Tape (JSON-LD) + XMP + Scout outcome  | Gatekeeper needs orderbook, gamed-ness, prices; Scout never sees these        |
| **Output**           | item_id (discrete, from trie)     | PICK_ID / PICK_0 (binary/token)       | Different vocab and decode paths                                              |
| **Training**         | Next-token prediction on sequences| GRPO reward on veto/strike             | Different loss; different data (sequences vs scenarios)                       |
| **Latency**          | ~200–280 ms warm                  | Low                                   | Scout dominates; Gatekeeper is cheap once Scout returns                       |

**Core constraint:** Gatekeeper's decision depends on **Tape + XMP** (orderbook, liquidity, gamed-ness). Scout has no access to that. A merged model would need both item-sequence and Tape/XMP as input. That implies:

1. Extending RecGPT to accept multimodal (sequence + structured Tape) — major architecture change
2. Or replacing RecGPT with an LLM that does both recommend and veto — we lose RecGPT's efficient item-id output and FSQ/trie pipeline

---

## Merge Paths (If We Insist)

**Path A: LLM does both** — Use Qwen (or similar) for end-to-end: input = (context items as text, Tape, XMP), output = (PICK_ID + item_id) or PICK_0. Problems: LLM must output valid item_ids; no FSQ/trie constrains output; likely slower than Scout alone. We'd need to teach the LLM the catalog (item titles, outcome_ids) and enforce valid picks.

**Path B: RecGPT + veto head** — Add a binary head to RecGPT: output (item_id, veto_score). RecGPT would need Tape/XMP as auxiliary input. We'd have to add an encoder for Tape (e.g. embed Tape JSON or key fields) and fuse with RecGPT hidden. Training: joint next-token + veto reward (multi-task). Complex; RecGPT was never designed for multimodal.

**Path C: Keep separate** — Scout and Gatekeeper stay distinct. Minimal change; proven pipeline. Gatekeeper is lightweight (one LLM call per Scout candidate); Scout dominates latency.

**Recommendation:** Keep separate unless we have strong evidence that a unified model improves veto-adjusted expectancy and latency. The merge adds complexity without clear payoff.
