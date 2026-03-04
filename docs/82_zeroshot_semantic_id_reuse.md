# Zero-Shot RecGPT and Semantic ID Reuse

What else can do zero-shot recommendation so we reuse the semantic id pipeline?

Related: [11 RecGPT paradigm](11_recgpt_paradigm.md), [62 Ablation tensor graph](62_ablation_tensor_graph.md), [81 Scout vs Gatekeeper merge](81_scout_gatekeeper_merge_feasibility.md).

---

## What We Have

**Semantic id** = 4-token FSQ path per item. Pipeline:

1. **Embedding** — `RecGPT.Embedding` (Bumblebee, sentence-transformers/all-mpnet-base-v2): item text → 768-d vector
2. **FSQ** — `RecGPT.FSQEncoder`: 768-d → 4 discrete tokens (vocab ~15,360)
3. **Fixture** — `token_id_list`: list of [t0, t1, t2, t3] per item_id
4. **Trie** — Maps 4-token path → item_id for constrained decode
5. **Decode** — RecGPT autoregressive over tokens; trie restricts next-token to valid paths

Zero-shot = pretrained checkpoint + fixture (no training on this catalog). New items: embed text → FSQ → add to fixture; model can recommend them without retrain.

---

## What Else Can Produce the Same Format?

Any system that outputs **4-token sequences** in the FSQ vocab (or compatible indices) can plug into our decode path. Reusable pieces:

| Component       | What it does                          | Reuse condition                                                     |
| --------------- | -------------------------------------- | ------------------------------------------------------------------- |
| **Fixture build**| Embed + FSQ → token_id_list            | Same Embedding + FSQ; output format unchanged                       |
| **Trie**        | 4-token path → item_id                 | Any model that emits 4-token paths uses same trie                    |
| **Decode**      | Beam search, trie restriction, item_at_leaf | Works with any model that outputs logits over FSQ vocab           |
| **Serve**       | Load fixture, trie, checkpoint; Predict| Swap checkpoint; keep fixture/trie if catalog unchanged               |

---

## Candidates for Zero-Shot Recommendation (Semantic ID Compatible)

| Approach                     | Output format          | Reuses our work?                         | Notes                                                                 |
| ---------------------------- | ---------------------- | ---------------------------------------- | --------------------------------------------------------------------- |
| **RecGPT (current)**         | 4-token path → item_id | Full pipeline                            | Pretrained + fixture; zero-shot or trained                            |
| **Different transformer**    | Logits over FSQ vocab  | Trie, decode, fixture, serve             | Same input (token_ids); same output (4 tokens per item); swap model  |
| **LLM fine-tuned to FSQ**    | Token generation       | FSQ encoder, trie, fixture               | LLM outputs 4-token sequence; we validate via trie                    |
| **Nearest-neighbor + FSQ**   | 4-token path           | Full FSQ, fixture, trie                  | Embed context; k-NN in embedding space; FSQ-encode neighbors; return  |
| **Rule-based / retrieval**   | item_id or 4-token     | Trie lookup, catalog                     | If it produces valid 4-token paths, trie resolves to item_id          |

**Key invariant:** The **semantic id** (4-token path) is the contract. Whatever produces it—RecGPT, another model, or retrieval—can use:

- `RecGPT.FixtureBuild` (Embedding + FSQ)
- `RecGPT.Trie` (path → item_id)
- `RecGPT.Serve` (if we generalize to accept non-RecGPT backends)

---

## Generalized Zero-Shot Backend

To support alternative recommenders:

1. **Fixture + trie** — Unchanged. Built from items + Embedding + FSQ.
2. **Recommendation interface** — Define `recommend(context_item_ids, top_k) -> [item_id]`. RecGPT implements it via forward + decode. Another backend (e.g. k-NN, different transformer) could implement the same interface.
3. **Output** — All backends return item_ids. Downstream (catalog, profit calc, Gatekeeper) unchanged.

If a backend produces **4-token sequences** directly (e.g. LLM that was fine-tuned to emit FSQ tokens), we run them through the trie to get item_ids. No change to fixture or trie.

---

## What We'd Need to Add

| Goal                         | Change                                                                 |
| ---------------------------- | ---------------------------------------------------------------------- |
| **Swap RecGPT for another transformer** | Load different checkpoint; same input/output shapes; reuse decode  |
| **k-NN zero-shot**           | `recommend/2` that: embed context items → aggregate → k-NN in embedding space → FSQ-encode neighbors → trie lookup → item_ids |
| **LLM as recommender**       | Fine-tune LLM to output 4-token sequences; wrap in `recommend/2`; trie validates |

The semantic id work (Embedding, FSQ, fixture, trie) is **backend-agnostic**. Any recommender that can produce valid 4-token paths reuses it.
