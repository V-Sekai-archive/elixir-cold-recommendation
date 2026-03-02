# Layer 5: Recommendation

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Prev: [19 Layer Model](19_layer_model.md). Next: [21 Layer Application](21_layer_application.md).

---

## Problem or limitation

Next-item recommendation must be catalog-constrained (trie, beam search) and loadable from fixture and checkpoint; without a documented surface, Serve and Decode roles are unclear.

---

## Proposed improvement

Document Layer 5 (Recommendation): responsibility, public surface, and how to test. Trie, Decode, and Serve; load_state + recommend.

Trie from token_id_list; Decode runs beam search using a logits function (from Inference); Serve loads fixture + checkpoint, builds trie and get_logits, and implements RecGPT.RecommendationService. **Public surface:** RecGPT.RecommendationService (behaviour; default impl Serve), RecGPT.Trie.build/1, RecGPT.Decode.beam_search_top_k/4, RecGPT.Serve.load_state/3, RecGPT.Serve.recommend/3. **How to test:** trie_test.exs, decode_test.exs, serve_test.exs. Stub get_logits for Trie/Decode; Serve tests can use stub state or full stack.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [19 Layer Model](19_layer_model.md) - Inference.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
