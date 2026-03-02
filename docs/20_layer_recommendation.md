# Layer 5: Recommendation

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Prev: [19 Layer Model](19_layer_model.md). Next: [21 Layer Application](21_layer_application.md).

---

## Problem or limitation

Next-item recommendation must be catalog-constrained (trie, beam search) and loadable from fixture and checkpoint; without a documented surface, Serve and Decode roles are unclear.

---

## Proposed improvement

Document Layer 5 (Recommendation): responsibility, public surface, and how to test. Trie, Decode, and Serve; load_state + recommend.

Trie from token_id_list; Decode runs SPMD-style beam search (trie tensors on device, single CPU sync); Serve loads fixture + checkpoint, builds trie tensors and get_logits_batch_tensor_fn, and implements RecGPT.RecommendationService. **Public surface:** RecGPT.RecommendationService (behaviour; default impl Serve), RecGPT.Trie.build/1, RecGPT.Trie.to_tensors/2, RecGPT.Decode.beam_search_top_k_spmd/7, RecGPT.Serve.load_state/3, RecGPT.Serve.recommend/3. **How to test:** trie_test.exs, decode_spmd_test.exs, serve_test.exs. See [32 SPMD decode flow](32_spmd_decode_flow.md) for the full decode pipeline.

---

## See also

- [32 SPMD decode flow](32_spmd_decode_flow.md) — Trie tensors, device-side beam search, single sync.
- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [19 Layer Model](19_layer_model.md) - Inference.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
