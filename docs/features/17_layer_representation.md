# Layer 2: Representation

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Prev: [16 Layer Artifacts](16_layer_artifacts.md). Next: [18 Layer Fixture](18_layer_fixture.md).

---

## Problem or limitation

Layer 2 must turn item text into token IDs for the pipeline; without a single documented surface (Embedding, FSQ, FSQEncoder), responsibility and testing are unclear.

---

## Proposed improvement

Document the Representation layer: responsibility, public surface, and how to test. Rely on Embedding (Bumblebee, MPNet) and FSQ for 768-d to 4 token IDs per item.

Turn item text into token IDs. Embedding (Bumblebee, MPNet) produces 768-d vectors; FSQ quantizes to 4 token IDs per item. No model weights beyond FSQ params. **Public surface:** RecGPT.Embedding.encode_item_text_dict/1, RecGPT.FSQ.load_params/1, RecGPT.FSQ.encode/2, RecGPT.FSQEncoder.encode_embeddings_to_token_id_list/3. **How to test:** Unit tests with stub or real FSQ params; Embedding tests may require Bumblebee.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
