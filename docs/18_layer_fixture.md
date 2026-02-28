# Layer 3: Fixture

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md). Prev: [17 Layer Representation](17_layer_representation.md). Next: [19 Layer Model](19_layer_model.md).

---

## Problem or limitation

Fixture (num_items, token_id_list) must be built from items and checkpoint in a single place; without a documented surface, build and test strategy are unclear.

---

## Proposed improvement

Document Layer 3 (Fixture): responsibility, public surface, and how to test. FixtureBuild depends on Artifacts and Representation.

Build fixture.json (num_items, token_id_list) from items JSON and checkpoint (for FSQ params). Depends on Layer 1 (CheckpointLoader) and Layer 2 (Embedding, FSQEncoder). **Public surface:** RecGPT.FixtureBuild.build/2, RecGPT.FixtureBuild.write_fixture/2. **How to test:** Stub Embedding/CheckpointLoader or use real fixture files; see fixture_build_test.exs.

---

## See also

- [15 Layers overview](15_layers_overview.md) - Diagram and table.
- [16 Layer Artifacts](16_layer_artifacts.md), [17 Layer Representation](17_layer_representation.md).
- [04 RecGPT library](04_recgpt_library.md) - Module reference.
