# Freeze inputs from full weights to isolate layers

Sub-proposal of the [documentation index](README.md). Overview: [15 Layers overview](15_layers_overview.md).

---

## Problem or limitation

Layers need a way to be tested or run in isolation without process boundaries (GenServers) or ad-hoc stubbing. Without a defined approach, tests duplicate stub state and the testing strategy is unclear.

---

## Proposed improvement

Instead of process boundaries (GenServers) or stubbing, **isolate layers by freezing the inputs** that each layer receives when the full pipeline runs with **full weights**. Run once with a real checkpoint and fixture; capture the inputs (and optionally outputs) at each layer boundary; then test or run each layer in isolation by feeding it only those **frozen inputs**. This works well for **unit tests** (one layer under test, frozen inputs from below) and **property-based tests** (e.g. generate many context item_ids or token lists, run Model or Recommendation with the same frozen params/state). No IPC, no shared mutable state; just pure functions with fixed inputs.

**Unit tests:** Freeze inputs once (from a full run or stub); call one layer's function with those inputs and assert on outputs. No live dependency on the layer below.

**Property testing:** Use the same frozen params/state as the invariant; generate many inputs (e.g. context item_ids, token lists) with StreamData; run the layer with each and check properties (e.g. recommend returns in-catalog IDs, forward_model shape).

---

## Unit tests and property testing

- **Unit tests:** Freeze inputs once (from a full run or stub); call one layer with those inputs and assert on outputs. No live dependency on the layer below.
- **Property testing:** Keep frozen params/state fixed; generate many inputs (e.g. context item_ids, token lists) with StreamData; run the layer with each and check properties (e.g. recommend returns in-catalog IDs, forward_model shape).

## How it works

1. **Full-weights run:** Load real checkpoint and fixture (e.g. Serve.load_state), optionally run recommend or FixtureBuild.build.
2. **Capture at boundaries:** At each layer boundary, record the inputs that the upper layer receives from the layer below.
3. **Isolate:** To test Layer N alone, load the frozen inputs for Layer N and call Layer N functions with those inputs only.

---

## Layer boundaries and what to freeze

| Layer             | Inputs from below          | Frozen form                           |
| ----------------- | -------------------------- | ------------------------------------- |
| 1. Artifacts      | Paths                      | Paths or in-memory loaded data        |
| 2. Representation | item_text_dict, FSQ params | Map + params or ckpt path             |
| 3. Fixture        | items_path, ckpt_dir       | Paths or (item_text_dict, fsq_params) |
| 4. Model          | params, token_list         | Full params + token list(s)           |
| 5. Recommendation | state, context item_ids    | Serve state + fixed item_ids          |
| 6. Application    | Same as 5                  | Same frozen state + test_cases        |

---

## Implementation

- **Manual:** In tests, call Serve.load_state once, keep state, pass it and fixed context item_ids into the function under test.
- **Record/replay helper:** `RecGPT.LayerFreeze.record_from_state/2` records from a full run; `forward_model/2` and `recommend/3` run Model or Recommendation layer with frozen inputs. Tests use `RecGPT.TestSupport.FrozenHelpers.build_stub_state/0`, `build_frozen/1`, and `write_stub_ckpt!/1` for a single shared stub and frozen snapshot.
- **Property tests:** See `test/recgpt/property_test.exs` (Trie round-trip). For Recommendation or Model, use a frozen state from `FrozenHelpers.build_frozen/1` and generate context item_ids or token lists with StreamData; assert invariants (e.g. recommend results in catalog, logits shape).

Layers stay pure functions; isolation comes from fixed inputs from a full-weights run, not from processes or stubs.

---

## See also

- [15 Layers overview](15_layers_overview.md)
- [16-21] Per-layer docs.
