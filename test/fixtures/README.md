# Test fixtures

## sample.pt

Minimal PyTorch `.pt` checkpoint (zip format) for testing `RecGPT.PtLoader`.

**PtLoader** supports both zip-based `.pt` (PyTorch 1.6+) and legacy (single-pickle) `.pt` files.

**Fixture choice (in order):** The `:pt_fixture` test uses (1) `data/recgpt_layer_3_weight.pt` if present, else (2) `test/fixtures/sample.pt`. If `sample.pt` is missing, it is generated with `mix recgpt.generate_pt_fixture`. If the known-good file exists but is not a valid .pt (e.g. a redirect page from Hugging Face), the test fails with a message to re-download the weights.

**Generate sample.pt:** `mix recgpt.generate_pt_fixture`

**Run the test:** (excluded by default)

```bash
mix test --include pt_fixture test/recgpt/pt_loader_test.exs
```
