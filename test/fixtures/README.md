# Test fixtures

## sample.pt

Minimal PyTorch `.pt` checkpoint (zip format) for testing `RecGPT.PtLoader`.

**Generate** (requires Python and PyTorch):

```bash
python scripts/generate_pt_fixture.py
```

This writes `test/fixtures/sample.pt` with a small state_dict: `wte` (4×8), `pred_head.weight` (8×4), `pred_head.bias` (4×1).

The fixture is used by the test tagged `:pt_fixture` in `test/recgpt/pt_loader_test.exs`. That test is excluded by default; run it with:

```bash
mix test --include pt_fixture test/recgpt/pt_loader_test.exs
```

Note: PyTorch pickle format can vary by version; if the fixture fails to load, try regenerating with a different PyTorch or use an export from `mix recgpt.export_ckpt --from-pt ...` for integration tests.
