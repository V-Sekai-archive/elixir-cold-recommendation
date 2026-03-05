# Tests

```bash
mix test
```

- Excluded by default: `integration`, `eval`.
- **Include integration:** `mix test --include integration`
- **Eval (fixture + ckpt + test file):** `mix test test/recgpt/eval_test.exs --include eval --include integration`

See [06 Evaluation and testing](06_evaluation_and_testing.md) and [04 RecGPT library](04_recgpt_library.md).
