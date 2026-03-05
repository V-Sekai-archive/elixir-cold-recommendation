# Plan: EXLA JIT disk cache

**Status:** Done | **Profit:** 3.0 | **Effort:** Low | **Gain:** High (setup)

Profile after change: `mix recgpt.trace_predict --runs 50 --jitter-ms 3`

---

## Goal

Setup 20–30s to 2–5s by persisting compiled XLA binaries so restarts skip JIT recompilation.

---

## Changes

- config/config.exs: `config :recgpt, :exla_jit_cache_dir, "tmp/exla_cache"`
- lib/recgpt/checkpoint_loader.ex: Add `get_sha256(export_dir)` returning checkpoint SHA256
- lib/recgpt/serve.ex: Pass `ckpt_export_dir` to `build_get_logits_batch_fn`. Compute cache key from `sha256("exla:#{exla_v}:nx:#{nx_v}:ckpt:#{ckpt_sha}:dtype:#{dtype}") |> slice(0,16)`. Use `Nx.Defn.jit(..., compiler: EXLA, cache: path)` for `forward_with_cache` and `forward_incremental`.
- Add `tmp/exla_cache/` to `.gitignore`

## Cache keying

Derive subdirectory from: EXLA version, Nx version, checkpoint SHA256, inference dtype, max_cache_len. Invalidates when any change.

**Stable canonicalization:** Incremental forward uses a padded KV cache (fixed shape) so the args signature stays constant across beam steps; otherwise varying seq_len (4, 5, 6…) would cause repeated recompilation.

---

## Profile

Two runs; second run should show cache hit (faster setup).
