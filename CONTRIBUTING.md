# Contributing to RecGPT (Elixir)

Thanks for your interest in contributing. This guide covers setup, tests, code quality, and how to run performance/accuracy and embedding checks.

---

## Prerequisites

- **Elixir** ~> 1.14 (tested on 1.19.x)
- **Erlang** (matching your Elixir install)
- For **GPU (CUDA)** builds: CUDA Toolkit and (on Windows) Visual Studio Build Tools; see [Building with CUDA](#building-with-cuda) below.

---

## Setup

```bash
git clone <repo-url>
cd elixir-recgpt
mix deps.get
mix compile
```

- Default build is **CPU-only** (Nx/Torchx with `LIBTORCH_TARGET=cpu`).
- For **CUDA**, set `LIBTORCH_TARGET` and (on Windows) `CUDA_PATH` before compiling; see next section.

---

## Building with CUDA

To use the GPU backend (e.g. for faster training or inference):

1. **Set the LibTorch target** (e.g. CUDA 12.9):
   ```bash
   export LIBTORCH_TARGET=cu129   # Linux/macOS
   set LIBTORCH_TARGET=cu129      # Windows cmd
   $env:LIBTORCH_TARGET = "cu129" # PowerShell
   ```

2. **Point to your CUDA install** (required so CMake finds `nvcc` and libraries):
   - **Linux:** Ensure `nvcc` is on `PATH` or set `CUDAToolkit_ROOT` / `CUDA_PATH`.
   - **Windows:** Set `CUDA_PATH` to your toolkit root, e.g.:
     ```powershell
     $env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
     ```

3. **Compile:**
   ```bash
   mix compile --force
   ```

On **Windows**, if you use **Visual Studio 2026 (or 18)** and see *"unsupported Microsoft Visual Studio version"* from `nvcc`, the project may rely on a **local patch** in `deps/torchx/mix.exs` that:

- Passes the parent environment (e.g. `CUDA_PATH`, `PATH`) into the CMake build,
- Sets `CMAKE_GENERATOR_TOOLSET=cuda=<CUDA_PATH>` for the Visual Studio generator,
- Adds `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` so `nvcc` accepts the newer host compiler.

That patch lives only in your checkout; it is overwritten if you run `mix deps.get` or `mix deps.update torchx`. Re-apply the same changes if you update the dependency.

After a successful CUDA build, you can confirm the GPU is used with:

```bash
mix recgpt.check_gpu
```

(First run may take a while while CUDA initializes.)

---

## Running tests

```bash
mix test
```

- **Excluded by default:** `integration`, `eval` (require fixture/checkpoint/test data or network).
- **Include eval-only tests** (no fixture):
  ```bash
  mix test --include eval
  ```
- **Include integration tests** (need Steam data + optional fixture/ckpt):
  ```bash
  mix test --include integration
  ```
- **Full eval with integration** (fixture + ckpt + test file; set `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT`, `RECGPT_TEST_SEQUENCES`):
  ```bash
  mix test test/recgpt/eval_test.exs --include eval --include integration
  ```

See [docs/06_evaluation_and_testing.md](docs/06_evaluation_and_testing.md) and [README.md](README.md#tests).

---

## Code quality

- **Credo** (style and consistency):
  ```bash
  mix credo
  ```
  Config: [.credo.exs](.credo.exs) (e.g. relaxed nesting/complexity for some modules).

- **Dialyzer** (types and contracts):
  ```bash
  mix dialyzer
  ```
  Uses `priv/plts/dialyzer.plt` and `.dialyzer_ignore.exs`; first run builds the PLT and can take several minutes.

---

## Performance and accuracy

- **Serve benchmark** (recommendation latency):
  ```bash
  mix run bench/recgpt_serve_bench.exs
  ```

- **Accuracy (Hit@k, MRR)** on a held-out test set requires fixture, checkpoint, and test sequences (e.g. after `mix recgpt.fetch_steam data/steam` and `mix recgpt.build_fixture`):
  ```bash
  mix recgpt.eval --data-dir data/steam --ckpt data/recgpt_ckpt_export --test data/steam/test_sequences.json
  ```
  See [mix recgpt.eval](mix/tasks/recgpt.eval.ex) and [docs/06_evaluation_and_testing.md](docs/06_evaluation_and_testing.md). Eval runs in Elixir (RecGPT.Serve + RecGPT.Eval).

- **Divide:** Generating embeddings and testing recommendation performance are separate concerns. See [docs/embedding_vs_eval.md](docs/embedding_vs_eval.md).
- **Embedding parity** (our Bumblebee embeddings vs dataset `item_text_embeddings.npy`):
  - Inspect the reference text shape: `mix recgpt.inspect_item_text --steam-dir data/steam --limit 5`
  - Compare embeddings: `mix recgpt.compare_embeddings --steam-dir data/steam --limit 500`
  - Try `--text-format title_only` to test plain title vs dict-style string (dict-style gives higher similarity).
  Reports per-item cosine similarity (mean, min, max). A mean below 0.95 is a large mismatch and can explain poor eval when using the released checkpoint. **Workaround:** use the original dataset’s `item_text_embeddings.npy` when building the fixture: `mix recgpt.build_fixture --embeddings-npy data/steam/item_text_embeddings.npy` so `token_id_list` matches the checkpoint. See [docs/26_embedding_mismatch.md](docs/26_embedding_mismatch.md) for the text-format pattern and debugging (e.g. `--dump-row 0`).
  - **Local dataset clone:** If you have [hkuds/RecGPT_dataset](https://huggingface.co/datasets/hkuds/RecGPT_dataset) cloned locally, run `mix recgpt.fetch_steam <clone>/test/steam` then use `--steam-dir <clone>/test/steam` for inspect/compare so the large .npy is read from disk instead of downloaded.

---

## Documentation

- **MVP guard rails:** [docs/25_mvp_guard_rails.md](docs/25_mvp_guard_rails.md) — tombstones to keep the rope bridge on track (no multi-rank SPMD / sharding until the minimal loop is closed).
- **First step plan:** [docs/24_first_step_plan.md](docs/24_first_step_plan.md) — Steam baseline; one-shot: `mix recgpt.first_step` (fetch + build_fixture + eval in Elixir). Requires checkpoint.
- **Index:** [docs/README.md](docs/README.md) — topics, pipeline, API, parity, and links to other docs.
- **Library and API:** [docs/04_recgpt_library.md](docs/04_recgpt_library.md).
- **Parity and embedding gap:** [docs/09_parity_overview.md](docs/09_parity_overview.md), [docs/10_parity_layers.md](docs/10_parity_layers.md), [docs/26_embedding_mismatch.md](docs/26_embedding_mismatch.md).

When adding features or changing behavior, update the relevant doc and the index if needed.

---

## Submitting changes

1. **Tests:** Ensure `mix test` passes (and, if you touch integration/eval paths, run with `--include eval` or `--include integration` as appropriate).
2. **Style:** Run `mix credo` and fix reported issues (or document why an exception is needed).
3. **Types:** Run `mix dialyzer` and fix or extend typespecs; update `.dialyzer_ignore.exs` only when necessary and with a short comment.
4. **Changelog:** Add an entry to [CHANGELOG.md](CHANGELOG.md) under an “Unreleased” or version heading.
5. **Versioning:** For releases, bump the version in [mix.exs](mix.exs) and tag (e.g. `v0.2.0`); see [README.md](README.md#versioning).

If you change the CUDA/build path (e.g. torchx patch or env vars), document it in this file or in the README so others can reproduce the build.
