# Proposal: Infrastructure and serving

Sub-proposal of the [documentation index](README.md). How to run and scale the recommendation server.

---

## Problem or limitation

Serving and deployment must be specified: how to run the server, what it depends on, and how it could scale (e.g. ETS, external inference). Without this, operators and SREs lack a single reference.

---

## Proposed improvement

**Current:** Single BEAM process; inference runs in-process with Nx. **Optional:** external inference (e.g., Triton) for GPU batching; multi-region/edge for lower latency. Document the run command, defaults, and optional scaling paths.

---

## In-process inference

At startup, `RecGPT.Serve.load_state/3` loads the checkpoint (`RecGPT.CheckpointLoader`, manifest + `.npy`), builds the trie from the fixture, and builds a logits function. `RecGPT.Inference.forward/4` runs the full forward pass (embedding, aux, GPT-2, head) in Nx. No separate inference server.

**Run the server:**

```bash
mix recgpt.serve --fixture <path> --ckpt <path> [--grpc-port 50051]
```

Defaults: `--fixture` → `data/serve_e2e_fixture.json` (or `RECGPT_FIXTURE`), `--ckpt` → `data/recgpt_ckpt_export` (or `RECGPT_CKPT_EXPORT`). Both the fixture and checkpoint export directory are required. API contract: [recommendation.proto](../priv/proto/recgpt/v1/recommendation.proto).

Nx can use the default backend (CPU) or Torchx if configured. For moderate traffic this is enough.

---

## Sub-proposals

- **In-process inference** (above) — load_state, forward, run serve.
- **Optional:** Triton/edge (above) — External inference; multi-region.

---

**See also:** [docs README](README.md), [02 Pipeline reference](02_pipeline_reference.md).
