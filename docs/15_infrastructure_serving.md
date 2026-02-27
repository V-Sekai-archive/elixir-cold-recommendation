# Infrastructure and Serving

**Current:** Single BEAM process; inference runs in-process with Nx. **Optional:** external inference (e.g. Triton) for GPU batching; multi-region/edge for lower latency.

---

## In-process inference

At startup, `RecGPT.Serve.load_state/4` loads the checkpoint (`RecGPT.CheckpointLoader`, manifest + `.npy`), builds the trie from the fixture, and builds a logits function. `RecGPT.Inference.forward/4` runs the full forward pass (embedding, aux, GPT-2, head) in Nx. No separate inference server.

**Run the server:**

```bash
mix recgpt.serve --fixture data/clickstream/fixture.json --ckpt data/ckpt_out [--port 8000]
```

Nx can use the default backend (CPU) or Torchx if configured. For moderate traffic this is enough.

---

## Optional: external inference and edge

- **Triton (or similar):** Offload the transformer to an external server; Elixir keeps trie and beam search and sends/receives logits. Not implemented in this repo; would require a client and config.
- **Edge:** Run `mix recgpt.serve` in multiple regions or edge locations; replicate fixture and checkpoint to each node (or load from shared storage at startup). No specific cloud or provider is prescribed.

**Next:** [16_architecture_conclusion.md](16_architecture_conclusion.md).
