# Architecting a Next-Generation Recommender System

This section describes how RecGPT-style recommendation fits into a production-style design: live catalog and an **Elixir-only** pipeline (no Python). API: unified gRPC+REST ([13](13_grpc_rest_api.md), [14](14_api_schemas.md)); REST implemented first. Docs 11–16 cover algorithm, state, APIs, deployment; pipeline: [08](08_pipeline_reference.md).

---

## Why this architecture

Traditional recommenders rely on ID-based embedding tables and suffer from cold-start: new items need many interactions before they are placed accurately in latent space. **RecGPT** (arXiv:2506.06270) reframes sequential recommendation as text-driven, autoregressive token generation. Item representations come from text (titles, descriptions, categories) via embeddings and Finite Scalar Quantization (FSQ), so the system can generalize zero-shot and recommend new items immediately.

In this repo the full stack runs in **Elixir**: Bumblebee (MPNet) for text embeddings, FSQ and fixture building, Axon training, Nx inference, and Plug/Cowboy serving. Production use implies keeping catalog and fixture in sync and scaling state (e.g. trie in ETS) and inference as needed.

**Next:** [11_recgpt_paradigm.md](11_recgpt_paradigm.md) — FSQ, attention, and how the pipeline fits the RecGPT model.
