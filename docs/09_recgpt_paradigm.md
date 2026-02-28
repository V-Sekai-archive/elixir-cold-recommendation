# Proposal: Algorithmic foundations (RecGPT paradigm)

Sub-proposal of the [documentation index](README.md). Zero-shot recommendation as text-driven, autoregressive token generation (analogous to LLMs). The repo implements the stack in **Elixir**: FSQ, Bumblebee/MPNet embeddings, Axon training, Nx inference.

---

## Problem or limitation

The RecGPT paradigm (FSQ, hybrid attention, pipeline) must be documented so that implementers and extenders understand the algorithmic choices. Without a clear foundation doc, “why FSQ” and “why this pipeline” are unclear.

---

## Proposed improvement

Document **algorithmic foundations**: FSQ and semantic tokenization, hybrid bidirectional–causal attention, and the pipeline/module mapping. Pipeline order and commands: [02_pipeline_reference.md](02_pipeline_reference.md).

---

## Finite Scalar Quantization (FSQ) and Semantic Tokenization

Continuous embeddings are rich but not directly usable for discrete token-by-token decoding. FSQ gives a fixed-length discrete token sequence per item (e.g., 4 tokens, vocab 15 360).

- **Text → vector:** `RecGPT.Embedding` (Bumblebee, **sentence-transformers/all-mpnet-base-v2**) produces 768-d vectors from item text. All runs in the BEAM VM.
- **Vector → tokens:** `RecGPT.FSQ` and `RecGPT.FSQEncoder` project embeddings into token IDs. Quantization is set up so gradients flow correctly during training.

User histories are then expressed as universal token sequences rather than dataset-specific IDs, giving a domain-invariant space. New items are encoded with `RecGPT.Embedding.encode_item_text_dict/1` and passed through FSQ, so they are immediately recommendable (no cold-start).

---

## Hybrid Bidirectional–Causal Attention

The RecGPT transformer uses **bidirectional** attention over the tokens of a single item and **causal** attention across items in the sequence. The implementation loads the model from the RecGPT checkpoint and runs it via `RecGPT.Inference` (see [07_recgpt_checkpoint_layout.md](07_recgpt_checkpoint_layout.md)).

---

## Pipeline and modules

Data → fixture → pretrain → eval. Fixture building uses `RecGPT.FixtureBuild.build/2` (Embedding + FSQ → `token_id_list`); pretraining uses `RecGPT.AxonTrain` with the same checkpoint layout as inference. Checkpoints are `manifest.json` + `.npy`; they can be imported from PyTorch with `mix recgpt.export_ckpt`.

---

## Sub-proposals

- **FSQ and semantic tokenization** (above) — Text → vector → tokens; Embedding, FSQ, FSQEncoder.
- **Hybrid attention** (above) — Bidirectional within item, causal across items.
- **Pipeline and modules** (above) — Data → fixture → pretrain → eval; module roles.
- [10_dynamic_state_ets.md](10_dynamic_state_ets.md) — Trie, beam search, optional ETS.
