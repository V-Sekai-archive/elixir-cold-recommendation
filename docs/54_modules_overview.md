# Modules (overview)

| Module | Purpose |
|--------|---------|
| **RecGPT.FSQ** | FSQ quantizer (levels [8,8,8,6,5], 4 tokens/item, vocab 15360). `load_params/1`, `encode/2`. |
| **RecGPT.FSQEncoder** | Embeddings (num_items, 768) + FSQ params → `token_id_list` (list of 4-token lists). |
| **RecGPT.Embedding** | Text → 768-d via Bumblebee (all-mpnet-base-v2). `encode_item_text_dict/1`. |
| **RecGPT.FixtureBuild** | Build fixture from items.json. `build/2`, `write_fixture/2`. |
| **RecGPT.Training** | `build_train_batch/4`, `encode_aux/3`, `loss_shifted_ce/2`. |
| **RecGPT.AxonTrain** | Training loop: `stream_batches/4`, `run/3` (Polaris optimizer). |
| **RecGPT.Inference** | Forward pass (training): token embed + aux + GPT-2 + head. `forward/4`, `forward_full_sequence/4`. |
| **RecGPT.InferenceParams** | Build defn-friendly full params (atom keys). Stub checkpoints get identity layers so one code path. |
| **RecGPT.InferenceDefn** | Defn entry points for serve: `forward_with_cache/4`, `forward_incremental/5` (EXLA JIT). |
| **RecGPT.Serve** | Load state (fixture + checkpoint); EXLA. Implements `RecGPT.RecommendationService`. |
| **RecGPT.CheckpointLoader** | Load export dir → `%{key => Nx.Tensor}`. |
| **RecGPT.CheckpointExport** | Write params to export dir (manifest + .npy). |
| **RecGPT.Steam.Fetch** | Steam test split → items + train/test/cold sequences (HuggingFace hkuds/RecGPT_dataset). |

Full list and details: [04 RecGPT library](04_recgpt_library.md).
