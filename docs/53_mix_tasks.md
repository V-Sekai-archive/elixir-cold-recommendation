# Mix tasks

| Task                       | Purpose                                                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mix recgpt.refetch`       | Refetch all bulk data: fetch_ckpt → export_ckpt → fetch_vae_ckpt → fetch_steam. `--force` to clear first.                                                |
| `mix recgpt.fetch_ckpt`    | Download RecGPT PyTorch checkpoint from Hugging Face (hkuds/RecGPT_model).                                                                              |
| `mix recgpt.export_ckpt`   | Export checkpoint to `manifest.json` + `.npy` (from `--from-pt`).                                                                                       |
| `mix recgpt.export_fuxi_ckpt` | Export FuXi-Linear init params for serve/pretrain. `--out` required; `--n-blocks`, `--max-seq-len`.                                                  |
| `mix recgpt.fetch_steam`   | Fetch Steam test split from HuggingFace (hkuds/RecGPT_dataset); write items + train/test/cold sequences.                                                |
| `mix recgpt.convert_trajectories` | Convert MovieLens/KuaiRand raw data to canonical JSON (items, train/test sequences). `--from`, `--out`, `--format movielens`, `--train-limit`, `--test-limit`. See [86](86_training_signal_test_dataset_plan.md). |
| `mix recgpt.training_signal_test` | Full pipeline: convert (optional) → build_fixture → pretrain → eval (zero-shot vs pretrained + cold). `--convert-from`, `--data-dir`, `--train-limit 0`, `--test-limit 0`, `--regime` (single/10min/5epochs/compare), `--fuxi`, `--epochs`, `--iterations`, `--fixture-limit`. See [86](86_training_signal_test_dataset_plan.md). |
| `mix recgpt.build_fixture` | Build `fixture.json` from `items.json` (Embedding + FSQ). Options: `--items`, `--out`, `--ckpt`.                                                        |
| `mix recgpt.pretrain`      | Pretrain on `train_sequences.json` with fixture + checkpoint; write updated params to `--out`. `--epochs 5`, `--save-every N` for periodic checkpoints. |
| `mix recgpt.eval`          | Run next-item eval in Elixir (`--data-dir`, `--ckpt`, `--fixture`, `--test`).                                                                           |
| `mix recgpt.serve`         | Start gRPC server (port 50051): Predict uses RecGPT.RecommendationService (default: Serve). Set `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT`.                 |

Paths default to `data/steam/` and `thirdparty/checkpoints/recgpt`. Env: `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT` for serve.

**Catalog storage** uses object-store semantics; options are BEAM-native (file path, [CubDB](https://hex.pm/packages/cubdb), or RabbitMQ's [Khepri](https://hex.pm/packages/khepri)). See [13 Infrastructure and serving](13_infrastructure_serving.md#catalog-storage-object-store-semantics).
