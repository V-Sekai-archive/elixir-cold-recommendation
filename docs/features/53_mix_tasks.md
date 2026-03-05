# Mix tasks

| Task                       | Purpose                                                                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mix recgpt.refetch`       | Refetch bulk data: export_fuxi_ckpt → fetch_vae_ckpt → fetch_steam. `--force` to clear first.                                                            |
| `mix recgpt.export_ckpt`   | Export checkpoint to `manifest.json` + `.npy` (from `--from-pt`; for custom .pt sources).                                                                 |
| `mix recgpt.export_fuxi_ckpt` | Export FuXi-Linear init params for serve/pretrain. `--out` required; `--n-blocks`, `--max-seq-len`.                                                  |
| `mix recgpt.fetch_steam`   | Fetch Steam test split from HuggingFace (hkuds/RecGPT_dataset); write items + train/test/cold sequences.                                                |
| `mix recgpt.convert_trajectories` | Convert KuaiRand (default) or MovieLens to canonical JSON. `--from`, `--out`, `--format`. See [93](93_pretraining_plan.md). |
| `mix recgpt.training_signal_test` | Full pipeline: convert (optional) → build_fixture → pretrain → eval (zero-shot vs pretrained + cold). `--convert-from`, `--data-dir`, `--train-limit 0`, `--test-limit 0`, `--regime` (single/10min/5epochs/compare), `--fuxi`, `--epochs`, `--iterations`, `--fixture-limit`.  |
| `mix recgpt.build_fixture` | Build `fixture.json` from `items.json` (Embedding + FSQ). Options: `--items`, `--out`, `--ckpt`.                                                        |
| `mix recgpt.pretrain`      | Pretrain on train sequences with fixture + checkpoint; write to `--out`. File-based: `--train`, `--items` paths; DB: `--train db --items db`. See [93](93_pretraining_plan.md). `--epochs 5`, `--eval-test-every N --test PATH` for train/test loss. |
| `mix recgpt.eval`          | Run next-item eval in Elixir (`--data-dir`, `--ckpt`, `--fixture`, `--test`).                                                                           |
| `mix recgpt.figgie_simulate` | Simulate Figgie games for training data. `--games N` (default 1000), `--output PATH` (default priv/figgie_fixture.json). Generates fixture for arbitrage training. |
| `mix recgpt.serve`         | Start gRPC server (port 50051): Predict uses RecGPT.RecommendationService (default: Serve). Set `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT`.                 |

Paths default to `data/steam/` and `data/fuxi_ckpt_export` (FuXi-Linear). Env: `RECGPT_FIXTURE`, `RECGPT_CKPT_EXPORT` for serve. **Decode strategy:** `RECGPT_DECODE_STRATEGY=mtp` uses Multi-Token Prediction (model predicts K tokens at once; acceleration in weights, no draft/cache). Default `beam_search`. See [65](65_latency_flow.md).

**Catalog storage** uses object-store semantics; options are BEAM-native (file path, [CubDB](https://hex.pm/packages/cubdb), or RabbitMQ's [Khepri](https://hex.pm/packages/khepri)). See [13 Infrastructure and serving](13_infrastructure_serving.md#catalog-storage-object-store-semantics).
