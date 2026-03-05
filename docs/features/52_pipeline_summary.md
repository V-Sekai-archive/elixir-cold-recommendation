# Pipeline summary

| Step            | Command / API                                                             | Outputs                                                                                                              |
| --------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **1. Data**     | `mix recgpt.fetch_steam data/steam` or `RecGPT.Steam.Fetch.run/1`         | `items.json`, `train_sequences.json`, `test_sequences.json`, `cold_test_sequences.json`, `cold_train_sequences.json` |
| **2. Fixture**  | `mix recgpt.build_fixture` or `RecGPT.FixtureBuild.build/2`               | `fixture.json` (`num_items`, `token_id_list`)                                                                        |
| **3. Pretrain** | `mix recgpt.pretrain` (uses `AxonTrain.stream_batches` + `run/3`)         | Updated checkpoint in `--out`                                                                                        |
| **4. Eval**     | `mix recgpt.eval` (Elixir; `--data-dir`, `--ckpt`, `--fixture`, `--test`) | Hit@k, MRR, etc.                                                                                                     |

For best quality, **pretrain then eval**; zero-shot (pretrained ckpt only) is a baseline. See [07 Steam splits and pretraining](07_steam_splits_and_pretraining.md).

## Canonical item texts (same input as official)

By default, `build_fixture` and `compare_embeddings` read item text from the `canonical_item_texts` SQLite table so both use the same bytes. To populate that table from Python (byte-exact match with the official script), run once:

```bash
mix ecto.migrate
uv run python scripts/dump_canonical_to_sqlite.py --pkl data/steam/item_text_dict.pkl --verify
```

Use the same `--db` or `RECGPT_SQLITE_PATH` as Elixir. Then both inputs are from Python; no Python at runtime.
