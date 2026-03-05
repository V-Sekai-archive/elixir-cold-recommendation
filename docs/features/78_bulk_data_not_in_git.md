# Bulk Data: Not in Git

This doc lists **bulk data** that lives locally, is **gitignored**, and should be **backed up elsewhere** (e.g. external drive, cloud). Do not commit it to git.

---

## Inventory (as of this writing)

| Location | Contents | Size (approx) | How to get | Back up? |
|----------|----------|---------------|------------|-----------|
| `data/steam/` | Steam dataset: items, sequences, embeddings (.npy), fixture (from build) | ~100s MB | `mix recgpt.fetch_steam data/steam` | Optional; refetchable from HuggingFace |
| `data/fuxi_ckpt_export/` | FuXi-Linear checkpoint (manifest + *.npy) for eval/serve | ~100s MB | `mix recgpt.export_fuxi_ckpt --out data/fuxi_ckpt_export` | Optional; refetchable |
| `thirdparty/checkpoints/vae/` | VAE checkpoint for FSQ | ~10 MB | `mix recgpt.fetch_vae_ckpt` | Optional; refetchable |
| `thirdparty/prediction-market-analysis/data/` | Jon-Becker Polymarket Parquet (markets, trades, blocks) | ~36 GiB | `git submodule update --init` then `cd thirdparty/prediction-market-analysis && make setup` | Optional; for Becker convert only |

**Total under `data/`:** ~2+ GB (depends on what you have). Add ~36 GiB if using Jon-Becker.

---

## What is gitignored

From [.gitignore](../../.gitignore):

- `data/` — entire data directory
- `data/**/*.db`, `*.db-shm`, `*.db-wal` — SQLite
- `data/fuxi_ckpt_export/*`, `data/**/*.pt`
- `thirdparty/checkpoints/vae/*.pt`, `thirdparty/checkpoints/recgpt/*.pt`, `*.npy`, `manifest.json`

So all bulk data is correctly excluded from git.

---

## Refetch (one command)

To recreate all bulk data from a clean clone:

```bash
mix recgpt.refetch
```

Runs in order: `export_fuxi_ckpt` → `fetch_vae_ckpt` → `fetch_steam`.

Use `--force` to remove existing `data/steam` and `thirdparty/checkpoints/` before refetch.
Next: `mix recgpt.first_step` (requires canonical texts; see [51 Quick start](51_quick_start.md)).

---

## How to “save but not in git”

1. **Refetch on demand** — `mix recgpt.refetch` gets everything. Or run individual tasks: `export_fuxi_ckpt`, `fetch_steam`, `fetch_vae_ckpt`.

2. **Back up locally** — Copy `data/` and `thirdparty/checkpoints/` to an external drive or NAS. Restore by copying back.

3. **Cloud / object storage** — Archive as tarballs; document bucket/path. Restore or refetch.

4. **What to back up first** — Checkpoint exports and Steam fixture take time to regenerate. Raw .pt and datasets can be refetched.

---

## See also

- [.gitignore](../../.gitignore) — what’s excluded from git
- [thirdparty/checkpoints/README.md](../../thirdparty/checkpoints/README.md) — checkpoint layout
- [51 Quick start](51_quick_start.md) — pipeline after refetch
- [53 Mix tasks](53_mix_tasks.md) — `refetch`, `export_ckpt`, `export_fuxi_ckpt`, `fetch_steam`, `fetch_vae_ckpt`
