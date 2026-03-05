# Quick start

1. **Get a checkpoint** (export dir with `manifest.json` + `.npy` tensors):

   ```bash
   mix recgpt.fetch_ckpt
   mix recgpt.export_ckpt --from-pt data/recgpt_layer_3_weight.pt --out data/recgpt_ckpt_export
   ```

2. **Generate data and run first step** (Steam baseline, Elixir eval):

   ```bash
   mix recgpt.first_step                     # fetch → build_fixture → eval (Elixir)
   # Or: mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture && mix recgpt.eval
   ```

3. **Serve recommendations** (gRPC; Predict uses Elixir Serve):

   ```bash
   RECGPT_FIXTURE=data/steam/fixture.json RECGPT_CKPT_EXPORT=data/recgpt_ckpt_export mix recgpt.serve
   ```

See [52 Pipeline summary](52_pipeline_summary.md), [02 Pipeline overview](02_pipeline_overview.md), and [03 Pipeline steps](03_pipeline_steps.md) for the full sequence and options.
