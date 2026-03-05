# Quick start

1. **Get checkpoint and data** (FuXi-Linear init + VAE + Steam):

   ```bash
   mix recgpt.refetch
   ```

2. **Run first step** (Steam baseline, Elixir eval):

   ```bash
   mix recgpt.first_step                     # fetch → build_fixture → eval (Elixir)
   # Or: mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture && mix recgpt.eval
   ```

3. **Serve recommendations** (gRPC; Predict uses Elixir Serve):

   ```bash
   RECGPT_FIXTURE=data/steam/fixture.json RECGPT_CKPT_EXPORT=data/fuxi_ckpt_export mix recgpt.serve
   ```

See [02 Pipeline overview](02_pipeline_overview.md) and [03 Pipeline steps](03_pipeline_steps.md) for the full sequence and options.
