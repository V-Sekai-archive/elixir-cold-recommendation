# Profile one Predict call. Run: mix profile.fprof scripts/profile_predict.exs
# Uses real checkpoint, runs setup (warmup) then profiles one recommend.
# Output: profile data with ACC/OWN times per function.

fixture_path = Path.expand("data/steam/fixture.json", File.cwd!())
ckpt_dir = Path.expand("data/recgpt_ckpt_export", File.cwd!())
catalog_path = Path.expand("data/steam/items.json", File.cwd!())

IO.puts("Loading state...")
{:ok, state} = RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path)

IO.puts("Setup (warmup)...")
{:ok, _} = RecGPT.Serve.recommend(state, [0, 1], 10)

IO.puts("Profiling one recommend...")
_ = RecGPT.Serve.recommend(state, [0, 1], 10)
