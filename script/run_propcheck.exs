# Run PropCheck tests by compiling test/recgpt/propcheck_test.exs.skip in a VM
# where PropCheck (CounterStrike) is already started. Use: MIX_ENV=test mix run script/run_propcheck.exs
Application.ensure_all_started(:nx)
Application.ensure_all_started(:propcheck)

ExUnit.start(exclude: [:embedding, :integration, :eval, :e2e_serve, :serve_parity, :pt_fixture])

src = Path.join([File.cwd!(), "test", "recgpt", "propcheck_test.exs.skip"])
tmp_dir = Path.join(Mix.Project.build_path(), "tmp")
File.mkdir_p!(tmp_dir)
tmp = Path.join(tmp_dir, "propcheck_test.exs")
File.write!(tmp, File.read!(src))
Code.require_file(tmp)

ExUnit.run()
