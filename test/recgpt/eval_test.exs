# Eval test: load fixture + checkpoint + test_sequences, run RecGPT.Eval, assert reject null.
# test_sequences.json must be held-out (e.g. last-item-out). Run with --include eval --include integration.
defmodule RecGPT.EvalTest do
  use ExUnit.Case, async: false

  alias RecGPT.Eval
  alias RecGPT.Serve

  defp fixture_path do
    from_env = System.get_env("RECGPT_FIXTURE")
    if from_env && from_env != "", do: from_env, else: resolve_path("data/clickstream/fixture.json")
  end

  defp ckpt_dir do
    from_env = System.get_env("RECGPT_CKPT_EXPORT")
    if from_env && from_env != "", do: from_env, else: resolve_path("data/recgpt_ckpt_export")
  end

  defp test_path do
    from_env = System.get_env("RECGPT_TEST_SEQUENCES")
    if from_env && from_env != "", do: from_env, else: resolve_path("data/clickstream/test_sequences.json")
  end

  defp resolve_path(path) do
    cwd = File.cwd!()
    from_cwd = Path.join(cwd, path)
    from_parent = Path.join(Path.dirname(cwd), path)
    if File.regular?(from_cwd) or File.dir?(from_cwd), do: from_cwd, else: from_parent
  end

  @tag :integration
  @tag :eval
  @tag timeout: 60_000
  test "eval on held-out test set rejects null (Hit@1 > random)" do
    fixture = fixture_path()
    ckpt = ckpt_dir()
    test_file = test_path()

    unless File.regular?(fixture) do
      flunk("""
      Skipped (missing data): Fixture not found: #{fixture}
      Set RECGPT_FIXTURE or provide data/clickstream/fixture.json.
      Build fixture from items (Embedding + FSQ) then run with --include eval --include integration.
      """)
    end

    unless File.dir?(ckpt) and File.regular?(Path.join(ckpt, "manifest.json")) do
      flunk("""
      Skipped (missing data): Checkpoint not found: #{ckpt}
      Set RECGPT_CKPT_EXPORT or export checkpoint to data/recgpt_ckpt_export.
      See docs/02_recgpt_checkpoint_layout.md.
      """)
    end

    unless File.regular?(test_file) do
      flunk("""
      Skipped (missing data): Test sequences not found: #{test_file}
      Set RECGPT_TEST_SEQUENCES or provide data/clickstream/test_sequences.json.
      From test env run RecGPT.Clickstream.Fetch.run() to build data/clickstream (test-only).
      """)
    end

    Application.ensure_all_started(:nx)

    assert {:ok, state} = Serve.load_state(fixture, ckpt, nil)
    assert {:ok, test_cases} = Eval.load_test_cases(test_file)
    assert length(test_cases) >= 1, "need at least one test case"

    metrics = Eval.evaluate(state, test_cases, top_k: 10)

    assert metrics.n >= 1
    assert metrics.catalog_size > 0
    assert metrics.hit_at_1 >= 0.0 and metrics.hit_at_1 <= 1.0
    assert metrics.random_hit_at_1 > 0.0

    assert metrics.rejects_null,
           "expected Hit@1 (#{metrics.hit_at_1}) > random_hit_at_1 (#{metrics.random_hit_at_1}) to reject null hypothesis"
  end
end
