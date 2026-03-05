# Eval test: load_test_cases with synthetic data; evaluate with frozen inputs (stub state).
# Integration: build data/steam with mix recgpt.fetch_steam data/steam.
defmodule RecGPT.EvalTest do
  use ExUnit.Case, async: false

  alias RecGPT.Eval
  alias RecGPT.Serve
  alias RecGPT.TestSupport.FrozenHelpers

  @top_k 10
  # Minimal test cases to have reasonable power to reject null (Hit@1 > random). Rule of thumb:
  # under H1 expect Hit@1 ~ 5% when random=1%; 100 cases => ~5 hits vs 1 under H0.
  @min_n_reject_null 100

  describe "evaluate/3" do
    test "returns metrics with frozen inputs (stub state) and test cases" do
      frozen = FrozenHelpers.build_frozen([0])
      test_cases = [%{"context" => [0], "next_item" => 0}, %{"context" => [0], "next_item" => 1}]
      metrics = Eval.evaluate(frozen.state, test_cases, top_k: 5)
      assert metrics.n >= 1
      assert metrics.hit_at_1 >= 0.0 and metrics.hit_at_1 <= 1.0
      assert metrics.hit_at_5 >= 0.0 and metrics.hit_at_5 <= 1.0
      assert metrics.mrr >= 0.0 and metrics.mrr <= 1.0
      assert metrics.catalog_size == 2
      assert metrics.random_hit_at_1 == 0.5
      assert is_boolean(metrics.rejects_null)
    end

    test "skips test cases with empty context or nil next_item" do
      frozen = FrozenHelpers.build_frozen([0])

      test_cases = [
        %{"context" => [], "next_item" => 0},
        %{"context" => [0], "next_item" => nil}
      ]

      metrics = Eval.evaluate(frozen.state, test_cases, top_k: 5)
      assert metrics.n == 0 or metrics.n == 1
    end

    test "evaluate with recommend_fn (gRPC path) uses PredictionService.Server" do
      state = FrozenHelpers.build_stub_state()
      Application.put_env(:recgpt, :serve_state, state)

      case Process.whereis(RecGPT.PredictBatchCollector) do
        nil ->
          {:ok, _} =
            GenServer.start_link(RecGPT.PredictBatchCollector, [],
              name: RecGPT.PredictBatchCollector
            )

        _ ->
          :ok
      end

      grpc_fn = fn ctx, k ->
        request = %Recgpt.V1.PredictRequest{context_item_ids: ctx, max_results: k}
        response = Recgpt.V1.PredictionService.Server.predict(request, nil)
        {:ok, response.item_ids || []}
      end

      test_cases = [
        %{"context" => [0], "next_item" => 1},
        %{"context" => [0, 1], "next_item" => 0}
      ]

      metrics = Eval.evaluate(state, test_cases, top_k: 5, recommend_fn: grpc_fn)

      assert metrics.n == 2
      assert metrics.hit_at_1 >= 0.0 and metrics.hit_at_1 <= 1.0
      assert metrics.catalog_size == 2
    after
      Application.delete_env(:recgpt, :serve_state)
    end
  end

  describe "load_test_cases/1" do
    test "returns error when file does not exist" do
      path =
        Path.join(System.tmp_dir!(), "nonexistent_#{:erlang.unique_integer([:positive])}.json")

      assert {:error, msg} = Eval.load_test_cases(path)
      assert msg =~ "not found" or msg =~ path
    end
  end

  @tag :eval
  test "Fetch output test_sequences.json fits top-k form (one next_item per case, k=10)" do
    path = Path.join(File.cwd!(), "data/steam/test_sequences.json")

    unless File.regular?(path) do
      flunk("Run mix recgpt.fetch_steam data/steam to build data/steam/test_sequences.json")
    end

    assert {:ok, cases} = Eval.load_test_cases(path)
    assert cases != []

    raw = File.read!(path) |> Jason.decode!()
    num_items = raw["num_items"] || 0
    assert num_items >= 1

    for tc <- cases do
      context = tc["context"] || tc[:context] || []
      next_item = tc["next_item"] || tc[:next_item]
      assert is_list(context), "each test case must have context list"
      assert is_integer(next_item), "each test case must have single next_item (top-k form)"
      assert next_item >= 0 and next_item < num_items, "next_item must be in 0..num_items-1"
    end
  end

  @tag :eval
  test "load_test_cases loads test_sequences JSON" do
    num_items = 20

    test_cases =
      for i <- 0..14 do
        %{"context" => [rem(i, num_items)], "next_item" => rem(i + 1, num_items)}
      end

    payload = %{"num_items" => num_items, "test_cases" => test_cases}

    path =
      Path.join(
        System.tmp_dir!(),
        "recgpt_eval_test_sequences_#{:erlang.unique_integer([:positive])}.json"
      )

    File.write!(path, Jason.encode!(payload))

    try do
      assert {:ok, cases} = Eval.load_test_cases(path)
      assert length(cases) == 15

      for tc <- cases do
        assert is_list(tc["context"])
        assert is_integer(tc["next_item"])
        assert tc["next_item"] >= 0 and tc["next_item"] < num_items
      end
    after
      File.rm(path)
    end
  end

  @tag :integration
  @tag :eval
  @tag timeout: 300_000
  test "eval on held-out test set rejects null (Hit@1 > random)" do
    fixture = System.get_env("RECGPT_FIXTURE")
    ckpt = System.get_env("RECGPT_CKPT_EXPORT")
    test_file = System.get_env("RECGPT_TEST_SEQUENCES")

    unless is_binary(fixture) and fixture != "" and File.regular?(fixture) do
      flunk("""
      Skipped (missing data): Set RECGPT_FIXTURE to path to fixture.json (token_id_list + num_items).
      """)
    end

    unless is_binary(ckpt) and ckpt != "" and File.dir?(ckpt) and
             File.regular?(Path.join(ckpt, "manifest.json")) do
      flunk("""
      Skipped (missing data): Set RECGPT_CKPT_EXPORT to checkpoint export dir. See docs/features/08_recgpt_checkpoint_layout.md.
      """)
    end

    unless is_binary(test_file) and test_file != "" and File.regular?(test_file) do
      flunk("""
      Skipped (missing data): Set RECGPT_TEST_SEQUENCES to path to test_sequences.json (test_cases + num_items).
      See docs/features/05_eval_data_shapes.md for test_sequences.json format.
      """)
    end

    Application.ensure_all_started(:nx)

    assert {:ok, state} = Serve.load_state(fixture, ckpt, nil)
    assert {:ok, raw_cases} = Eval.load_test_cases(test_file)
    assert raw_cases != [], "need at least one test case"

    test_cases =
      raw_cases
      |> Eval.filter_to_catalog(state.num_items)
      |> Enum.take(@min_n_reject_null)

    if test_cases == [] do
      flunk(
        "No in-catalog test cases (catalog size #{state.num_items}). Build fixture with --limit 100 or more."
      )
    end

    metrics = Eval.evaluate(state, test_cases, top_k: @top_k)

    assert metrics.n >= 1
    assert metrics.catalog_size > 0
    assert metrics.hit_at_1 >= 0.0 and metrics.hit_at_1 <= 1.0
    assert metrics.random_hit_at_1 > 0.0

    assert metrics.rejects_null,
           "expected Hit@1 (#{metrics.hit_at_1}) > random_hit_at_1 (#{metrics.random_hit_at_1}) to reject null hypothesis"
  end

  @tag :integration
  @tag :eval
  @tag timeout: 600_000
  test "pretrained (on catalogue) does not regress vs zero-shot and rejects null (Steam top-k)" do
    # Compare zero-shot (base ckpt) vs pretrained (ckpt after pretrain on this catalogue).
    fixture = System.get_env("RECGPT_FIXTURE")
    zero_shot_ckpt = System.get_env("RECGPT_CKPT_ZEROSHOT")
    pretrained_ckpt = System.get_env("RECGPT_CKPT_EXPORT")
    test_file = System.get_env("RECGPT_TEST_SEQUENCES")

    unless is_binary(fixture) and fixture != "" and File.regular?(fixture) do
      flunk("Set RECGPT_FIXTURE to path to fixture.json.")
    end

    unless is_binary(zero_shot_ckpt) and zero_shot_ckpt != "" and File.dir?(zero_shot_ckpt) and
             File.regular?(Path.join(zero_shot_ckpt, "manifest.json")) do
      flunk(
        "Set RECGPT_CKPT_ZEROSHOT to zero-shot checkpoint (base model, no training on this catalogue)."
      )
    end

    unless is_binary(pretrained_ckpt) and pretrained_ckpt != "" and File.dir?(pretrained_ckpt) and
             File.regular?(Path.join(pretrained_ckpt, "manifest.json")) do
      flunk("Set RECGPT_CKPT_EXPORT to pretrained checkpoint (after pretrain on this catalogue).")
    end

    unless is_binary(test_file) and test_file != "" and File.regular?(test_file) do
      flunk("Set RECGPT_TEST_SEQUENCES to path to test_sequences.json.")
    end

    Application.ensure_all_started(:nx)

    assert {:ok, zero_shot_state} = Serve.load_state(fixture, zero_shot_ckpt, nil)
    assert {:ok, pretrained_state} = Serve.load_state(fixture, pretrained_ckpt, nil)
    assert {:ok, raw_cases} = Eval.load_test_cases(test_file)
    assert raw_cases != [], "need at least one test case"

    test_cases =
      raw_cases
      |> Eval.filter_to_catalog(zero_shot_state.num_items)
      |> Enum.take(@min_n_reject_null)

    if test_cases == [] do
      flunk(
        "No in-catalog test cases (catalog size #{zero_shot_state.num_items}). Build fixture with --limit 100 or more."
      )
    end

    zero_shot_metrics = Eval.evaluate(zero_shot_state, test_cases, top_k: @top_k)
    pretrained_metrics = Eval.evaluate(pretrained_state, test_cases, top_k: @top_k)

    assert pretrained_metrics.rejects_null,
           "pretrained (on catalogue) Hit@1 (#{pretrained_metrics.hit_at_1}) must be > random (#{pretrained_metrics.random_hit_at_1})"

    assert pretrained_metrics.hit_at_1 >= zero_shot_metrics.hit_at_1,
           "pretrained (on catalogue) Hit@1 (#{pretrained_metrics.hit_at_1}) must be >= zero-shot Hit@1 (#{zero_shot_metrics.hit_at_1}); " <>
             "when zero-shot equals baseline, pretrain on catalogue must improve or we fail"
  end
end
