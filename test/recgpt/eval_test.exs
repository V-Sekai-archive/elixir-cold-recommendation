# Eval test: load_test_cases with synthetic data; evaluate with stub state; Fetch output (top-k).
# Integration: build data/steam with mix recgpt.fetch_steam data/steam.
defmodule RecGPT.EvalTest do
  use ExUnit.Case, async: false

  alias RecGPT.Eval
  alias RecGPT.Serve

  @top_k 10

  describe "evaluate/3" do
    test "returns metrics with stub state and test cases" do
      state = build_eval_stub_state()
      test_cases = [%{"context" => [0], "next_item" => 0}, %{"context" => [0], "next_item" => 1}]
      metrics = Eval.evaluate(state, test_cases, top_k: 5)
      assert metrics.n >= 1
      assert metrics.hit_at_1 >= 0.0 and metrics.hit_at_1 <= 1.0
      assert metrics.hit_at_5 >= 0.0 and metrics.hit_at_5 <= 1.0
      assert metrics.mrr >= 0.0 and metrics.mrr <= 1.0
      assert metrics.catalog_size == 2
      assert metrics.random_hit_at_1 == 0.5
      assert is_boolean(metrics.rejects_null)
    end

    test "skips test cases with empty context or nil next_item" do
      state = build_eval_stub_state()

      test_cases = [
        %{"context" => [], "next_item" => 0},
        %{"context" => [0], "next_item" => nil}
      ]

      metrics = Eval.evaluate(state, test_cases, top_k: 5)
      assert metrics.n == 0 or metrics.n == 1
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

    if not File.regular?(path) do
      ExUnit.skip(
        "Run mix recgpt.fetch_steam data/steam to build data/steam/test_sequences.json"
      )
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
  test "load_test_cases loads synthetic test_sequences JSON" do
    num_items = 20
    payload = RecGPT.EvalFixtures.generate_test_sequences_json(num_items, 15)

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
  @tag timeout: 60_000
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
      Skipped (missing data): Set RECGPT_CKPT_EXPORT to checkpoint export dir. See docs/02_recgpt_checkpoint_layout.md.
      """)
    end

    unless is_binary(test_file) and test_file != "" and File.regular?(test_file) do
      flunk("""
      Skipped (missing data): Set RECGPT_TEST_SEQUENCES to path to test_sequences.json (test_cases + num_items).
      Use synthetic data: RecGPT.EvalFixtures.generate_test_sequences_json/3; see docs/06_eval_data_shapes.md.
      """)
    end

    Application.ensure_all_started(:nx)

    assert {:ok, state} = Serve.load_state(fixture, ckpt, nil)
    assert {:ok, test_cases} = Eval.load_test_cases(test_file)
    assert test_cases != [], "need at least one test case"

    metrics = Eval.evaluate(state, test_cases, top_k: @top_k)

    assert metrics.n >= 1
    assert metrics.catalog_size > 0
    assert metrics.hit_at_1 >= 0.0 and metrics.hit_at_1 <= 1.0
    assert metrics.random_hit_at_1 > 0.0

    assert metrics.rejects_null,
           "expected Hit@1 (#{metrics.hit_at_1}) > random_hit_at_1 (#{metrics.random_hit_at_1}) to reject null hypothesis"
  end

  defp build_eval_stub_state do
    token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
    trie = RecGPT.Trie.build(token_id_list)

    params = %{
      "wte" => Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
      "pred_head.weight" =>
        Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
      "pred_head.bias" => Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    }

    get_logits_fn = fn token_list ->
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      seq_len = length(token_list)
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      RecGPT.Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end

    %Serve{
      params: params,
      trie: trie,
      token_id_list: token_id_list,
      item_text: %{},
      num_items: 2,
      get_logits_fn: get_logits_fn
    }
  end
end
