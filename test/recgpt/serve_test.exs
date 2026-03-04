defmodule RecGPT.ServeTest do
  use ExUnit.Case, async: false

  alias RecGPT.LayerFreeze
  alias RecGPT.Serve
  alias RecGPT.TestSupport.FrozenHelpers

  describe "item_ids_to_context_token_ids/3" do
    test "left-pads to seq_token_capacity 1024" do
      token_id_list = for _ <- 1..10, do: [1, 2, 3, 4]
      ids = [0, 1]
      context = Serve.item_ids_to_context_token_ids(ids, token_id_list)
      assert length(context) == 1024
      # First tokens should be padding (15360), then 8 tokens from 2 items
      padding_count = 1024 - 8
      assert Enum.take(context, padding_count) == List.duplicate(15_360, padding_count)
      assert Enum.drop(context, padding_count) == [1, 2, 3, 4, 1, 2, 3, 4]
    end

    test "truncates to max_length items (255)" do
      token_id_list = for _ <- 1..300, do: [0, 0, 0, 0]
      ids = Enum.to_list(0..299)
      context = Serve.item_ids_to_context_token_ids(ids, token_id_list)
      assert length(context) == 1024
    end
  end

  describe "recommend/3" do
    test "returns error when item_ids empty" do
      frozen = FrozenHelpers.build_frozen([0])
      assert LayerFreeze.recommend(frozen, [], 5) == {:error, "item_ids must be non-empty"}
    end

    test "returns up to top_k recommended item_ids (best first) via frozen inputs" do
      frozen = FrozenHelpers.build_frozen([0])
      assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 5)
      assert is_list(list)
      assert length(list) <= 5
      assert length(list) <= 2, "stub catalog has 2 items"
      Enum.each(list, fn id -> assert id in [0, 1] end)
    end

    test "top_k=1 returns at most one item" do
      frozen = FrozenHelpers.build_frozen([0])
      assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 1)
      assert length(list) <= 1
    end

    test "top_k is capped at 20" do
      frozen = FrozenHelpers.build_frozen([0])
      assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 100)
      assert length(list) <= 20
    end
  end

  describe "LayerFreeze (frozen inputs for layer isolation)" do
    test "forward_model with frozen params returns logits (Model layer in isolation)" do
      frozen = FrozenHelpers.build_frozen([0])
      token_list = Enum.take(frozen.context_token_ids, 8)
      logits = LayerFreeze.forward_model(frozen, token_list)
      assert Nx.shape(logits) == {1, 15_361}
    end

    test "recommend via frozen matches Serve.recommend with same state" do
      state = FrozenHelpers.build_stub_state()
      frozen = LayerFreeze.record_from_state(state, [0])
      assert {:ok, a} = LayerFreeze.recommend(frozen, [0], 5)
      assert {:ok, b} = Serve.recommend(state, [0], 5)
      assert a == b
    end
  end

  describe "load_state/3" do
    test "returns error when fixture missing" do
      assert {:error, _} =
               Serve.load_state("/nonexistent/fixture.json", "data/recgpt_ckpt_export", nil)
    end

    test "returns error when checkpoint dir has no manifest" do
      dir =
        Path.join(System.tmp_dir!(), "recgpt_no_manifest_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(dir)

      fixture =
        Path.join(System.tmp_dir!(), "recgpt_fixture_#{:erlang.unique_integer([:positive])}.json")

      File.write!(fixture, Jason.encode!(%{"token_id_list" => [[1, 2, 3, 4]], "num_items" => 1}))

      try do
        assert {:error, _} = Serve.load_state(fixture, dir, nil)
      after
        File.rm_rf(dir)
        File.rm(fixture)
      end
    end

    @tag :integration
    test "loads state when fixture and checkpoint exist" do
      fixture = Path.expand("../data/serve_e2e_fixture.json", File.cwd!())
      ckpt = Path.expand("../data/recgpt_ckpt_export", File.cwd!())

      if File.regular?(fixture) and File.regular?(Path.join(ckpt, "manifest.json")) do
        assert {:ok, state} = Serve.load_state(fixture, ckpt, nil)
        assert state.num_items > 0
        assert is_list(state.token_id_list)
      else
        raise "Skip: need data/serve_e2e_fixture.json and data/recgpt_ckpt_export (or run Serve E2E from your local setup; see CONTRIBUTING)"
      end
    end

    test "load_state with built fixture and stub checkpoint returns state; recommend returns valid item IDs" do
      Application.ensure_all_started(:nx)
      Application.put_env(:recgpt, :ckpt_expected_sha256, nil)

      base =
        Path.join(System.tmp_dir!(), "recgpt_serve_built_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(base)
      fixture_path = Path.join(base, "fixture.json")
      ckpt_dir = Path.join(base, "ckpt")

      try do
        num_items = 2
        token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]

        File.write!(
          fixture_path,
          Jason.encode!(%{"num_items" => num_items, "token_id_list" => token_id_list})
        )

        FrozenHelpers.write_stub_ckpt!(ckpt_dir)

        assert {:ok, state} = Serve.load_state(fixture_path, ckpt_dir, nil)
        assert state.num_items == num_items
        frozen = LayerFreeze.record_from_state(state, [0])
        assert {:ok, list} = LayerFreeze.recommend(frozen, [0], 5)
        assert is_list(list)
        assert length(list) <= 5
        assert Enum.all?(list, fn id -> is_integer(id) and id >= 0 and id < num_items end)
      after
        File.rm_rf(base)
      end
    end

    @tag timeout: 180_000
    test "load_state with FuXi checkpoint uses FuxiLinearInferenceDefn; recommend returns valid item IDs" do
      Application.ensure_all_started(:nx)
      Application.put_env(:recgpt, :ckpt_expected_sha256, nil)

      base =
        Path.join(System.tmp_dir!(), "recgpt_serve_fuxi_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(base)
      fixture_path = Path.join(base, "fixture.json")
      ckpt_dir = Path.join(base, "ckpt")

      try do
        num_items = 2
        token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]

        File.write!(
          fixture_path,
          Jason.encode!(%{"num_items" => num_items, "token_id_list" => token_id_list})
        )

        FrozenHelpers.write_fuxi_stub_ckpt!(ckpt_dir)

        assert {:ok, state} = Serve.load_state(fixture_path, ckpt_dir, nil)
        assert state.num_items == num_items
        assert {:ok, list} = Serve.recommend(state, [0], 5)
        assert is_list(list)
        assert length(list) <= 5
        assert Enum.all?(list, fn id -> is_integer(id) and id >= 0 and id < num_items end)
      after
        File.rm_rf(base)
      end
    end

    @tag :integration
    test "pipeline: load_state + recommend + Eval.evaluate returns metrics" do
      Application.ensure_all_started(:nx)

      base =
        Path.join(System.tmp_dir!(), "recgpt_serve_eval_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(base)
      fixture_path = Path.join(base, "fixture.json")
      ckpt_dir = Path.join(base, "ckpt")

      try do
        num_items = 2
        token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]

        File.write!(
          fixture_path,
          Jason.encode!(%{"num_items" => num_items, "token_id_list" => token_id_list})
        )

        FrozenHelpers.write_stub_ckpt!(ckpt_dir)

        assert {:ok, state} = Serve.load_state(fixture_path, ckpt_dir, nil)

        test_cases = [
          %{"context" => [0], "next_item" => 1},
          %{"context" => [1], "next_item" => 0}
        ]

        metrics = RecGPT.Eval.evaluate(state, test_cases, top_k: 10)
        assert metrics.n == 2
        assert metrics.catalog_size == 2
        assert is_float(metrics.hit_at_1)
        assert is_float(metrics.mrr)
      after
        File.rm_rf(base)
      end
    end
  end
end
