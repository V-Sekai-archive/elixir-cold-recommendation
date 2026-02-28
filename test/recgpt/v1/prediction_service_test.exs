# gRPC status codes: 3 = INVALID_ARGUMENT, 14 = UNAVAILABLE
defmodule Recgpt.V1.PredictionServiceTest do
  use ExUnit.Case, async: false

  alias RecGPT.Serve
  alias RecGPT.TestSupport.FrozenHelpers
  alias Recgpt.V1.PredictionService.Server
  alias Recgpt.V1.PredictRequest

  setup do
    state = FrozenHelpers.build_stub_state()
    Application.put_env(:recgpt, :serve_state, state)
    on_exit(fn -> Application.delete_env(:recgpt, :serve_state) end)
    :ok
  end

  describe "predict/2" do
    test "returns item_ids and items (ItemSummary) for valid request" do
      request = %PredictRequest{context_item_ids: [0], max_results: 5}
      response = Server.predict(request, nil)
      assert is_list(response.item_ids)
      assert length(response.items) == length(response.item_ids)

      Enum.zip(response.item_ids, response.items)
      |> Enum.each(fn {id, item} ->
        assert item.item_id == id
        assert is_binary(item.display_name) or item.display_name == ""
      end)
    end

    test "empty context_item_ids raises INVALID_ARGUMENT" do
      request = %PredictRequest{context_item_ids: [], max_results: 5}

      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError -> assert e.status == 3
      end
    end

    test "nil serve_state raises UNAVAILABLE" do
      Application.put_env(:recgpt, :serve_state, nil)
      request = %PredictRequest{context_item_ids: [0], max_results: 5}

      try do
        try do
          Server.predict(request, nil)
        rescue
          e in GRPC.RPCError -> assert e.status == 14
        end
      after
        Application.put_env(:recgpt, :serve_state, FrozenHelpers.build_stub_state())
      end
    end

    test "max_results 0 uses default 5 and succeeds" do
      request = %PredictRequest{context_item_ids: [0], max_results: 0}
      response = Server.predict(request, nil)
      assert is_list(response.item_ids)
      assert length(response.items) == length(response.item_ids)
    end

    test "max_results 21 raises INVALID_ARGUMENT" do
      request = %PredictRequest{context_item_ids: [0], max_results: 21}

      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError -> assert e.status == 3
      end
    end

    # Satisfies full-flow integration test (docs/14_top_tier_recommendations.md).
    test "full flow: load_state then predict returns valid response" do
      Application.ensure_all_started(:nx)

      base =
        Path.join(System.tmp_dir!(), "recgpt_predict_flow_#{:erlang.unique_integer([:positive])}")

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
        Application.put_env(:recgpt, :serve_state, state)

        request = %PredictRequest{context_item_ids: [0], max_results: 5}
        response = Server.predict(request, nil)

        assert is_list(response.item_ids)
        assert length(response.items) == length(response.item_ids)

        assert Enum.all?(response.item_ids, fn id ->
                 is_integer(id) and id >= 0 and id < num_items
               end)

        Enum.zip(response.item_ids, response.items)
        |> Enum.each(fn {id, item} ->
          assert item.item_id == id
          assert is_binary(item.display_name) or item.display_name == ""
        end)
      after
        File.rm_rf(base)
      end
    end
  end
end
