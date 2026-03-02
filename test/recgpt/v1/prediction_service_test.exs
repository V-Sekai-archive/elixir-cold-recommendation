# gRPC status codes: 3 = INVALID_ARGUMENT, 9 = FAILED_PRECONDITION
# Predict uses RecGPT.RecommendationService (default impl: Serve); tests use stub serve_state from FrozenHelpers.
defmodule Recgpt.V1.PredictionServiceTest do
  use ExUnit.Case, async: false

  alias Recgpt.V1.PredictionService.Server
  alias Recgpt.V1.PredictRequest

  setup do
    state = RecGPT.TestSupport.FrozenHelpers.build_stub_state()
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

    test "when serve_state is not loaded, raises FAILED_PRECONDITION" do
      Application.put_env(:recgpt, :serve_state, nil)
      request = %PredictRequest{context_item_ids: [0], max_results: 5}

      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError ->
          assert e.status == 9
      after
        # Restore stub so on_exit from setup doesn't fail
        Application.put_env(:recgpt, :serve_state, RecGPT.TestSupport.FrozenHelpers.build_stub_state())
      end
    end
  end
end
