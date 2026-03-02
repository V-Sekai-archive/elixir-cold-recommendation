# StaffService gRPC server tests. Delegates to RecGPT.StaffApi (default impl may need Repo for list_items from DB).
defmodule Recgpt.V1.StaffServiceTest do
  use ExUnit.Case, async: false

  alias Recgpt.V1.StaffService.Server
  alias Recgpt.V1.{SyncSequencesRequest, UpsertItemsRequest}

  describe "sync_sequences/2" do
    test "empty data_dir raises INVALID_ARGUMENT" do
      request = %SyncSequencesRequest{data_dir: ""}

      assert_raise GRPC.RPCError, fn ->
        Server.sync_sequences(request, nil)
      end
    end

    test "nil data_dir raises INVALID_ARGUMENT" do
      request = %SyncSequencesRequest{data_dir: nil}

      assert_raise GRPC.RPCError, fn ->
        Server.sync_sequences(request, nil)
      end
    end
  end

  describe "upsert_items/2" do
    test "empty items raises INVALID_ARGUMENT" do
      request = %UpsertItemsRequest{items: []}

      assert_raise GRPC.RPCError, fn ->
        Server.upsert_items(request, nil)
      end
    end
  end
end
