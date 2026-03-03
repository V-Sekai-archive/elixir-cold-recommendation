# StaffService gRPC server tests. Delegates to RecGPT.StaffApi (default impl may need Repo for list_items from DB).
defmodule Recgpt.V1.StaffServiceTest do
  use ExUnit.Case, async: false

  alias Recgpt.V1.StaffService.Server
  alias Recgpt.V1.{SyncSequencesRequest, UpsertItemsRequest}

  # gRPC status 3 = INVALID_ARGUMENT
  @invalid_argument_status 3

  describe "sync_sequences/2" do
    test "empty data_dir returns INVALID_ARGUMENT" do
      request = %SyncSequencesRequest{data_dir: ""}

      assert {:error, %GRPC.RPCError{status: @invalid_argument_status}} =
               Server.sync_sequences(request, nil)
    end

    test "nil data_dir returns INVALID_ARGUMENT" do
      request = %SyncSequencesRequest{data_dir: nil}

      assert {:error, %GRPC.RPCError{status: @invalid_argument_status}} =
               Server.sync_sequences(request, nil)
    end
  end

  describe "upsert_items/2" do
    test "empty items returns INVALID_ARGUMENT" do
      request = %UpsertItemsRequest{items: []}

      assert {:error, %GRPC.RPCError{status: @invalid_argument_status}} =
               Server.upsert_items(request, nil)
    end
  end
end
