defmodule Recgpt.V1.StaffService.Service do
  @moduledoc false
  use GRPC.Service, name: "recgpt.v1.StaffService"

  rpc(:ListItems, Recgpt.V1.ListItemsRequest, Recgpt.V1.ListItemsResponse)
  rpc(:GetItem, Recgpt.V1.GetItemRequest, Recgpt.V1.GetItemResponse)
  rpc(:UpsertItems, Recgpt.V1.UpsertItemsRequest, Recgpt.V1.UpsertItemsResponse)
  rpc(:SyncItemsFromJson, Recgpt.V1.SyncItemsFromJsonRequest, Recgpt.V1.SyncItemsFromJsonResponse)
  rpc(:WriteItemsJson, Recgpt.V1.WriteItemsJsonRequest, Recgpt.V1.WriteItemsJsonResponse)
  rpc(:SyncSequences, Recgpt.V1.SyncSequencesRequest, Recgpt.V1.SyncSequencesResponse)
  rpc(:BuildFixture, Recgpt.V1.BuildFixtureRequest, Recgpt.V1.BuildFixtureResponse)
  rpc(:WriteFixture, Recgpt.V1.WriteFixtureRequest, Recgpt.V1.WriteFixtureResponse)
  rpc(:Pretrain, Recgpt.V1.PretrainRequest, Recgpt.V1.PretrainResponse)
  rpc(:SetCanonicalTexts, Recgpt.V1.SetCanonicalTextsRequest, Recgpt.V1.SetCanonicalTextsResponse)
end

defmodule Recgpt.V1.StaffService.Server do
  @moduledoc """
  gRPC server for RecGPT StaffService. Delegates to RecGPT.StaffApi.
  """
  use GRPC.Server, service: Recgpt.V1.StaffService.Service

  alias Recgpt.V1.{
    BuildFixtureResponse,
    CatalogItem,
    GetItemResponse,
    ListItemsResponse,
    PretrainResponse,
    SetCanonicalTextsResponse,
    SyncItemsFromJsonResponse,
    SyncSequencesResponse,
    UpsertItemsResponse,
    WriteItemsJsonResponse
  }

  def list_items(request, _stream) do
    source = if request.path != nil and request.path != "", do: {:path, request.path}, else: :db

    case RecGPT.StaffApi.list_items(source) do
      {:ok, items} ->
        catalog_items =
          Enum.map(items, fn m ->
            %CatalogItem{
              item_id: m[:item_id] || m["item_id"],
              title: m[:title] || m["title"] || ""
            }
          end)

        %ListItemsResponse{items: catalog_items}

      {:error, reason} ->
        {:error,
         GRPC.RPCError.exception(
           status: :internal,
           message: "ListItems failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
         )}
    end
  end

  def get_item(request, _stream) do
    item_id = request.item_id || 0

    case RecGPT.StaffApi.get_item(item_id) do
      {:ok, nil} ->
        %GetItemResponse{item_id: item_id, title: "", found: false}

      {:ok, item} ->
        title = item[:title] || item["title"] || ""
        %GetItemResponse{item_id: item_id, title: title, found: true}

      {:error, reason} ->
        {:error,
         GRPC.RPCError.exception(
           status: :internal,
           message: "GetItem failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
         )}
    end
  end

  def upsert_items(request, _stream) do
    entries =
      (request.items || []) |> Enum.map(fn c -> %{item_id: c.item_id, title: c.title || ""} end)

    if entries == [] do
      {:error,
       GRPC.RPCError.exception(status: :invalid_argument, message: "items must not be empty")}
    else
      case RecGPT.StaffApi.upsert_items(entries) do
        :ok ->
          %UpsertItemsResponse{}

        {:error, reason} ->
          {:error,
           GRPC.RPCError.exception(
             status: :internal,
             message: "UpsertItems failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
           )}
      end
    end
  end

  def sync_items_from_json(request, _stream) do
    path = request.path || ""

    if path == "" do
      {:error, GRPC.RPCError.exception(status: :invalid_argument, message: "path is required")}
    else
      case RecGPT.StaffApi.sync_items_from_json(path) do
        :ok ->
          %SyncItemsFromJsonResponse{}

        {:error, reason} ->
          {:error,
           GRPC.RPCError.exception(
             status: :internal,
             message: "SyncItemsFromJson failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
           )}
      end
    end
  end

  def write_items_json(request, _stream) do
    path = request.path || ""

    items =
      (request.items || [])
      |> Enum.map(fn c -> %{"id" => c.item_id, "title" => c.title || ""} end)

    if path == "" do
      {:error, GRPC.RPCError.exception(status: :invalid_argument, message: "path is required")}
    else
      case RecGPT.StaffApi.write_items_json(path, items) do
        :ok ->
          %WriteItemsJsonResponse{}

        {:error, reason} ->
          {:error,
           GRPC.RPCError.exception(
             status: :internal,
             message: "WriteItemsJson failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
           )}
      end
    end
  end

  def sync_sequences(request, _stream) do
    data_dir = request.data_dir || ""

    if data_dir == "" do
      {:error,
       GRPC.RPCError.exception(status: :invalid_argument, message: "data_dir is required")}
    else
      case RecGPT.StaffApi.sync_sequences(data_dir) do
        :ok ->
          %SyncSequencesResponse{}

        {:error, reason} ->
          {:error,
           GRPC.RPCError.exception(
             status: :internal,
             message: "SyncSequences failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
           )}
      end
    end
  end

  def build_fixture(request, _stream) do
    items_path = request.items_path || ""
    ckpt_dir = request.ckpt_dir || ""

    if items_path == "" or ckpt_dir == "" do
      {:error,
       GRPC.RPCError.exception(
         status: :invalid_argument,
         message: "items_path and ckpt_dir are required"
       )}
    else
      opts = []

      opts =
        if request.limit != nil and request.limit > 0,
          do: Keyword.put(opts, :limit, request.limit),
          else: opts

      opts =
        if request.canonical_texts == true,
          do: Keyword.put(opts, :canonical_texts, true),
          else: opts

      opts =
        if request.vae_ckpt != nil and request.vae_ckpt != "",
          do: Keyword.put(opts, :vae_ckpt, request.vae_ckpt),
          else: opts

      case RecGPT.StaffApi.build_fixture(items_path, ckpt_dir, opts) do
        {:ok, fixture} ->
          num_items = fixture["num_items"] || 0
          out_path = request.out_path || ""

          if out_path != "" do
            case RecGPT.StaffApi.write_fixture(fixture, out_path) do
              :ok ->
                %BuildFixtureResponse{num_items: num_items, out_path: out_path}

              {:error, reason} ->
                {:error,
                 GRPC.RPCError.exception(
                   status: :internal,
                   message: "WriteFixture failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
                 )}
            end
          else
            %BuildFixtureResponse{num_items: num_items, out_path: ""}
          end

        {:error, reason} ->
          {:error,
           GRPC.RPCError.exception(
             status: :internal,
             message: "BuildFixture failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
           )}
      end
    end
  end

  def write_fixture(request, _stream) do
    path = request.path || ""

    if path == "" do
      {:error, GRPC.RPCError.exception(status: :invalid_argument, message: "path is required")}
    else
      {:error,
       GRPC.RPCError.exception(
         status: :unimplemented,
         message:
           "WriteFixture requires a prior BuildFixture with out_path; use BuildFixture with out_path set instead"
       )}
    end
  end

  def pretrain(request, _stream) do
    ckpt_dir = request.ckpt_dir || ""
    fixture_path = request.fixture_path || ""
    train_path = request.train_path || ""
    items_path = request.items_path || ""
    out_dir = request.out_dir || ""

    if ckpt_dir == "" or fixture_path == "" or train_path == "" or items_path == "" or
         out_dir == "" do
      {:error,
       GRPC.RPCError.exception(
         status: :invalid_argument,
         message: "ckpt_dir, fixture_path, train_path, items_path, out_dir are required"
       )}
    else
      opts =
        [
          ckpt_dir: ckpt_dir,
          fixture_path: fixture_path,
          train_path: train_path,
          items_path: items_path,
          out_dir: out_dir,
          iterations: request.iterations || 100,
          batch_size: request.batch_size || 8,
          learning_rate: request.learning_rate || 1.0e-4,
          limit: request.limit
        ]
        |> Enum.reject(fn {_k, v} -> v == nil end)

      case RecGPT.StaffApi.pretrain(opts) do
        :ok ->
          %PretrainResponse{}

        {:error, reason} ->
          {:error,
           GRPC.RPCError.exception(
             status: :internal,
             message: "Pretrain failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
           )}
      end
    end
  end

  def set_canonical_texts(request, _stream) do
    entries =
      (request.entries || []) |> Enum.map(fn e -> %{item_id: e.item_id, text: e.text || <<>>} end)

    case RecGPT.StaffApi.set_canonical_texts(entries) do
      :ok ->
        %SetCanonicalTextsResponse{}

      {:error, reason} ->
        {:error,
         GRPC.RPCError.exception(
           status: :internal,
           message: "SetCanonicalTexts failed: #{RecGPT.SafeInspect.safe_inspect(reason)}"
         )}
    end
  end
end
