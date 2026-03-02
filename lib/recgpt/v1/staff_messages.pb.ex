# Staff API message definitions (from staff.proto).
# Keep in sync with priv/proto/recgpt/v1/staff.proto

defmodule Recgpt.V1.CatalogItem do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.CatalogItem"

  field(:item_id, 1, type: :int32)
  field(:title, 2, type: :string)
end

defmodule Recgpt.V1.ListItemsRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.ListItemsRequest"

  field(:from_db, 1, type: :bool)
  field(:path, 2, type: :string)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.ListItemsResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.ListItemsResponse"

  field(:items, 1, repeated: true, type: Recgpt.V1.CatalogItem)
end

defmodule Recgpt.V1.GetItemRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.GetItemRequest"

  field(:item_id, 1, type: :int32)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.GetItemResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.GetItemResponse"

  field(:item_id, 1, type: :int32)
  field(:title, 2, type: :string)
  field(:found, 3, type: :bool)
end

defmodule Recgpt.V1.UpsertItemsRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.UpsertItemsRequest"

  field(:items, 1, repeated: true, type: Recgpt.V1.CatalogItem)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.UpsertItemsResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.UpsertItemsResponse"
end

defmodule Recgpt.V1.SyncItemsFromJsonRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.SyncItemsFromJsonRequest"

  field(:path, 1, type: :string)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.SyncItemsFromJsonResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.SyncItemsFromJsonResponse"
end

defmodule Recgpt.V1.WriteItemsJsonRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.WriteItemsJsonRequest"

  field(:path, 1, type: :string)
  field(:items, 2, repeated: true, type: Recgpt.V1.CatalogItem)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.WriteItemsJsonResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.WriteItemsJsonResponse"
end

defmodule Recgpt.V1.SyncSequencesRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.SyncSequencesRequest"

  field(:data_dir, 1, type: :string)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.SyncSequencesResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.SyncSequencesResponse"
end

defmodule Recgpt.V1.BuildFixtureRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.BuildFixtureRequest"

  field(:items_path, 1, type: :string)
  field(:ckpt_dir, 2, type: :string)
  field(:out_path, 3, type: :string)
  field(:limit, 4, type: :int32)
  field(:canonical_texts, 5, type: :bool)
  field(:vae_ckpt, 6, type: :string)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.BuildFixtureResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.BuildFixtureResponse"

  field(:num_items, 1, type: :int32)
  field(:out_path, 2, type: :string)
end

defmodule Recgpt.V1.WriteFixtureRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.WriteFixtureRequest"

  field(:path, 1, type: :string)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.WriteFixtureResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.WriteFixtureResponse"
end

defmodule Recgpt.V1.PretrainRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.PretrainRequest"

  field(:ckpt_dir, 1, type: :string)
  field(:fixture_path, 2, type: :string)
  field(:train_path, 3, type: :string)
  field(:items_path, 4, type: :string)
  field(:out_dir, 5, type: :string)
  field(:iterations, 6, type: :int32)
  field(:batch_size, 7, type: :int32)
  field(:learning_rate, 8, type: :double)
  field(:limit, 9, type: :int32)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.PretrainResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.PretrainResponse"
end

defmodule Recgpt.V1.CanonicalTextEntry do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.CanonicalTextEntry"

  field(:item_id, 1, type: :int32)
  field(:text, 2, type: :bytes)
end

defmodule Recgpt.V1.SetCanonicalTextsRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.SetCanonicalTextsRequest"

  field(:entries, 1, repeated: true, type: Recgpt.V1.CanonicalTextEntry)
  field(:rank, 15, type: :int32)
end

defmodule Recgpt.V1.SetCanonicalTextsResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.SetCanonicalTextsResponse"
end
