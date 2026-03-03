defmodule Recgpt.V1.PredictResponse do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.PredictResponse"

  field(:item_ids, 1, repeated: true, type: :int32)
  field(:items, 2, repeated: true, type: Recgpt.V1.ItemSummary)
end
