defmodule Recgpt.V1.ItemSummary do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.ItemSummary"

  field(:item_id, 1, type: :int32)
  field(:display_name, 2, type: :string)
end
