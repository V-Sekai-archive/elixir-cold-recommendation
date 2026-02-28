defmodule Recgpt.V1.PredictRequest do
  @moduledoc false
  use Protobuf,
    syntax: :proto3,
    full_name: "recgpt.v1.PredictRequest"

  field(:context_item_ids, 1, repeated: true, type: :int32)
  field(:max_results, 2, type: :int32)
end
