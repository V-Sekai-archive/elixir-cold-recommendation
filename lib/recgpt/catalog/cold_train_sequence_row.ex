defmodule RecGPT.Catalog.ColdTrainSequenceRow do
  @moduledoc "ETNF: cold_train_sequence_rows(seq_id, pos, item_id)."
  use Ecto.Schema

  @primary_key false
  schema "cold_train_sequence_rows" do
    field(:seq_id, :integer)
    field(:pos, :integer)
    field(:item_id, :integer)
  end
end
