defmodule RecGPT.Catalog.TrainSequenceRow do
  @moduledoc "ETNF: train_sequence_rows(seq_id, pos, item_id)."
  use Ecto.Schema

  @primary_key false
  schema "train_sequence_rows" do
    field(:seq_id, :integer)
    field(:pos, :integer)
    field(:item_id, :integer)
  end
end
