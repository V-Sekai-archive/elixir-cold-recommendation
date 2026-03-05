defmodule RecGPT.Catalog.ColdTestContext do
  @moduledoc "ETNF: cold_test_context(case_id, pos, item_id)."
  use Ecto.Schema

  @primary_key false
  schema "cold_test_context" do
    field(:case_id, :integer)
    field(:pos, :integer)
    field(:item_id, :integer)
    field(:time_ms, :integer)
  end
end
