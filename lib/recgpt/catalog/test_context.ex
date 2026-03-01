defmodule RecGPT.Catalog.TestContext do
  @moduledoc "ETNF: test_context(case_id, pos, item_id)."
  use Ecto.Schema

  @primary_key false
  schema "test_context" do
    field(:case_id, :integer)
    field(:pos, :integer)
    field(:item_id, :integer)
  end
end
