defmodule RecGPT.Catalog.ColdTestCase do
  @moduledoc "ETNF: cold_test_cases(case_id, next_item)."
  use Ecto.Schema

  @primary_key {:case_id, :integer, autogenerate: false}
  schema "cold_test_cases" do
    field(:next_item, :integer)
  end
end
