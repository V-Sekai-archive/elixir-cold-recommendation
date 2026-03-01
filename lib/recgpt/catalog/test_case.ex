defmodule RecGPT.Catalog.TestCase do
  @moduledoc "ETNF: test_cases(case_id, next_item)."
  use Ecto.Schema

  @primary_key {:case_id, :integer, autogenerate: false}
  schema "test_cases" do
    field(:next_item, :integer)
  end
end
