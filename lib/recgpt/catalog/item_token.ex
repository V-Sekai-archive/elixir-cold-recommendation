defmodule RecGPT.Catalog.ItemToken do
  @moduledoc "ETNF: item_tokens(item_id, t0..t3) — FSQ output for trie."
  use Ecto.Schema

  @primary_key {:item_id, :integer, autogenerate: false}
  schema "item_tokens" do
    field(:t0, :integer)
    field(:t1, :integer)
    field(:t2, :integer)
    field(:t3, :integer)
  end
end
