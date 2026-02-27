defmodule RecGPT.Repo.Migrations.DropLegacyClickstreamTables do
  @moduledoc "Drop legacy items/events tables; eval uses ETNF catalog_item + event only."
  use Ecto.Migration

  def change do
    drop_if_exists table(:events)
    drop_if_exists table(:items)
  end
end
