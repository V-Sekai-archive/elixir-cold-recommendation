defmodule RecGPT.Repo.Migrations.CreateClickstreamTables do
  @moduledoc "UCI Clickstream (e-shop clothing): items + events for small sequential rec. CC BY 4.0."
  use Ecto.Migration

  def change do
    create table(:items, primary_key: false) do
      add :id, :integer, primary_key: true
      add :title, :string, null: false
      timestamps(type: :utc_datetime_usec)
    end

    create table(:events) do
      add :session_id, :integer, null: false
      add :ord, :integer, null: false
      add :item_id, :integer, null: false
      timestamps(type: :utc_datetime_usec)
    end

    create index(:events, [:session_id])
    create index(:events, [:session_id, :ord])
  end
end
