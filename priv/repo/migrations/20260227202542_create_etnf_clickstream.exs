defmodule RecGPT.Repo.Migrations.CreateEtnfClickstream do
  @moduledoc "ETNF/BCNF catalog_item + event; view catalog_item_xmp_jsonld for Dublin Core XMP JSON-LD."
  use Ecto.Migration

  def change do
    create table(:catalog_item, primary_key: false) do
      add :item_id, :integer, null: false, primary_key: true
      add :source_dataset, :string, null: false, primary_key: true
      add :dc_title, :string, null: false
      add :dc_description, :string, null: false
      add :dc_type, :string
      add :dcterms_source, :string, null: false
      timestamps(type: :utc_datetime_usec)
    end

    create table(:event, primary_key: false) do
      add :session_id, :integer, null: false, primary_key: true
      add :ord, :integer, null: false, primary_key: true
      add :item_id, :integer, null: false
      add :source_dataset, :string, null: false
      timestamps(type: :utc_datetime_usec)
    end

    create index(:event, [:session_id])
    create index(:event, [:session_id, :ord])
    create index(:event, [:item_id])

    execute(
      """
      CREATE VIEW catalog_item_xmp_jsonld AS
      SELECT
        item_id,
        source_dataset,
        json_object(
          '@context', json_object('dc', 'http://purl.org/dc/elements/1.1/', 'dcterms', 'http://purl.org/dc/terms/'),
          'dc:title', dc_title,
          'dc:description', dc_description,
          'dc:identifier', source_dataset || ':' || item_id,
          'dcterms:source', dcterms_source
        ) AS item_xmp_jsonld
      FROM catalog_item
      """,
      "DROP VIEW catalog_item_xmp_jsonld"
    )
  end
end
