defmodule RecGPT.Repo.Migrations.CreateItemEmbeddingTexts do
  use Ecto.Migration

  def up do
    create table(:item_embedding_texts, primary_key: false) do
      add :item_id, :integer, primary_key: true
      add :embedding_text, :text, null: false
    end
  end

  def down do
    drop table(:item_embedding_texts)
  end
end
