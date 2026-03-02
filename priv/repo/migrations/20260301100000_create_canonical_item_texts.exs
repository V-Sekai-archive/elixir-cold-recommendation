defmodule RecGPT.Repo.Migrations.CreateCanonicalItemTexts do
  use Ecto.Migration

  def change do
    # Canonical item text (RecGPT official str(dict).replace) per item_id; BLOB so bytes are not lost.
    create table(:canonical_item_texts, primary_key: false) do
      add :item_id, :integer, primary_key: true
      add :text, :binary, null: false
    end
  end
end
