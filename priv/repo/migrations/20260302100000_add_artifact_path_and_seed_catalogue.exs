defmodule RecGPT.Repo.Migrations.AddArtifactPathAndSeedCatalogue do
  use Ecto.Migration

  def up do
    alter table(:artifacts) do
      add :path, :string
    end

    # Seed existing pipeline artifact kinds (name + default path)
    execute """
    INSERT INTO artifacts (name, path, inserted_at, updated_at) VALUES
      ('fixture', 'data/steam/fixture.json', datetime('now'), datetime('now')),
      ('checkpoint', 'data/recgpt_ckpt_export', datetime('now'), datetime('now')),
      ('train_sequences', 'data/steam/train_sequences.json', datetime('now'), datetime('now')),
      ('cold_train_sequences', 'data/steam/cold_train_sequences.json', datetime('now'), datetime('now')),
      ('test_sequences', 'data/steam/test_sequences.json', datetime('now'), datetime('now')),
      ('cold_test_sequences', 'data/steam/cold_test_sequences.json', datetime('now'), datetime('now')),
      ('items', 'data/steam/items.json', datetime('now'), datetime('now'))
    """
  end

  def down do
    execute """
    DELETE FROM artifacts WHERE name IN (
      'fixture', 'checkpoint', 'train_sequences', 'cold_train_sequences',
      'test_sequences', 'cold_test_sequences', 'items'
    )
    """
    alter table(:artifacts) do
      remove :path
    end
  end
end
