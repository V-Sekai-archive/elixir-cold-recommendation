defmodule RecGPT.Repo.Migrations.CreateSteamEtnfTables do
  use Ecto.Migration

  def change do
    # ETNF: items(item_id PK, title) - catalog
    create table(:items, primary_key: false) do
      add :item_id, :integer, primary_key: true
      add :title, :text, null: false
    end

    # ETNF: item_embeddings(item_id PK, embedding BLOB) - 768-d per item
    create table(:item_embeddings, primary_key: false) do
      add :item_id, :integer, primary_key: true
      add :embedding, :binary, null: false
    end

    # ETNF: item_tokens(item_id PK, t0..t3) - FSQ output for trie
    create table(:item_tokens, primary_key: false) do
      add :item_id, :integer, primary_key: true
      add :t0, :integer, null: false
      add :t1, :integer, null: false
      add :t2, :integer, null: false
      add :t3, :integer, null: false
    end

    # ETNF: train_sequence_rows(seq_id, pos PK, item_id)
    create table(:train_sequence_rows, primary_key: false) do
      add :seq_id, :integer, null: false
      add :pos, :integer, null: false
      add :item_id, :integer, null: false
    end
    create unique_index(:train_sequence_rows, [:seq_id, :pos])

    create table(:cold_train_sequence_rows, primary_key: false) do
      add :seq_id, :integer, null: false
      add :pos, :integer, null: false
      add :item_id, :integer, null: false
    end
    create unique_index(:cold_train_sequence_rows, [:seq_id, :pos])

    # ETNF: test_cases(case_id PK, next_item); test_context(case_id, pos PK, item_id)
    create table(:test_cases, primary_key: false) do
      add :case_id, :integer, primary_key: true
      add :next_item, :integer, null: false
    end

    create table(:test_context, primary_key: false) do
      add :case_id, :integer, null: false
      add :pos, :integer, null: false
      add :item_id, :integer, null: false
    end
    create unique_index(:test_context, [:case_id, :pos])

    create table(:cold_test_cases, primary_key: false) do
      add :case_id, :integer, primary_key: true
      add :next_item, :integer, null: false
    end

    create table(:cold_test_context, primary_key: false) do
      add :case_id, :integer, null: false
      add :pos, :integer, null: false
      add :item_id, :integer, null: false
    end
    create unique_index(:cold_test_context, [:case_id, :pos])
  end
end
