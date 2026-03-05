defmodule RecGPT.Repo.Migrations.AddTimeMsToSequenceTables do
  use Ecto.Migration

  def change do
    alter table(:train_sequence_rows) do
      add :time_ms, :bigint
    end

    alter table(:cold_train_sequence_rows) do
      add :time_ms, :bigint
    end

    alter table(:test_context) do
      add :time_ms, :bigint
    end

    alter table(:cold_test_context) do
      add :time_ms, :bigint
    end
  end
end
