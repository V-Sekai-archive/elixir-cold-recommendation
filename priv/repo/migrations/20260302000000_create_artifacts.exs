defmodule RecGPT.Repo.Migrations.CreateArtifacts do
  use Ecto.Migration

  def change do
    create table(:artifacts) do
      add :name, :string, null: false
      add :file, :string
      timestamps()
    end
  end
end
