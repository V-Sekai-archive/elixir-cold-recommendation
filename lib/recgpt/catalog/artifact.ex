defmodule RecGPT.Catalog.Artifact do
  @moduledoc """
  Ecto schema for blob/artifact storage with Waffle (local or S3).
  Stores a name and an optional file attachment; use `cast_attachments/3` in changesets.
  Optional `path` stores a default filesystem path for pipeline artifacts (fixture, checkpoint, etc.).
  Use `resolve_path/2` to get the path for a named artifact from the catalogue (then fall back to env/defaults).
  """
  use Ecto.Schema
  use Waffle.Ecto.Schema

  schema "artifacts" do
    field :name, :string
    field :path, :string
    field :file, RecGPT.ArtifactUploader.Type
    timestamps()
  end

  @doc """
  Resolve the filesystem path for a named artifact from the catalogue.
  Returns an absolute path (expanded with cwd) if the artifact exists and has a path; otherwise nil.
  When the app has no Repo or the artifact is missing, returns nil so callers can fall back to env/defaults.
  """
  @spec resolve_path(String.t(), keyword()) :: String.t() | nil
  def resolve_path(name, opts \\ []) when is_binary(name) do
    repo = Keyword.get(opts, :repo, RecGPT.Repo)
    cwd = Keyword.get(opts, :cwd, File.cwd!())

    try do
      case repo.get_by(__MODULE__, name: name) do
        nil -> nil
        %__MODULE__{path: nil} -> nil
        %__MODULE__{path: path} when is_binary(path) ->
          if absolute_path?(path), do: path, else: Path.expand(path, cwd)
      end
    rescue
      _ -> nil
    end
  end

  defp absolute_path?(p), do: String.starts_with?(p, "/") or p =~ ~r/^[a-zA-Z]:/

  @doc "Known artifact kinds used by the pipeline (fixture, checkpoint, train, etc.)."
  def default_artifact_kinds do
    [
      {"fixture", "data/steam/fixture.json"},
      {"checkpoint", "data/recgpt_ckpt_export"},
      {"train_sequences", "data/steam/train_sequences.json"},
      {"cold_train_sequences", "data/steam/cold_train_sequences.json"},
      {"test_sequences", "data/steam/test_sequences.json"},
      {"cold_test_sequences", "data/steam/cold_test_sequences.json"},
      {"items", "data/steam/items.json"}
    ]
  end

  def changeset(artifact, params \\ %{}) do
    artifact
    |> Ecto.Changeset.cast(params, [:name, :path])
    |> cast_attachments(params, [:file])
    |> Ecto.Changeset.validate_required([:name])
  end
end
