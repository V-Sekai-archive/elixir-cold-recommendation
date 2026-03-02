defmodule RecGPT.ArtifactUploader do
  @moduledoc """
  Waffle definition for blob/artifact storage (fixture refs, checkpoint refs, exports).
  Use with Ecto via `RecGPT.Catalog.Artifact` and `cast_attachments/3`.
  """
  use Waffle.Definition
  use Waffle.Ecto.Definition

  @versions [:original]

  def filename(version, {file, _scope}) do
    "#{version}-#{file.file_name}"
  end

  def storage_dir(_version, {_file, scope}) do
    base = "uploads/artifacts"
    if is_struct(scope) and Map.get(scope, :id) do
      "#{base}/#{scope.id}"
    else
      "#{base}/temp"
    end
  end
end
