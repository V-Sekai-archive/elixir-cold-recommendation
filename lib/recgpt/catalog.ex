defmodule RecGPT.Catalog do
  @moduledoc """
  Atomic catalog/fixture file write for SSD-safe updates.
  Use when writing catalog or fixture JSON from the app (e.g. a task that generates or updates items).
  """
  @spec write!(String.t(), map() | binary()) :: :ok
  def write!(path, content) when is_map(content) do
    write!(path, Jason.encode!(content, pretty: true))
  end

  def write!(path, content) when is_binary(content) do
    tmp = path <> ".tmp"
    File.mkdir_p!(Path.dirname(path))
    {:ok, fd} = :file.open(tmp, [:write, :raw, :binary])
    :file.write(fd, content)
    :file.sync(fd)
    :file.close(fd)
    File.rename!(tmp, path)
    :ok
  end
end
