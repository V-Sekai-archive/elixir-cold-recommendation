defmodule Mix.Tasks.Recgpt.DebugPt do
  @shortdoc "Debug .pt zip: list entries and find which one fails CRC"
  use Mix.Task

  @impl true
  def run([]), do: Mix.raise("Usage: mix recgpt.debug_pt path/to/file.pt")
  def run([path]) when is_binary(path), do: run_pt(path)
  def run(_), do: Mix.raise("Usage: mix recgpt.debug_pt path/to/file.pt")

  defp run_pt(path) do
    unless File.regular?(path), do: Mix.raise("Not a file: #{path}")
    binary = File.read!(path)
    magic = binary_part(binary, 0, min(4, byte_size(binary)))
    if magic != <<0x50, 0x4B, 0x03, 0x04>>, do: Mix.raise("Not a zip file")
    {:ok, unzip} = Unzip.new(binary)
    entries = Unzip.list_entries(unzip)
    Mix.shell().info("Zip entries: #{length(entries)}")
    for e <- entries do
      Mix.shell().info("  #{e.file_name}  comp=#{e.compressed_size}  uncomp=#{e.uncompressed_size}")
    end
    Mix.shell().info("")
    for {p, _} <- unzip.cd_list do
      try do
        unzip |> Unzip.file_stream!(p) |> Enum.reduce(0, fn c, a -> a + byte_size(IO.iodata_to_binary(c)) end)
        Mix.shell().info("  OK: #{p}")
      rescue
        err in [Unzip.Error] -> Mix.shell().error("  FAIL: #{p} - #{err.message}")
      end
    end
    Mix.shell().info("Done.")
  end
end
