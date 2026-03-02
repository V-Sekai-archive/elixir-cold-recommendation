defmodule RecGPT.CheckpointLoader do
  @moduledoc """
  Load RecGPT checkpoint from an export directory (manifest.json + .npy files).

  Export dir can be produced by:
  - `RecGPT.CheckpointExport.write_export/2` or `mix recgpt.export_ckpt --from-export DIR --out DIR`
  - `mix recgpt.export_ckpt --from-pt path.pt --out DIR` (requires Python + torch)
  - Formerly: `scripts/inspect_recgpt_checkpoint.py --export DIR`

  The export dir must contain:
  - manifest.json: map of state_dict key -> %{"file" => "key.npy", "shape" => [dims]}
  - One .npy file per tensor.

  Returns a map of key => Nx.Tensor. The inference model (RecGPT.Inference) expects
  keys such as: wte (FSQ embed table), ae.* (aux encoder), gpt2model.* (GPT-2), pred_head.* (head).
  """

  @doc """
  Load checkpoint from an export directory. Returns %{key => Nx.Tensor}.
  """
  def load_from_export(export_dir) when is_binary(export_dir) do
    do_load_from_export(export_dir)
  end

  defp do_load_from_export(export_dir) do
    manifest_path = Path.join(export_dir, "manifest.json")

    if not File.regular?(manifest_path) do
      raise File.Error, path: manifest_path, reason: :enoent
    end

    manifest = File.read!(manifest_path) |> Jason.decode!()

    Enum.reduce(manifest, %{}, fn {key, meta}, acc ->
      fname = meta["file"]
      path = Path.join(export_dir, fname)

      case Npy.load(path, :nx) do
        {:ok, tensor} -> Map.put(acc, key, tensor)
        {:error, reason} -> raise "Failed to load #{path}: #{inspect(reason)}"
      end
    end)
  end
end
