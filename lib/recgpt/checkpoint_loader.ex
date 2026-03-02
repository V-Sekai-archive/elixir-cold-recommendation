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

  Tensors are created with Nx.BinaryBackend so loading works regardless
  of the default Nx backend; callers can then transfer params to EXLA (e.g. in Serve).
  """

  @doc """
  Load checkpoint from an export directory. Returns %{key => Nx.Tensor}.
  Uses BinaryBackend; transfer to EXLA in Serve if desired.
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

      case Npy.load(path, :npy) do
        {:ok, npy} ->
          tensor = npy_to_tensor_binary_backend(npy)
          Map.put(acc, key, tensor)
        {:error, reason} ->
          raise "Failed to load #{path}: #{inspect(reason)}"
      end
    end)
  end

  # Build Nx tensor from %Npy{} using BinaryBackend only. Same descr→type
  # mapping as Npy.npy2tensor/1; caller can backend_transfer to EXLA.
  defp npy_to_tensor_binary_backend(%Npy{descr: descr, shape: shape, data: data}) do
    type = npy_descr_to_nx_type(descr)
    prev = Nx.default_backend()
    Nx.default_backend(Nx.BinaryBackend)
    try do
      data
      |> Nx.from_binary(type)
      |> Nx.reshape(shape)
    after
      Nx.default_backend(prev)
    end
  end

  defp npy_descr_to_nx_type(descr) do
    case descr do
      "<i1" -> {:s, 8}
      "<i2" -> {:s, 16}
      "<i4" -> {:s, 32}
      "<i8" -> {:s, 64}
      "<u1" -> {:u, 8}
      "<u2" -> {:u, 16}
      "<u4" -> {:u, 32}
      "<u8" -> {:u, 64}
      "<f4" -> {:f, 32}
      "<f8" -> {:f, 64}
      "<f2" -> {:bf, 16}
      # big-endian variants
      ">i1" -> {:s, 8}
      ">i2" -> {:s, 16}
      ">i4" -> {:s, 32}
      ">i8" -> {:s, 64}
      ">u1" -> {:u, 8}
      ">u2" -> {:u, 16}
      ">u4" -> {:u, 32}
      ">u8" -> {:u, 64}
      ">f4" -> {:f, 32}
      ">f8" -> {:f, 64}
      ">f2" -> {:bf, 16}
      other -> raise "Unsupported npy descr: #{inspect(other)}"
    end
  end
end
