defmodule RecGPT.ParamFlatten do
  @moduledoc """
  Flatten and unflatten RecGPT param map for use with Nx.Defn.grad.
  Produces an ordered list of trainable tensors and a single flattened Nx tensor;
  can unflatten a gradient vector back into a map keyed like the original.
  """

  @doc """
  Build a spec from a param map and an ordered list of canonical keys.
  Each canonical key may map to one or more checkpoint keys (first found wins).
  Returns [%{key: key, shape: shape, size: size}, ...] and total size.
  Tensors are normalized to inference-ready shapes (e.g. pred_head.weight as {768, 15361}).
  """
  def spec_from_params(params, canonical_keys, key_aliases \\ %{}) do
    key_aliases =
      key_aliases
      |> Map.merge(default_key_aliases())

    Enum.reduce_while(canonical_keys, {[], 0}, fn key, {spec_acc, offset} ->
      checkpoint_keys = Map.get(key_aliases, key, [key])
      tensor = find_param(params, checkpoint_keys)

      if tensor == nil do
        {:halt, {:error, "missing param for #{key}, tried #{inspect(checkpoint_keys)}"}}
      else
        tensor = ensure_canonical_shape(key, tensor)
        shape = Nx.shape(tensor)
        size = Nx.size(tensor)
        spec = %{key: key, shape: shape, size: size, offset: offset}
        {:cont, {[spec | spec_acc] |> Enum.reverse(), offset + size}}
      end
    end)
    |> case do
      {:error, _} = err -> err
      {spec, total} -> {:ok, spec, total}
    end
  end

  defp default_key_aliases do
    %{
      "wte" => ["gpt2model.wte", "gpt2model.wte.weight", "wte", "wte.weight"],
      "ae.linear.weight" =>
        ["ae.linear.weight", "ae.weight", "linear_layer.weight"],
      "ae.linear.bias" => ["ae.linear.bias", "ae.bias"],
      "pred_head.weight" => ["pred_head.weight"],
      "pred_head.bias" => ["pred_head.bias"]
    }
  end

  defp find_param(params, keys) when is_list(keys) do
    Enum.find_value(keys, fn k -> Map.get(params, k) end)
  end

  defp ensure_canonical_shape("wte", tensor) do
    {rows, _} = Nx.shape(tensor)
    if rows >= 15361, do: Nx.slice_along_axis(tensor, 0, 15361, axis: 0), else: tensor
  end

  defp ensure_canonical_shape("ae.linear.weight", tensor) do
    ensure_shape(tensor, {768, 192})
  end

  defp ensure_canonical_shape("pred_head.weight", tensor) do
    ensure_shape(tensor, {768, 15361})
  end

  defp ensure_canonical_shape(_, tensor), do: tensor

  defp ensure_shape(tensor, expected) do
    if Nx.shape(tensor) == expected, do: tensor, else: Nx.transpose(tensor)
  end

  @doc """
  Flatten params to a single 1-D tensor and return the spec.
  params: map from CheckpointLoader.
  canonical_keys: ordered list of keys to include (e.g. stub: wte, ae.linear.weight, ae.linear.bias, pred_head.weight, pred_head.bias).
  """
  def flatten(params, canonical_keys, key_aliases \\ %{}) do
    aliases = Map.merge(default_key_aliases(), key_aliases)

    case spec_from_params(params, canonical_keys, aliases) do
      {:ok, spec, _total} ->
        parts =
          Enum.map(spec, fn %{key: key, size: size} ->
            keys = Map.get(aliases, key, [key]) |> List.wrap()
            tensor = find_param(params, keys)
            tensor = ensure_canonical_shape(key, tensor)
            Nx.reshape(tensor, {size})
          end)

        flat = Nx.concatenate(parts, axis: 0)
        {:ok, flat, spec}

      err ->
        err
    end
  end

  @doc """
  Unflatten a flat 1-D tensor (e.g. gradients) back into a map keyed by canonical key.
  """
  def unflatten(flat_tensor, spec) do
    Enum.reduce(spec, %{}, fn %{key: key, shape: shape, offset: offset, size: size}, acc ->
      slice = Nx.slice(flat_tensor, [offset], [size])
      tensor = Nx.reshape(slice, shape)
      Map.put(acc, key, tensor)
    end)
  end

  @doc """
  Update the param map with new values from an unflattened map (e.g. after optimizer step).
  Only updates keys present in updated; uses checkpoint key names if provided.
  """
  def update_params(params, unflattened, key_to_checkpoint \\ %{}) do
    default_aliases = default_key_aliases()

    Enum.reduce(unflattened, params, fn {canonical_key, tensor}, acc ->
      checkpoint_keys = Map.get(key_to_checkpoint, canonical_key) || Map.get(default_aliases, canonical_key) || [canonical_key]
      checkpoint_key = List.first(List.wrap(checkpoint_keys))
      Map.put(acc, checkpoint_key, tensor)
    end)
  end
end
