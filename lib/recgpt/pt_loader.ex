defmodule RecGPT.PtLoader do
  @moduledoc """
  Load PyTorch .pt checkpoint in pure Elixir using Unzip + Unpickler.

  Supports the zip-based format (default since PyTorch 1.6): unzips the .pt,
  reads `data.pkl` with Unpickler, resolves storage via `persistent_id` (reads
  `data/0`, `data/1`, ... from the zip), and resolves tensors to Nx.Tensor via
  `object_resolver`. Returns a map of string key => Nx.Tensor (state_dict).
  """

  @zip_magic <<0x50, 0x4B, 0x03, 0x04>>

  @doc """
  Loads a .pt file and returns %{key => Nx.Tensor} (state_dict).

  Raises if the file is not a supported PyTorch zip checkpoint or unpickling fails.
  """
  def load!(path) when is_binary(path) do
    binary = File.read!(path)

    if binary_starts_with?(binary, @zip_magic) do
      load_zip!(binary)
    else
      raise ArgumentError,
            "RecGPT.PtLoader only supports zip-based .pt files (PyTorch 1.6+). Got non-zip file."
    end
  end

  defp binary_starts_with?(binary, prefix) do
    byte_size(binary) >= byte_size(prefix) and binary_part(binary, 0, byte_size(prefix)) == prefix
  end

  defp load_zip!(binary) do
    {:ok, unzip} = Unzip.new(binary)
    entries = Unzip.list_entries(unzip) |> Enum.map(& &1.file_name)
    storage_map = build_storage_map(unzip, entries)
    data_pkl_path = find_data_pkl(entries)
    unless data_pkl_path, do: raise("data.pkl not found in zip (entries: #{inspect(entries)})")
    data_pkl = read_zip_file!(unzip, data_pkl_path)

    persistent_id_resolver = fn id ->
      resolve_persistent_id(id, storage_map)
    end

    object_resolver = fn obj ->
      resolve_tensor_object(obj)
    end

    {root, _rest} =
      Unpickler.load!(data_pkl,
        persistent_id_resolver: persistent_id_resolver,
        object_resolver: object_resolver
      )

    state_dict(root)
  end

  defp find_data_pkl(entries) do
    Enum.find(entries, fn p ->
      p |> String.trim_trailing("/") |> Path.basename() == "data.pkl"
    end)
  end

  defp build_storage_map(unzip, entries) do
    # Storage files are .../data/0, .../data/1, etc. (PyTorch may use a top-level folder e.g. sample/data/0)
    entries
    |> Enum.filter(fn p ->
      trimmed = String.trim_trailing(p, "/")

      case String.split(trimmed, "/") do
        [_, "data", key] when key != "" -> true
        ["data", key] when key != "" -> true
        _ -> false
      end
    end)
    |> Enum.reduce(%{}, fn path, acc ->
      key = path |> String.trim_trailing("/") |> String.split("/") |> List.last()
      data = read_zip_file!(unzip, path)
      Map.put(acc, key, data)
    end)
  end

  defp read_zip_file!(unzip, path) do
    unzip
    |> Unzip.file_stream!(path)
    |> Enum.reduce(<<>>, fn chunk, acc -> acc <> IO.iodata_to_binary(chunk) end)
  end

  defp resolve_persistent_id(id, storage_map) when is_tuple(id) do
    case Tuple.to_list(id) do
      ["storage", storage_type, storage_key, _location, _storage_numel] ->
        key = to_string(storage_key)
        data = Map.get(storage_map, key)
        unless data, do: raise("Storage not found: #{inspect(storage_key)}")
        dtype = storage_type_to_nx(storage_type)
        %{data: data, dtype: dtype, key: key}

      ["module", _mod, _source_file, _source] ->
        # Placeholder so Unpickler doesn't error; we only care about state_dict tensors
        %{__module: true}

      _ ->
        raise "Unsupported persistent_id: #{inspect(id)}"
    end
  end

  defp resolve_persistent_id(id, _) do
    raise "Unexpected persistent_id: #{inspect(id)}"
  end

  defp storage_type_to_nx(type) when is_binary(type) do
    type = type |> String.split(".") |> List.last()

    case type do
      "FloatStorage" -> {:f, 32}
      "DoubleStorage" -> {:f, 64}
      "HalfStorage" -> {:f, 16}
      "LongStorage" -> {:s, 64}
      "IntStorage" -> {:s, 32}
      "ShortStorage" -> {:s, 16}
      "ByteStorage" -> {:u, 8}
      "BoolStorage" -> {:u, 8}
      _ -> {:f, 32}
    end
  end

  defp storage_type_to_nx(type) when is_atom(type), do: storage_type_to_nx(to_string(type))
  defp storage_type_to_nx(_), do: {:f, 32}

  defp resolve_tensor_object(%Unpickler.Object{} = obj) do
    # torch._utils._rebuild_tensor(storage, offset, shape, stride) or Tensor from reduce
    constructor = obj.constructor |> to_string()
    args = obj.args || []
    {storage, offset, shape, stride} = extract_tensor_args(args)

    cond do
      constructor =~ "rebuild_tensor" and storage != nil ->
        rebuild_tensor(storage, offset, shape, stride)

      (constructor =~ "Tensor" or constructor =~ "_TensorBase") and storage != nil ->
        rebuild_tensor(storage, offset, shape, stride)

      true ->
        :error
    end
  end

  defp resolve_tensor_object(_), do: :error

  # PyTorch REDUCE args: (storage, offset, shape, stride) as list, or single tuple;
  # tuple order may be (storage, shape, offset, stride).
  defp extract_tensor_args([single]) when is_tuple(single) and tuple_size(single) == 4 do
    [s, x, y, _z] = Tuple.to_list(single)

    # If x is a small int (offset) and y is a list/tuple (shape), then order is (storage, offset, shape, stride).
    # If x is list/tuple (shape) and y is int (offset), then order is (storage, shape, offset, stride).
    {offset, shape} =
      if is_number(x) and (is_list(y) or is_tuple(y)) do
        {x, y}
      else
        {y, x}
      end

    shape_tuple = parse_shape(shape) |> List.to_tuple()
    {s, offset, shape_tuple, nil}
  end

  defp extract_tensor_args(args) when is_list(args) and length(args) >= 4 do
    [a, b, c, d] = Enum.take(args, 4)
    # Identify storage (map with :data) and offset (integer); the other two are shape and stride.
    {storage, offset, shape, _stride} =
      cond do
        storage?(a) and offset?(b) -> order_storage_offset_shape_stride(a, b, c, d)
        storage?(b) and offset?(a) -> order_storage_offset_shape_stride(b, a, c, d)
        storage?(a) and offset?(c) -> order_storage_offset_shape_stride(a, c, b, d)
        storage?(b) and offset?(c) -> order_storage_offset_shape_stride(b, c, a, d)
        storage?(c) and offset?(a) -> order_storage_offset_shape_stride(c, a, b, d)
        storage?(c) and offset?(b) -> order_storage_offset_shape_stride(c, b, a, d)
        true -> order_storage_offset_shape_stride(a, b, c, d)
      end

    {storage, offset, shape, nil}
  end

  defp extract_tensor_args(_), do: {nil, nil, nil, nil}

  defp storage?(%{data: _}), do: true
  defp storage?(_), do: false

  defp offset?(x) when is_integer(x), do: true
  defp offset?(_), do: false

  defp order_storage_offset_shape_stride(s, o, a, b) do
    # PyTorch may pass (storage, offset, shape, stride) or (storage, offset, stride, shape).
    # Shape has product = num_elements; stride often has 1s.
    a_list = parse_shape(a)
    b_list = parse_shape(b)
    num_a = Enum.reduce(a_list, 1, &*/2)
    num_b = Enum.reduce(b_list, 1, &*/2)
    # Use the one with larger product as shape (num elements in tensor).
    if num_b > num_a do
      {s, o, List.to_tuple(b_list), List.to_tuple(a_list)}
    else
      {s, o, List.to_tuple(a_list), List.to_tuple(b_list)}
    end
  end

  defp rebuild_tensor(%{data: data, dtype: dtype}, offset, shape, _stride) do
    elem_bytes = nx_dtype_to_bytes(dtype)
    shape_list = parse_shape(shape)
    shape_tuple = List.to_tuple(shape_list)
    num_elems = Enum.reduce(shape_list, 1, &*/2)
    offset_bytes = offset * elem_bytes
    # When pickle reports shape with product 1 but storage has more, use remaining bytes as 1-D
    # (handles some PyTorch pickle variants).
    {num_elems, shape_tuple} =
      if num_elems == 1 do
        remaining = byte_size(data) - offset_bytes
        n = max(1, div(remaining, elem_bytes))
        {n, List.to_tuple([n])}
      else
        {num_elems, shape_tuple}
      end

    need_bytes = num_elems * elem_bytes
    slice = binary_part(data, offset_bytes, need_bytes)
    tensor = binary_to_nx(slice, dtype, shape_tuple)
    {:ok, tensor}
  end

  defp rebuild_tensor(_, _, _, _), do: :error

  defp parse_shape(shape) when is_list(shape), do: Enum.map(shape, &elem_to_int/1)
  defp parse_shape(shape) when is_tuple(shape), do: shape |> Tuple.to_list() |> parse_shape()
  defp parse_shape(_), do: []

  defp elem_to_int(x) when is_integer(x), do: x
  defp elem_to_int(x) when is_binary(x), do: String.to_integer(x)
  defp elem_to_int(_), do: 0

  defp nx_dtype_to_bytes({f, bits}) when f in [:f, :s, :u], do: div(bits, 8)
  defp nx_dtype_to_bytes(_), do: 4

  defp maybe_fix_shape(shape, size) when is_tuple(shape) do
    product = Enum.reduce(Tuple.to_list(shape), 1, &*/2)
    if product == 1 and size > 1, do: List.to_tuple([size]), else: shape
  end

  defp maybe_fix_shape(shape, _), do: shape

  defp binary_to_nx(binary, {:f, 32}, shape) do
    size = div(byte_size(binary), 4)
    out_shape = maybe_fix_shape(shape, size)
    flat_shape = List.duplicate(1, size) |> List.to_tuple()
    Nx.from_binary(binary, :f32) |> Nx.reshape(flat_shape) |> Nx.reshape(out_shape)
  end

  defp binary_to_nx(binary, {:f, 64}, shape) do
    size = div(byte_size(binary), 8)
    flat_shape = List.duplicate(1, size) |> List.to_tuple()
    Nx.from_binary(binary, :f64) |> Nx.reshape(flat_shape) |> Nx.reshape(shape)
  end

  defp binary_to_nx(binary, {:s, 64}, shape) do
    size = div(byte_size(binary), 8)
    flat_shape = List.duplicate(1, size) |> List.to_tuple()
    Nx.from_binary(binary, :s64) |> Nx.reshape(flat_shape) |> Nx.reshape(shape)
  end

  defp binary_to_nx(binary, {:s, 32}, shape) do
    size = div(byte_size(binary), 4)
    flat_shape = List.duplicate(1, size) |> List.to_tuple()
    Nx.from_binary(binary, :s32) |> Nx.reshape(flat_shape) |> Nx.reshape(shape)
  end

  defp binary_to_nx(binary, {:f, 16}, shape) do
    size = div(byte_size(binary), 2)
    flat_shape = List.duplicate(1, size) |> List.to_tuple()
    Nx.from_binary(binary, :f16) |> Nx.reshape(flat_shape) |> Nx.reshape(shape)
  end

  defp binary_to_nx(binary, {:u, 8}, shape) do
    size = byte_size(binary)
    flat_shape = List.duplicate(1, size) |> List.to_tuple()
    Nx.from_binary(binary, :u8) |> Nx.reshape(flat_shape) |> Nx.reshape(shape)
  end

  defp binary_to_nx(binary, _, shape), do: binary_to_nx(binary, {:f, 32}, shape)

  defp state_dict(root) when is_map(root) do
    root
    |> Enum.reject(fn {_k, v} -> is_map(v) and Map.get(v, :__module) end)
    |> Enum.filter(fn {_k, v} -> is_struct(v, Nx.Tensor) end)
    |> Map.new()
  end

  # Python OrderedDict may unpickle as list of {k, v} pairs
  defp state_dict(root) when is_list(root) do
    root
    |> Enum.reject(fn {_k, v} -> is_map(v) and Map.get(v, :__module) end)
    |> Enum.filter(fn {_k, v} -> is_struct(v, Nx.Tensor) end)
    |> Map.new()
  end

  defp state_dict(root) do
    cond do
      is_map(root) and Map.get(root, :__module) ->
        %{}

      function_exported?(root, :state_dict, 0) ->
        root.state_dict() |> state_dict()

      true ->
        raise "Unsupported .pt root: expected map (state_dict) or list of pairs, got #{inspect(root)}"
    end
  end
end
