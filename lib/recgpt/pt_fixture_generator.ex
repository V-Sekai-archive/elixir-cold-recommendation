defmodule RecGPT.PtFixtureGenerator do
  @moduledoc """
  Generates a minimal PyTorch .pt (zip) fixture for testing RecGPT.PtLoader.

  Produces a zip containing `data.pkl` (pickle state_dict) and `data/0`, `data/1`, `data/2`
  (FloatStorage blobs). State dict keys: "wte" (4×8), "pred_head.weight" (8×4), "pred_head.bias" (4).
  No Python or PyTorch required; uses minimal pickle opcodes and Erlang :zip.
  """

  # Pickle protocol 4 opcodes (decimal)
  @proto 128
  @proto_version 4
  @mark 40
  @tuple 116
  @global 99
  @reduce 82
  @binpersid 81
  @empty_dict 125
  @stop 46
  @binint1 75
  @short_binunicode 140
  @binput 113
  @binget 104
  @pop 48

  @doc """
  Generates a minimal .pt zip binary (PyTorch 1.6+ zip format).

  Writes nothing to disk; returns the binary. Use `generate_to_path/1` to write to a file.
  """
  def generate! do
    # Storage blobs: f32, row-major. wte (4,8)=32 floats, pred_head.weight (8,4)=32, pred_head.bias (4)=4.
    wte = for _ <- 1..32, do: <<0.0::float-32-little>>, into: <<>>
    head_w = for _ <- 1..32, do: <<0.0::float-32-little>>, into: <<>>
    head_b = for _ <- 1..4, do: <<0.0::float-32-little>>, into: <<>>

    pkl =
      build_data_pkl([
        {"wte", "0", {4, 8}, wte},
        {"pred_head.weight", "1", {8, 4}, head_w},
        {"pred_head.bias", "2", {4}, head_b}
      ])

    files = [
      {"data/0", wte},
      {"data/1", head_w},
      {"data/2", head_b},
      {"data.pkl", pkl}
    ]

    # Erlang :zip expects list of {filename, binary}; filename as charlist for compatibility
    file_list = Enum.map(files, fn {name, bin} -> {String.to_charlist(name), bin} end)
    {:ok, {_name, zip_bin}} = :zip.create(~c"mem", file_list, [:memory])
    zip_bin
  end

  @doc "Writes the fixture to `path`. Creates parent dirs. Returns :ok or raises."
  def generate_to_path(path) when is_binary(path) do
    path |> Path.dirname() |> File.mkdir_p!()
    File.write!(path, generate!())
  end

  defp build_data_pkl(tensors) do
    # Protocol 4 header
    acc = <<@proto, @proto_version>>

    # Build each tensor and memoize (BINPUT 0,1,2); stack becomes [t3, t2, t1].
    # Use memo 10+ for globals inside tensor_pickle so we don't overwrite tensor memos.
    {acc, _} =
      Enum.reduce(Enum.with_index(tensors, 0), {acc, 0}, fn {{_key, storage_key, shape, _blob}, idx}, {acc, _} ->
        numel = shape_to_numel(shape)
        stride = shape_to_stride(shape)
        acc = acc <> tensor_pickle(storage_key, numel, shape, stride, 10 + idx) <> <<@binput, idx>>
        {acc, idx + 1}
      end)

    # Pop the 3 tensors, EMPTY_DICT, then for each: key, BINGET idx, SETITEM (stack: dict, key, value)
    acc =
      acc <>
        <<@pop, @pop, @pop>> <>
        <<@empty_dict>>

    acc =
      Enum.reduce(Enum.with_index(tensors, 0), acc, fn {{key, _sid, _shape, _blob}, idx}, acc ->
        acc <> push_short_binunicode(key) <> <<@binget, idx>> <> <<115>>
      end)

    acc <> <<@stop>>
  end

  defp shape_to_numel(shape) when is_tuple(shape) do
    shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)
  end

  defp shape_to_stride(shape) when is_tuple(shape) do
    dims = Tuple.to_list(shape)
    dims
    |> Enum.with_index()
    |> Enum.map(fn {_d, i} ->
      Enum.drop(dims, i + 1) |> Enum.reduce(1, &*/2)
    end)
    |> List.to_tuple()
  end

  defp tensor_pickle(storage_key, numel, shape, stride, global_memo) do
    # Args tuple for _rebuild_tensor: (storage, offset, shape, stride).
    # Push storage via BINPERSID with id = ("storage", "torch.FloatStorage", storage_key, "cpu", numel)
    storage_id = build_storage_id_tuple(storage_key, numel)
    # Stack: storage (from BINPERSID), 0, shape_tuple, stride_tuple; then MARK + TUPLE to build 4-tuple
    args =
      storage_id <>
        <<@binpersid>> <>
        <<@binint1, 0>> <>
        push_shape_tuple(shape) <>
        push_shape_tuple(stride) <>
        <<@mark, @tuple>>

    # Push global, BINPUT (memo), MARK (saves [ref], clears), args -> [ref, 4-tuple];
    # BINGET global, POP -> [ref, 4-tuple] (drop duplicate global), REDUCE.
    global = <<@global>> <> line("torch._utils") <> line("_rebuild_tensor")
    reorder =
      global <>
        <<@binput, global_memo::unsigned-8>> <>
        <<@mark>> <>
        args <>
        <<@binget, global_memo::unsigned-8>> <>
        <<@pop>>

    <<reorder::binary, @reduce>>
  end

  defp build_storage_id_tuple(storage_key, numel) do
    # Tuple ("storage", "torch.FloatStorage", storage_key, "cpu", numel)
    <<@mark>> <>
      push_short_binunicode("storage") <>
      push_short_binunicode("torch.FloatStorage") <>
      push_short_binunicode(storage_key) <>
      push_short_binunicode("cpu") <>
      <<@binint1, numel::integer-8>> <>
      <<@tuple>>
  end

  defp push_short_binunicode(s) when is_binary(s) do
    bin = s
    len = byte_size(bin)
    if len > 255, do: raise("SHORT_BINUNICODE length > 255")
    <<@short_binunicode, len::integer-8, bin::binary>>
  end

  defp push_shape_tuple(shape) when is_tuple(shape) do
    elems = Tuple.to_list(shape)
    acc = <<@mark>>
    acc =
      Enum.reduce(elems, acc, fn n, acc ->
        n = min(255, max(0, n))
        acc <> <<@binint1, n::unsigned-8>>
      end)
    acc <> <<@tuple>>
  end

  defp line(s), do: s <> "\n"
end
