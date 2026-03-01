defmodule RecGPT.FSQ do
  @moduledoc """
  Finite Scalar Quantization (FSQ) for RecGPT item tokens.

  Port of RecGPT utils/fsq.py: levels [8,8,8,6,5] -> 15360 codes, 4 tokens per item.
  Each token is one index in 0..15359. Padding id is 15360.

  Requires Nx. Weights (project_in, project_out) come from the **VAE** checkpoint (the RecGPT
  .pt export does not include them). Use `load_params_from_vae_pt/1` with the path to
  `vae_len4_fsq88865_ep90.pt` so token_id_list matches the Python pipeline; or `load_params/1`
  with a map that has `quantizer.project_in.weight` / `quantizer.project_out.weight` (e.g. from
  `RecGPT.PtLoader.load!/1` on the VAE .pt).
  """

  @seq_len 4
  @dim 192
  @level_list [8, 8, 8, 6, 5]
  @vocab_size 15_360
  @padding_id 15_360

  def seq_len, do: @seq_len
  def vocab_size, do: @vocab_size
  def padding_id, do: @padding_id
  def dim, do: @dim

  def basis do
    [1 | Enum.take(@level_list, 4)]
    |> Enum.scan(1, fn l, acc -> acc * l end)
    |> Nx.tensor(names: [:dim])
  end

  def levels do
    Nx.tensor(@level_list, type: {:s, 32}, names: [:dim])
  end

  def bound(z, eps \\ 1.0e-3) do
    levels = levels()
    half_l = Nx.multiply(Nx.subtract(levels, 1), 1 - eps) |> Nx.divide(2)
    zero = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), levels)
    half = Nx.broadcast(Nx.tensor(0.5, type: {:f, 32}), levels)
    offset = Nx.select(Nx.equal(Nx.remainder(levels, 2), 0), half, zero)
    shift = Nx.divide(offset, half_l) |> Nx.tanh()
    Nx.subtract(Nx.multiply(Nx.tanh(Nx.add(z, shift)), half_l), offset)
  end

  def round_ste(z) do
    zhat = Nx.round(z)
    Nx.add(z, Nx.subtract(zhat, z))
  end

  def quantize(z) do
    bounded = bound(z)
    quantized = round_ste(bounded)
    half_width = Nx.divide(levels(), 2)
    Nx.divide(quantized, half_width)
  end

  def scale_and_shift(zhat_normalized) do
    half_width = Nx.divide(levels(), 2)
    Nx.add(Nx.multiply(zhat_normalized, half_width), half_width)
  end

  def scale_and_shift_inverse(zhat) do
    half_width = Nx.divide(levels(), 2)
    Nx.divide(Nx.subtract(zhat, half_width), half_width)
  end

  def codes_to_indices(codes) do
    zhat = scale_and_shift(codes)
    b = Nx.reshape(basis(), {1, 1, 5})
    raw = Nx.multiply(zhat, b) |> Nx.sum(axes: [-1]) |> Nx.round() |> Nx.as_type({:s, 32})
    max_idx = vocab_size() - 1
    Nx.clip(raw, 0, max_idx)
  end

  def indices_to_codes(indices, params) do
    {batch, _} = Nx.shape(indices)
    indices_5d = Nx.reshape(indices, {batch, 4, 1})
    b = Nx.reshape(basis(), {1, 1, 5})
    l = Nx.reshape(levels(), {1, 1, 5})
    codes_non_centered = Nx.remainder(Nx.quotient(indices_5d, b), l)
    codes = scale_and_shift_inverse(codes_non_centered)

    Nx.dot(codes, [2], params["project_out"]["kernel"], [0])
    |> Nx.add(params["project_out"]["bias"] || 0)
  end

  def encode(z, params) do
    z_proj = Nx.dot(z, [2], params["project_in"]["kernel"], [0])
    z_proj = if b = params["project_in"]["bias"], do: Nx.add(z_proj, b), else: z_proj
    codes = quantize(z_proj)
    indices = codes_to_indices(codes)
    out = Nx.dot(codes, [2], params["project_out"]["kernel"], [0])
    out = if b = params["project_out"]["bias"], do: Nx.add(out, b), else: out
    {out, indices}
  end

  @doc """
  Load FSQ params from a VAE checkpoint .pt file (e.g. vae_len4_fsq88865_ep90.pt).

  The VAE state_dict uses keys like `quantizer.project_in.weight`; they are extracted
  and normalized to the same format as `load_params/1`. Use this when building a fixture
  so token_id_list matches the Python RecGPT pipeline (same FSQ codebook as eval/predict).
  """
  @spec load_params_from_vae_pt(String.t()) :: map()
  def load_params_from_vae_pt(vae_pt_path) when is_binary(vae_pt_path) do
    vae_pt_path = Path.expand(vae_pt_path)
    unless File.regular?(vae_pt_path), do: raise(File.Error, path: vae_pt_path, reason: :enoent)
    tensor_map = RecGPT.PtLoader.load!(vae_pt_path)
    load_params(tensor_map)
  end

  def load_params(tensor_map) do
    # Export / manifest keys, RecGPT export keys, and VAE .pt state_dict keys
    project_in_k =
      tensor_map["project_in/kernel"] || tensor_map["fsq.project_in.weight"] ||
        tensor_map["quantizer.project_in.weight"]
    project_in_b =
      tensor_map["project_in/bias"] || tensor_map["fsq.project_in.bias"] ||
        tensor_map["quantizer.project_in.bias"]
    project_out_k =
      tensor_map["project_out/kernel"] || tensor_map["fsq.project_out.weight"] ||
        tensor_map["quantizer.project_out.weight"]
    project_out_b =
      tensor_map["project_out/bias"] || tensor_map["fsq.project_out.bias"] ||
        tensor_map["quantizer.project_out.bias"]

    project_in_k =
      if project_in_k && Nx.shape(project_in_k) == {5, 192},
        do: Nx.transpose(project_in_k),
        else: project_in_k

    # PyTorch Linear(5, 192) stores weight (192, 5). Nx.dot(codes, [2], kernel, [0]) needs kernel (5, 192).
    project_out_k =
      if project_out_k && Nx.shape(project_out_k) == {192, 5},
        do: Nx.transpose(project_out_k),
        else: project_out_k

    # When checkpoint has no FSQ params, use dummies so build_fixture runs (token_id_list will be placeholder)
    project_in_k = project_in_k || Nx.broadcast(0.0, {192, 5})
    project_out_k = project_out_k || Nx.broadcast(0.0, {5, 192})

    %{
      "project_in" => %{"kernel" => project_in_k, "bias" => project_in_b},
      "project_out" => %{"kernel" => project_out_k, "bias" => project_out_b}
    }
  end
end
