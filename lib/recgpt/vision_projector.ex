defmodule RecGPT.VisionProjector do
  @moduledoc """
  Industry-standard vision projector for contrastive image–text alignment.

  Maps DINOv2 vision embeddings (768-d) into the same 768-d space as MPNet text embeddings.
  Both encoders are frozen; only this projector is trained with contrastive loss (e.g. InfoNCE).

  Architecture: 2-layer MLP (768 → 768, GELU, 768 → 768) + L2 norm, matching common
  CLIP/LLaVA-style projection heads. Params use string keys for compatibility with
  checkpoint and training loops.

  ## Usage

  Load DINOv2 image embeddings (768-d), call `VisionProjector.forward(proj_params, vision_768)`,
  and train `proj_params` with contrastive loss against MPNet text embeddings (768-d),
  with DINOv2 and MPNet frozen.
  """

  @doc """
  Returns new params for the projector (flat map, string keys).

  Keys: `"vision_proj.fc1.weight"`, `"vision_proj.fc1.bias"`,
  `"vision_proj.fc2.weight"`, `"vision_proj.fc2.bias"`.
  Weight shapes: fc1 (768, 768), fc2 (768, 768); bias (768).
  """
  @spec init_params() :: %{String.t() => Nx.Tensor.t()}
  def init_params do
    scale = 0.02

    # Xavier-like: (out, in) for Nx.dot(x, w, [axes: [1], [0]])
    # x [batch, 768], w [768, 768] -> out [batch, 768]
    fc1_w = init_linear_weight({768, 768}, scale)
    fc1_b = Nx.broadcast(0, {768}) |> Nx.as_type({:f, 32})
    fc2_w = init_linear_weight({768, 768}, scale)
    fc2_b = Nx.broadcast(0, {768}) |> Nx.as_type({:f, 32})

    %{
      "vision_proj.fc1.weight" => fc1_w,
      "vision_proj.fc1.bias" => fc1_b,
      "vision_proj.fc2.weight" => fc2_w,
      "vision_proj.fc2.bias" => fc2_b
    }
  end

  defp init_linear_weight({out, in_dim}, scale) do
    Nx.iota({out, in_dim}, type: {:f, 32})
    |> Nx.divide(max(out * in_dim, 1))
    |> Nx.multiply(scale * 4)
    |> Nx.subtract(scale * 2)
  end

  @doc """
  Forward pass: vision embedding [batch, 768] -> projected 768-d (L2-normalized).

  Params can use string keys (`"vision_proj.fc1.weight"`) or atom keys
  (`:vision_proj_fc1_weight`) for Defn compatibility.
  """
  @spec forward(map(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward(params, x) do
    # x: [batch, 768]
    fc1_w = get_param(params, :fc1, :weight)
    fc1_b = get_param(params, :fc1, :bias)
    fc2_w = get_param(params, :fc2, :weight)
    fc2_b = get_param(params, :fc2, :bias)

    h = Nx.dot(x, [1], fc1_w, [0])
    h = Nx.add(h, fc1_b)
    h = gelu(h)
    h = Nx.dot(h, [1], fc2_w, [0])
    h = Nx.add(h, fc2_b)
    l2_norm(h)
  end

  defp get_param(params, layer, :weight) do
    params["vision_proj.#{layer}.weight"] || params[:"vision_proj_#{layer}_weight"] ||
      raise("missing vision_proj.#{layer}.weight")
  end

  defp get_param(params, layer, :bias) do
    params["vision_proj.#{layer}.bias"] || params[:"vision_proj_#{layer}_bias"] ||
      raise("missing vision_proj.#{layer}.bias")
  end

  defp gelu(x) do
    # GELU(x) ≈ x * Φ(x); common approximation 0.5 * x * (1 + tanh(sqrt(2/π)(x + 0.044715 x^3)))
    t = Nx.type(x)
    half = Nx.tensor(0.5, type: t)
    one = Nx.tensor(1.0, type: t)
    coeff = Nx.tensor(0.044715, type: t)
    sqrt_2_pi = Nx.tensor(0.7978845608, type: t)

    inner = Nx.add(x, Nx.multiply(coeff, Nx.pow(x, 3)))
    Nx.multiply(Nx.multiply(half, x), Nx.add(one, Nx.tanh(Nx.multiply(sqrt_2_pi, inner))))
  end

  defp l2_norm(x) do
    norm = Nx.LinAlg.norm(x, axes: [-1], keep_axes: true)
    norm = Nx.max(norm, Nx.tensor(1.0e-8, type: Nx.type(x)))
    Nx.divide(x, norm)
  end
end
