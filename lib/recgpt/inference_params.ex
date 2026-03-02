defmodule RecGPT.InferenceParams do
  @moduledoc """
  Builds defn-friendly full param maps (atom keys) for RecGPT.InferenceDefn.

  Always returns a single full_params map. When n_layers is 0 (stub checkpoint),
  the 12 transformer layers get identity weights so the same forward_with_cache /
  forward_incremental entry points apply.
  """

  @n_embd 768
  @vocab_size 15_361
  @max_pos 1024

  @doc """
  Build full params for Defn from checkpoint string-key map.

  - `params_map`: from `RecGPT.CheckpointLoader.load_from_export/1`
  - `n_layers`: 0 (stub), or 1..12. Use `RecGPT.Inference.n_layers_from_params/1` to get it.
    When 1..11, layers 0..(n_layers-1) use checkpoint params; layers n_layers..11 get identity.
  - `dtype`: optional, default `{:f, 32}`. Use `{:bf, 16}` for BF16 inference (Tensor Cores).

  Returns a map of atom keys. When n_layers is 0, all 12 layer slots get identity tensors.
  """
  @spec build_defn_params(map(), 0..12, keyword() | tuple()) :: map()
  def build_defn_params(params_map, n_layers, dtype \\ {:f, 32})
      when n_layers in 0..12 do
    wte = get_wte(params_map) |> as_dtype(dtype)
    wpe = get_wpe(params_map)
    ln_f = get_ln_f(params_map, dtype)
    ae = get_ae(params_map, dtype)
    pred_head = get_pred_head(params_map, dtype)

    base = %{
      wte: wte,
      wpe: wpe |> as_dtype(dtype),
      ln_f_weight: ln_f.weight |> as_dtype(dtype),
      ln_f_bias: ln_f.bias |> as_dtype(dtype),
      ae_linear_weight: ae.linear_weight |> as_dtype(dtype),
      ae_linear_bias: ae.linear_bias |> as_dtype(dtype),
      ae_norm_weight: ae.norm_weight |> as_dtype(dtype),
      ae_norm_bias: ae.norm_bias |> as_dtype(dtype),
      pred_head_weight: pred_head.weight |> as_dtype(dtype),
      pred_head_bias: pred_head.bias |> as_dtype(dtype)
    }

    layers =
      if n_layers == 0 do
        identity_layers(dtype)
      else
        real_layers_partial(params_map, n_layers, dtype)
      end

    Map.merge(base, layers)
  end

  defp as_dtype(tensor, dtype) when is_tuple(dtype), do: Nx.as_type(tensor, dtype)
  defp as_dtype(tensor, _), do: tensor

  defp get_wte(params) do
    wte =
      params["gpt2model.wte"] || params["gpt2model.wte.weight"] || params["wte"] ||
        params["wte.weight"]

    if is_nil(wte), do: raise("missing wte in params")
    {rows, _} = Nx.shape(wte)
    if rows >= @vocab_size, do: Nx.slice_along_axis(wte, 0, @vocab_size, axis: 0), else: wte
  end

  defp get_wpe(params) do
    wpe =
      params["gpt2model.wpe"] || params["gpt2model.wpe.weight"] || params["transformer.wpe"] ||
        params["transformer.wpe.weight"]

    case wpe do
      nil ->
        Nx.broadcast(0.0, {@max_pos, @n_embd}) |> Nx.as_type({:f, 32})

      t ->
        {rows, cols} = Nx.shape(t)
        t = if cols == @n_embd and rows != @n_embd, do: t, else: Nx.transpose(t)
        {rows, _} = Nx.shape(t)

        if rows >= @max_pos do
          Nx.slice_along_axis(t, 0, @max_pos, axis: 0)
        else
          padded = Nx.broadcast(0.0, {@max_pos, @n_embd}) |> Nx.as_type({:f, 32})
          Nx.put_slice(padded, [0, 0], t)
        end
    end
  end

  defp get_ln_f(params, dtype) do
    w = params["gpt2model.ln_f.weight"] || params["transformer.ln_f.weight"]
    b = params["gpt2model.ln_f.bias"] || params["transformer.ln_f.bias"]

    %{
      weight: (w && ensure_shape(w, {@n_embd})) || ones({@n_embd}, dtype),
      bias: (b && ensure_shape(b, {@n_embd})) || zeros({@n_embd}, dtype)
    }
  end

  defp get_ae(params, dtype) do
    w = params["ae.linear.weight"] || params["ae.weight"] || params["linear_layer.weight"]
    b = params["ae.linear.bias"] || params["ae.bias"]
    nw = params["ae.norm.weight"] || params["norm_aux.weight"]
    nb = params["ae.norm.bias"] || params["norm_aux.bias"]

    %{
      linear_weight: (w && ensure_shape(w, {@n_embd, 192})) || zeros({@n_embd, 192}, dtype),
      linear_bias: (b && ensure_shape(b, {@n_embd})) || zeros({@n_embd}, dtype),
      norm_weight: (nw && ensure_shape(nw, {@n_embd})) || ones({@n_embd}, dtype),
      norm_bias: (nb && ensure_shape(nb, {@n_embd})) || zeros({@n_embd}, dtype)
    }
  end

  defp get_pred_head(params, dtype) do
    w = params["pred_head.weight"]
    b = params["pred_head.bias"]
    if is_nil(w), do: raise("missing pred_head.weight in params")

    %{
      weight: ensure_shape(w, {@n_embd, @vocab_size}),
      bias: (b && ensure_shape(b, {@vocab_size})) || zeros({@vocab_size}, dtype)
    }
  end

  defp identity_layers(dtype) do
    Enum.reduce(0..11, %{}, fn i, acc ->
      prefix = "layer_#{i}_"

      Map.merge(acc, %{
        :"#{prefix}ln_1_weight" => ones({@n_embd}, dtype),
        :"#{prefix}ln_1_bias" => zeros({@n_embd}, dtype),
        :"#{prefix}attn_c_attn_weight" => zeros({2304, @n_embd}, dtype),
        :"#{prefix}attn_c_attn_bias" => zeros({2304}, dtype),
        :"#{prefix}attn_c_proj_weight" => zeros({@n_embd, @n_embd}, dtype),
        :"#{prefix}attn_c_proj_bias" => zeros({@n_embd}, dtype),
        :"#{prefix}ln_2_weight" => ones({@n_embd}, dtype),
        :"#{prefix}ln_2_bias" => zeros({@n_embd}, dtype),
        :"#{prefix}mlp_c_fc_weight" => zeros({3072, @n_embd}, dtype),
        :"#{prefix}mlp_c_fc_bias" => zeros({3072}, dtype),
        :"#{prefix}mlp_c_proj_weight" => zeros({@n_embd, 3072}, dtype),
        :"#{prefix}mlp_c_proj_bias" => zeros({@n_embd}, dtype)
      })
    end)
  end

  defp real_layers_partial(params_map, n_layers, dtype) do
    prefix = gpt2_prefix(params_map)
    if is_nil(prefix), do: raise("expected full checkpoint with gpt2model.h or transformer.h")

    Enum.reduce(0..11, %{}, fn i, acc ->
      layer_map =
        if i < n_layers do
          base = prefix <> "h.#{i}."

          %{
            :"layer_#{i}_ln_1_weight" =>
              get_param(params_map, base <> "ln_1.weight", {@n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_ln_1_bias" =>
              get_param(params_map, base <> "ln_1.bias", {@n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_attn_c_attn_weight" =>
              get_param(params_map, base <> "attn.c_attn.weight", {2304, @n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_attn_c_attn_bias" =>
              get_param(params_map, base <> "attn.c_attn.bias", {2304}) |> as_dtype(dtype),
            :"layer_#{i}_attn_c_proj_weight" =>
              get_param(params_map, base <> "attn.c_proj.weight", {@n_embd, @n_embd})
              |> as_dtype(dtype),
            :"layer_#{i}_attn_c_proj_bias" =>
              get_param(params_map, base <> "attn.c_proj.bias", {@n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_ln_2_weight" =>
              get_param(params_map, base <> "ln_2.weight", {@n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_ln_2_bias" =>
              get_param(params_map, base <> "ln_2.bias", {@n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_mlp_c_fc_weight" =>
              get_param(params_map, base <> "mlp.c_fc.weight", {3072, @n_embd}) |> as_dtype(dtype),
            :"layer_#{i}_mlp_c_fc_bias" =>
              get_param(params_map, base <> "mlp.c_fc.bias", {3072}) |> as_dtype(dtype),
            :"layer_#{i}_mlp_c_proj_weight" =>
              get_param(params_map, base <> "mlp.c_proj.weight", {@n_embd, 3072})
              |> as_dtype(dtype),
            :"layer_#{i}_mlp_c_proj_bias" =>
              get_param(params_map, base <> "mlp.c_proj.bias", {@n_embd}) |> as_dtype(dtype)
          }
        else
          identity_layer_at(i, dtype)
        end

      Map.merge(acc, layer_map)
    end)
  end

  defp identity_layer_at(i, dtype) do
    prefix = "layer_#{i}_"

    %{
      :"#{prefix}ln_1_weight" => ones({@n_embd}, dtype),
      :"#{prefix}ln_1_bias" => zeros({@n_embd}, dtype),
      :"#{prefix}attn_c_attn_weight" => zeros({2304, @n_embd}, dtype),
      :"#{prefix}attn_c_attn_bias" => zeros({2304}, dtype),
      :"#{prefix}attn_c_proj_weight" => zeros({@n_embd, @n_embd}, dtype),
      :"#{prefix}attn_c_proj_bias" => zeros({@n_embd}, dtype),
      :"#{prefix}ln_2_weight" => ones({@n_embd}, dtype),
      :"#{prefix}ln_2_bias" => zeros({@n_embd}, dtype),
      :"#{prefix}mlp_c_fc_weight" => zeros({3072, @n_embd}, dtype),
      :"#{prefix}mlp_c_fc_bias" => zeros({3072}, dtype),
      :"#{prefix}mlp_c_proj_weight" => zeros({@n_embd, 3072}, dtype),
      :"#{prefix}mlp_c_proj_bias" => zeros({@n_embd}, dtype)
    }
  end

  defp get_param(params_map, key, expected_shape) do
    t = params_map[key]
    if is_nil(t), do: raise("missing #{key} in params")
    ensure_shape(t, expected_shape)
  end

  defp gpt2_prefix(params) do
    keys = Map.keys(params)

    cond do
      Enum.any?(keys, &String.starts_with?(&1, "gpt2model.h.")) -> "gpt2model."
      Enum.any?(keys, &String.starts_with?(&1, "transformer.h.")) -> "transformer."
      true -> nil
    end
  end

  defp ensure_shape(tensor, expected) do
    shape = Nx.shape(tensor)
    if shape == expected, do: tensor, else: Nx.transpose(tensor)
  end

  defp zeros(shape, dtype), do: Nx.broadcast(0.0, shape) |> Nx.as_type(dtype)
  defp ones(shape, dtype), do: Nx.broadcast(1.0, shape) |> Nx.as_type(dtype)
end
