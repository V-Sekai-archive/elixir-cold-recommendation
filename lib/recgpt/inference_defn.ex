defmodule RecGPT.InferenceDefn do
  @moduledoc """
  Defn entry points for Torchx JIT: forward_with_cache/4 and forward_incremental/5 only.

  Params must be from RecGPT.InferenceParams.build_defn_params/2 (atom keys, full structure).
  Cache is a tuple of 12 `{k, v}` elements, each k/v shape `(batch, n_head, seq_len, head_dim)`.
  """

  import Nx.Defn

  @n_embd 768
  @n_head 12
  @head_dim 64

  defn forward_with_cache(batch_token_ids, batch_aux, embed_mask, params) do
    {hidden, cache} = forward_hidden_with_cache(batch_token_ids, batch_aux, embed_mask, params)
    last_idx = elem(Nx.shape(batch_token_ids), 1) - 1
    last_hidden = hidden |> Nx.slice_along_axis(last_idx, 1, axis: 1) |> Nx.squeeze(axes: [1])
    logits = apply_head(last_hidden, params)
    {logits, cache}
  end

  defn forward_incremental(batch_token_ids, batch_aux, embed_mask, params, past_cache) do
    {hidden, new_cache} =
      forward_hidden_incremental(batch_token_ids, batch_aux, embed_mask, params, past_cache)

    last_hidden = Nx.squeeze(hidden, axes: [1])
    logits = apply_head(last_hidden, params)
    {logits, new_cache}
  end

  defnp forward_hidden_with_cache(batch_token_ids, batch_aux, embed_mask, params) do
    wte = params[:wte]
    {batch, seq_len} = Nx.shape(batch_token_ids)
    flat_ids = Nx.reshape(batch_token_ids, {batch * seq_len})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, seq_len, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux, embed_mask, params)
    combined = Nx.add(token_embeds, aux_768)
    h = add_wpe(combined, params, seq_len)
    {h, c0} = block_with_cache_0(h, params)
    {h, c1} = block_with_cache_1(h, params)
    {h, c2} = block_with_cache_2(h, params)
    {h, c3} = block_with_cache_3(h, params)
    {h, c4} = block_with_cache_4(h, params)
    {h, c5} = block_with_cache_5(h, params)
    {h, c6} = block_with_cache_6(h, params)
    {h, c7} = block_with_cache_7(h, params)
    {h, c8} = block_with_cache_8(h, params)
    {h, c9} = block_with_cache_9(h, params)
    {h, c10} = block_with_cache_10(h, params)
    {h, c11} = block_with_cache_11(h, params)
    cache = {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11}
    h = apply_ln_f(h, params)
    {h, cache}
  end

  defnp forward_hidden_incremental(batch_token_ids, batch_aux, embed_mask, params, past_cache) do
    wte = params[:wte]
    {batch, _seq_len} = Nx.shape(batch_token_ids)
    flat_ids = Nx.reshape(batch_token_ids, {batch})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, 1, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux, embed_mask, params)
    combined = Nx.add(token_embeds, aux_768)
    past_len = elem(Nx.shape(elem(elem(past_cache, 0), 0)), 2)
    h = add_wpe_at_position(combined, past_len, params)
    {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11} = past_cache
    {h, c0} = block_incremental_0(h, params, c0)
    {h, c1} = block_incremental_1(h, params, c1)
    {h, c2} = block_incremental_2(h, params, c2)
    {h, c3} = block_incremental_3(h, params, c3)
    {h, c4} = block_incremental_4(h, params, c4)
    {h, c5} = block_incremental_5(h, params, c5)
    {h, c6} = block_incremental_6(h, params, c6)
    {h, c7} = block_incremental_7(h, params, c7)
    {h, c8} = block_incremental_8(h, params, c8)
    {h, c9} = block_incremental_9(h, params, c9)
    {h, c10} = block_incremental_10(h, params, c10)
    {h, c11} = block_incremental_11(h, params, c11)
    new_cache = {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11}
    h = apply_ln_f(h, params)
    {h, new_cache}
  end

  defnp apply_aux_encoder(aux_192, mask, params) do
    w = params[:ae_linear_weight]
    b = params[:ae_linear_bias]
    nw = params[:ae_norm_weight]
    nb = params[:ae_norm_bias]
    out = Nx.dot(aux_192, [2], w, [1])
    out = Nx.add(out, Nx.reshape(b, {1, 1, @n_embd}))
    out = layer_norm(out, nw, nb)
    Nx.multiply(out, mask)
  end

  defnp add_wpe(hidden, params, seq_len) do
    wpe = params[:wpe]
    indices = Nx.iota({seq_len}, type: {:s, 32})
    pe = Nx.gather(wpe, Nx.new_axis(indices, -1))
    pe = Nx.reshape(pe, {1, seq_len, @n_embd})
    Nx.add(hidden, pe)
  end

  defnp add_wpe_at_position(hidden, position, params) do
    wpe = params[:wpe]
    pe_row = Nx.slice_along_axis(wpe, position, 1, axis: 0)
    pe = Nx.reshape(pe_row, {1, 1, @n_embd})
    Nx.add(hidden, pe)
  end

  defnp apply_ln_f(hidden, params) do
    w = params[:ln_f_weight]
    b = params[:ln_f_bias]
    layer_norm(hidden, w, b)
  end

  defnp apply_head(hidden, params) do
    w = params[:pred_head_weight]
    b = params[:pred_head_bias]
    # hidden (batch, 768)
    logits = Nx.dot(hidden, [1], w, [0])
    Nx.add(logits, b)
  end

  defnp layer_norm(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    x = Nx.divide(Nx.subtract(x, mean), Nx.add(Nx.sqrt(var), 1.0e-5))
    Nx.add(Nx.multiply(x, weight), bias)
  end

  # One block per layer so param keys are literal (defn cannot use dynamic keys)
  defnp block_with_cache_0(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_0(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_1(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_1(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_2(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_2(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_3(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_3(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_4(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_4(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_5(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_5(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_6(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_6(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_7(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_7(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_8(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_8(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_9(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_9(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_10(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_10(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_11(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_11(params)
    block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
      ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_with_cache_impl(hidden, ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b,
         ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b) do
    attn_in = layer_norm(hidden, ln1_w, ln1_b)
    {attn_out, kv} = attn_with_cache(attn_in, c_attn_w, c_attn_b, c_proj_w, c_proj_b)
    h = Nx.add(hidden, attn_out)
    mlp_in = layer_norm(h, ln2_w, ln2_b)
    mlp_out = mlp(mlp_in, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
    {Nx.add(h, mlp_out), kv}
  end

  defnp get_layer_params_0(params) do
    {params[:layer_0_ln_1_weight], params[:layer_0_ln_1_bias],
     params[:layer_0_attn_c_attn_weight], params[:layer_0_attn_c_attn_bias],
     params[:layer_0_attn_c_proj_weight], params[:layer_0_attn_c_proj_bias],
     params[:layer_0_ln_2_weight], params[:layer_0_ln_2_bias], params[:layer_0_mlp_c_fc_weight],
     params[:layer_0_mlp_c_fc_bias], params[:layer_0_mlp_c_proj_weight],
     params[:layer_0_mlp_c_proj_bias]}
  end

  defnp get_layer_params_1(params) do
    {params[:layer_1_ln_1_weight], params[:layer_1_ln_1_bias],
     params[:layer_1_attn_c_attn_weight], params[:layer_1_attn_c_attn_bias],
     params[:layer_1_attn_c_proj_weight], params[:layer_1_attn_c_proj_bias],
     params[:layer_1_ln_2_weight], params[:layer_1_ln_2_bias], params[:layer_1_mlp_c_fc_weight],
     params[:layer_1_mlp_c_fc_bias], params[:layer_1_mlp_c_proj_weight],
     params[:layer_1_mlp_c_proj_bias]}
  end

  defnp get_layer_params_2(params) do
    {params[:layer_2_ln_1_weight], params[:layer_2_ln_1_bias],
     params[:layer_2_attn_c_attn_weight], params[:layer_2_attn_c_attn_bias],
     params[:layer_2_attn_c_proj_weight], params[:layer_2_attn_c_proj_bias],
     params[:layer_2_ln_2_weight], params[:layer_2_ln_2_bias], params[:layer_2_mlp_c_fc_weight],
     params[:layer_2_mlp_c_fc_bias], params[:layer_2_mlp_c_proj_weight],
     params[:layer_2_mlp_c_proj_bias]}
  end

  defnp get_layer_params_3(params) do
    {params[:layer_3_ln_1_weight], params[:layer_3_ln_1_bias],
     params[:layer_3_attn_c_attn_weight], params[:layer_3_attn_c_attn_bias],
     params[:layer_3_attn_c_proj_weight], params[:layer_3_attn_c_proj_bias],
     params[:layer_3_ln_2_weight], params[:layer_3_ln_2_bias], params[:layer_3_mlp_c_fc_weight],
     params[:layer_3_mlp_c_fc_bias], params[:layer_3_mlp_c_proj_weight],
     params[:layer_3_mlp_c_proj_bias]}
  end

  defnp get_layer_params_4(params) do
    {params[:layer_4_ln_1_weight], params[:layer_4_ln_1_bias],
     params[:layer_4_attn_c_attn_weight], params[:layer_4_attn_c_attn_bias],
     params[:layer_4_attn_c_proj_weight], params[:layer_4_attn_c_proj_bias],
     params[:layer_4_ln_2_weight], params[:layer_4_ln_2_bias], params[:layer_4_mlp_c_fc_weight],
     params[:layer_4_mlp_c_fc_bias], params[:layer_4_mlp_c_proj_weight],
     params[:layer_4_mlp_c_proj_bias]}
  end

  defnp get_layer_params_5(params) do
    {params[:layer_5_ln_1_weight], params[:layer_5_ln_1_bias],
     params[:layer_5_attn_c_attn_weight], params[:layer_5_attn_c_attn_bias],
     params[:layer_5_attn_c_proj_weight], params[:layer_5_attn_c_proj_bias],
     params[:layer_5_ln_2_weight], params[:layer_5_ln_2_bias], params[:layer_5_mlp_c_fc_weight],
     params[:layer_5_mlp_c_fc_bias], params[:layer_5_mlp_c_proj_weight],
     params[:layer_5_mlp_c_proj_bias]}
  end

  defnp get_layer_params_6(params) do
    {params[:layer_6_ln_1_weight], params[:layer_6_ln_1_bias],
     params[:layer_6_attn_c_attn_weight], params[:layer_6_attn_c_attn_bias],
     params[:layer_6_attn_c_proj_weight], params[:layer_6_attn_c_proj_bias],
     params[:layer_6_ln_2_weight], params[:layer_6_ln_2_bias], params[:layer_6_mlp_c_fc_weight],
     params[:layer_6_mlp_c_fc_bias], params[:layer_6_mlp_c_proj_weight],
     params[:layer_6_mlp_c_proj_bias]}
  end

  defnp get_layer_params_7(params) do
    {params[:layer_7_ln_1_weight], params[:layer_7_ln_1_bias],
     params[:layer_7_attn_c_attn_weight], params[:layer_7_attn_c_attn_bias],
     params[:layer_7_attn_c_proj_weight], params[:layer_7_attn_c_proj_bias],
     params[:layer_7_ln_2_weight], params[:layer_7_ln_2_bias], params[:layer_7_mlp_c_fc_weight],
     params[:layer_7_mlp_c_fc_bias], params[:layer_7_mlp_c_proj_weight],
     params[:layer_7_mlp_c_proj_bias]}
  end

  defnp get_layer_params_8(params) do
    {params[:layer_8_ln_1_weight], params[:layer_8_ln_1_bias],
     params[:layer_8_attn_c_attn_weight], params[:layer_8_attn_c_attn_bias],
     params[:layer_8_attn_c_proj_weight], params[:layer_8_attn_c_proj_bias],
     params[:layer_8_ln_2_weight], params[:layer_8_ln_2_bias], params[:layer_8_mlp_c_fc_weight],
     params[:layer_8_mlp_c_fc_bias], params[:layer_8_mlp_c_proj_weight],
     params[:layer_8_mlp_c_proj_bias]}
  end

  defnp get_layer_params_9(params) do
    {params[:layer_9_ln_1_weight], params[:layer_9_ln_1_bias],
     params[:layer_9_attn_c_attn_weight], params[:layer_9_attn_c_attn_bias],
     params[:layer_9_attn_c_proj_weight], params[:layer_9_attn_c_proj_bias],
     params[:layer_9_ln_2_weight], params[:layer_9_ln_2_bias], params[:layer_9_mlp_c_fc_weight],
     params[:layer_9_mlp_c_fc_bias], params[:layer_9_mlp_c_proj_weight],
     params[:layer_9_mlp_c_proj_bias]}
  end

  defnp get_layer_params_10(params) do
    {params[:layer_10_ln_1_weight], params[:layer_10_ln_1_bias],
     params[:layer_10_attn_c_attn_weight], params[:layer_10_attn_c_attn_bias],
     params[:layer_10_attn_c_proj_weight], params[:layer_10_attn_c_proj_bias],
     params[:layer_10_ln_2_weight], params[:layer_10_ln_2_bias],
     params[:layer_10_mlp_c_fc_weight], params[:layer_10_mlp_c_fc_bias],
     params[:layer_10_mlp_c_proj_weight], params[:layer_10_mlp_c_proj_bias]}
  end

  defnp get_layer_params_11(params) do
    {params[:layer_11_ln_1_weight], params[:layer_11_ln_1_bias],
     params[:layer_11_attn_c_attn_weight], params[:layer_11_attn_c_attn_bias],
     params[:layer_11_attn_c_proj_weight], params[:layer_11_attn_c_proj_bias],
     params[:layer_11_ln_2_weight], params[:layer_11_ln_2_bias],
     params[:layer_11_mlp_c_fc_weight], params[:layer_11_mlp_c_fc_bias],
     params[:layer_11_mlp_c_proj_weight], params[:layer_11_mlp_c_proj_bias]}
  end

  defnp block_incremental_0(hidden, params, past_kv) do
    block_incremental_impl_0(hidden, params, past_kv)
  end

  defnp block_incremental_1(hidden, params, past_kv) do
    block_incremental_impl_1(hidden, params, past_kv)
  end

  defnp block_incremental_2(hidden, params, past_kv) do
    block_incremental_impl_2(hidden, params, past_kv)
  end

  defnp block_incremental_3(hidden, params, past_kv) do
    block_incremental_impl_3(hidden, params, past_kv)
  end

  defnp block_incremental_4(hidden, params, past_kv) do
    block_incremental_impl_4(hidden, params, past_kv)
  end

  defnp block_incremental_5(hidden, params, past_kv) do
    block_incremental_impl_5(hidden, params, past_kv)
  end

  defnp block_incremental_6(hidden, params, past_kv) do
    block_incremental_impl_6(hidden, params, past_kv)
  end

  defnp block_incremental_7(hidden, params, past_kv) do
    block_incremental_impl_7(hidden, params, past_kv)
  end

  defnp block_incremental_8(hidden, params, past_kv) do
    block_incremental_impl_8(hidden, params, past_kv)
  end

  defnp block_incremental_9(hidden, params, past_kv) do
    block_incremental_impl_9(hidden, params, past_kv)
  end

  defnp block_incremental_10(hidden, params, past_kv) do
    block_incremental_impl_10(hidden, params, past_kv)
  end

  defnp block_incremental_11(hidden, params, past_kv) do
    block_incremental_impl_11(hidden, params, past_kv)
  end

  defnp block_incremental_impl_0(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_0(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_1(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_1(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_2(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_2(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_3(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_3(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_4(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_4(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_5(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_5(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_6(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_6(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_7(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_7(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_8(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_8(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_9(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_9(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_10(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_10(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_11(hidden, params, past_kv) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_11(params)
    block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
      c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
  end

  defnp block_incremental_impl_body(hidden, past_kv, ln1_w, ln1_b, c_attn_w, c_attn_b,
         c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b) do
    {past_k, past_v} = past_kv
    attn_in = layer_norm(hidden, ln1_w, ln1_b)
    {attn_out, new_kv} =
      attn_incremental(attn_in, c_attn_w, c_attn_b, c_proj_w, c_proj_b, past_k, past_v)
    h = Nx.add(hidden, attn_out)
    mlp_in = layer_norm(h, ln2_w, ln2_b)
    mlp_out = mlp(mlp_in, c_fc_w, c_fc_b, c_proj_mlp_w, c_proj_mlp_b)
    {Nx.add(h, mlp_out), new_kv}
  end

  defnp attn_with_cache(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b) do
    qkv = Nx.dot(x, [2], c_attn_w, [1])
    qkv = Nx.add(qkv, Nx.reshape(c_attn_b, {1, 1, 2304}))
    {batch, seq, _} = Nx.shape(qkv)
    q = qkv |> Nx.slice_along_axis(0, @n_embd, axis: 2)
    k = qkv |> Nx.slice_along_axis(@n_embd, @n_embd, axis: 2)
    v = qkv |> Nx.slice_along_axis(2 * @n_embd, @n_embd, axis: 2)
    q = Nx.reshape(q, {batch, seq, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = Nx.reshape(k, {batch, seq, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = Nx.reshape(v, {batch, seq, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    scale = Nx.sqrt(Nx.tensor(@head_dim, type: {:f, 32}))
    k_t = Nx.transpose(k, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], k_t, [2], [0, 1]) |> Nx.divide(scale)
    row = Nx.iota({seq}, type: {:s, 32}) |> Nx.new_axis(-1)
    col = Nx.iota({seq}, type: {:s, 32}) |> Nx.new_axis(0)
    mask = Nx.greater(col, row)

    mask =
      Nx.select(
        mask,
        Nx.broadcast(Nx.tensor(-1.0e10, type: {:f, 32}), {seq, seq}),
        Nx.broadcast(0.0, {seq, seq})
      )

    mask = Nx.reshape(mask, {1, 1, seq, seq}) |> Nx.as_type({:f, 32})
    scores = Nx.add(scores, mask)
    e = Nx.exp(scores)
    probs = Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
    out = Nx.dot(probs, [3], [0, 1], v, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq, @n_embd})
    out = Nx.dot(out, [2], c_proj_w, [1])
    out = Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd}))
    {out, {k, v}}
  end

  defnp attn_incremental(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, past_k, past_v) do
    qkv = Nx.dot(x, [2], c_attn_w, [1])
    qkv = Nx.add(qkv, Nx.reshape(c_attn_b, {1, 1, 2304}))
    {batch, _seq, _} = Nx.shape(qkv)
    q = qkv |> Nx.slice_along_axis(0, @n_embd, axis: 2)
    k = qkv |> Nx.slice_along_axis(@n_embd, @n_embd, axis: 2)
    v = qkv |> Nx.slice_along_axis(2 * @n_embd, @n_embd, axis: 2)
    q = Nx.reshape(q, {batch, 1, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = Nx.reshape(k, {batch, 1, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = Nx.reshape(v, {batch, 1, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    new_k = Nx.concatenate([past_k, k], axis: 2)
    new_v = Nx.concatenate([past_v, v], axis: 2)
    scale = Nx.sqrt(Nx.tensor(@head_dim, type: {:f, 32}))
    new_k_t = Nx.transpose(new_k, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], new_k_t, [2], [0, 1]) |> Nx.divide(scale)
    e = Nx.exp(scores)
    probs = Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
    out = Nx.dot(probs, [3], [0, 1], new_v, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3]) |> Nx.reshape({batch, 1, @n_embd})
    out = Nx.dot(out, [2], c_proj_w, [1])
    out = Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd}))
    {out, {new_k, new_v}}
  end

  defnp mlp(x, c_fc_w, c_fc_b, c_proj_w, c_proj_b) do
    h = Nx.dot(x, [2], c_fc_w, [1])
    h = Nx.add(h, Nx.reshape(c_fc_b, {1, 1, 3072}))
    h = gelu(h)
    out = Nx.dot(h, [2], c_proj_w, [1])
    Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd}))
  end

  defnp gelu(x) do
    half = Nx.tensor(0.5, type: {:f, 32})
    x_scaled = Nx.divide(x, Nx.sqrt(Nx.tensor(2.0, type: {:f, 32})))
    Nx.multiply(Nx.multiply(half, x), Nx.add(1.0, Nx.erf(x_scaled)))
  end
end
