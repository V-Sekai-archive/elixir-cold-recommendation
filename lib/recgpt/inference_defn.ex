defmodule RecGPT.InferenceDefn do
  @moduledoc """
  Defn entry points for EXLA JIT: forward_with_cache/4 and forward_incremental/5 only.

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

  defn forward_incremental(batch_token_ids, batch_aux, embed_mask, params, past_cache, past_len) do
    {hidden, new_cache} =
      forward_hidden_incremental(
        batch_token_ids,
        batch_aux,
        embed_mask,
        params,
        past_cache,
        past_len
      )

    last_hidden = Nx.squeeze(hidden, axes: [1])
    logits = apply_head(last_hidden, params)
    {logits, new_cache}
  end

  @doc """
  Fused beam search: one Defn for step 0 + steps 1–3 (four forwards in a single graph).
  Reduces kernel launch overhead. All inputs must be on the same EXLA backend.

  context_len_scalar: scalar tensor (type {:s, 32}) with the context length (same as axis 1 of context_tokens).
  Returns {item_ids, beam_scores, prefix_tokens} for the single sync and post-decode.
  Does not support initial_step0 (context cache); use unfused path when cache hit.
  """
  defn beam_search_fused(
        context_tokens,
        context_len_scalar,
        past_len_offset_1,
        past_len_offset_2,
        past_len_offset_3,
        batch_aux_0,
        embed_mask_0,
        params,
        next_state,
        item_at_leaf,
        root_state,
        neg_inf,
        vocab_t,
        beam_width
      ) do
    # Step 0: full forward, then trie/top_k
    {logits_0, cache} = forward_with_cache(context_tokens, batch_aux_0, embed_mask_0, params)
    # logits_0 {1, vocab_size}
    context_len = Nx.axis_size(context_tokens, 1)
    vocab_size = elem(Nx.shape(next_state), 1)

    valid_0 = Nx.gather(next_state, Nx.reshape(root_state, {1, 1})) |> Nx.reshape({:auto})
    valid_mask_0 = Nx.greater_equal(valid_0, 0)
    scores_0 = Nx.select(valid_mask_0, Nx.reshape(logits_0, {:auto}), neg_inf)
    {top_scores_0, top_indices_0} = Nx.top_k(scores_0, k: beam_width)
    top_token_ids_0 = Nx.reshape(top_indices_0, {:auto}) |> Nx.as_type({:s, 32})
    state_ids_0 = gather_2d_defn(next_state, root_state, top_token_ids_0)
    prefix_tokens_0 = Nx.reshape(top_token_ids_0, {beam_width, 1})
    beam_scores_0 = Nx.as_type(top_scores_0, {:f, 32})

    cache_rep = replicate_cache_defn(cache, beam_width)
    aux_incr = Nx.broadcast(Nx.tensor(0, type: Nx.type(batch_aux_0)), {beam_width, 1, 192})
    mask_incr = Nx.broadcast(Nx.tensor(1, type: Nx.type(embed_mask_0)), {beam_width, 1, 1})

    # Step 1
    {state_ids_1, prefix_tokens_1, beam_scores_1, cache_1} =
      beam_step_defn(
        next_state,
        item_at_leaf,
        context_tokens,
        context_len,
        context_len_scalar,
        past_len_offset_1,
        1,
        beam_width,
        vocab_size,
        state_ids_0,
        prefix_tokens_0,
        beam_scores_0,
        cache_rep,
        params,
        aux_incr,
        mask_incr,
        neg_inf,
        vocab_t
      )

    # Step 2
    {state_ids_2, prefix_tokens_2, beam_scores_2, cache_2} =
      beam_step_defn(
        next_state,
        item_at_leaf,
        context_tokens,
        context_len,
        context_len_scalar,
        past_len_offset_2,
        2,
        beam_width,
        vocab_size,
        state_ids_1,
        prefix_tokens_1,
        beam_scores_1,
        cache_1,
        params,
        aux_incr,
        mask_incr,
        neg_inf,
        vocab_t
      )

    # Step 3: returns item_ids and prefix_tokens
    {_state_ids_3, prefix_tokens_3, beam_scores_3, _cache_3, item_ids} =
      beam_step_defn(
        next_state,
        item_at_leaf,
        context_tokens,
        context_len,
        context_len_scalar,
        past_len_offset_3,
        3,
        beam_width,
        vocab_size,
        state_ids_2,
        prefix_tokens_2,
        beam_scores_2,
        cache_2,
        params,
        aux_incr,
        mask_incr,
        neg_inf,
        vocab_t
      )

    {item_ids, beam_scores_3, prefix_tokens_3}
  end

  defnp replicate_cache_defn(cache_tuple, batch_size) do
    replicate_one = fn {k, v} ->
      {_b, n_head, len, hd} = Nx.shape(k)
      {Nx.broadcast(k, {batch_size, n_head, len, hd}), Nx.broadcast(v, {batch_size, n_head, len, hd})}
    end
    {
      replicate_one.(elem(cache_tuple, 0)),
      replicate_one.(elem(cache_tuple, 1)),
      replicate_one.(elem(cache_tuple, 2)),
      replicate_one.(elem(cache_tuple, 3)),
      replicate_one.(elem(cache_tuple, 4)),
      replicate_one.(elem(cache_tuple, 5)),
      replicate_one.(elem(cache_tuple, 6)),
      replicate_one.(elem(cache_tuple, 7)),
      replicate_one.(elem(cache_tuple, 8)),
      replicate_one.(elem(cache_tuple, 9)),
      replicate_one.(elem(cache_tuple, 10)),
      replicate_one.(elem(cache_tuple, 11))
    }
  end

  defnp gather_2d_defn(tensor, row_indices, col_indices) do
    # row_indices {1} or {k}, col_indices {k}; tensor {num_states, vocab_size} or 3D
    row_2d = Nx.new_axis(Nx.reshape(row_indices, {:auto}), -1)
    rows = Nx.gather(tensor, row_2d)
    rows = if Nx.rank(rows) == 1, do: Nx.reshape(rows, {1, :auto}), else: rows
    rows =
      if Nx.rank(rows) == 3 do
        {a, _b, c} = Nx.shape(rows)
        Nx.reshape(rows, {a, c})
      else
        rows
      end
    k = Nx.axis_size(col_indices, 0)
    {_num_rows, vocab_size} = Nx.shape(rows)
    rows = if k > 1, do: Nx.broadcast(rows, {k, vocab_size}), else: rows
    indices = Nx.reshape(col_indices, {k, 1})
    Nx.take_along_axis(rows, indices, axis: 1)
  end

  defnp beam_step_defn(
         next_state,
         item_at_leaf,
         context_tokens,
         context_len,
         context_len_scalar,
         past_len_offset,
         step,
         beam_width,
         vocab_size,
         state_ids,
         prefix_tokens,
         beam_scores,
         cache,
         params,
         aux_incr,
         mask_incr,
         neg_inf,
         vocab_t
       ) do
    prefix_len = step
    context_broadcast = Nx.broadcast(context_tokens, {beam_width, context_len})
    prefix_slice = prefix_tokens |> Nx.slice_along_axis(0, prefix_len, axis: 1)
    batch = Nx.concatenate([context_broadcast, prefix_slice], axis: 1)
    past_len = Nx.add(context_len_scalar, Nx.as_type(past_len_offset, {:s, 32}))
    last_tokens = Nx.slice_along_axis(batch, context_len + prefix_len - 1, 1, axis: 1)
    {logits, new_cache} = forward_incremental(last_tokens, aux_incr, mask_incr, params, cache, past_len)

    state_ids_safe = Nx.max(state_ids, 0)
    idx_2d = Nx.new_axis(state_ids_safe, -1)

    {valid_rows, transition_tensor} =
      if step == 3 do
        {Nx.gather(item_at_leaf, idx_2d) |> Nx.reshape({beam_width, vocab_size}), item_at_leaf}
      else
        {Nx.gather(next_state, idx_2d) |> Nx.reshape({beam_width, vocab_size}), next_state}
      end

    valid_mask = Nx.greater_equal(valid_rows, 0)
    scores_per_token = Nx.select(valid_mask, logits, neg_inf)
    beam_scores_broadcast = Nx.new_axis(beam_scores, 1)
    scores_per_token = Nx.add(scores_per_token, beam_scores_broadcast)
    flat = Nx.reshape(scores_per_token, {:auto})
    {top_scores, top_flat} = Nx.top_k(flat, k: beam_width)
    top_flat = Nx.as_type(top_flat, {:s, 32})
    batch_indices = Nx.quotient(top_flat, vocab_t)
    token_ids = Nx.remainder(top_flat, vocab_t)

    state_at_top = Nx.gather(state_ids, Nx.new_axis(batch_indices, -1)) |> Nx.reshape({:auto})
    state_at_top_safe = Nx.max(state_at_top, 0)

    new_state_ids =
      if step == 3 do
        gather_2d_defn(item_at_leaf, state_at_top_safe, token_ids) |> Nx.squeeze(axes: [1])
      else
        gather_2d_defn(transition_tensor, state_at_top_safe, token_ids) |> Nx.squeeze(axes: [1])
      end

    old_prefixes =
      Nx.gather(prefix_tokens, Nx.new_axis(batch_indices, -1))
      |> Nx.reshape({beam_width, prefix_len})

    new_col = Nx.reshape(token_ids, {beam_width, 1})
    new_prefix_tokens = Nx.concatenate([old_prefixes, new_col], axis: 1)

    item_ids = if step == 3, do: new_state_ids, else: new_state_ids
    {new_state_ids, new_prefix_tokens, top_scores, new_cache, item_ids}
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

  defnp forward_hidden_incremental(
          batch_token_ids,
          batch_aux,
          embed_mask,
          params,
          past_cache,
          past_len
        ) do
    wte = params[:wte]
    {batch, _seq_len} = Nx.shape(batch_token_ids)
    flat_ids = Nx.reshape(batch_token_ids, {batch})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, 1, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux, embed_mask, params)
    combined = Nx.add(token_embeds, aux_768)
    h = add_wpe_at_position(combined, past_len, params)
    {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11} = past_cache
    {h, c0} = block_incremental_0(h, params, c0, past_len)
    {h, c1} = block_incremental_1(h, params, c1, past_len)
    {h, c2} = block_incremental_2(h, params, c2, past_len)
    {h, c3} = block_incremental_3(h, params, c3, past_len)
    {h, c4} = block_incremental_4(h, params, c4, past_len)
    {h, c5} = block_incremental_5(h, params, c5, past_len)
    {h, c6} = block_incremental_6(h, params, c6, past_len)
    {h, c7} = block_incremental_7(h, params, c7, past_len)
    {h, c8} = block_incremental_8(h, params, c8, past_len)
    {h, c9} = block_incremental_9(h, params, c9, past_len)
    {h, c10} = block_incremental_10(h, params, c10, past_len)
    {h, c11} = block_incremental_11(h, params, c11, past_len)
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

  defnp add_wpe_at_position(hidden, past_len, params) do
    wpe = params[:wpe]
    # past_len is a scalar tensor; Nx.slice supports dynamic start indices
    pe_row = Nx.slice(wpe, [past_len, 0], [1, @n_embd])
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

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_1(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_1(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_2(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_2(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_3(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_3(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_4(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_4(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_5(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_5(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_6(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_6(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_7(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_7(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_8(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_8(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_9(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_9(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_10(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_10(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_11(hidden, params) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_11(params)

    block_with_cache_impl(
      hidden,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_with_cache_impl(
          hidden,
          ln1_w,
          ln1_b,
          c_attn_w,
          c_attn_b,
          c_proj_w,
          c_proj_b,
          ln2_w,
          ln2_b,
          c_fc_w,
          c_fc_b,
          c_proj_mlp_w,
          c_proj_mlp_b
        ) do
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

  defnp block_incremental_0(hidden, params, past_kv, past_len) do
    block_incremental_impl_0(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_1(hidden, params, past_kv, past_len) do
    block_incremental_impl_1(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_2(hidden, params, past_kv, past_len) do
    block_incremental_impl_2(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_3(hidden, params, past_kv, past_len) do
    block_incremental_impl_3(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_4(hidden, params, past_kv, past_len) do
    block_incremental_impl_4(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_5(hidden, params, past_kv, past_len) do
    block_incremental_impl_5(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_6(hidden, params, past_kv, past_len) do
    block_incremental_impl_6(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_7(hidden, params, past_kv, past_len) do
    block_incremental_impl_7(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_8(hidden, params, past_kv, past_len) do
    block_incremental_impl_8(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_9(hidden, params, past_kv, past_len) do
    block_incremental_impl_9(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_10(hidden, params, past_kv, past_len) do
    block_incremental_impl_10(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_11(hidden, params, past_kv, past_len) do
    block_incremental_impl_11(hidden, params, past_kv, past_len)
  end

  defnp block_incremental_impl_0(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_0(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_1(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_1(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_2(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_2(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_3(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_3(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_4(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_4(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_5(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_5(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_6(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_6(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_7(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_7(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_8(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_8(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_9(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_9(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_10(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_10(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_11(hidden, params, past_kv, past_len) do
    {ln1_w, ln1_b, c_attn_w, c_attn_b, c_proj_w, c_proj_b, ln2_w, ln2_b, c_fc_w, c_fc_b,
     c_proj_mlp_w, c_proj_mlp_b} = get_layer_params_11(params)

    block_incremental_impl_body(
      hidden,
      past_kv,
      past_len,
      ln1_w,
      ln1_b,
      c_attn_w,
      c_attn_b,
      c_proj_w,
      c_proj_b,
      ln2_w,
      ln2_b,
      c_fc_w,
      c_fc_b,
      c_proj_mlp_w,
      c_proj_mlp_b
    )
  end

  defnp block_incremental_impl_body(
          hidden,
          past_kv,
          past_len,
          ln1_w,
          ln1_b,
          c_attn_w,
          c_attn_b,
          c_proj_w,
          c_proj_b,
          ln2_w,
          ln2_b,
          c_fc_w,
          c_fc_b,
          c_proj_mlp_w,
          c_proj_mlp_b
        ) do
    {past_k, past_v} = past_kv
    attn_in = layer_norm(hidden, ln1_w, ln1_b)

    {attn_out, new_kv} =
      attn_incremental(attn_in, c_attn_w, c_attn_b, c_proj_w, c_proj_b, past_k, past_v, past_len)

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
    ttype = Nx.type(c_attn_w)
    scale = Nx.sqrt(Nx.tensor(@head_dim, type: ttype))
    k_t = Nx.transpose(k, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], k_t, [2], [0, 1]) |> Nx.divide(scale)
    row = Nx.iota({seq}, type: {:s, 32}) |> Nx.new_axis(-1)
    col = Nx.iota({seq}, type: {:s, 32}) |> Nx.new_axis(0)
    mask = Nx.greater(col, row)

    mask =
      Nx.select(
        mask,
        Nx.broadcast(Nx.tensor(-1.0e10, type: ttype), {seq, seq}),
        Nx.broadcast(Nx.tensor(0.0, type: ttype), {seq, seq})
      )

    mask = Nx.reshape(mask, {1, 1, seq, seq}) |> Nx.as_type(ttype)
    scores = Nx.add(scores, mask)
    e = Nx.exp(scores)
    probs = Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
    out = Nx.dot(probs, [3], [0, 1], v, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq, @n_embd})
    out = Nx.dot(out, [2], c_proj_w, [1])
    out = Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd}))
    {out, {k, v}}
  end

  defnp attn_incremental(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, past_k, past_v, past_len) do
    qkv = Nx.dot(x, [2], c_attn_w, [1])
    qkv = Nx.add(qkv, Nx.reshape(c_attn_b, {1, 1, 2304}))
    {batch, _seq, _} = Nx.shape(qkv)
    q = qkv |> Nx.slice_along_axis(0, @n_embd, axis: 2)
    k = qkv |> Nx.slice_along_axis(@n_embd, @n_embd, axis: 2)
    v = qkv |> Nx.slice_along_axis(2 * @n_embd, @n_embd, axis: 2)
    q = Nx.reshape(q, {batch, 1, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = Nx.reshape(k, {batch, 1, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = Nx.reshape(v, {batch, 1, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    # Padded cache: write new k,v at position past_len (stable shape for EXLA JIT)
    new_k = Nx.put_slice(past_k, [0, 0, past_len, 0], k)
    new_v = Nx.put_slice(past_v, [0, 0, past_len, 0], v)
    ttype = Nx.type(c_attn_w)
    scale = Nx.sqrt(Nx.tensor(@head_dim, type: ttype))
    new_k_t = Nx.transpose(new_k, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], new_k_t, [2], [0, 1]) |> Nx.divide(scale)
    # Mask positions > past_len (padded slots) for stable attention
    max_len = elem(Nx.shape(past_k), 2)
    indices = Nx.iota({max_len}, type: {:s, 32})
    invalid = Nx.greater(indices, past_len)

    mask =
      Nx.select(
        invalid,
        Nx.broadcast(Nx.tensor(-1.0e10, type: ttype), {max_len}),
        Nx.broadcast(Nx.tensor(0.0, type: ttype), {max_len})
      )

    mask = Nx.reshape(mask, {1, 1, 1, max_len})
    scores = Nx.add(scores, mask)
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
    ttype = Nx.type(x)
    half = Nx.tensor(0.5, type: ttype)
    x_scaled = Nx.divide(x, Nx.sqrt(Nx.tensor(2.0, type: ttype)))
    Nx.multiply(Nx.multiply(half, x), Nx.add(1.0, Nx.erf(x_scaled)))
  end
end
