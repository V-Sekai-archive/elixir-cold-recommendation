defmodule RecGPT.Inference do
  @moduledoc """
  RecGPT inference: token embed + aux fusion + GPT-2 body + prediction head.

  Params from RecGPT.CheckpointLoader.load_from_export/1. Expected keys
  (see docs/07_recgpt_checkpoint_layout.md):
  - wte or gpt2model.wte: (15_361, 768) token embedding table
  - gpt2model.wpe (optional): position embeddings; added to hidden if present
  - gpt2model.h.{i}.attn.*, ln_1, mlp.*, ln_2: transformer blocks (when present)
  - gpt2model.ln_f: final layer norm
  - ae.*: aux encoder linear 192->768 + optional LayerNorm
  - pred_head.weight, pred_head.bias: Linear(768, 15_361)

  forward/4 returns logits (batch, 15_361) for the last position. When checkpoint
  includes GPT-2 layer params (e.g. gpt2model.h.0.attn.c_attn.weight), the full
  backbone runs; otherwise a stub (last-position combined embed) is used.
  """

  @n_embd 768
  @n_head 12
  @head_dim 64

  @doc """
  Forward pass. batch_token_ids: (batch, seq_len), batch_aux_embeds: (batch, seq_len, 192),
  embed_mask: (batch, seq_len, 1), params: map from CheckpointLoader.
  Returns logits (batch, 15_361) for the last position.
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def forward(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    hidden = forward_hidden(batch_token_ids, batch_aux_embeds, embed_mask, params)
    # Last position only: (batch, 768) -> (batch, 15_361)
    last_idx = elem(Nx.shape(batch_token_ids), 1) - 1
    last_hidden = hidden |> Nx.slice_along_axis(last_idx, 1, axis: 1) |> Nx.squeeze(axes: [1])
    apply_head(last_hidden, params)
  end

  @doc """
  Full-sequence forward for training. Same as forward/4 but returns logits for every position.
  Returns logits (batch, seq_len, 15_361) for use with Training.loss_shifted_ce/2.
  """
  @spec forward_full_sequence(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def forward_full_sequence(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    hidden = forward_hidden(batch_token_ids, batch_aux_embeds, embed_mask, params)
    apply_head(hidden, params)
  end

  @doc """
  Full forward and return KV-cache for the sequence. Use for the first decode step.
  Returns `{logits, past_key_values}`. `past_key_values` is a list of `{k, v}` per layer,
  each shape `(batch, n_head, seq_len, head_dim)`. When `gpt2_n_layers` is 0, returns `{logits, []}`.
  """
  @spec forward_with_cache(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) ::
          {Nx.Tensor.t(), list({Nx.Tensor.t(), Nx.Tensor.t()})}
  def forward_with_cache(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    case gpt2_n_layers(params) do
      0 ->
        hidden = forward_hidden(batch_token_ids, batch_aux_embeds, embed_mask, params)
        last_idx = elem(Nx.shape(batch_token_ids), 1) - 1
        last_hidden = hidden |> Nx.slice_along_axis(last_idx, 1, axis: 1) |> Nx.squeeze(axes: [1])
        {apply_head(last_hidden, params), []}

      n_layers ->
        {hidden, cache} =
          forward_hidden_with_cache(
            batch_token_ids,
            batch_aux_embeds,
            embed_mask,
            params,
            n_layers
          )

        last_idx = elem(Nx.shape(batch_token_ids), 1) - 1
        last_hidden = hidden |> Nx.slice_along_axis(last_idx, 1, axis: 1) |> Nx.squeeze(axes: [1])
        {apply_head(last_hidden, params), cache}
    end
  end

  @doc """
  Incremental forward for one new token using KV-cache. `batch_token_ids` must be `(batch, 1)`.
  Returns `{logits, new_past_key_values}`. When `gpt2_n_layers` is 0, runs full forward on the single token.
  """
  @spec forward_incremental(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          map(),
          list({Nx.Tensor.t(), Nx.Tensor.t()})
        ) :: {Nx.Tensor.t(), list({Nx.Tensor.t(), Nx.Tensor.t()})}
  def forward_incremental(batch_token_ids, batch_aux_embeds, embed_mask, params, past_key_values) do
    case gpt2_n_layers(params) do
      0 ->
        logits = forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
        {logits, []}

      _n_layers when past_key_values == [] ->
        logits = forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
        {logits, []}

      n_layers ->
        {hidden, new_cache} =
          forward_hidden_incremental(
            batch_token_ids,
            batch_aux_embeds,
            embed_mask,
            params,
            past_key_values,
            n_layers
          )

        last_hidden = hidden |> Nx.squeeze(axes: [1])
        {apply_head(last_hidden, params), new_cache}
    end
  end

  defp forward_hidden(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    wte = get_wte(params)
    {batch, seq_len} = Nx.shape(batch_token_ids)

    # 1. Token embedding lookup
    flat_ids = Nx.reshape(batch_token_ids, {batch * seq_len})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, seq_len, @n_embd})

    # 2. Aux: linear 192->768, optional LayerNorm, mask; add to token embeds
    aux_768 = apply_aux_encoder(batch_aux_embeds, embed_mask, params)
    combined = Nx.add(token_embeds, aux_768)

    # 3. GPT-2 backbone (or stub) -> (batch, seq_len, 768)
    case gpt2_n_layers(params) do
      0 ->
        combined

      n_layers ->
        h = add_wpe(combined, params, seq_len)
        h = run_gpt2_blocks(h, params, n_layers)
        apply_ln_f(h, params)
    end
  end

  defp forward_hidden_with_cache(batch_token_ids, batch_aux_embeds, embed_mask, params, n_layers) do
    wte = get_wte(params)
    {batch, seq_len} = Nx.shape(batch_token_ids)
    flat_ids = Nx.reshape(batch_token_ids, {batch * seq_len})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, seq_len, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux_embeds, embed_mask, params)
    combined = Nx.add(token_embeds, aux_768)
    h = add_wpe(combined, params, seq_len)
    {h, cache} = run_gpt2_blocks_with_cache(h, params, n_layers)
    {apply_ln_f(h, params), cache}
  end

  defp forward_hidden_incremental(
         batch_token_ids,
         batch_aux_embeds,
         embed_mask,
         params,
         past_key_values,
         n_layers
       ) do
    wte = get_wte(params)
    {batch, _seq_len} = Nx.shape(batch_token_ids)
    # (batch, 1) -> embed
    flat_ids = Nx.reshape(batch_token_ids, {batch})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, 1, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux_embeds, embed_mask, params)
    combined = Nx.add(token_embeds, aux_768)
    past_len = elem(Nx.shape(elem(Enum.at(past_key_values, 0), 0)), 2)
    h = add_wpe_at_position(combined, params, past_len)
    {h, new_cache} = run_gpt2_blocks_incremental(h, params, n_layers, past_key_values)
    {apply_ln_f(h, params), new_cache}
  end

  defp add_wpe(hidden, params, seq_len) do
    wpe =
      params["gpt2model.wpe"] || params["gpt2model.wpe.weight"] || params["transformer.wpe"] ||
        params["transformer.wpe.weight"]

    case wpe do
      nil ->
        hidden

      _ ->
        # wpe: (max_pos, 768); we need (seq_len, 768)
        indices = Nx.iota({seq_len}, type: {:s, 32})
        pe = Nx.gather(wpe, Nx.new_axis(indices, -1))
        pe = Nx.reshape(pe, {1, seq_len, @n_embd})
        Nx.add(hidden, pe)
    end
  end

  defp add_wpe_at_position(hidden, params, position) do
    wpe =
      params["gpt2model.wpe"] || params["gpt2model.wpe.weight"] || params["transformer.wpe"] ||
        params["transformer.wpe.weight"]

    case wpe do
      nil ->
        hidden

      _ ->
        # hidden (batch, 1, 768); add pe at position
        pe_row = Nx.slice_along_axis(wpe, position, 1, axis: 0)
        pe = Nx.reshape(pe_row, {1, 1, @n_embd})
        Nx.add(hidden, pe)
    end
  end

  @doc """
  Returns the number of GPT-2 transformer layers in params (0 when no layer keys present).
  Used by Serve/InferenceParams to build full defn params.
  """
  @spec n_layers_from_params(map()) :: non_neg_integer()
  def n_layers_from_params(params) do
    prefix = gpt2_prefix(params)
    if is_nil(prefix), do: 0, else: count_gpt2_layers(params, prefix)
  end

  @doc """
  Returns true when params are a FuXi-Linear checkpoint (fuxi.block.0.* keys).
  Used by Serve to choose FuxiLinearInferenceDefn over InferenceDefn.
  """
  @spec fuxi_checkpoint?(map()) :: boolean()
  def fuxi_checkpoint?(params) when is_map(params) do
    Enum.any?(Map.keys(params), fn
      k when is_binary(k) -> String.starts_with?(k, "fuxi.block.")
      _ -> false
    end)
  end

  defp gpt2_n_layers(params) do
    prefix = gpt2_prefix(params)
    if is_nil(prefix), do: 0, else: count_gpt2_layers(params, prefix)
  end

  defp gpt2_prefix(params) do
    keys = Map.keys(params)

    cond do
      Enum.any?(keys, &String.starts_with?(&1, "gpt2model.h.")) -> "gpt2model."
      Enum.any?(keys, &String.starts_with?(&1, "transformer.h.")) -> "transformer."
      true -> nil
    end
  end

  defp count_gpt2_layers(params, prefix) do
    pattern = prefix <> "h."

    layer_indices =
      params
      |> Map.keys()
      |> Enum.filter(&String.starts_with?(&1, pattern))
      |> Enum.map(fn k ->
        rest = String.replace_prefix(k, pattern, "")

        case Integer.parse(rest) do
          {i, _} -> i
          :error -> -1
        end
      end)
      |> Enum.reject(&(&1 < 0))

    max_idx = Enum.max(layer_indices ++ [-1])
    max_idx + 1
  end

  defp run_gpt2_blocks(hidden, params, n_layers) do
    prefix = gpt2_prefix(params)

    Enum.reduce(0..(n_layers - 1), hidden, fn i, h ->
      gpt2_block(h, params, prefix, i)
    end)
  end

  defp run_gpt2_blocks_with_cache(hidden, params, n_layers) do
    prefix = gpt2_prefix(params)

    Enum.reduce(0..(n_layers - 1), {hidden, []}, fn i, {h, acc_cache} ->
      {out, kv} = gpt2_block_with_cache(h, params, prefix, i)
      {out, acc_cache ++ [kv]}
    end)
  end

  defp run_gpt2_blocks_incremental(hidden, params, n_layers, past_key_values) do
    prefix = gpt2_prefix(params)

    Enum.reduce(Enum.zip(0..(n_layers - 1), past_key_values), {hidden, []}, fn {i,
                                                                                {past_k, past_v}},
                                                                               {h, acc_cache} ->
      {out, new_kv} = gpt2_block_incremental(h, params, prefix, i, past_k, past_v)
      {out, acc_cache ++ [new_kv]}
    end)
  end

  defp gpt2_block(hidden, params, prefix, i) do
    base = prefix <> "h.#{i}."
    # Pre-norm attention: h = h + attn(ln_1(h))
    ln1_w = params[base <> "ln_1.weight"]
    ln1_b = params[base <> "ln_1.bias"]
    attn_in = if ln1_w && ln1_b, do: layer_norm(hidden, ln1_w, ln1_b), else: hidden
    attn_out = gpt2_attn(attn_in, params, base)
    h = Nx.add(hidden, attn_out)
    # Pre-norm MLP: h = h + mlp(ln_2(h))
    ln2_w = params[base <> "ln_2.weight"]
    ln2_b = params[base <> "ln_2.bias"]
    mlp_in = if ln2_w && ln2_b, do: layer_norm(h, ln2_w, ln2_b), else: h
    mlp_out = gpt2_mlp(mlp_in, params, base)
    Nx.add(h, mlp_out)
  end

  defp gpt2_block_with_cache(hidden, params, prefix, i) do
    base = prefix <> "h.#{i}."
    ln1_w = params[base <> "ln_1.weight"]
    ln1_b = params[base <> "ln_1.bias"]
    attn_in = if ln1_w && ln1_b, do: layer_norm(hidden, ln1_w, ln1_b), else: hidden
    {attn_out, kv} = gpt2_attn_with_cache(attn_in, params, base)
    h = Nx.add(hidden, attn_out)
    ln2_w = params[base <> "ln_2.weight"]
    ln2_b = params[base <> "ln_2.bias"]
    mlp_in = if ln2_w && ln2_b, do: layer_norm(h, ln2_w, ln2_b), else: h
    mlp_out = gpt2_mlp(mlp_in, params, base)
    {Nx.add(h, mlp_out), kv}
  end

  defp gpt2_block_incremental(hidden, params, prefix, i, past_k, past_v) do
    base = prefix <> "h.#{i}."
    ln1_w = params[base <> "ln_1.weight"]
    ln1_b = params[base <> "ln_1.bias"]
    attn_in = if ln1_w && ln1_b, do: layer_norm(hidden, ln1_w, ln1_b), else: hidden
    {attn_out, new_kv} = gpt2_attn_incremental(attn_in, params, base, past_k, past_v)
    h = Nx.add(hidden, attn_out)
    ln2_w = params[base <> "ln_2.weight"]
    ln2_b = params[base <> "ln_2.bias"]
    mlp_in = if ln2_w && ln2_b, do: layer_norm(h, ln2_w, ln2_b), else: h
    mlp_out = gpt2_mlp(mlp_in, params, base)
    {Nx.add(h, mlp_out), new_kv}
  end

  # Scaled dot-product attention (GPT-2): scores = Q K^T / sqrt(d), probs = softmax(scores + mask), out = probs @ V.
  # Layout: q,k,v (batch, n_head, seq, head_dim). Batch axes [0,1] in Nx.dot/6 so we get batched matmul, not outer product.
  # Scores: contract head_dim (axis 3 of q, axis 2 of k_t). Output: contract seq_k (axis 3 of probs, axis 2 of v).
  defp gpt2_attn(x, params, base) do
    # c_attn: (batch, seq, 768) -> (batch, seq, 2304) for q,k,v
    c_attn_w = params[base <> "attn.c_attn.weight"]
    c_attn_b = params[base <> "attn.c_attn.bias"]
    unless c_attn_w, do: raise("missing #{base}attn.c_attn.weight")
    c_attn_w = ensure_shape(c_attn_w, {2304, @n_embd})
    qkv = Nx.dot(x, [2], c_attn_w, [1])
    qkv = if c_attn_b, do: Nx.add(qkv, Nx.reshape(c_attn_b, {1, 1, 2304})), else: qkv
    {batch, seq, _} = Nx.shape(qkv)
    q = qkv |> Nx.slice_along_axis(0, @n_embd, axis: 2)
    k = qkv |> Nx.slice_along_axis(@n_embd, @n_embd, axis: 2)
    v = qkv |> Nx.slice_along_axis(2 * @n_embd, @n_embd, axis: 2)
    # Reshape to (batch, n_head, seq, head_dim)
    q = Nx.reshape(q, {batch, seq, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = Nx.reshape(k, {batch, seq, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = Nx.reshape(v, {batch, seq, @n_head, @head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    scale = Nx.sqrt(Nx.tensor(@head_dim, type: {:f, 32}))

    # Batched attention: q (batch, n_head, seq, head_dim), k (batch, n_head, seq, head_dim) -> (batch, n_head, seq, seq)
    # Use dot/6 with batch_axes [0, 1] so we get batched matmul; dot/4 would outer-product to 6D.
    k_t = Nx.transpose(k, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], k_t, [2], [0, 1]) |> Nx.divide(scale)
    # Causal mask: position i may attend only to j <= i. Mask (seq, seq): -1e10 where j > i
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

    # Batched: probs @ v; contract seq_k (axis 3 of probs, axis 2 of v)
    out = Nx.dot(probs, [3], [0, 1], v, [2], [0, 1])

    out =
      out
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq, @n_embd})

    c_proj_w = params[base <> "attn.c_proj.weight"]
    c_proj_b = params[base <> "attn.c_proj.bias"]
    c_proj_w = ensure_shape(c_proj_w, {@n_embd, @n_embd})
    out = Nx.dot(out, [2], c_proj_w, [1])
    if c_proj_b, do: Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd})), else: out
  end

  defp gpt2_attn_with_cache(x, params, base) do
    c_attn_w = params[base <> "attn.c_attn.weight"]
    c_attn_b = params[base <> "attn.c_attn.bias"]
    unless c_attn_w, do: raise("missing #{base}attn.c_attn.weight")
    c_attn_w = ensure_shape(c_attn_w, {2304, @n_embd})
    qkv = Nx.dot(x, [2], c_attn_w, [1])
    qkv = if c_attn_b, do: Nx.add(qkv, Nx.reshape(c_attn_b, {1, 1, 2304})), else: qkv
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

    out =
      out
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({batch, seq, @n_embd})

    c_proj_w = params[base <> "attn.c_proj.weight"]
    c_proj_b = params[base <> "attn.c_proj.bias"]
    c_proj_w = ensure_shape(c_proj_w, {@n_embd, @n_embd})
    out = Nx.dot(out, [2], c_proj_w, [1])
    out = if c_proj_b, do: Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd})), else: out
    {out, {k, v}}
  end

  defp gpt2_attn_incremental(x, params, base, past_k, past_v) do
    # x: (batch, 1, 768); past_k, past_v: (batch, n_head, past_len, head_dim)
    c_attn_w = params[base <> "attn.c_attn.weight"]
    c_attn_b = params[base <> "attn.c_attn.bias"]
    unless c_attn_w, do: raise("missing #{base}attn.c_attn.weight")
    c_attn_w = ensure_shape(c_attn_w, {2304, @n_embd})
    qkv = Nx.dot(x, [2], c_attn_w, [1])
    qkv = if c_attn_b, do: Nx.add(qkv, Nx.reshape(c_attn_b, {1, 1, 2304})), else: qkv
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

    # Batched: q (batch, n_head, 1, head_dim), new_k (batch, n_head, seq_len, head_dim) -> (batch, n_head, 1, seq_len)
    new_k_t = Nx.transpose(new_k, axes: [0, 1, 3, 2])
    scores = Nx.dot(q, [3], [0, 1], new_k_t, [2], [0, 1]) |> Nx.divide(scale)
    e = Nx.exp(scores)
    probs = Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))

    # Batched: probs @ new_v; contract seq_len (axis 3 of probs, axis 2 of new_v)
    out = Nx.dot(probs, [3], [0, 1], new_v, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3]) |> Nx.reshape({batch, 1, @n_embd})
    c_proj_w = params[base <> "attn.c_proj.weight"]
    c_proj_b = params[base <> "attn.c_proj.bias"]
    c_proj_w = ensure_shape(c_proj_w, {@n_embd, @n_embd})
    out = Nx.dot(out, [2], c_proj_w, [1])
    out = if c_proj_b, do: Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd})), else: out
    {out, {new_k, new_v}}
  end

  defp gpt2_mlp(x, params, base) do
    c_fc_w = params[base <> "mlp.c_fc.weight"]
    c_fc_b = params[base <> "mlp.c_fc.bias"]
    unless c_fc_w, do: raise("missing #{base}mlp.c_fc.weight")
    c_fc_w = ensure_shape(c_fc_w, {3072, @n_embd})
    h = Nx.dot(x, [2], c_fc_w, [1])
    h = if c_fc_b, do: Nx.add(h, Nx.reshape(c_fc_b, {1, 1, 3072})), else: h
    h = gelu(h)
    c_proj_w = params[base <> "mlp.c_proj.weight"]
    c_proj_b = params[base <> "mlp.c_proj.bias"]
    c_proj_w = ensure_shape(c_proj_w, {@n_embd, 3072})
    out = Nx.dot(h, [2], c_proj_w, [1])
    if c_proj_b, do: Nx.add(out, Nx.reshape(c_proj_b, {1, 1, @n_embd})), else: out
  end

  defp gelu(x) do
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    half = Nx.tensor(0.5, type: {:f, 32})
    x_scaled = Nx.divide(x, Nx.sqrt(Nx.tensor(2.0, type: {:f, 32})))
    Nx.multiply(Nx.multiply(half, x), Nx.add(1.0, Nx.erf(x_scaled)))
  end

  defp apply_ln_f(hidden, params) do
    ln_f_w = params["gpt2model.ln_f.weight"] || params["transformer.ln_f.weight"]
    ln_f_b = params["gpt2model.ln_f.bias"] || params["transformer.ln_f.bias"]
    if ln_f_w && ln_f_b, do: layer_norm(hidden, ln_f_w, ln_f_b), else: hidden
  end

  defp get_wte(params) do
    wte =
      params["gpt2model.wte"] || params["gpt2model.wte.weight"] || params["wte"] ||
        params["wte.weight"]

    if is_nil(wte), do: raise("missing wte in params")
    # RecGPT checkpoint may have GPT-2 vocab (50257); use first 15_361 for FSQ vocab
    {rows, _} = Nx.shape(wte)
    if rows >= 15_361, do: Nx.slice_along_axis(wte, 0, 15_361, axis: 0), else: wte
  end

  defp apply_aux_encoder(aux_192, mask, params) do
    weight = params["ae.linear.weight"] || params["ae.weight"] || params["linear_layer.weight"]
    bias = params["ae.linear.bias"] || params["ae.bias"]
    norm_w = params["ae.norm.weight"] || params["norm_aux.weight"]
    norm_b = params["ae.norm.bias"] || params["norm_aux.bias"]

    if weight do
      weight = ensure_shape(weight, {768, 192})
      out = Nx.dot(aux_192, [2], weight, [1])
      out = if bias, do: Nx.add(out, Nx.reshape(bias, {1, 1, 768})), else: out
      out = if norm_w && norm_b, do: layer_norm(out, norm_w, norm_b), else: out
      Nx.multiply(out, mask)
    else
      # No aux params: zeros (batch, seq_len, 768) so add to token_embeds is no-op when masked
      {b, s, _} = Nx.shape(aux_192)
      Nx.multiply(Nx.broadcast(0.0, {b, s, 768}), mask)
    end
  end

  defp layer_norm(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    x = Nx.divide(Nx.subtract(x, mean), Nx.add(Nx.sqrt(var), 1.0e-5))
    Nx.add(Nx.multiply(x, weight), bias)
  end

  defp apply_head(hidden, params) do
    weight = params["pred_head.weight"]
    bias = params["pred_head.bias"]

    if weight do
      # PyTorch Linear(768, 15_361) is (15_361, 768); Nx.dot needs (768, 15_361)
      weight = ensure_shape(weight, {768, 15_361})
      # hidden: (batch, 768) or (batch, seq_len, 768)
      shape = Nx.shape(hidden)

      logits =
        if tuple_size(shape) == 2 do
          Nx.dot(hidden, [1], weight, [0])
        else
          {batch, seq_len, _} = shape
          flat = Nx.reshape(hidden, {batch * seq_len, 768})
          out = Nx.dot(flat, [1], weight, [0])
          Nx.reshape(out, {batch, seq_len, 15_361})
        end

      if bias, do: Nx.add(logits, bias), else: logits
    else
      raise "missing pred_head.weight in params"
    end
  end

  defp ensure_shape(tensor, expected) do
    shape = Nx.shape(tensor)
    if shape == expected, do: tensor, else: Nx.transpose(tensor)
  end
end
