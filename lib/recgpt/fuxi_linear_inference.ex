defmodule RecGPT.FuxiLinearInference do
  @moduledoc """
  Full FuXi-Linear model port, using RecGPT semantic ID structure.

  **RecGPT interface (unchanged):**
  - Input: batch_token_ids (batch, seq_len), batch_aux (batch, seq_len, 192), embed_mask
  - Output: logits (batch, 15_361) for last position

  **FuXi-Linear body (full model as reference):**
  - Retention + LinearTemporalChannel + LinearPositionalChannel per block
  - uvqk projection (SiLU), u*gate, Multistage FFN
  - Padded sequences (no jagged); position indices as timestamps when none provided

  Reference: USTC-StarTeam/fuxi-linear (FuXiLinearBlockJagged, Retention, LinearTemporalChannel, LinearPositionalChannel).
  """

  @n_embd 768
  @n_head 4
  @head_dim 32
  @value_dim 128
  @n_blocks 4
  @vocab_size 15_361
  @channel_t_heads 8
  @channel_p_dim 32

  # attn_dim = value_dim * 3 (Retention + ChannelT + ChannelP)
  @attn_dim 384
  # uvqk: attn_dim + q + k + v = 384 + 128 + 128 + 128
  @uvqk_out 768

  @doc """
  Forward pass. Returns logits (batch, 15_361) for last position.
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def forward(batch_token_ids, batch_aux, embed_mask, params) do
    hidden = forward_hidden(batch_token_ids, batch_aux, embed_mask, params)
    {_batch, _seq_len, _} = Nx.shape(hidden)
    last_idx = elem(Nx.shape(batch_token_ids), 1) - 1
    last_hidden = Nx.slice_along_axis(hidden, last_idx, 1, axis: 1) |> Nx.squeeze(axes: [1])
    apply_head(last_hidden, params)
  end

  defp forward_hidden(batch_token_ids, batch_aux, embed_mask, params) do
    wte = get_wte(params)
    {batch, seq_len} = Nx.shape(batch_token_ids)

    # RecGPT semantic ID: token embed + aux encoder
    flat_ids = Nx.reshape(batch_token_ids, {batch * seq_len})
    token_embeds = Nx.gather(wte, Nx.new_axis(flat_ids, -1))
    token_embeds = Nx.reshape(token_embeds, {batch, seq_len, @n_embd})
    aux_768 = apply_aux_encoder(batch_aux, embed_mask, params)
    x = Nx.add(token_embeds, aux_768)

    # Position indices as timestamps (no real timestamps yet)
    all_timestamps = position_timestamps(batch, seq_len)
    invalid_attn_mask = causal_mask(seq_len)

    # FuXi-Linear: N blocks
    h = x
    for i <- 0..(@n_blocks - 1), reduce: h do
      acc -> fuxi_block(acc, i, seq_len, all_timestamps, invalid_attn_mask, params)
    end

    apply_ln_f(h, params)
  end

  defp position_timestamps(batch, seq_len) do
    pos = Nx.iota({seq_len}, type: {:f, 32})
    Nx.broadcast(Nx.reshape(pos, {1, seq_len, 1}), {batch, seq_len, @channel_t_heads})
  end

  defp causal_mask(seq_len) do
    row = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(-1)
    col = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(0)
    Nx.less_equal(col, row) |> Nx.as_type({:f, 32})
  end

  defp fuxi_block(hidden, block_idx, seq_len, all_timestamps, invalid_attn_mask, params) do
    base = "fuxi.block.#{block_idx}."
    normed = layer_norm(hidden, params[base <> "ln.weight"], params[base <> "ln.bias"])

    uvqk_w = params[base <> "uvqk"] || params[:"fuxi_block_#{block_idx}_uvqk"]
    if uvqk_w == nil, do: raise("missing #{base}uvqk")

    mm = Nx.dot(normed, [2], uvqk_w, [0])
    mm = silu(mm)

    u = Nx.slice_along_axis(mm, 0, @attn_dim, axis: 2)
    q = Nx.slice_along_axis(mm, @attn_dim, @value_dim, axis: 2)
    k = Nx.slice_along_axis(mm, @attn_dim + @value_dim, @value_dim, axis: 2)
    v = Nx.slice_along_axis(mm, @attn_dim + 2 * @value_dim, @value_dim, axis: 2)

    outputs = []

    # Retention
    ret_out = retention_forward(q, k, v, seq_len, invalid_attn_mask, base, params)
    outputs = [ret_out | outputs]

    # LinearTemporalChannel (position-as-time)
    ct_out = channel_t_forward(normed, seq_len, all_timestamps, invalid_attn_mask, base, params)
    outputs = [ct_out | outputs]

    # LinearPositionalChannel
    cp_out = channel_p_forward(normed, seq_len, invalid_attn_mask, base, params)
    outputs = [cp_out | outputs]

    combined = Nx.concatenate(Enum.reverse(outputs), axis: 2)
    attn_out = Nx.multiply(u, combined)

    mffn_out = mffn_forward(attn_out, hidden, base, params)
    Nx.add(hidden, mffn_out)
  end

  defp silu(x), do: Nx.multiply(x, Nx.sigmoid(x))

  # Retention: qk_attn * ts_attn, out = attn @ v
  defp retention_forward(q, k, v, seq_len, invalid_attn_mask, base, params) do
    {batch, _n, _} = Nx.shape(q)
    q = Nx.reshape(q, {batch, seq_len, @n_head, @head_dim})
    k = Nx.reshape(k, {batch, seq_len, @n_head, @head_dim})
    v = Nx.reshape(v, {batch, seq_len, @n_head, @head_dim})

    gamma_raw = params[base <> "retention.gamma"] || params[:"#{base}retention_gamma"]
    gamma = if gamma_raw do
      g = Nx.log(Nx.add(1, Nx.exp(gamma_raw)))
      g = Nx.cumulative_sum(g, axis: 0)
      Nx.exp(Nx.negate(g))
    else
      Nx.broadcast(Nx.tensor(0.9, type: {:f, 32}), {@n_head})
    end

    q_t = Nx.transpose(q, axes: [0, 2, 1, 3])
    k_t = Nx.transpose(k, axes: [0, 2, 3, 1])
    qk = Nx.dot(q_t, [3], [0, 1], k_t, [2], [0, 1])

    row = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(-1)
    col = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(0)
    diff = Nx.max(Nx.subtract(row, col), 0)
    gamma_bc = Nx.reshape(gamma, {@n_head, 1, 1})
    diff_bc = Nx.reshape(diff, {1, seq_len, seq_len})
    ts_attn = Nx.exp(Nx.negate(Nx.multiply(gamma_bc, diff_bc)))
    ts_attn = Nx.multiply(ts_attn, Nx.reshape(invalid_attn_mask, {1, seq_len, seq_len}))
    ts_attn = Nx.new_axis(ts_attn, 0)
    qk = Nx.multiply(qk, ts_attn)

    v_t = Nx.transpose(v, axes: [0, 2, 1, 3])
    out = Nx.dot(qk, [3], [0, 1], v_t, [2], [0, 1])
    out = Nx.transpose(out, axes: [0, 2, 1, 3])
    out = Nx.reshape(out, {batch, seq_len, @value_dim})
    layer_norm(out, params[base <> "retention.ln.weight"], params[base <> "retention.ln.bias"])
  end

  # LinearTemporalChannel: sinusoidal Q/K from timestamps, decay by intervals, attn @ v
  defp channel_t_forward(normed_x, seq_len, all_timestamps, invalid_attn_mask, base, params) do
    {batch, _n, _} = Nx.shape(normed_x)
    proj_w = params[base <> "channel_t.proj_v.weight"]
    v = Nx.dot(normed_x, [2], proj_w, [0])

    if seq_len < 2 do
      alpha = params[base <> "channel_t.alpha"]
      beta = params[base <> "channel_t.beta"]
      if alpha && beta, do: Nx.add(Nx.multiply(v, alpha), Nx.multiply(v, beta)), else: v
    else
      base_val = 2
      idx = Nx.iota({@channel_t_heads}, type: {:s, 64})
      intervals = Nx.pow(base_val, idx)
      scale_factor = Nx.multiply(2 * :math.pi(), Nx.pow(1 / base_val, Nx.as_type(idx, {:f, 32})))
      gamma_t = params[base <> "channel_t.gamma"] || Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {@channel_t_heads})
      gamma = Nx.sigmoid(gamma_t)

      theta = Nx.multiply(
        Nx.remainder(all_timestamps, Nx.reshape(intervals, {1, 1, @channel_t_heads})),
        Nx.reshape(scale_factor, {1, 1, @channel_t_heads})
      )
      cos_t = Nx.cos(theta)
      sin_t = Nx.sin(theta)
      k = Nx.concatenate([cos_t, cos_t, sin_t, sin_t], axis: 2)

      q_sin = Nx.concatenate([
        Nx.slice(sin_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]),
        Nx.negate(Nx.slice(cos_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]))
      ], axis: 2)
      q_cos = Nx.concatenate([
        Nx.slice(cos_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]),
        Nx.slice(sin_t, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads])
      ], axis: 2)
      q_part = Nx.concatenate([q_sin, q_cos], axis: 2)
      q_last_idx = max(0, seq_len - 2)
      q_last = Nx.slice(q_part, [0, q_last_idx, 0], [batch, 1, @channel_t_heads * 4])
      q = Nx.concatenate([q_part, q_last], axis: 1)

      interval_diff = Nx.clip(
        Nx.subtract(
          Nx.slice(all_timestamps, [0, 1, 0], [batch, seq_len - 1, @channel_t_heads]),
          Nx.slice(all_timestamps, [0, 0, 0], [batch, seq_len - 1, @channel_t_heads])
        ),
        0,
        1.0e9
      )
      interval_diff = Nx.concatenate([Nx.broadcast(0.0, {batch, 1, @channel_t_heads}), interval_diff], axis: 1)
      hinterval = Nx.multiply(interval_diff, Nx.reshape(scale_factor, {1, 1, @channel_t_heads}))
      log_decay_pos = Nx.multiply(hinterval, Nx.negate(Nx.log(gamma)))
      log_decay_pos = Nx.concatenate([log_decay_pos, log_decay_pos], axis: 2)

      decay_map = ext_decay_attn_map(log_decay_pos, seq_len)
      decay_map = Nx.multiply(decay_map, Nx.reshape(invalid_attn_mask, {1, seq_len, seq_len}))
      decay_map = Nx.reshape(decay_map, {batch, 16, 1, seq_len, seq_len}) |> Nx.broadcast({batch, 16, 2, seq_len, seq_len}) |> Nx.reshape({batch, 32, seq_len, seq_len})

      q_4d = Nx.reshape(q, {batch, seq_len, @channel_t_heads * 4, 1})
      k_4d = Nx.reshape(k, {batch, seq_len, @channel_t_heads * 4, 1})
      q_t = Nx.transpose(q_4d, axes: [0, 2, 1, 3])
      k_t = Nx.transpose(k_4d, axes: [0, 2, 3, 1])
      qk = Nx.dot(q_t, [3], [0, 1], k_t, [2], [0, 1])
      attn_maps = Nx.multiply(qk, decay_map)

      v_per_head = div(@value_dim, @channel_t_heads * 4)
      v_4d = Nx.reshape(v, {batch, seq_len, @channel_t_heads * 4, v_per_head})
      v_4d = Nx.transpose(v_4d, axes: [0, 2, 1, 3])
      out = Nx.dot(attn_maps, [3], [0, 1], v_4d, [2], [0, 1])
      out = Nx.transpose(out, axes: [0, 2, 1, 3])
      out = Nx.reshape(out, {batch, seq_len, @value_dim})

      alpha = params[base <> "channel_t.alpha"]
      beta = params[base <> "channel_t.beta"]
      if alpha && beta do
        Nx.add(Nx.multiply(out, alpha), Nx.multiply(v, beta))
      else
        out
      end
    end
  end

  # decay[i,j] = exp(-(cumsum[j+1]-cumsum[i])); cumsum over log_decay with leading 0
  defp ext_decay_attn_map(log_decay, seq_len) do
    {batch, _n, n_h} = Nx.shape(log_decay)
    ext_log = Nx.concatenate([Nx.broadcast(0.0, {batch, 1, n_h}), log_decay], axis: 1)
    cumsum = Nx.cumulative_sum(ext_log, axis: 1)
    cumsum = Nx.transpose(cumsum, axes: [0, 2, 1])
    cs_j = Nx.slice(cumsum, [0, 0, 1], [batch, n_h, seq_len])
    cs_i = Nx.slice(cumsum, [0, 0, 0], [batch, n_h, seq_len])
    cs_j = Nx.reshape(cs_j, {batch, n_h, 1, seq_len})
    cs_i = Nx.reshape(cs_i, {batch, n_h, seq_len, 1})
    log_map = Nx.max(Nx.subtract(cs_j, cs_i), 0)
    Nx.exp(Nx.negate(log_map))
  end

  # LinearPositionalChannel: attn = emb @ emb.T / dim * mask, y = attn @ v
  defp channel_p_forward(normed_x, seq_len, invalid_attn_mask, base, params) do
    {batch, _n, _} = Nx.shape(normed_x)
    proj_w = params[base <> "channel_p.proj_p.weight"]
    v = if proj_w do
      Nx.dot(normed_x, [2], proj_w, [0])
    else
      Nx.slice(normed_x, [0, 0, 0], [batch, seq_len, @value_dim])
    end

    emb = params[base <> "channel_p.emb"] || params[:"#{base}channel_p_emb"]
    emb = if emb do
      Nx.slice(emb, [0, 0], [seq_len, @channel_p_dim])
    else
      half = div(@channel_p_dim, 2)
      theta = Nx.pow(10000, Nx.negate(Nx.divide(Nx.iota({half}), half)))
      pos = Nx.iota({seq_len}, type: {:f, 32}) |> Nx.new_axis(-1)
      Nx.concatenate([Nx.sin(Nx.multiply(pos, theta)), Nx.cos(Nx.multiply(pos, theta))], axis: 1)
    end

    attn_w = Nx.dot(emb, [1], emb, [1])
    attn_w = Nx.divide(attn_w, max(div(@channel_p_dim, 2), 1))
    attn_w = Nx.multiply(attn_w, invalid_attn_mask)
    attn_w = Nx.reshape(attn_w, {1, seq_len, seq_len}) |> Nx.broadcast({batch, seq_len, seq_len})

    # (batch,n,n) @ (batch,n,d) -> (batch,n,d)
    out = Nx.dot(attn_w, [2], [0], v, [1], [0])

    alpha = params[base <> "channel_p.alpha"]
    beta = params[base <> "channel_p.beta"]
    if alpha && beta do
      Nx.add(Nx.multiply(out, alpha), Nx.multiply(v, beta))
    else
      out
    end
  end

  defp mffn_forward(x, x0, base, params) do
    lin0_w = params[base <> "mffn.lin0.weight"]
    lin1_w = params[base <> "mffn.lin1.weight"]
    lin2_w = params[base <> "mffn.lin2.weight"]
    lin3_w = params[base <> "mffn.lin3.weight"]

    if lin0_w == nil, do: raise("missing #{base}mffn.lin0.weight")

    h = Nx.dot(x, [2], lin0_w, [0])
    h = Nx.add(h, x0)

    if lin1_w && lin2_w && lin3_w do
      normed = rms_norm(h)
      x1 = Nx.multiply(silu(Nx.dot(normed, [2], lin1_w, [0])), Nx.dot(normed, [2], lin3_w, [0]))
      Nx.add(Nx.dot(x1, [2], lin2_w, [0]), h)
    else
      h
    end
  end

  defp rms_norm(x) do
    rms = Nx.sqrt(Nx.add(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true), 1.0e-6))
    Nx.divide(x, rms)
  end

  defp apply_aux_encoder(batch_aux, embed_mask, params) do
    w = params["ae.linear.weight"] || params["ae_linear_weight"]
    b = params["ae.linear.bias"] || params["ae_linear_bias"]
    nw = params["ae.norm.weight"] || params["ae_norm_weight"]
    nb = params["ae.norm.bias"] || params["ae_norm_bias"]
    if w == nil, do: raise("FuXi requires ae.* params (RecGPT semantic ID)")
    out = Nx.dot(batch_aux, [2], w, [0])
    out = Nx.add(out, Nx.reshape(b, {1, 1, @n_embd}))
    out = layer_norm(out, nw, nb)
    Nx.multiply(out, embed_mask)
  end

  defp apply_ln_f(hidden, params) do
    w = params["ln_f.weight"] || params["gpt2model.ln_f.weight"]
    b = params["ln_f.bias"] || params["gpt2model.ln_f.bias"]
    if w, do: layer_norm(hidden, w, b), else: hidden
  end

  defp apply_head(hidden, params) do
    w = params["pred_head.weight"] || params["pred_head_weight"]
    b = params["pred_head.bias"] || params["pred_head_bias"]
    Nx.dot(hidden, [1], w, [0]) |> Nx.add(b)
  end

  defp layer_norm(x, weight, bias) do
    if !weight || !bias, do: x, else: layer_norm_impl(x, weight, bias)
  end

  defp layer_norm_impl(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    var = Nx.variance(x, axes: [-1], keep_axes: true)
    x_norm = Nx.divide(Nx.subtract(x, mean), Nx.add(Nx.sqrt(var), 1.0e-6))
    Nx.add(Nx.multiply(x_norm, weight), bias)
  end

  defp get_wte(params) do
    wte = params["wte"] || params["gpt2model.wte.weight"] || params["gpt2model.wte"]
    if wte == nil, do: raise("params must include wte")
    {rows, _} = Nx.shape(wte)
    if rows >= @vocab_size, do: Nx.slice_along_axis(wte, 0, @vocab_size, axis: 0), else: wte
  end

  @doc """
  Build full model params: FuXi blocks + RecGPT semantic ID (wte, ae.*, pred_head, ln_f).
  Use for unit tests or training from scratch. No stubs; all components from upstream.
  """
  @spec init_full_params(Keyword.t()) :: map()
  def init_full_params(opts \\ []) do
    init = fn shape, std ->
      n = shape |> Tuple.to_list() |> Enum.reduce(1, &Kernel.*/2)
      Nx.iota(shape, type: {:f, 32}) |> Nx.divide(max(n, 1)) |> Nx.multiply(std * 4) |> Nx.subtract(std * 2)
    end

    block_params = init_params(opts)

    # RecGPT semantic ID components
    base =
      %{}
      |> Map.put("wte", init.({@vocab_size, @n_embd}, 0.02))
      |> Map.put("ae.linear.weight", init.({192, @n_embd}, 0.02))
      |> Map.put("ae.linear.bias", Nx.broadcast(0, {@n_embd}) |> Nx.as_type({:f, 32}))
      |> Map.put("ae.norm.weight", Nx.iota({@n_embd}, type: {:f, 32}) |> Nx.add(1))
      |> Map.put("ae.norm.bias", Nx.broadcast(0, {@n_embd}) |> Nx.as_type({:f, 32}))
      |> Map.put("ln_f.weight", Nx.iota({@n_embd}, type: {:f, 32}) |> Nx.add(1))
      |> Map.put("ln_f.bias", Nx.broadcast(0, {@n_embd}) |> Nx.as_type({:f, 32}))
      |> Map.put("pred_head.weight", init.({@n_embd, @vocab_size}, 0.02))
      |> Map.put("pred_head.bias", Nx.broadcast(0, {@vocab_size}) |> Nx.as_type({:f, 32}))

    Map.merge(base, block_params)
  end

  @doc """
  Build FuXi-Linear block params only (Retention, LinearTemporalChannel, LinearPositionalChannel, MFFN).
  Use with existing RecGPT wte/ae/pred_head. Keys match forward/4 expectations.
  """
  @spec init_params(Keyword.t()) :: map()
  def init_params(opts \\ []) do
    n_blocks = Keyword.get(opts, :n_blocks, @n_blocks)
    max_seq_len = Keyword.get(opts, :max_seq_len, 1024)

    init = fn shape, std ->
      n = shape |> Tuple.to_list() |> Enum.reduce(1, &Kernel.*/2)
      Nx.iota(shape, type: {:f, 32}) |> Nx.divide(max(n, 1)) |> Nx.multiply(std * 4) |> Nx.subtract(std * 2)
    end

    Enum.reduce(0..(n_blocks - 1), %{}, fn i, params ->
      base = "fuxi.block.#{i}."
      half = div(@channel_p_dim, 2)
      theta = Nx.pow(10000, Nx.negate(Nx.divide(Nx.iota({half}), half)))
      pos = Nx.iota({min(max_seq_len, 2048)}, type: {:f, 32}) |> Nx.new_axis(-1)
      emb = Nx.concatenate([Nx.sin(Nx.multiply(pos, theta)), Nx.cos(Nx.multiply(pos, theta))], axis: 1)

      params
      |> Map.put(base <> "ln.weight", Nx.iota({@n_embd}, type: {:f, 32}) |> Nx.add(1))
      |> Map.put(base <> "ln.bias", Nx.broadcast(0, {@n_embd}) |> Nx.as_type({:f, 32}))
      |> Map.put(base <> "uvqk", init.({@n_embd, @uvqk_out}, 0.02))
      |> Map.put(base <> "retention.gamma", init.({@n_head}, 0.02))
      |> Map.put(base <> "retention.ln.weight", Nx.iota({@value_dim}, type: {:f, 32}) |> Nx.add(1))
      |> Map.put(base <> "retention.ln.bias", Nx.broadcast(0, {@value_dim}) |> Nx.as_type({:f, 32}))
      |> Map.put(base <> "channel_t.proj_v.weight", init.({@n_embd, @value_dim}, 0.02))
      |> Map.put(base <> "channel_t.gamma", Nx.broadcast(0, {@channel_t_heads}) |> Nx.as_type({:f, 32}))
      |> Map.put(base <> "channel_t.alpha", init.({1}, 0.02))
      |> Map.put(base <> "channel_t.beta", Nx.broadcast(1, {1}) |> Nx.as_type({:f, 32}))
      |> Map.put(base <> "channel_p.proj_p.weight", init.({@n_embd, @value_dim}, 0.02))
      |> Map.put(base <> "channel_p.emb", emb)
      |> Map.put(base <> "channel_p.alpha", init.({1}, 0.02))
      |> Map.put(base <> "channel_p.beta", Nx.broadcast(1, {1}) |> Nx.as_type({:f, 32}))
      |> Map.put(base <> "mffn.lin0.weight", init.({@attn_dim, @n_embd}, 0.02))
      |> Map.put(base <> "mffn.lin1.weight", init.({@n_embd, @n_embd}, 0.02))
      |> Map.put(base <> "mffn.lin2.weight", init.({@n_embd, @n_embd}, 0.02))
      |> Map.put(base <> "mffn.lin3.weight", init.({@n_embd, @n_embd}, 0.02))
    end)
  end
end
