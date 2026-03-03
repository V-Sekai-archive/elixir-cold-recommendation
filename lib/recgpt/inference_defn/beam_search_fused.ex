defmodule RecGPT.InferenceDefn.BeamSearchFused do
  @moduledoc false
  # Generates beam_search_fused_k_N/13 and beam_step_defn_k_N (defnp) for N in 4..20
  # so Nx.top_k(..., k: k) receives a literal integer and Nx.Shape.top_k does not
  # trigger String.Chars on an Nx.Defn.Expr.

  defmacro __using__(_opts) do
    k_range = 4..20

    replicate_quotes =
      for k <- k_range do
        quote_replicate_cache_defn_k(k)
      end

    defnp_quotes =
      for k <- k_range do
        quote_beam_step_defn_k(k)
      end

    defn_quotes =
      for k <- k_range do
        quote_beam_search_fused_k(k)
      end

    quote do
      unquote(replicate_quotes)
      unquote(defnp_quotes)
      unquote(defn_quotes)
    end
  end

  defp quote_replicate_cache_defn_k(k) do
    name = :"replicate_cache_defn_k_#{k}"
    quote do
      defnp unquote(name)(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11) do
        replicate_one = fn {k, v} ->
          {_b, n_head, len, hd} = Nx.shape(k)
          {Nx.broadcast(k, {unquote(k), n_head, len, hd}), Nx.broadcast(v, {unquote(k), n_head, len, hd})}
        end
        # Return map so beam_step can pass to forward_incremental_map without tuple/element/2.
        %{
          c0: replicate_one.(c0), c1: replicate_one.(c1), c2: replicate_one.(c2), c3: replicate_one.(c3),
          c4: replicate_one.(c4), c5: replicate_one.(c5), c6: replicate_one.(c6), c7: replicate_one.(c7),
          c8: replicate_one.(c8), c9: replicate_one.(c9), c10: replicate_one.(c10), c11: replicate_one.(c11)
        }
      end
    end
  end

  defp quote_beam_step_defn_k(k) do
    name = :"beam_step_defn_k_#{k}"
    quote do
      defnp unquote(name)(
              next_state,
              item_at_leaf,
              context_tokens,
              context_len,
              context_len_scalar,
              past_len_offset,
              step,
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
        context_broadcast = Nx.broadcast(context_tokens, {unquote(k), context_len})
        prefix_slice = prefix_tokens |> Nx.slice_along_axis(0, prefix_len, axis: 1)
        batch = Nx.concatenate([context_broadcast, prefix_slice], axis: 1)
        past_len = Nx.add(context_len_scalar, Nx.as_type(past_len_offset, {:s, 32}))
        # Last token index = context_len + prefix_len - 1 = past_len - 1 (past_len_offset is 1,2,3 for steps 1,2,3).
        slice_start = Nx.subtract(past_len, Nx.tensor(1, type: {:s, 32}))
        last_tokens = Nx.slice_along_axis(batch, slice_start, 1, axis: 1)
        incr_res = forward_incremental_map(last_tokens, aux_incr, mask_incr, params, cache, past_len)
        logits = incr_res.logits
        new_cache = incr_res

        state_ids_safe = Nx.max(state_ids, 0)
        idx_2d = Nx.new_axis(state_ids_safe, -1)

        # Use tensor condition so defn does not invoke :erlang.==/2.
        is_step_3 = Nx.equal(past_len_offset, Nx.tensor(3, type: {:s, 32}))
        valid_rows_step3 = Nx.gather(item_at_leaf, idx_2d) |> Nx.reshape({unquote(k), vocab_size})
        valid_rows_other = Nx.gather(next_state, idx_2d) |> Nx.reshape({unquote(k), vocab_size})
        valid_rows = Nx.select(is_step_3, valid_rows_step3, valid_rows_other)
        transition_tensor = Nx.select(is_step_3, item_at_leaf, next_state)

        valid_mask = Nx.greater_equal(valid_rows, 0)
        scores_per_token = Nx.select(valid_mask, logits, neg_inf)
        beam_scores_broadcast = Nx.new_axis(beam_scores, 1)
        scores_per_token = Nx.add(scores_per_token, beam_scores_broadcast)
        flat = Nx.reshape(scores_per_token, {:auto})
        {top_scores, top_flat} = Nx.top_k(flat, k: unquote(k))
        top_flat = Nx.as_type(top_flat, {:s, 32})
        batch_indices = Nx.quotient(top_flat, vocab_t)
        token_ids = Nx.remainder(top_flat, vocab_t)

        state_at_top = Nx.gather(state_ids, Nx.new_axis(batch_indices, -1)) |> Nx.reshape({:auto})
        state_at_top_safe = Nx.max(state_at_top, 0)

        new_state_ids_step3 = gather_2d_defn(item_at_leaf, state_at_top_safe, token_ids) |> Nx.squeeze(axes: [1])
        new_state_ids_other = gather_2d_defn(transition_tensor, state_at_top_safe, token_ids) |> Nx.squeeze(axes: [1])
        new_state_ids = Nx.select(is_step_3, new_state_ids_step3, new_state_ids_other)

        old_prefixes =
          Nx.gather(prefix_tokens, Nx.new_axis(batch_indices, -1))
          |> Nx.reshape({unquote(k), prefix_len})

        new_col = Nx.reshape(token_ids, {unquote(k), 1})
        new_prefix_tokens = Nx.concatenate([old_prefixes, new_col], axis: 1)

        item_ids = new_state_ids
        {new_state_ids, new_prefix_tokens, top_scores, new_cache, item_ids}
      end
    end
  end

  defp quote_beam_search_fused_k(k) do
    name = :"beam_search_fused_k_#{k}"
    step_defn = :"beam_step_defn_k_#{k}"
    replicate_defn = :"replicate_cache_defn_k_#{k}"

    quote do
      defn unquote(name)(
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
            vocab_t
          ) do
        res = forward_with_cache_map(context_tokens, batch_aux_0, embed_mask_0, params)
        context_len = Nx.axis_size(context_tokens, 1)
        vocab_size = Nx.axis_size(next_state, 1)

        valid_0 = Nx.gather(next_state, Nx.reshape(root_state, {1, 1})) |> Nx.reshape({:auto})
        valid_mask_0 = Nx.greater_equal(valid_0, 0)
        scores_0 = Nx.select(valid_mask_0, Nx.reshape(res.logits, {:auto}), neg_inf)
        {top_scores_0, top_indices_0} = Nx.top_k(scores_0, k: unquote(k))
        top_token_ids_0 = Nx.reshape(top_indices_0, {:auto}) |> Nx.as_type({:s, 32})
        state_ids_0 = gather_2d_defn(next_state, root_state, top_token_ids_0)
        prefix_tokens_0 = Nx.reshape(top_token_ids_0, {unquote(k), 1})
        beam_scores_0 = Nx.as_type(top_scores_0, {:f, 32})

        cache_rep =
          unquote(replicate_defn)(
            res.c0, res.c1, res.c2, res.c3, res.c4, res.c5,
            res.c6, res.c7, res.c8, res.c9, res.c10, res.c11
          )
        aux_incr = Nx.broadcast(Nx.tensor(0, type: Nx.type(batch_aux_0)), {unquote(k), 1, 192})
        mask_incr = Nx.broadcast(Nx.tensor(1, type: Nx.type(embed_mask_0)), {unquote(k), 1, 1})

        {state_ids_1, prefix_tokens_1, beam_scores_1, cache_1} =
          unquote(step_defn)(
            next_state,
            item_at_leaf,
            context_tokens,
            context_len,
            context_len_scalar,
            past_len_offset_1,
            1,
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

        {state_ids_2, prefix_tokens_2, beam_scores_2, cache_2} =
          unquote(step_defn)(
            next_state,
            item_at_leaf,
            context_tokens,
            context_len,
            context_len_scalar,
            past_len_offset_2,
            2,
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

        {_state_ids_3, prefix_tokens_3, beam_scores_3, _cache_3, item_ids} =
          unquote(step_defn)(
            next_state,
            item_at_leaf,
            context_tokens,
            context_len,
            context_len_scalar,
            past_len_offset_3,
            3,
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
    end
  end
end
