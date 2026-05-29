defmodule RecGPT.InferenceDefn do
  @moduledoc """
  Defn entry points. Delegates to FuXiLinearInferenceDefn (the sole architecture).

  Same interface: (batch_token_ids, batch_aux, embed_mask, params) -> logits_4 (batch, 4, vocab_size).
  """

  import Nx.Defn

  @doc """
  Single forward returning logits for the last 4 positions. Used by serve/decode.
  """
  defn forward_last_4_logits(batch_token_ids, batch_aux, embed_mask, params) do
    RecGPT.FuxiLinearInferenceDefn.forward_last_4_logits(batch_token_ids, batch_aux, embed_mask, params)
  end
end
