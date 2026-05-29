defmodule RecGPT.Inference do
  @moduledoc """
  Unified inference interface. Delegates to FuXi-Linear (the sole architecture).

  Preserves the original RecGPT API: forward/4, forward_full_sequence/4,
  forward_with_cache/4, forward_hidden/4, n_layers_from_params/1.
  """

  alias RecGPT.FuxiLinearInference

  @doc """
  Forward pass. batch_token_ids: (batch, seq_len), batch_aux_embeds: (batch, seq_len, 192),
  embed_mask: (batch, seq_len, 1), params: map from CheckpointLoader.
  Returns logits (batch, 15_361) for the last position.
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def forward(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    FuxiLinearInference.forward(batch_token_ids, batch_aux_embeds, embed_mask, params, [])
  end

  @doc """
  Full-sequence forward for training. Returns logits (batch, seq_len, 15_361).
  """
  @spec forward_full_sequence(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def forward_full_sequence(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    FuxiLinearInference.forward_full_sequence(batch_token_ids, batch_aux_embeds, embed_mask, params, [])
  end

  @doc """
  Forward with KV-cache (no-op for FuXi-Linear). Returns {logits, []} for API compatibility.
  """
  @spec forward_with_cache(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) ::
          {Nx.Tensor.t(), list()}
  def forward_with_cache(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    logits = forward(batch_token_ids, batch_aux_embeds, embed_mask, params)
    {logits, []}
  end

  @doc """
  Forward hidden states. Returns (batch, seq_len, 768).
  """
  @spec forward_hidden(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), map()) :: Nx.Tensor.t()
  def forward_hidden(batch_token_ids, batch_aux_embeds, embed_mask, params) do
    FuxiLinearInference.forward_hidden(batch_token_ids, batch_aux_embeds, embed_mask, params, [])
  end

  @doc """
  Returns number of layers (FuXi-Linear always uses 4 blocks).
  """
  @spec n_layers_from_params(map()) :: non_neg_integer()
  def n_layers_from_params(_params) do
    4
  end
end
