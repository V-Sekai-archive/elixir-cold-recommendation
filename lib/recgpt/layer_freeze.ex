defmodule RecGPT.LayerFreeze do
  @moduledoc """
  Freeze inputs from a full-weights run to isolate layers without GenServers or stubbing.
  See docs/22_freeze_inputs_layer_isolation.md.
  """
  alias RecGPT.Inference
  alias RecGPT.Serve

  defstruct [:params, :state, :context_item_ids, :context_token_ids]

  @type t :: %__MODULE__{
          params: map(),
          state: Serve.state(),
          context_item_ids: [non_neg_integer()],
          context_token_ids: [integer()]
        }

  @doc """
  Records frozen inputs from a full Serve state and one context (item_ids).
  """
  @spec record_from_state(Serve.state(), [non_neg_integer()]) :: t()
  def record_from_state(state, context_item_ids \\ [0]) do
    context_token_ids = Serve.item_ids_to_context_token_ids(context_item_ids, state.token_id_list)
    %__MODULE__{
      params: state.params,
      state: state,
      context_item_ids: context_item_ids,
      context_token_ids: context_token_ids
    }
  end

  @doc """
  Runs the Model layer in isolation with frozen params and the given token list.
  """
  @spec forward_model(t(), [integer()]) :: Nx.Tensor.t()
  def forward_model(frozen, token_list) do
    seq_len = length(token_list)
    batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
    batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
    embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
    Inference.forward(batch_token_ids, batch_aux, embed_mask, frozen.params)
  end

  @doc """
  Runs the Recommendation layer in isolation with frozen state.
  """
  @spec recommend(t(), [non_neg_integer()], pos_integer()) :: {:ok, [non_neg_integer()]} | {:error, String.t()}
  def recommend(frozen, item_ids, top_k \\ 5) do
    Serve.recommend(frozen.state, item_ids, top_k)
  end
end