defmodule RecGPT.RecommendationService do
  @moduledoc """
  Behaviour for next-item recommendation. Default implementation: `RecGPT.Serve`.

  Used by `Recgpt.V1.PredictionService.Server` and `RecGPT.Eval` so callers can use
  the configured implementation (e.g. Serve or a test stub).
  """
  @callback recommend(
              state :: term(),
              context_item_ids :: [non_neg_integer()],
              top_k :: pos_integer()
            ) ::
              {:ok, [non_neg_integer()]} | {:error, term()}
end
