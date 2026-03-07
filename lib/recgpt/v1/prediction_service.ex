defmodule Recgpt.V1.PredictionService.Service do
  @moduledoc false
  use GRPC.Service, name: "recgpt.v1.PredictionService"

  rpc(:Predict, Recgpt.V1.PredictRequest, Recgpt.V1.PredictResponse)
end

defmodule Recgpt.V1.PredictionService.Server do
  @moduledoc """
  gRPC server for recgpt.v1.PredictionService/Predict.
  Delegates to RecGPT.PredictBatchCollector for batching and RecGPT.Serve.recommend.
  """
  use GRPC.Server, service: Recgpt.V1.PredictionService.Service

  @doc """
  Handles Predict RPC: context_item_ids + max_results → item_ids + items (ItemSummary).
  When called with a stream (gRPC), the framework sends the returned struct as the response.
  When called with nil (e.g. from mix recgpt.eval_grpc), the struct is returned directly.
  """
  def predict(request, _stream) do
    context_ids = request.context_item_ids || []
    max_results = max(1, min(request.max_results || 10, 100))

    case RecGPT.PredictBatchCollector.predict(context_ids, max_results, false) do
      {:ok, item_ids, items} ->
        %Recgpt.V1.PredictResponse{item_ids: item_ids, items: items}

      {:error, :serve_state_not_loaded} ->
        raise GRPC.RPCError, status: :failed_precondition, message: "serve state not loaded"

      {:error, :timeout} ->
        raise GRPC.RPCError, status: :deadline_exceeded, message: "recommendation timeout"

      {:error, reason} ->
        raise GRPC.RPCError,
          status: :internal,
          message: "recommendation failed: #{inspect(reason)}"
    end
  end
end
