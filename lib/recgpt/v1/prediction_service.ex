defmodule Recgpt.V1.PredictionService.Service do
  @moduledoc false
  use GRPC.Service, name: "recgpt.v1.PredictionService"

  rpc :Predict, Recgpt.V1.PredictRequest, Recgpt.V1.PredictResponse
end

defmodule Recgpt.V1.PredictionService.Server do
  @moduledoc """
  gRPC server for RecGPT PredictionService. Delegates to `RecGPT.Serve`.
  """
  use GRPC.Server, service: Recgpt.V1.PredictionService.Service

  alias Recgpt.V1.{ItemSummary, PredictResponse}

  def predict(request, _stream) do
    state = Application.get_env(:recgpt, :serve_state)
    if is_nil(state), do: raise(GRPC.RPCError, status: :unavailable, message: "Service not ready")

    context_ids = request.context_item_ids || []
    max_results = request.max_results || 5
    max_results = max(1, min(20, max_results))

    if context_ids == [] do
      raise GRPC.RPCError, status: :invalid_argument, message: "context_item_ids must not be empty"
    end

    case RecGPT.Serve.recommend(state, context_ids, max_results) do
      {:ok, item_ids} ->
        items =
          Enum.map(item_ids, fn id ->
            display = display_name(Map.get(state.item_text, id))
            %ItemSummary{item_id: id, display_name: display}
          end)

        %PredictResponse{item_ids: item_ids, items: items}

      {:error, msg} ->
        raise GRPC.RPCError, status: :invalid_argument, message: to_string(msg)
    end
  end

  defp display_name(nil), do: ""
  defp display_name(s) when is_binary(s), do: s
  defp display_name(m) when is_map(m), do: inspect(m)
  defp display_name(x), do: to_string(x)
end
