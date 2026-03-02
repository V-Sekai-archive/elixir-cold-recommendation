defmodule Recgpt.V1.PredictionService.Service do
  @moduledoc false
  use GRPC.Service, name: "recgpt.v1.PredictionService"

  rpc(:Predict, Recgpt.V1.PredictRequest, Recgpt.V1.PredictResponse)
end

defmodule Recgpt.V1.PredictionService.Server do
  @moduledoc """
  gRPC server for RecGPT PredictionService. Uses Elixir inference (RecGPT.Serve.recommend).
  State is loaded by `mix recgpt.serve` and stored in config :recgpt, :serve_state.
  """
  use GRPC.Server, service: Recgpt.V1.PredictionService.Service

  alias Recgpt.V1.{ItemSummary, PredictResponse}

  def predict(request, _stream) do
    context_ids = request.context_item_ids || []
    raw_max = request.max_results || 0
    max_results = if raw_max in 1..20, do: raw_max, else: 5

    if raw_max != 0 and (raw_max < 1 or raw_max > 20) do
      raise GRPC.RPCError,
        status: :invalid_argument,
        message: "max_results must be between 1 and 20"
    end

    if context_ids == [] do
      raise GRPC.RPCError,
        status: :invalid_argument,
        message: "context_item_ids must not be empty"
    end

    case Application.get_env(:recgpt, :serve_state) do
      nil ->
        raise GRPC.RPCError,
          status: :failed_precondition,
          message: "Serve state not loaded. Start with mix recgpt.serve (fixture + checkpoint)."

      state ->
        case RecGPT.Serve.recommend(state, context_ids, max_results) do
          {:ok, item_ids} ->
            items =
              Enum.map(item_ids, fn id ->
                name =
                  case Map.get(state.item_text, id) do
                    t when is_binary(t) -> t
                    m when is_map(m) -> m["title"] || m["name"] || to_string(id)
                    _ -> to_string(id)
                  end
                %ItemSummary{item_id: id, display_name: name}
              end)
            %PredictResponse{item_ids: item_ids, items: items}

          {:error, reason} ->
            raise GRPC.RPCError, status: :internal, message: "Recommend failed: #{inspect(reason)}"
        end
    end
  end
end
