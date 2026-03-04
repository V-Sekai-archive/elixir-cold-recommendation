defmodule Recgpt.V1.PredictionService.Service do
  @moduledoc false
  use GRPC.Service, name: "recgpt.v1.PredictionService"

  rpc(:Predict, Recgpt.V1.PredictRequest, Recgpt.V1.PredictResponse)
end

defmodule Recgpt.V1.PredictionService.Server do
  @moduledoc """
  gRPC server for RecGPT PredictionService. Uses RecGPT.PredictBatchCollector, which
  calls RecGPT.Serve.recommend (optionally batching by predict_batch_size / predict_batch_timeout_ms).
  State is loaded by `mix recgpt.serve` and stored in config :recgpt, :serve_state.

  Set config `:recgpt, :trace_predict` to `true` to log per-request timings.
  For a full breakdown (context, inference, beam_search), run `mix recgpt.trace_predict`.
  """
  use GRPC.Server, service: Recgpt.V1.PredictionService.Service

  alias Recgpt.V1.PredictResponse

  defp response_item_id_to_int(id) when is_integer(id), do: id

  defp response_item_id_to_int(%Nx.Tensor{} = t),
    do: t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number() |> round()

  defp response_item_id_to_int(x) when is_number(x), do: round(x)

  defp normalize_item_summary(%{item_id: id, display_name: name}) do
    id_int = response_item_id_to_int(id)
    name_str = if is_binary(name), do: name, else: Integer.to_string(id_int)
    %Recgpt.V1.ItemSummary{item_id: id_int, display_name: name_str}
  end

  # Avoid inspect(reason) when reason contains Nx.Tensor (Protocol.String.Chars not implemented).
  defp safe_reason_string(s) when is_binary(s), do: s
  defp safe_reason_string(reason), do: RecGPT.SafeInspect.safe_inspect(reason)

  def predict(request, _stream) do
    context_ids = request.context_item_ids || []
    raw_max = request.max_results || 0
    max_results = if raw_max in 1..20, do: raw_max, else: 5

    cond do
      raw_max != 0 and (raw_max < 1 or raw_max > 20) ->
        {:error,
         GRPC.RPCError.exception(
           status: :invalid_argument,
           message: "max_results must be between 1 and 20"
         )}

      context_ids == [] ->
        {:error,
         GRPC.RPCError.exception(
           status: :invalid_argument,
           message: "context_item_ids must not be empty"
         )}

      true ->
        trace? = Application.get_env(:recgpt, :trace_predict, false)
        result = RecGPT.PredictBatchCollector.predict(context_ids, max_results, trace?)

        case result do
          {:ok, item_ids, items} ->
            item_ids = Enum.map(item_ids, &response_item_id_to_int/1)
            items = Enum.map(items, &normalize_item_summary/1)
            %PredictResponse{item_ids: item_ids, items: items}

          {:error, :serve_state_not_loaded} ->
            {:error,
             GRPC.RPCError.exception(
               status: :failed_precondition,
               message:
                 "Serve state not loaded. Start with mix recgpt.serve (fixture + checkpoint)."
             )}

          {:error, :timeout} ->
            {:error,
             GRPC.RPCError.exception(
               status: :deadline_exceeded,
               message: "Predict timed out (increase predict_timeout_ms or warm server)."
             )}

          {:error, reason} ->
            {:error,
             GRPC.RPCError.exception(
               status: :internal,
               message: "Recommend failed: #{safe_reason_string(reason)}"
             )}
        end
    end
  end
end
