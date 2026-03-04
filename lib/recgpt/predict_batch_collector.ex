defmodule RecGPT.PredictBatchCollector do
  @moduledoc """
  Collects Predict requests and processes them in batches (by size or timeout).

  When `predict_batch_size` > 1 or `predict_batch_timeout_ms` > 0, requests are
  queued and flushed when the batch is full or the first request has waited
  the timeout. Each request is processed with `Serve.recommend` and the result
  is replied to the caller. With defaults (batch_size 1, timeout 0), each
  request is processed immediately.
  """
  use GenServer

  @doc "Start the batch collector (registered under __MODULE__)."
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc "Call from PredictionService: returns {:ok, item_ids} or {:error, reason}."
  def predict(context_ids, max_results, trace?) do
    timeout_ms = Application.get_env(:recgpt, :predict_timeout_ms, 120_000)
    GenServer.call(__MODULE__, {:predict, context_ids, max_results, trace?}, timeout_ms)
  end

  @impl true
  def init(_) do
    batch_size = Application.get_env(:recgpt, :predict_batch_size, 1)
    timeout_ms = Application.get_env(:recgpt, :predict_batch_timeout_ms, 0)
    state = %{queue: [], timer_ref: nil, batch_size: batch_size, timeout_ms: timeout_ms}
    {:ok, state}
  end

  @impl true
  def handle_call({:predict, context_ids, max_results, trace?}, from, state) do
    entry = {context_ids, max_results, from, trace?}
    queue = state.queue ++ [entry]

    state =
      state
      |> Map.put(:queue, [])
      |> maybe_flush_or_schedule(queue)

    {:noreply, state}
  end

  @impl true
  def handle_info(:flush_timeout, state) do
    state =
      state
      |> Map.put(:timer_ref, nil)
      |> flush(state.queue)

    {:noreply, state}
  end

  defp maybe_flush_or_schedule(state, queue) do
    batch_size = state.batch_size
    timeout_ms = state.timeout_ms

    if length(queue) >= batch_size do
      # Batch full: flush immediately
      cancel_timer(state.timer_ref)
      flush(state, queue)
    else
      # Schedule flush on timeout if this is the first in the batch
      timer_ref =
        if timeout_ms > 0 and state.timer_ref == nil do
          Process.send_after(self(), :flush_timeout, timeout_ms)
        else
          state.timer_ref
        end

      %{state | queue: queue, timer_ref: timer_ref}
    end
  end

  defp flush(state, queue) do
    cancel_timer(state.timer_ref)
    state = %{state | queue: [], timer_ref: nil}
    process_batch(queue)
    state
  end

  defp recommend_with_timeout(state, context_ids, max_results, timeout_ms) do
    parent = self()
    ref = make_ref()

    child =
      spawn(fn ->
        result = RecGPT.Serve.recommend(state, context_ids, max_results)
        send(parent, {ref, result})
      end)

    receive do
      {^ref, result} -> result
    after
      timeout_ms ->
        Process.exit(child, :kill)
        {:error, :timeout}
    end
  end

  defp process_batch(entries) do
    case Application.get_env(:recgpt, :serve_state) do
      nil ->
        for {_context_ids, _max_results, from, _trace?} <- entries do
          GenServer.reply(from, {:error, :serve_state_not_loaded})
        end

      state ->
        timeout_ms = Application.get_env(:recgpt, :predict_timeout_ms, 120_000)

        for {context_ids, max_results, from, trace?} <- entries do
          {recommend_result, recommend_us} =
            if trace? do
              {us, res} =
                :timer.tc(fn ->
                  recommend_with_timeout(state, context_ids, max_results, timeout_ms)
                end)

              {res, us}
            else
              {recommend_with_timeout(state, context_ids, max_results, timeout_ms), nil}
            end

          reply = build_reply(recommend_result, recommend_us, state, context_ids, max_results)
          GenServer.reply(from, reply)
        end
    end
  end

  defp build_reply({:ok, item_ids}, recommend_us, state, context_ids, max_results) do
    # Coerce to plain integers (decode can leave scalar tensors in rare cases)
    item_ids = Enum.map(item_ids, &item_id_to_int/1)

    items =
      Enum.map(item_ids, fn id ->
        id_int = item_id_to_int(id)

        name =
          case Map.get(state.item_text, id_int) do
            t when is_binary(t) -> t
            m when is_map(m) -> m["title"] || m["name"] || Integer.to_string(id_int)
            _ -> Integer.to_string(id_int)
          end

        %Recgpt.V1.ItemSummary{item_id: id_int, display_name: ensure_display_string(name, id_int)}
      end)

    if recommend_us do
      require Logger

      Logger.info(
        "Predict trace context=#{RecGPT.SafeInspect.safe_inspect(context_ids)} top_k=#{max_results} " <>
          "recommend_us=#{recommend_us} total_ms=#{Float.round(recommend_us / 1000, 2)}"
      )

      RecGPT.LatencyStats.record(recommend_us)
    end

    {:ok, item_ids, items}
  end

  defp build_reply({:error, _} = err, _recommend_us, _state, _context_ids, _max_results), do: err

  defp item_id_to_int(id) when is_integer(id), do: id

  defp item_id_to_int(%Nx.Tensor{} = t),
    do: t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number() |> round()

  defp item_id_to_int(x) when is_number(x), do: round(x)

  # Proto/GRPC require display_name to be a string; avoid String.Chars on Nx.Tensor or other types.
  defp ensure_display_string(x, _id_int) when is_binary(x), do: x
  defp ensure_display_string(%Nx.Tensor{}, id_int), do: Integer.to_string(id_int)
  defp ensure_display_string(x, _id_int) when is_integer(x), do: Integer.to_string(x)

  defp ensure_display_string(x, id_int) do
    if String.Chars.impl_for(x), do: to_string(x), else: Integer.to_string(id_int)
  end

  defp cancel_timer(nil), do: :ok

  defp cancel_timer(ref) do
    Process.cancel_timer(ref)
    :ok
  end
end
