defmodule RecGPT.Serve.Plug do
  @moduledoc false
  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    state = Application.get_env(:recgpt, :serve_state)
    if is_nil(state) do
      send_resp(conn, 503, "Service not ready")
    else
      dispatch(conn, state)
    end
  end

  defp dispatch(conn, state) do
    conn = conn |> fetch_query_params() |> assign(:recgpt_state, state)
    path = conn.request_path

    case {conn.method, path} do
      {"GET", "/search"} -> handle_search(conn)
      {"POST", "/recommend"} -> handle_recommend(conn)
      {"GET", "/health"} -> send_json(conn, 200, %{"status" => "ok"})
      _ -> send_resp(conn, 404, "Not Found")
    end
  end

  defp handle_search(conn) do
    state = conn.assigns.recgpt_state
    q = get_param(conn, "q") || ""
    limit = get_param(conn, "limit") |> parse_int(20) |> min(100)
    matches = RecGPT.Serve.search(state, q, limit)
    send_json(conn, 200, %{"matches" => matches})
  end

  defp handle_recommend(conn) do
    state = conn.assigns.recgpt_state

    body =
      case Plug.Conn.read_body(conn) do
        {:ok, raw, _conn} -> Jason.decode(raw)
        {:more, _, _conn} -> Jason.decode("{}")
        {:error, _reason} -> {:error, :body}
      end

    case body do
      {:ok, %{"item_ids" => item_ids} = req} when is_list(item_ids) ->
        top_k = (req["top_k"] || 5) |> min(20)
        case RecGPT.Serve.recommend(state, item_ids, top_k) do
          {:ok, rec_ids} ->
            item_texts = Enum.map(rec_ids, fn id -> item_response(state, id) end)
            send_json(conn, 200, %{"item_ids" => rec_ids, "item_texts" => item_texts})
          {:error, msg} ->
            send_json(conn, 400, %{"error" => to_string(msg)})
        end
      {:ok, _} ->
        send_json(conn, 400, %{"error" => "body must contain item_ids (list)"})
      _ ->
        send_json(conn, 400, %{"error" => "invalid JSON body"})
    end
  end

  defp item_response(state, item_id) do
    text = Map.get(state.item_text, item_id)
    %{"item_id" => item_id, "raw" => safe_str(text)}
  end

  defp safe_str(nil), do: ""
  defp safe_str(s) when is_binary(s), do: s
  defp safe_str(m) when is_map(m), do: inspect(m)
  defp safe_str(x), do: to_string(x)

  defp get_param(conn, key) do
    conn.query_params[key] || conn.params[key]
  end

  defp parse_int(nil, default), do: default
  defp parse_int(s, default) when is_binary(s) do
    case Integer.parse(s) do
      {n, _} -> n
      :error -> default
    end
  end
  defp parse_int(_, default), do: default

  defp send_json(conn, status, body) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(body))
  end
end
