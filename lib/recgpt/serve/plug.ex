defmodule RecGPT.Serve.Plug do
  @moduledoc """
  RESTful HTTP API for RecGPT (Google API Design Guide).

  Serves only versioned routes under /v1/:
  - GET  /v1/catalog/items     — List (search) catalog items; query params: q, pageSize.
  - POST /v1/catalog:recommend — Custom method: get next-item recommendations; body: context_item_ids, max_results.
  - GET  /v1/health            — Readiness.

  All errors return a Google-style JSON error body (error.code, error.message, error.status).
  """
  import Plug.Conn

  alias RecGPT.Serve.REST

  def init(opts), do: opts

  def call(conn, _opts) do
    state = Application.get_env(:recgpt, :serve_state)

    if is_nil(state) do
      send_error(conn, 503, "Service not ready. Load fixture and checkpoint first.")
    else
      conn = conn |> fetch_query_params() |> assign(:recgpt_state, state)
      dispatch(conn)
    end
  end

  defp dispatch(conn) do
    path = conn.request_path
    method = conn.method
    base = "/" <> api_prefix()

    cond do
      method == "GET" and path == base <> "/catalog/items" ->
        handle_list_items(conn)

      method == "POST" and path == base <> "/catalog:recommend" ->
        handle_recommend(conn)

      method == "GET" and path == base <> "/health" ->
        handle_health(conn)

      true ->
        send_error(
          conn,
          404,
          "Not Found. Supported: GET #{base}/catalog/items, POST #{base}/catalog:recommend, GET #{base}/health."
        )
    end
  end

  defp api_prefix do
    Application.get_env(:recgpt, :api_prefix, "v1")
  end

  defp handle_health(conn) do
    send_json(conn, 200, %{"status" => "ok"})
  end

  defp handle_list_items(conn) do
    state = conn.assigns.recgpt_state
    q = get_param(conn, "q") || ""
    page_size = get_param(conn, "pageSize") || get_param(conn, "page_size")
    page_size = parse_positive_int(page_size, 20) |> min(100)

    matches = RecGPT.Serve.search(state, q, page_size)
    body = REST.list_items_response(matches, state)
    send_json(conn, 200, body)
  end

  defp handle_recommend(conn) do
    state = conn.assigns.recgpt_state

    body =
      case Plug.Conn.read_body(conn) do
        {:ok, raw, _conn} -> Jason.decode(raw)
        {:more, _, _conn} -> {:error, :body}
        {:error, _} -> {:error, :body}
      end

    case body do
      {:ok, nil} -> send_error(conn, 400, "Request body must be JSON object.")
      {:ok, req} when is_map(req) -> handle_recommend_parsed(conn, state, req)
      _ -> send_error(conn, 400, "Request body must be valid JSON object.")
    end
  end

  defp handle_recommend_parsed(conn, state, req) do
    case REST.parse_recommend_request(req) do
      {:ok, %{context_item_ids: ids, max_results: max_results}} ->
        case RecGPT.Serve.recommend(state, ids, max_results) do
          {:ok, rec_ids} -> send_json(conn, 200, REST.recommend_response(rec_ids, state))
          {:error, msg} -> send_error(conn, 400, to_string(msg))
        end

      {:error, _kind, msg} ->
        send_error(conn, 400, msg)
    end
  end

  defp get_param(conn, key) do
    conn.query_params[key] || conn.params[key]
  end

  defp parse_positive_int(nil, default), do: default

  defp parse_positive_int(s, default) when is_binary(s) do
    case Integer.parse(s) do
      {n, _} when n > 0 -> n
      _ -> default
    end
  end

  defp parse_positive_int(_, default), do: default

  defp send_error(conn, code, message) do
    body = REST.error_body(code, message)

    conn
    |> put_resp_content_type("application/json")
    |> send_resp(code, Jason.encode!(body))
  end

  defp send_json(conn, status, body) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(status, Jason.encode!(body))
  end
end
