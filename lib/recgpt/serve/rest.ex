defmodule RecGPT.Serve.REST do
  @moduledoc """
  RESTful API for RecGPT recommendation service.

  Follows [Google API Design Guide](https://cloud.google.com/apis/design):
  resource-oriented URLs, versioned API (`/v1/`), custom method `catalog:recommend`,
  and standard error format (google.rpc.Status–style).
  """

  @api_version "v1"
  @default_error_domain "recgpt.googleapis.com"

  # Google RPC code names for HTTP (AIP-193)
  @status_invalid_argument "INVALID_ARGUMENT"
  @status_not_found "NOT_FOUND"
  @status_failed_precondition "FAILED_PRECONDITION"
  @status_unavailable "UNAVAILABLE"

  def api_version, do: @api_version

  defp error_domain do
    Application.get_env(:recgpt, :rest_error_domain, @default_error_domain)
  end

  @doc """
  Builds a Google-style error response body.

  Code: 400 INVALID_ARGUMENT, 404 NOT_FOUND, 503 UNAVAILABLE, etc.
  Error domain is configurable via :recgpt, :rest_error_domain (e.g. for embedding in reflex-logic-market).
  """
  def error_body(code, message, reason \\ nil) when is_integer(code) and is_binary(message) do
    status_str =
      reason ||
        case code do
          400 -> @status_invalid_argument
          404 -> @status_not_found
          412 -> @status_failed_precondition
          503 -> @status_unavailable
          _ -> "UNKNOWN"
        end

    %{
      "error" => %{
        "code" => code,
        "message" => message,
        "status" => status_str,
        "details" => [
          %{"@type" => "type.googleapis.com/recgpt.v1.ErrorInfo", "domain" => error_domain()}
        ]
      }
    }
  end

  @doc """
  Parses Recommend request body per API spec.

  Required: `context_item_ids` (list of int). Optional: `max_results` (int, default 5, max 20).
  Other keys (e.g. metadata, user_id, filter) are ignored for forward compatibility.
  Returns `{:ok, %{context_item_ids: ..., max_results: ...}}` or `{:error, :invalid_argument, message}`.
  """
  def parse_recommend_request(body) when is_map(body) do
    ids = body["context_item_ids"] || body["contextItemIds"]
    max_results = body["max_results"] || body["maxResults"] || 5

    cond do
      not is_list(ids) ->
        {:error, :invalid_argument, "context_item_ids must be a list of item IDs"}

      ids == [] ->
        {:error, :invalid_argument, "context_item_ids must not be empty"}

      true ->
        case coerce_ids_and_max_results(ids, max_results) do
          {:ok, coerced_ids, mr} -> {:ok, %{context_item_ids: coerced_ids, max_results: mr}}
          {:error, msg} -> {:error, :invalid_argument, msg}
        end
    end
  end

  def parse_recommend_request(_),
    do: {:error, :invalid_argument, "Request body must be JSON object"}

  defp coerce_ids_and_max_results(ids, max_results) do
    coerced = Enum.map(ids, &coerce_int/1)

    if Enum.any?(coerced, &(&1 == :error)) do
      {:error, "context_item_ids must contain integers"}
    else
      mr = coerce_int(max_results)
      mr = if mr == :error, do: 5, else: min(max(mr, 1), 20)
      {:ok, coerced, mr}
    end
  end

  defp coerce_int(x) when is_integer(x), do: x

  defp coerce_int(x) when is_binary(x) do
    case Integer.parse(x) do
      {n, _} -> n
      :error -> :error
    end
  end

  defp coerce_int(_), do: :error

  @doc """
  Builds Recommend response body per API spec.

  Returns: `{"item_ids": [...], "items": [{"item_id": id, "display_name": "...", ...}, ...]}`.
  If state has :item_extra (map or function), those fields are merged into each item (e.g. asset_id, slug for Polymarket).
  """
  def recommend_response(item_ids, state) do
    items =
      Enum.map(item_ids, fn id ->
        base = %{
          "item_id" => id,
          "display_name" => RecGPT.Serve.safe_str(Map.get(state.item_text, id))
        }

        merge_item_extra(base, id, state)
      end)

    %{"item_ids" => item_ids, "items" => items}
  end

  @doc """
  Builds List (search) items response per API spec.

  Returns: `{"items": [{"item_id": id, "display_name": "...", ...}, ...]}`.
  If state has :item_extra, those fields are merged into each item.
  """
  def list_items_response(matches, state) do
    items =
      Enum.map(matches, fn %{"item_id" => id, "raw" => raw} ->
        base = %{"item_id" => id, "display_name" => raw}
        merge_item_extra(base, id, state)
      end)

    %{"items" => items}
  end

  defp merge_item_extra(base, item_id, state) do
    case Map.get(state, :item_extra) do
      nil -> base
      extra when is_map(extra) -> Map.merge(base, Map.get(extra, item_id, %{}))
      fun when is_function(fun, 2) -> Map.merge(base, fun.(item_id, state))
      _ -> base
    end
  end
end
