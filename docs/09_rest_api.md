# REST API (Google API Design Guide)

The RecGPT recommendation service is exposed as a **RESTful API** that follows [Google's API Design Guide](https://cloud.google.com/apis/design): resource-oriented URLs, versioned base path, custom methods, and a standard error format.

---

## Base URL and version

All endpoints are under **`/v1/`**. Only these routes are served; any other path returns `404` with a JSON error body.

---

## Endpoints

### GET /v1/catalog/items

List (search) catalog items by string query.

| Query parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `q` | string | `""` | Search string (case-insensitive substring match on item text). |
| `pageSize` | int | 20 | Maximum number of items to return (capped at 100). |

**Response (200):**

```json
{
  "items": [
    { "item_id": 0, "display_name": "Product title" }
  ]
}
```

---

### POST /v1/catalog:recommend

Custom method: get next-item recommendations given a context sequence of item IDs.

**Request body:**

```json
{
  "context_item_ids": [0, 1, 2],
  "max_results": 5
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `context_item_ids` | array of int | Yes | Ordered list of catalog item IDs (context). Must be non-empty. |
| `max_results` | int | No | Maximum number of recommendations to return (default 5, max 20). |

Snake_case and camelCase (`contextItemIds`, `maxResults`) are both accepted.

**Response (200):**

```json
{
  "item_ids": [3, 1, 4],
  "items": [
    { "item_id": 3, "display_name": "Title A" },
    { "item_id": 1, "display_name": "Title B" }
  ]
}
```

`item_ids` is the ordered list of recommended item IDs; `items` gives `item_id` and `display_name` for each.

---

### GET /v1/health

Readiness check.

**Response (200):** `{"status": "ok"}`

---

## Errors

Errors use a Google-style JSON body (see [AIP-193](https://google.aip.dev/193)):

```json
{
  "error": {
    "code": 400,
    "message": "context_item_ids must not be empty.",
    "status": "INVALID_ARGUMENT",
    "details": [{ "@type": "type.googleapis.com/recgpt.v1.ErrorInfo", "domain": "recgpt.googleapis.com" }]
  }
}
```

| HTTP code | status | When |
|-----------|--------|------|
| 400 | INVALID_ARGUMENT | Missing or invalid request body or parameters. |
| 404 | NOT_FOUND | Path not supported. |
| 503 | UNAVAILABLE | Service not ready (fixture/checkpoint not loaded). |

---

## Running the server

```bash
mix recgpt.serve --port 8000
```

Requires fixture and checkpoint export dir (see [08 Pipeline reference](08_pipeline_reference.md)). Only the versioned routes are served (default base path `/v1/`).

---

## Flexibility for embedding (e.g. reflex-logic-market)

The same API can serve **different domains** (Polymarket outcomes, Booth catalog, UCI Clickstream, etc.) by loading a different fixture and catalog at startup. Resource names stay generic (`catalog`, `items`); the **content** is determined by the loaded data.

### Application config

When embedding RecGPT in another app (e.g. reflex-logic-market / Polymarket Scout), you can set:

| Config key | Default | Description |
|------------|---------|-------------|
| `:api_prefix` | `"v1"` | Path prefix before endpoint segments. Use `"recgpt/v1"` to mount at `/recgpt/v1/catalog/items`, etc. |
| `:rest_error_domain` | `"recgpt.googleapis.com"` | Error `details[].domain` in JSON errors. Set to your service domain for consistent error handling. |

Example (e.g. in reflex-logic-market or polymarket app):

```elixir
# config/config.exs or config/runtime.exs
config :recgpt,
  api_prefix: "recgpt/v1",
  rest_error_domain: "polymarket.reflex-logic-market"
```

### Optional request fields

Recommend request body may include extra keys (e.g. `metadata`, `user_id`, `filter`). They are **ignored** by the server but allowed so that clients (e.g. Scout in reflex-logic-market) can send them for logging or future use without breaking the API.

### Response enrichment (item_extra)

To add domain-specific fields to each item in list and recommend responses (e.g. `asset_id`, `slug` for Polymarket), pass `:item_extra` when loading state:

```elixir
# Map: item_id => %{"asset_id" => "...", "slug" => "..."}
item_extra = build_index_to_asset_map()  # your mapping
{:ok, state} = RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path, item_extra: item_extra)

# Or function: (item_id, state) => %{...}
{:ok, state} = RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path,
  item_extra: fn id, _state -> %{"asset_id" => index_to_asset(id)} end)
```

Response items will then include `item_id`, `display_name`, and any keys from `item_extra`. Clients can use `item_id` as the stable index (0..num_items-1) and map to domain IDs (e.g. asset_id) via your mapping or via the enriched response.

### Catalog = fixture + catalog at load time

- **Fixture** (token_id_list + num_items) and **checkpoint** define the model and item set.
- **Catalog** (optional JSON) adds display text; **item_extra** (optional) adds domain fields.
- One server instance = one catalog. For multiple catalogs (e.g. Polymarket vs Booth), run separate processes or use different config per deployment.

---

## See also

- [Google API Design Guide](https://cloud.google.com/apis/design)
- [00 RecGPT library](00_recgpt_library.md) — Serve, Serve.Plug, Serve.REST.
- [08 Pipeline reference](08_pipeline_reference.md) — Serve step and file layout.
