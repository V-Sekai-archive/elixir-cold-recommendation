# ETNF catalog metadata: Dublin Core + XMP JSON-LD

Catalog item metadata in the ETNF FOSS datasets DB is stored in normalized tables and exposed as **Dublin Core** ([Wikipedia](https://en.wikipedia.org/wiki/Dublin_Core)) **XMP JSON-LD** via a **view** (not a stored column), in the style of [Khronos KHR_xmp_json_ld](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_xmp_json_ld).

---

## Schema (this repo)

- **`catalog_item`** — One row per item; composite key `(item_id, source_dataset)`. Columns: `dc_title`, `dc_description`, `dc_type`, `dcterms_source`, timestamps.
- **`event`** — One row per click; composite key `(session_id, ord)`. Columns: `item_id`, `source_dataset`, timestamps.
- **`catalog_item_xmp_jsonld`** — A **view** (read-only). Selects from `catalog_item` and builds the XMP JSON-LD document per row using SQLite `json_object`. Columns: `item_id`, `source_dataset`, `item_xmp_jsonld` (TEXT). No stored JSON-LD column; the view computes it from normalized columns.

---

## Dublin Core

We use the **Dublin Core Metadata Element Set** (15 elements) and **DCMI Metadata Terms** where needed:

| Element         | Use in catalog                                                          |
| --------------- | ----------------------------------------------------------------------- |
| **title**       | Abbreviated description (short): first line or first 200 characters.    |
| **description** | Full item text (no truncation). Used for RecGPT/MPNet and full content. |
| **identifier**  | Same as table `item_id` (global id, e.g. `uci_clickstream:42`).              |
| **source**      | Provenance: `source_dataset` (e.g. uci_clickstream, merrec). |
| **type**        | Optional: e.g. `item`, `product`, `movie`.                              |
| **language**    | Optional: content language if known.                                    |
| **format**      | Optional: e.g. `text/plain`.                                            |
| **rights**      | Optional: license (e.g. CC BY 4.0).                                     |

Other terms (creator, date, subject, relation, etc.) may be added when the source dataset provides them.

**Namespaces:**

- Dublin Core 1.1 elements: `http://purl.org/dc/elements/1.1/`
- DCMI terms: `http://purl.org/dc/terms/`

---

## XMP JSON-LD encoding

The view `catalog_item_xmp_jsonld` exposes a column `item_xmp_jsonld` (TEXT) computed from `catalog_item`. Each row is a single JSON-LD document that:

1. Uses a `@context` that maps `dc` and `dcterms` to the DCMI URIs.
2. Uses Dublin Core properties for all item metadata.
3. Conforms to a restricted, XMP-style JSON-LD subset (no nested @graph required; simple key–value structure) so that both plain JSON and JSON-LD parsers can read it, per KHR_xmp_json_ld.

**Example:**

```json
{
  "@context": {
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/"
  },
  "dc:title": "category 1 product 42 colour blue",
  "dc:description": "category 1 product 42 colour blue",
  "dc:identifier": "uci_clickstream:42",
  "dcterms:source": "uci_clickstream"
}
```

When the full text is longer than one line or 200 characters, `dc:title` is an abbreviation and `dc:description` holds the full text. **Deriving plain text for RecGPT:** Use `dc:description` (full text) when building RecGPT training data from the ETNF DB.

---

## References

- [Dublin Core](https://en.wikipedia.org/wiki/Dublin_Core) — vocabulary and history.
- [DCMI Metadata Terms](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) — formal term definitions.
- [KHR_xmp_json_ld](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_xmp_json_ld) — glTF extension for XMP metadata as JSON-LD (ISO 16684-3 style).
