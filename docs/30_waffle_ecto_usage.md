# waffle_ecto usage — blob storage with Ecto and optional object store

Sub-proposal of the [documentation index](README.md). How to use [waffle_ecto](https://hex.pm/packages/waffle_ecto) with Ecto for storing file/object references and uploading blobs to local storage or S3/GCS.

---

## Problem or limitation

Artifacts (fixture, checkpoint, product media, exports) need **mass storage** with a single code path: local file today, object store (S3/GCS) when required. Without a documented approach, apps either hardcode file paths, duplicate logic for local vs S3, or defer object-store support. Ecto schemas should hold **references** (path/URL); the actual storage backend should be swappable by config.

---

## Proposed improvement

Use **waffle_ecto** with **Waffle** so that:

- Ecto schemas store attachment path/URL via a custom Ecto type (the Waffle uploader’s type).
- **Waffle** performs upload/download; backend is **local** or **S3** (and S3-compatible providers) via config.
- One code path: `cast_attachments` in changesets; switch backends by changing Waffle config (no app code change).

Official docs: [waffle_ecto (Hex)](https://hex.pm/packages/waffle_ecto), [hexdocs](https://hexdocs.pm/waffle_ecto), [Waffle (Hex)](https://hex.pm/packages/waffle).

---

## Installation

Add to `mix.exs`:

```elixir
defp deps do
  [
    {:waffle_ecto, "~> 0.0"},
    {:waffle, "~> 1.1"}
    # If using S3:
    # {:ex_aws, "~> 2.1"},
    # {:ex_aws_s3, "~> 2.0"},
    # {:hackney, "~> 1.9"},
    # {:sweet_xml, "~> 0.6"}
  ]
end
```

Run `mix deps.get`. **waffle_ecto** depends on `ecto ~> 3.0` and `waffle ~> 1.0`.

---

## Waffle uploader (definition module)

Define a **Waffle definition module** that specifies storage path, URL, and optional transformations. Either create it by hand or generate with `mix waffle.g <name>` (if your project uses that generator).

Example (e.g. `lib/my_app/uploaders/artifact.ex`):

```elixir
defmodule MyApp.ArtifactUploader do
  use Waffle.Definition

  @versions [:original]

  def filename(version, {file, _scope}) do
    "#{version}-#{file.file_name}"
  end

  def storage_dir(_version, {_file, scope}) do
    "uploads/artifacts/#{scope.id}"
  end

  # Optional: override for private URLs, custom host, etc.
  # def acl(version, {file, scope}), do: :private
end
```

The **Ecto type** for this uploader is `MyApp.ArtifactUploader.Type` (convention: `YourUploader.Type`). Use that type in your Ecto schema.

---

## Ecto schema and changeset

1. **use Waffle.Ecto.Schema** in the schema module.
2. Add a field with type `YourUploader.Type`.
3. In the changeset, call **cast_attachments/3** for that field (same pattern as `cast/3`).

Example:

```elixir
defmodule MyApp.Artifact do
  use Ecto.Schema
  use Waffle.Ecto.Schema

  schema "artifacts" do
    field :name, :string
    field :file, MyApp.ArtifactUploader.Type
    timestamps()
  end

  def changeset(artifact, params \\ %{}) do
    artifact
    |> Ecto.Changeset.cast(params, [:name])
    |> cast_attachments(params, [:file])
    |> Ecto.Changeset.validate_required([:name])
  end
end
```

- **Params:** Pass a map with `"file"` (or your field name) set to a `%Plug.Upload{}` (from a form or controller), or a path/URL if you use the options below.
- **cast_attachments/3** options (see [hexdocs](https://hexdocs.pm/waffle_ecto/Waffle.Ecto.Schema.html)):
  - **allow_paths: true** — accept a local file path as the source.
  - **allow_urls: true** — accept an HTTP/HTTPS URL; Waffle will fetch and store the file.

After insert/update, the schema’s `file` field holds the stored path or URL (depending on the uploader and storage backend). Use your uploader’s `url/2` or `url/3` to generate URLs for serving or download.

---

## Storage backend (config)

Configure Waffle so the same code uses **local** or **S3** storage.

**Local:**

```elixir
# config/config.exs or config/dev.exs
config :waffle,
  storage: Waffle.Storage.Local,
  asset_host: "http://localhost:4000"  # or {:system, "ASSET_HOST"}
```

**S3:**

```elixir
# config/config.exs or config/runtime.exs
config :waffle,
  storage: Waffle.Storage.S3,
  bucket: "my-bucket",  # or {:system, "AWS_S3_BUCKET"}
  asset_host: "https://my-bucket.s3.region.amazonaws.com"

config :ex_aws, :s3,
  region: "us-east-1"

config :ex_aws, json_codec: Jason
```

Use `{:system, "VAR"}` for bucket/host so secrets and env-specific values stay out of the repo. For **GCS** or other S3-compatible backends, see [Waffle docs](https://hexdocs.pm/waffle) and community adapters (e.g. [waffle_gcs](https://github.com/elixir-waffle/waffle_gcs)).

---

## When to use it in this repo

- **Catalog or artifact tables** that reference blobs (e.g. checkpoint manifest, fixture path, product media): store the reference in Ecto; let Waffle handle the file or object.
- **RecGPT artifacts:** Fixture, checkpoint, train data are read/written as bulk blobs; one GET/PUT per artifact. waffle_ecto fits when you persist **metadata** in Ecto and the blob in local or object store. See [13 Infrastructure and serving](13_infrastructure_serving.md#catalog-storage-object-store-semantics) and [thirdparty polymarket_sportradar_arb_reference](../thirdparty/polymarket_sportradar_arb_reference.md) (blob/artifact storage).

---

## See also

- [waffle_ecto (Hex)](https://hex.pm/packages/waffle_ecto) — Package and deps.
- [waffle_ecto hexdocs](https://hexdocs.pm/waffle_ecto) — API (e.g. `cast_attachments`, `Waffle.Ecto.Schema`).
- [Waffle (Hex)](https://hex.pm/packages/waffle) — Upload definition, storage backends (local, S3).
- [13 Infrastructure and serving](13_infrastructure_serving.md) — Catalog storage and object-store options.
- [Documentation index](README.md).
