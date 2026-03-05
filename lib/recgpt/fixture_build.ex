defmodule RecGPT.FixtureBuild do
  @moduledoc """
  Build fixture.json from items.json: Embedding.encode_item_text_dict + FSQ → token_id_list.

  Item text is always built with RecGPT-style format (str(dict) with braces stripped) so embeddings
  match the dataset's item_text_embeddings.npy. Used by `mix recgpt.build_fixture`. Output format
  matches Serve.load_fixture (num_items, token_id_list). When RECGPT_SQLITE_PATH is set (or opts[:sqlite]),
  flushes items, embeddings, and tokens to SQLite per batch.
  """

  alias RecGPT.Catalog.Sync
  alias RecGPT.Embedding
  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder
  alias RecGPT.Repo
  alias RecGPT.Steam.CanonicalItemText

  @sqlite_batch_size 100

  @doc """
  Builds fixture from items path and checkpoint.

  - Reads items.json → item_text_dict (id => title for 0..num_items-1).
  - Item embeddings: if opts[:embeddings_npy] is set and the file exists, loads that .npy (original
    dataset embeddings) and uses rows 0..num_items-1. Otherwise encodes via Embedding.encode_item_text_dict/1.
    Using the dataset's item_text_embeddings.npy ensures token_id_list matches the released checkpoint.
  - Loads FSQ params from ckpt_dir.
  - Encodes embeddings to token_id_list via FSQEncoder.encode_embeddings_to_token_id_list/3.
  - When sqlite: processes in batches and flushes items, item_embeddings, item_tokens to SQLite each batch.
  - Returns %{"num_items" => n, "token_id_list" => token_id_list}.
  """
  @spec build(String.t(), String.t(), keyword()) :: %{
          String.t() => non_neg_integer() | [[non_neg_integer()]]
        }
  def build(items_path, ckpt_dir, opts \\ []) do
    item_text_dict =
      cond do
        opts[:canonical_texts] -> load_canonical_texts(opts[:limit])
        items_path in [:db, "db"] -> load_item_text_dict_from_db(opts[:limit])
        true -> load_item_text_dict(items_path, opts[:limit])
      end

    sqlite? = opts[:sqlite] || System.get_env("RECGPT_SQLITE_PATH") != nil

    if sqlite? do
      build_with_sqlite(item_text_dict, ckpt_dir, items_path, opts)
    else
      build_in_memory(item_text_dict, ckpt_dir, items_path, opts)
    end
  end

  defp build_in_memory(item_text_dict, ckpt_dir, items_path, opts) do
    num_items = map_size(item_text_dict)
    embeddings = load_item_embeddings(item_text_dict, num_items, items_path, opts)
    fsq_params = load_fsq_params(ckpt_dir, opts)
    token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)
    %{"num_items" => num_items, "token_id_list" => token_id_list}
  end

  defp build_with_sqlite(item_text_dict, ckpt_dir, items_path, opts) do
    Application.ensure_all_started(:recgpt)
    fsq_params = load_fsq_params(ckpt_dir, opts)
    ids = item_text_dict |> Map.keys() |> Enum.sort()
    num_items = length(ids)
    # When using dataset .npy, load once and slice per batch; otherwise encode per batch.
    preloaded =
      if items_path in [:db, "db"] do
        nil
      else
        if npy_path = resolve_embeddings_npy(items_path, opts),
          do: load_embeddings_npy(npy_path, num_items),
          else: nil
      end

    {token_id_list, _cleared} =
      ids
      |> Enum.chunk_every(@sqlite_batch_size)
      |> Enum.reduce({[], false}, fn batch_ids, {acc, cleared} ->
        embeddings =
          if preloaded do
            indices = Nx.tensor(Enum.map(batch_ids, & &1), type: {:s, 64})
            Nx.gather(preloaded, Nx.new_axis(indices, 1))
          else
            batch_dict = Map.take(item_text_dict, batch_ids)
            Embedding.encode_item_text_dict(batch_dict)
          end

        batch_tokens = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)

        unless cleared, do: Sync.clear_catalog_tables()

        entries_items =
          Enum.map(batch_ids, fn id -> %{item_id: id, title: Map.fetch!(item_text_dict, id)} end)

        entries_embeddings =
          batch_ids
          |> Enum.with_index()
          |> Enum.map(fn {id, i} ->
            bin = Nx.slice(embeddings, [i, 0], [1, 768]) |> Nx.to_binary()
            %{item_id: id, embedding: bin}
          end)

        entries_tokens =
          Enum.zip(batch_ids, batch_tokens)
          |> Enum.map(fn {id, [t0, t1, t2, t3]} ->
            %{item_id: id, t0: t0, t1: t1, t2: t2, t3: t3}
          end)

        Sync.insert_items(entries_items)
        Sync.insert_item_embeddings(entries_embeddings)
        Sync.insert_item_tokens(entries_tokens)

        {acc ++ batch_tokens, true}
      end)

    # Sequences: skip JSON sync when loading items from DB (data already in DB from Convert)
    unless items_path in [:db, "db"] do
      base = Path.dirname(items_path)

      Sync.sync_sequences_from_json(
        Path.join(base, "train_sequences.json"),
        Path.join(base, "cold_train_sequences.json")
      )

      Sync.sync_test_from_json(
        Path.join(base, "test_sequences.json"),
        Path.join(base, "cold_test_sequences.json")
      )
    end

    %{"num_items" => num_items, "token_id_list" => token_id_list}
  end

  @doc "Writes fixture map to path (JSON)."
  @spec write_fixture(map(), String.t()) :: :ok
  def write_fixture(fixture, path) do
    File.mkdir_p!(Path.dirname(path))
    File.write!(path, Jason.encode!(fixture, pretty: true))
    :ok
  end

  # Use dataset item_text_embeddings.npy when available so token_id_list matches the released checkpoint.
  defp load_item_embeddings(item_text_dict, _num_items, items_path, opts) do
    case resolve_embeddings_npy(items_path, opts) do
      npy_path when is_binary(npy_path) -> load_embeddings_npy(npy_path, map_size(item_text_dict))
      nil -> Embedding.encode_item_text_dict(item_text_dict)
    end
  end

  defp resolve_embeddings_npy(_items_path, opts) do
    path = opts[:embeddings_npy]
    if path && File.regular?(path), do: path, else: nil
  end

  defp load_embeddings_npy(npy_path, num_items) do
    {:ok, tensor} = Npy.load(npy_path, :nx)
    tensor = ensure_embeddings_2d(tensor)
    {rows, _} = Nx.shape(tensor)
    if rows < num_items, do: raise("Dataset .npy has #{rows} rows, need #{num_items}")
    Nx.slice(tensor, [0, 0], [num_items, 768])
  end

  defp ensure_embeddings_2d(tensor) do
    case Nx.shape(tensor) do
      {_n, 768} -> tensor
      {_n, 1, 768} -> Nx.squeeze(tensor, axes: [1])
      shape -> raise "Unexpected .npy shape #{inspect(shape)}, expected {n, 768} or {n, 1, 768}"
    end
  end

  defp load_canonical_texts(limit) do
    list = CanonicalItemText.load_from_repo(Repo)
    list = if limit, do: Enum.take(list, limit), else: list

    list
    |> Enum.with_index(0)
    |> Map.new(fn {text, idx} -> {idx, text} end)
  end

  defp load_item_text_dict_from_db(limit) do
    import Ecto.Query
    alias RecGPT.Catalog.Item
    alias RecGPT.Catalog.ItemEmbeddingText

    query = from(i in Item, order_by: [asc: i.item_id])
    query = if limit, do: limit(query, ^limit), else: query
    items = Repo.all(query)

    embed_map =
      from(e in ItemEmbeddingText, where: e.item_id in ^Enum.map(items, & &1.item_id))
      |> Repo.all()
      |> Map.new(&{&1.item_id, &1.embedding_text})

    items
    |> Map.new(fn item ->
      text = Map.get(embed_map, item.item_id) || item.title
      {item.item_id, Embedding.recgpt_item_text(%{title: text})}
    end)
  end

  defp load_item_text_dict(path, limit) do
    raw = File.read!(path) |> Jason.decode!()
    items = raw["items"] || []
    num_items = raw["num_items"] || length(items)
    num_items = if limit, do: min(num_items, limit), else: num_items

    items
    |> Enum.take(num_items)
    |> Enum.with_index()
    |> Map.new(fn {item, idx} -> {idx, Embedding.recgpt_item_text(item)} end)
  end

  @vae_default_filename "vae_len4_fsq88865_ep90.pt"

  defp load_fsq_params(_ckpt_dir, opts) do
    vae_path = resolve_vae_ckpt_path(opts)

    unless vae_path do
      raise "VAE checkpoint required for FSQ. No VAE path found (tried --vae-ckpt, RECGPT_VAE_CKPT, " <>
              "thirdparty/checkpoints/vae/#{@vae_default_filename}, data/#{@vae_default_filename}). " <>
              "Run: mix recgpt.fetch_vae_ckpt  or set RECGPT_VAE_CKPT=path/to/#{@vae_default_filename}"
    end

    params = FSQ.load_params_from_vae_pt(vae_path)

    if fsq_params_ok?(params) and not fsq_params_dummy?(params) do
      params
    else
      raise "FSQ params from VAE are invalid or dummy (zero kernels). Check #{vae_path}"
    end
  end

  # Prefer opts[:vae_ckpt], then RECGPT_VAE_CKPT, then default paths so FSQ is loaded when we have the VAE.
  defp resolve_vae_ckpt_path(opts) do
    cond do
      path = opts[:vae_ckpt] ->
        path = path |> to_string() |> String.trim()
        if path != "" and File.regular?(path), do: path, else: nil

      path = System.get_env("RECGPT_VAE_CKPT") ->
        path = path |> String.trim()
        if path != "" and File.regular?(path), do: path, else: nil

      true ->
        cwd = File.cwd!()

        [
          Path.join([cwd, "thirdparty", "checkpoints", "vae", @vae_default_filename]),
          Path.join([cwd, "data", @vae_default_filename])
        ]
        |> Enum.find(&File.regular?/1)
    end
  end

  defp fsq_params_dummy?(%{"project_in" => %{"kernel" => k}, "project_out" => %{"kernel" => o}}) do
    Nx.all_close(k, Nx.broadcast(0.0, Nx.shape(k))) |> Nx.to_number() == 1 and
      Nx.all_close(o, Nx.broadcast(0.0, Nx.shape(o))) |> Nx.to_number() == 1
  end

  defp fsq_params_dummy?(_), do: true

  defp fsq_params_ok?(%{"project_in" => %{"kernel" => k}, "project_out" => %{"kernel" => o}})
       when not is_nil(k) and not is_nil(o),
       do: true

  defp fsq_params_ok?(_), do: false
end
