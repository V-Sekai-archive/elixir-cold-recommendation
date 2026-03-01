defmodule RecGPT.EmbeddingCompare do
  @moduledoc """
  Compare our Bumblebee embeddings to the dataset's item_text_embeddings.npy.
  Reports cosine similarity (mean, min, max, std) and optionally FSQ token agreement.
  """

  alias RecGPT.Embedding
  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder
  alias RecGPT.CheckpointLoader

  @dataset_npy_url "https://huggingface.co/datasets/hkuds/RecGPT_dataset/resolve/main/test/steam/item_text_embeddings.npy"

  @doc """
  Runs the comparison: loads dataset .npy, generates our embeddings for the same items,
  computes per-row cosine similarity, prints report. Optionally compares FSQ token_id_list.

  Options:
  - :limit - max items to compare (default 500)
  - :ckpt_dir - if set, also compare FSQ token agreement
  - :dump_row - if set, write this row of our embeddings to :dump_path as raw float32 (for Python sanity check)
  - :dump_path - path for dump (default: item{N}_elixir.raw)
  """
  def run(steam_dir, opts \\ []) do
    limit = Keyword.get(opts, :limit, 500)
    steam_dir = Path.expand(steam_dir, File.cwd!())
    items_path = Path.join(steam_dir, "items.json")
    npy_path = Path.join(steam_dir, "item_text_embeddings.npy")

    unless File.regular?(items_path) do
      raise "items.json not found at #{items_path}. Run mix recgpt.fetch_steam first."
    end

    ensure_npy!(npy_path)

    Application.ensure_all_started(:nx)
    Application.ensure_all_started(:bumblebee)

    texts = load_ordered_texts(items_path, limit)
    n = length(texts)
    IO.puts("Comparing #{n} items...")
    report_first_strings(texts, 3)

    dataset = load_dataset_embeddings(npy_path, n)
    ours = Embedding.encode_item_text_dict(Map.new(Enum.with_index(texts, fn t, i -> {i, t} end)))

    # Ensure same shape: both {n, 768}
    dataset = ensure_2d(dataset)
    {^n, 768} = Nx.shape(ours)
    # Same type and backend so Nx ops work
    ours = ours |> Nx.as_type(:f32) |> Nx.backend_transfer(Nx.BinaryBackend)
    dataset = dataset |> Nx.as_type(:f32) |> Nx.backend_transfer(Nx.BinaryBackend)

    cos_sim = cosine_similarity_per_row(ours, dataset)
    report_row_order_check(ours, dataset)
    report_norm_diagnostic(ours, dataset)
    report_cosine(cos_sim, n)

    if dump_row = opts[:dump_row], do: dump_row_to_file(ours, dump_row, n, opts[:dump_path])
    if ckpt_dir = opts[:ckpt_dir], do: report_fsq(ours, dataset, ckpt_dir, n)
  end

  defp dump_row_to_file(ours, row_idx, n, path) do
    if row_idx < 0 or row_idx >= n do
      IO.puts("(dump-row #{row_idx} out of range [0, #{n - 1}]; skipping dump)")
    else
      path = path || "item#{row_idx}_elixir.raw"
      row = Nx.slice(ours, [row_idx, 0], [1, 768])
      bin = Nx.to_binary(row)
      File.write!(path, bin)
      IO.puts("")
      IO.puts("Dumped row #{row_idx} to #{path} (#{byte_size(bin)} bytes, float32).")
      IO.puts("  Python: np.fromfile(#{inspect(path)}, dtype=np.float32).reshape(1, 768)")
      IO.puts("")
    end
  end

  defp report_first_strings(texts, count) do
    IO.puts("")
    IO.puts("=== First #{count} encoded strings (compare with Python str(dict).replace) ===")
    texts
    |> Enum.take(count)
    |> Enum.with_index(0)
    |> Enum.each(fn {s, i} ->
      truncated = if String.length(s) > 120, do: String.slice(s, 0, 120) <> "...", else: s
      IO.puts("  [#{i}] #{truncated}")
    end)
    IO.puts("")
  end

  # If .npy row order differs from our items.json order, row 0's best match won't be dataset row 0.
  defp report_row_order_check(ours, dataset) do
    ours_0 = Nx.slice(ours, [0, 0], [1, 768])
    dots = Nx.sum(Nx.multiply(ours_0, dataset), axes: [1])
    norm_0 = Nx.sqrt(Nx.sum(Nx.multiply(ours_0, ours_0), axes: [1]))
    norm_ds = Nx.sqrt(Nx.sum(Nx.multiply(dataset, dataset), axes: [1]))
    product = Nx.multiply(norm_0, norm_ds)
    safe = Nx.select(Nx.greater(product, 1.0e-9), product, Nx.tensor(1.0, type: Nx.type(product)))
    cos_all = Nx.divide(dots, safe)
    flat = Nx.to_flat_list(cos_all)
    {max_sim, best_idx} = Enum.with_index(flat, 0) |> Enum.max_by(fn {v, _} -> v end)
    if best_idx != 0 do
      IO.puts("")
      IO.puts("=== Row order check ===")
      IO.puts("  Our item 0's best match in .npy is row #{best_idx} (cos=#{Float.round(max_sim, 4)}), not row 0.")
      IO.puts("  .npy row order may differ from items.json; consider aligning item order.")
      IO.puts("")
    end
  end

  # If norms differ a lot, pooling or normalization may differ (e.g. masked vs unmasked mean).
  defp report_norm_diagnostic(ours, dataset) do
    scalar = fn t -> t |> Nx.squeeze() |> Nx.to_number() end
    norm_ours_0 = Nx.slice(ours, [0, 0], [1, 768]) |> then(&Nx.sqrt(Nx.sum(Nx.multiply(&1, &1), axes: [1]))) |> scalar.()
    norm_ds_0 = Nx.slice(dataset, [0, 0], [1, 768]) |> then(&Nx.sqrt(Nx.sum(Nx.multiply(&1, &1), axes: [1]))) |> scalar.()
    mean_norm_ours = Nx.sqrt(Nx.sum(Nx.multiply(ours, ours), axes: [1])) |> Nx.mean() |> scalar.()
    mean_norm_ds = Nx.sqrt(Nx.sum(Nx.multiply(dataset, dataset), axes: [1])) |> Nx.mean() |> scalar.()
    IO.puts("")
    IO.puts("=== Scale / L2 norm (row 0 and mean) ===")
    IO.puts("  ours row 0: #{Float.round(norm_ours_0, 4)}  |  dataset row 0: #{Float.round(norm_ds_0, 4)}")
    IO.puts("  mean norm ours: #{Float.round(mean_norm_ours, 4)}  |  mean norm dataset: #{Float.round(mean_norm_ds, 4)}")
    IO.puts("  (Large ratio suggests different pooling, e.g. masked vs unmasked mean.)")
    IO.puts("  To compare with Python: encode the same string (e.g. [0] above) with sentence_transformers and diff embeddings.")
    IO.puts("")
  end

  defp ensure_npy!(path) do
    if File.regular?(path), do: :ok, else: download_npy!(path)
  end

  defp download_npy!(path) do
    IO.puts("Downloading item_text_embeddings.npy (~92 MB)...")
    Application.ensure_all_started(:req)
    case Req.get(@dataset_npy_url) do
      {:ok, %{status: 200, body: body}} ->
        File.mkdir_p!(Path.dirname(path))
        File.write!(path, body)
        :ok
      {:ok, %{status: code}} ->
        raise "HTTP #{code} for #{@dataset_npy_url}"
      {:error, reason} ->
        raise "Download failed: #{inspect(reason)}"
    end
  end

  defp load_ordered_texts(items_path, limit) do
    raw = File.read!(items_path) |> Jason.decode!()
    items = raw["items"] || []
    items
    |> Enum.take(limit)
    |> Enum.map(fn item -> Embedding.recgpt_item_text(item) end)
  end

  defp load_dataset_embeddings(npy_path, n) do
    case Npy.load(npy_path, :nx) do
      {:ok, tensor} ->
        tensor = ensure_2d(tensor)
        {rows, _} = Nx.shape(tensor)
        if rows < n, do: raise "Dataset .npy has #{rows} rows, need #{n}"
        Nx.slice(tensor, [0, 0], [n, 768])
      {:error, reason} ->
        raise "Failed to load #{npy_path}: #{inspect(reason)}"
    end
  end

  defp ensure_2d(tensor) do
    shape = Nx.shape(tensor)
    case shape do
      {_n, 768} -> tensor
      {_n, 1, 768} -> Nx.squeeze(tensor, axes: [1])
      _ -> raise "Unexpected .npy shape #{inspect(shape)}, expected {n, 768} or {n, 1, 768}"
    end
  end

  defp cosine_similarity_per_row(a, b) do
    # per row: dot(a[i], b[i]) / (norm(a[i]) * norm(b[i]))
    dots = Nx.sum(Nx.multiply(a, b), axes: [1])
    norm_a = Nx.sqrt(Nx.sum(Nx.multiply(a, a), axes: [1]))
    norm_b = Nx.sqrt(Nx.sum(Nx.multiply(b, b), axes: [1]))
    product = Nx.multiply(norm_a, norm_b)
    # avoid div by zero
    safe = Nx.select(Nx.greater(product, 1.0e-9), product, Nx.tensor(1.0, type: Nx.type(product)))
    Nx.divide(dots, safe)
  end

  defp report_cosine(cos_sim, n) do
    flat = Nx.to_flat_list(cos_sim)
    mean = Enum.sum(flat) / n
    min_val = Enum.min(flat)
    max_val = Enum.max(flat)
    variance = Enum.map(flat, &(&1 - mean) ** 2) |> Enum.sum() |> then(&(&1 / n))
    std = :math.sqrt(variance)

    IO.puts("")
    IO.puts("=== Embedding comparison (our Bumblebee vs dataset .npy) ===")
    IO.puts("  Cosine similarity (per item):")
    IO.puts("    mean = #{Float.round(mean, 4)}")
    IO.puts("    min  = #{Float.round(min_val, 4)}")
    IO.puts("    max  = #{Float.round(max_val, 4)}")
    IO.puts("    std  = #{Float.round(std, 4)}")
    IO.puts("")
    verdict =
      cond do
        mean >= 0.99 -> "very close (mean >= 0.99)."
        mean >= 0.95 -> "moderate match (0.95-0.99)."
        true -> "large mismatch (mean < 0.95). Likely explains poor eval."
      end
    IO.puts("  Verdict: #{verdict}")
    IO.puts("")
  end

  defp report_fsq(ours, dataset, ckpt_dir, n) do
    ckpt_params = CheckpointLoader.load_from_export(ckpt_dir)
    params = FSQ.load_params(ckpt_params)

    if fsq_ok?(params) do
      tokens_ours = FSQEncoder.encode_embeddings_to_token_id_list(ours, params)
      tokens_dataset = FSQEncoder.encode_embeddings_to_token_id_list(dataset, params)
      same = Enum.zip(tokens_ours, tokens_dataset) |> Enum.count(fn {a, b} -> a == b end)
      frac = same / n
      IO.puts("=== FSQ token agreement (same 4-token code per item) ===")
      IO.puts("  #{same} / #{n} = #{Float.round(frac * 100, 1)}%")
      IO.puts("")
    else
      IO.puts("(FSQ params not in checkpoint; skipping token comparison)")
    end
  end

  defp fsq_ok?(%{"project_in" => %{"kernel" => k}, "project_out" => %{"kernel" => o}})
       when not is_nil(k) and not is_nil(o), do: true
  defp fsq_ok?(_), do: false
end
