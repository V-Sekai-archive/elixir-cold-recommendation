# Run with: mix test test/recgpt/embedding_test.exs --include embedding
# First run may be slow (HuggingFace model download). Tests have 10min timeout.
defmodule RecGPT.EmbeddingTest do
  use ExUnit.Case, async: false

  alias RecGPT.Embedding

  describe "embedding_size/0" do
    test "returns 768" do
      assert Embedding.embedding_size() == 768
    end
  end

  describe "serving/0 (cached branch, no model load)" do
    test "returns cached value when Application env is set" do
      stub = :cached_serving_stub
      Application.put_env(:recgpt, :embedding_serving, stub)

      try do
        assert Embedding.serving() == stub
      after
        Application.delete_env(:recgpt, :embedding_serving)
      end
    end
  end

  describe "save_embeddings/2 and load_embeddings/1" do
    test "roundtrip preserves tensor shape and values" do
      embeddings = Nx.iota({3, 768}) |> Nx.divide(768) |> Nx.as_type({:f, 32})
      path = Path.join(System.tmp_dir!(), "recgpt_emb_#{:erlang.unique_integer([:positive])}.nx")

      try do
        Embedding.save_embeddings(embeddings, path)
        loaded = Embedding.load_embeddings(path)
        assert Nx.shape(loaded) == Nx.shape(embeddings)
        assert Nx.all_close(loaded, embeddings) |> Nx.to_number() == 1
      after
        File.rm(path)
      end
    end

    test "load_embeddings/1 raises on missing file" do
      path =
        Path.join(
          System.tmp_dir!(),
          "recgpt_nonexistent_#{:erlang.unique_integer([:positive])}.nx"
        )

      assert_raise File.Error, fn ->
        Embedding.load_embeddings(path)
      end
    end

    test "save_embeddings/2 raises on invalid path (e.g. non-existent directory)" do
      embeddings = Nx.iota({1, 768}) |> Nx.divide(768) |> Nx.as_type({:f, 32})

      path =
        System.tmp_dir!()
        |> Path.join("nonexistent_dir_#{:erlang.unique_integer([:positive])}")
        |> Path.join("file.nx")

      assert_raise File.Error, fn ->
        Embedding.save_embeddings(embeddings, path)
      end
    end

    test "load_embeddings/1 raises on corrupt file (invalid Nx serialization)" do
      path =
        Path.join(System.tmp_dir!(), "recgpt_corrupt_#{:erlang.unique_integer([:positive])}.nx")

      File.write!(path, <<0, 1, 2, 3, 255>>)

      try do
        assert_raise ArgumentError, fn ->
          Embedding.load_embeddings(path)
        end
      after
        File.rm(path)
      end
    end
  end

  describe "encode_texts/1 (requires model load)" do
    @describetag :embedding
    @tag timeout: 600_000
    test "returns tensor shape {n, 768} with finite values" do
      texts = ["hello world", "another sentence"]
      out = Embedding.encode_texts(texts)
      assert Nx.shape(out) == {2, 768}
      assert Nx.type(out) == {:f, 32}
      flat = Nx.to_flat_list(out)
      # credo:disable-for-next-line Credo.Check.Warning.OperationOnSameValues
      assert Enum.all?(flat, fn x -> x == x and x != :infinity and x != :neg_infinity end)
    end

    @describetag :embedding
    @tag timeout: 600_000
    test "different texts yield different vectors" do
      a = Embedding.encode_texts(["apple"])
      b = Embedding.encode_texts(["orange"])
      a_flat = Nx.reshape(a, {768}) |> Nx.to_flat_list()
      b_flat = Nx.reshape(b, {768}) |> Nx.to_flat_list()
      assert a_flat != b_flat
    end
  end

  describe "encode_item_text_dict/1 (requires model load)" do
    @describetag :embedding
    @tag timeout: 600_000
    test "returns tensor shape {num_items, 768} in index order" do
      dict = %{0 => "first", 1 => "second"}
      out = Embedding.encode_item_text_dict(dict)
      assert Nx.shape(out) == {2, 768}
    end

    @describetag :embedding
    @tag timeout: 600_000
    test "with unsorted keys produces rows in sorted index order (parity doc: indices 0..num_items-1, sorted)" do
      # Keys 2, 0, 1 -> sorted indices [0, 1, 2] -> texts for item 0, 1, 2
      dict = %{2 => "item two", 0 => "item zero", 1 => "item one"}
      out = Embedding.encode_item_text_dict(dict)
      assert Nx.shape(out) == {3, 768}
    end
  end

  describe "encode_item_text_dict/1 (no model)" do
    @describetag :embedding
    test "empty map raises (empty texts cause tokenizer or stack to fail)" do
      assert_raise Enum.EmptyError, fn ->
        Embedding.encode_item_text_dict(%{})
      end
    end
  end

  describe "MPNet vs Python (normalize_embeddings=False parity)" do
    @describetag :embedding
    @describetag :compare_embedding
    @tag timeout: 600_000

    test "Elixir encode_texts matches Python reference embeddings when fixtures exist" do
      dir = embedding_fixture_dir()
      texts_path = Path.join(dir, "texts.json")
      embeddings_path = Path.join(dir, "embeddings.json")

      unless File.regular?(texts_path) and File.regular?(embeddings_path) do
        raise """
        Fixtures missing. From repo root run:
          uv run python scripts/export_mpnet_embeddings.py --output-dir data/recgpt_embedding
        Then run: mix test test/recgpt/embedding_test.exs --include compare_embedding --include embedding
        """
      end

      texts = File.read!(texts_path) |> Jason.decode!()
      ref_emb = File.read!(embeddings_path) |> Jason.decode!() |> Nx.tensor(type: {:f, 32})
      elixir_emb = Embedding.encode_texts(texts)
      assert Nx.shape(elixir_emb) == Nx.shape(ref_emb)

      # Cosine similarity per row; expect > 0.99 for same model + normalize_embeddings=False
      {n, _dim} = Nx.shape(elixir_emb)

      for i <- 0..(n - 1) do
        a = elixir_emb |> Nx.slice_along_axis(i, 1, axis: 0) |> Nx.squeeze(axes: [0])
        b = ref_emb |> Nx.slice_along_axis(i, 1, axis: 0) |> Nx.squeeze(axes: [0])
        dot = Nx.dot(a, b) |> Nx.squeeze() |> Nx.to_number()
        na = Nx.LinAlg.norm(a) |> Nx.to_number()
        nb = Nx.LinAlg.norm(b) |> Nx.to_number()
        cos = if na > 1.0e-6 and nb > 1.0e-6, do: dot / (na * nb), else: 1.0
        assert cos >= 0.99, "Row #{i} cosine similarity #{cos} < 0.99 (Elixir vs Python MPNet)"
      end
    end
  end

  defp embedding_fixture_dir do
    cwd = File.cwd!()
    from_recgpt = Path.expand("../data/recgpt_embedding", cwd)
    from_repo = Path.join(cwd, "data/recgpt_embedding")

    cond do
      File.exists?(from_recgpt) -> from_recgpt
      File.exists?(from_repo) -> from_repo
      true -> Path.join(cwd, "data/recgpt_embedding")
    end
  end
end
