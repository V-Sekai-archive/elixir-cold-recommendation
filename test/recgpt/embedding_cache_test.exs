defmodule RecGPT.EmbeddingCacheTest do
  use ExUnit.Case, async: true

  alias RecGPT.EmbeddingCache
  alias RecGPT.Catalog.ItemEmbedding
  alias RecGPT.Catalog.ItemToken
  alias RecGPT.Repo

  setup do
    :ok = Ecto.Adapters.SQL.Sandbox.checkout(Repo)
    :ok
  end

  test "load_from_db creates ETS tables and populates them" do
    # Insert test data
    # 768 f32 values
    embedding1 = :crypto.strong_rand_bytes(768 * 4)
    embedding2 = :crypto.strong_rand_bytes(768 * 4)

    Repo.insert_all(ItemEmbedding, [
      %{item_id: 1, embedding: embedding1},
      %{item_id: 2, embedding: embedding2}
    ])

    Repo.insert_all(ItemToken, [
      %{item_id: 1, t0: 10, t1: 20, t2: 30, t3: 40},
      %{item_id: 2, t0: 11, t1: 21, t2: 31, t3: 41}
    ])

    # Load caches (use BinaryBackend so test runs without EXLA/CUDA)
    {:ok, {embeddings_table, tokens_table}} = EmbeddingCache.load_from_db({Nx.BinaryBackend, []})

    assert is_reference(embeddings_table)
    assert is_reference(tokens_table)

    # Test embedding lookup
    {:ok, tensor1} = EmbeddingCache.get_embedding(embeddings_table, 1)
    assert Nx.shape(tensor1) == {768}

    {:error, :not_found} = EmbeddingCache.get_embedding(embeddings_table, 999)

    # Test token lookup
    {:ok, tokens1} = EmbeddingCache.get_tokens(tokens_table, 1)
    assert tokens1 == [10, 20, 30, 40]

    {:ok, tokens2} = EmbeddingCache.get_tokens(tokens_table, 2)
    assert tokens2 == [11, 21, 31, 41]

    {:error, :not_found} = EmbeddingCache.get_tokens(tokens_table, 999)

    # Cleanup
    :ok = EmbeddingCache.cleanup(embeddings_table, tokens_table)
  end

  test "get_embedding and get_tokens return proper tuples" do
    embedding_bin = :crypto.strong_rand_bytes(768 * 4)

    Repo.insert_all(ItemEmbedding, [
      %{item_id: 5, embedding: embedding_bin}
    ])

    Repo.insert_all(ItemToken, [
      %{item_id: 5, t0: 100, t1: 101, t2: 102, t3: 103}
    ])

    {:ok, {_embeddings_table, tokens_table}} = EmbeddingCache.load_from_db({Nx.BinaryBackend, []})

    # Verify token lookup on missing item
    assert EmbeddingCache.get_tokens(tokens_table, 5) == {:ok, [100, 101, 102, 103]}
    assert EmbeddingCache.get_tokens(tokens_table, 6) == {:error, :not_found}
  end
end
