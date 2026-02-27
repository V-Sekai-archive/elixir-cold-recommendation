# Asserts constants and formats documented in docs/01_python_recgpt_parity_progress.md.
# Run: mix test test/recgpt/parity_constants_test.exs
defmodule RecGPT.ParityConstantsTest do
  use ExUnit.Case, async: true

  alias RecGPT.Embedding
  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder
  alias RecGPT.Training

  # Values from parity progress doc (01_python_recgpt_parity_progress.md)
  @doc "FSQ vocab size (0..15359); padding id is 15360."
  @parity_fsq_vocab_size 15_360
  @parity_fsq_padding_id 15_360
  @parity_fsq_tokens_per_item 4
  @parity_embedding_size 768
  @parity_seq_token_capacity 1024
  @parity_max_length 256
  @parity_label_ignore -100

  describe "FSQ constants (parity doc §2)" do
    test "vocab_size is 15360" do
      assert FSQ.vocab_size() == @parity_fsq_vocab_size
    end

    test "padding_id is 15360" do
      assert FSQ.padding_id() == @parity_fsq_padding_id
    end

    test "seq_len is 4 (tokens per item)" do
      assert FSQ.seq_len() == @parity_fsq_tokens_per_item
    end

    test "dim is 192 (FSQ internal dimension)" do
      assert FSQ.dim() == 192
    end

    test "basis matches levels [8,8,8,6,5] cumprod" do
      assert Nx.to_flat_list(FSQ.basis()) == [1, 8, 64, 512, 3072]
    end
  end

  describe "Embedding constants (parity doc §1)" do
    test "embedding_size is 768" do
      assert Embedding.embedding_size() == @parity_embedding_size
    end
  end

  describe "Training batch format (parity doc §3, matches Python GPT2RecBatchTrainAuxData)" do
    # Python: HKUDS/RecGPT utils/data.py GPT2RecBatchTrainAuxData
    # - max_length=256, padding_id=15360, id_list/label_list length 1024, label -100, right-padding
    # - encode_aux: 256 item_ids -> (1024, 192) aux embeds, (1024, 1) mask
    test "build_train_batch output shapes and padding match Python (seq_token_capacity 1024, max_length 256, padding_id 15360, label_ignore -100)" do
      num_items = 3
      seqs = [[0, 1]]
      token_id_list = for _ <- 1..num_items, do: [10, 20, 30, 40]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      batch_indices = [0]

      {batch_seq, batch_labels, batch_aux, embed_mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, batch_indices)

      # Doc: seq_token_capacity 1024, max_length 256
      assert Nx.shape(batch_seq) == {1, @parity_seq_token_capacity}
      assert Nx.shape(batch_labels) == {1, @parity_seq_token_capacity}
      assert Nx.shape(batch_aux) == {1, @parity_max_length * 4, 192}
      assert Nx.shape(embed_mask) == {1, @parity_max_length * 4, 1}

      # Padding region uses padding_id and label_ignore
      first_seq = batch_seq |> Nx.squeeze(axes: [0]) |> Nx.to_flat_list()
      first_labels = batch_labels |> Nx.squeeze(axes: [0]) |> Nx.to_flat_list()
      # First 8 tokens are from 2 items (2*4); rest padded
      assert Enum.at(first_seq, 8) == @parity_fsq_padding_id
      assert Enum.at(first_labels, 8) == @parity_label_ignore
    end
  end

  describe "FSQEncoder output format (parity doc §2)" do
    test "encode_embeddings_to_token_id_list returns 4 tokens per item in 0..vocab_size-1" do
      params = %{
        "project_in" => %{
          "kernel" => Nx.iota({192, 5}) |> Nx.divide(192 * 5),
          "bias" => Nx.broadcast(0.0, {5})
        },
        "project_out" => %{
          "kernel" => Nx.iota({5, 192}) |> Nx.divide(5 * 192),
          "bias" => Nx.broadcast(0.0, {192})
        }
      }

      embeddings = Nx.iota({2, 768}) |> Nx.divide(768 * 2)
      token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params)

      assert length(token_id_list) == 2
      assert Enum.all?(token_id_list, fn list -> length(list) == @parity_fsq_tokens_per_item end)

      assert Enum.all?(token_id_list, fn list ->
               Enum.all?(list, fn id -> id >= 0 and id < @parity_fsq_vocab_size end)
             end)
    end
  end
end
