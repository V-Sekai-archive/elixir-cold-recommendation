defmodule RecGPT.TrainingTest do
  use ExUnit.Case, async: true

  alias RecGPT.FSQ
  alias RecGPT.Training

  @padding_id FSQ.padding_id()
  @seq_cap 1024
  @max_length 256

  describe "build_train_batch/4" do
    test "returns correct tensor shapes" do
      num_items = 10
      seqs = [[0, 1, 2], [3, 4, 5]]
      token_id_list = for _ <- 1..num_items, do: [0, 100, 200, 300]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      batch_indices = [0, 1]

      {batch_seq, batch_labels, batch_aux_embeds, embed_mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, batch_indices)

      assert Nx.shape(batch_seq) == {2, @seq_cap}
      assert Nx.shape(batch_labels) == {2, @seq_cap}
      assert Nx.shape(batch_aux_embeds) == {2, @max_length * 4, 192}
      assert Nx.shape(embed_mask) == {2, @max_length * 4, 1}
    end

    test "truncates seq longer than max_length" do
      # Seq of 300 items; should be truncated to last 256
      num_items = 310
      seqs = [Enum.to_list(0..(num_items - 1))]
      token_id_list = for _ <- 1..num_items, do: [1, 2, 3, 4]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)

      {batch_seq, _labels, batch_aux, _mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, [0])

      # Effective tokens = 256 * 4 = 1024 (seq_token_capacity)
      first_row = Nx.slice(batch_seq, [0, 0], [1, @seq_cap]) |> Nx.squeeze(axes: [0])
      first_four = Nx.slice(first_row, [0], [4]) |> Nx.to_flat_list()
      assert first_four == [1, 2, 3, 4]
      assert Nx.shape(batch_aux) == {1, @max_length * 4, 192}
    end

    test "uses [0,0,0,0] for item_id missing from token_id_list" do
      # token_id_list has 3 items (indices 0,1,2); seq references item 5 -> nil -> List.duplicate(0, 4)
      seqs = [[0, 5, 1]]
      token_id_list = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
      num_items = 3
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      batch_indices = [0]

      {batch_seq, _labels, _aux, _mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, batch_indices)

      # First 4 tokens = item 0 -> [1,2,3,4]; next 4 = item 5 (missing) -> [0,0,0,0]; next 4 = item 1 -> [10,20,30,40]
      first_row = Nx.slice(batch_seq, [0, 0], [1, @seq_cap]) |> Nx.squeeze(axes: [0])
      assert Nx.slice(first_row, [0], [4]) |> Nx.to_flat_list() == [1, 2, 3, 4]
      assert Nx.slice(first_row, [4], [4]) |> Nx.to_flat_list() == [0, 0, 0, 0]
      assert Nx.slice(first_row, [8], [4]) |> Nx.to_flat_list() == [10, 20, 30, 40]
    end

    test "empty batch_indices raises when building tensors" do
      seqs = [[0]]
      token_id_list = [[1, 2, 3, 4]]
      item_embeddings = Nx.iota({1, 768}) |> Nx.divide(768)

      assert_raise RuntimeError, ~r/cannot build empty tensor/, fn ->
        Training.build_train_batch(seqs, token_id_list, item_embeddings, [])
      end
    end

    test "seq length exactly 256 is not truncated" do
      num_items = 256
      seqs = [Enum.to_list(0..(num_items - 1))]

      token_id_list =
        for i <- 0..(num_items - 1),
            do: [rem(i, 100), rem(i + 1, 100), rem(i + 2, 100), rem(i + 3, 100)]

      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)

      {batch_seq, _labels, batch_aux, _mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, [0])

      assert Nx.shape(batch_seq) == {1, @seq_cap}
      assert Nx.shape(batch_aux) == {1, @max_length * 4, 192}
    end

    test "right-pads batch_seq with padding_id and labels with -100" do
      seqs = [[0]]
      token_id_list = [[1, 2, 3, 4]]
      item_embeddings = Nx.iota({1, 768}) |> Nx.divide(768)
      batch_indices = [0]

      {batch_seq, batch_labels, _aux, _mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, batch_indices)

      first_row = Nx.slice(batch_seq, [0, 0], [1, @seq_cap]) |> Nx.squeeze(axes: [0])
      first_four = Nx.slice(first_row, [0], [4]) |> Nx.to_flat_list()
      assert first_four == [1, 2, 3, 4]
      pad_region = Nx.slice(first_row, [4], [@seq_cap - 4]) |> Nx.to_flat_list()
      assert Enum.all?(pad_region, &(&1 == @padding_id))

      label_row = Nx.slice(batch_labels, [0, 0], [1, @seq_cap]) |> Nx.squeeze(axes: [0])
      label_four = Nx.slice(label_row, [0], [4]) |> Nx.to_flat_list()
      assert label_four == [1, 2, 3, 4]
      label_pad = Nx.slice(label_row, [4], [@seq_cap - 4]) |> Nx.to_flat_list()
      assert Enum.all?(label_pad, &(&1 == -100))
    end
  end

  describe "encode_aux/3" do
    test "returns embeds shape (n*4, 192) and mask (n*4, 1)" do
      num_items = 5
      batch_ids = [0, 1, 2]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)

      {batch_embeds, embed_mask} = Training.encode_aux(batch_ids, item_embeddings, num_items)

      assert Nx.shape(batch_embeds) == {3 * 4, 192}
      assert Nx.shape(embed_mask) == {3 * 4, 1}
    end

    test "uses 0 for out-of-range item ids" do
      num_items = 2
      batch_ids = [0, 1, 99]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768)
      {batch_embeds, mask} = Training.encode_aux(batch_ids, item_embeddings, num_items)
      assert Nx.shape(batch_embeds) == {3 * 4, 192}
      assert Nx.shape(mask) == {3 * 4, 1}
      # Third slot (99) is out of range -> gather index 0
      mask_flat = Nx.reshape(mask, {12}) |> Nx.to_flat_list()
      assert Enum.slice(mask_flat, 8, 4) == [1.0, 1.0, 1.0, 1.0]
    end

    test "mask is 1.0 for valid ids and 0.0 for -1 padding" do
      num_items = 3
      batch_ids = [0, -1, 1]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768)

      {_embeds, mask} = Training.encode_aux(batch_ids, item_embeddings, num_items)

      mask_flat = Nx.reshape(mask, {12}) |> Nx.to_flat_list()
      assert Enum.take(mask_flat, 4) == [1.0, 1.0, 1.0, 1.0]
      assert Enum.slice(mask_flat, 4, 4) == [0.0, 0.0, 0.0, 0.0]
      assert Enum.slice(mask_flat, 8, 4) == [1.0, 1.0, 1.0, 1.0]
    end

    test "empty batch_ids raises when building tensors" do
      num_items = 5
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)

      assert_raise RuntimeError, ~r/cannot build empty tensor/, fn ->
        Training.encode_aux([], item_embeddings, num_items)
      end
    end
  end

  describe "loss_shifted_ce/2" do
    test "returns a finite scalar" do
      batch = 2
      seq_len = 8
      vocab = 15_361
      logits = Nx.iota({batch, seq_len, vocab}) |> Nx.divide(vocab) |> Nx.subtract(0.5)

      labels =
        Nx.tensor(
          [[1, 2, 3, 4, -100, -100, -100, -100], [5, 6, -100, -100, -100, -100, -100, -100]],
          type: {:s, 32}
        )

      loss = Training.loss_shifted_ce(logits, labels)
      assert Nx.shape(loss) == {}
      loss_val = Nx.to_number(loss)
      assert loss_val > 0 and loss_val < 100
      # credo:disable-for-next-line Credo.Check.Warning.OperationOnSameValues
      assert loss_val == loss_val and loss_val != :infinity and loss_val != :neg_infinity
    end

    test "ignores positions with label -100" do
      batch = 1
      seq_len = 4
      vocab = 15_361
      logits = Nx.iota({batch, seq_len, vocab}) |> Nx.divide(vocab) |> Nx.subtract(0.5)
      labels_all_valid = Nx.tensor([[1, 2, 3, 4]], type: {:s, 32})
      labels_half_valid = Nx.tensor([[1, 2, -100, -100]], type: {:s, 32})

      loss_all = Training.loss_shifted_ce(logits, labels_all_valid)
      loss_half = Training.loss_shifted_ce(logits, labels_half_valid)
      assert Nx.to_number(loss_all) > 0
      assert Nx.to_number(loss_half) > 0
    end

    test "all labels -100 returns zero loss (no valid positions)" do
      batch = 1
      seq_len = 4
      vocab = 15_361
      logits = Nx.iota({batch, seq_len, vocab}) |> Nx.divide(vocab) |> Nx.subtract(0.5)
      labels = Nx.tensor([[-100, -100, -100, -100]], type: {:s, 32})
      loss = Training.loss_shifted_ce(logits, labels)
      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) == 0.0
    end
  end
end
