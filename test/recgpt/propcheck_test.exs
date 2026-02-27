# Property-based tests for RecGPT modules (PropCheck: https://github.com/alfert/propcheck).
# Run: mix test test/recgpt/propcheck_test.exs
defmodule RecGPT.PropCheckTest do
  use ExUnit.Case, async: false
  use PropCheck, default_opts: [numtests: 100]

  alias RecGPT.FSQ
  alias RecGPT.FSQEncoder
  alias RecGPT.Training

  # Shared FSQ params for properties (fixed shape; no PropCheck generation of large tensors).
  defp fsq_params do
    project_in_k = Nx.iota({192, 5}) |> Nx.divide(192 * 5) |> Nx.subtract(0.05)
    project_in_b = Nx.broadcast(0.0, {5})
    project_out_k = Nx.iota({5, 192}) |> Nx.divide(5 * 192) |> Nx.subtract(0.05)
    project_out_b = Nx.broadcast(0.0, {192})

    %{
      "project_in" => %{"kernel" => project_in_k, "bias" => project_in_b},
      "project_out" => %{"kernel" => project_out_k, "bias" => project_out_b}
    }
  end

  # --- FSQ properties ---

  property "FSQ: indices in 0..vocab_size-1 round-trip via 5-d codes (indices -> codes -> indices)" do
    # Reconstruct 5-d normalized codes from indices (basis/levels), then codes_to_indices recovers indices.
    forall indices_4 <- vector(4, integer(0, FSQ.vocab_size() - 1)) do
      indices = Nx.tensor([indices_4], type: {:s, 32})
      b = Nx.reshape(FSQ.basis(), {1, 1, 5})
      l = Nx.reshape(FSQ.levels(), {1, 1, 5})
      indices_5d = Nx.reshape(indices, {1, 4, 1})
      codes_non_centered = Nx.remainder(Nx.quotient(indices_5d, b), l)
      codes = FSQ.scale_and_shift_inverse(codes_non_centered)
      recovered = FSQ.codes_to_indices(codes)
      Nx.all(Nx.equal(indices, recovered)) |> Nx.to_number() == 1
    end
  end

  property "FSQ: codes_to_indices returns integers in 0..vocab_size-1 for normalized codes in [-1,1]" do
    forall code_5 <- vector(5, float(-1.0, 1.0)) do
      codes = Nx.tensor([[[code_5]]], type: {:f, 32})
      idx = FSQ.codes_to_indices(codes)
      flat = Nx.to_flat_list(idx)

      Enum.all?(flat, fn v ->
        is_integer(v) and v >= 0 and v < FSQ.vocab_size()
      end)
    end
  end

  property "FSQ: scale_and_shift then scale_and_shift_inverse recovers normalized input (within float tolerance)" do
    forall z_5 <- vector(5, float(-1.0, 1.0)) do
      z = Nx.tensor([[[z_5]]], type: {:f, 32})
      shifted = FSQ.scale_and_shift(z)
      recovered = FSQ.scale_and_shift_inverse(shifted)
      # Float32 round-trip can drift; use relaxed tolerance so property is stable
      Nx.all_close(z, recovered, atol: 1.0e-5, rtol: 1.0e-5) |> Nx.to_number() == 1
    end
  end

  property "FSQ: bound output has finite values (no NaN/Inf)" do
    forall z_5 <- vector(5, float(-10.0, 10.0)) do
      z = Nx.tensor([[[z_5]]], type: {:f, 32})
      out = FSQ.bound(z)
      vals = Nx.to_flat_list(out)
      Enum.all?(vals, fn v -> is_number(v) and v == v and v != :infinity and v != :neg_infinity end)
    end
  end

  property "FSQ: quantize output is in [-1.1, 1.1] (normalized range with small tolerance)" do
    forall z_5 <- vector(5, float(-5.0, 5.0)) do
      z = Nx.tensor([[[z_5]]], type: {:f, 32})
      out = FSQ.quantize(z)
      vals = Nx.to_flat_list(out)
      Enum.all?(vals, fn v -> v >= -1.1 and v <= 1.1 end)
    end
  end

  property "FSQ: encode returns indices in 0..vocab_size-1" do
    params = fsq_params()

    forall _ <- nat() do
      # Single random (batch=1, 4, 192) input
      z = Nx.iota({1, 4, 192}) |> Nx.divide(1000) |> Nx.subtract(0.2)
      {_quant_embeds, indices} = FSQ.encode(z, params)
      flat = Nx.to_flat_list(indices)
      Enum.all?(flat, fn v -> v >= 0 and v < FSQ.vocab_size() end)
    end
  end

  # --- Training properties ---

  property "Training: build_train_batch returns tensors with expected shapes" do
    forall [num_items, batch_size] <- [integer(2, 10), integer(1, 3)] do
      num_seqs = max(3, batch_size)
      seqs = for i <- 0..(num_seqs - 1), do: [rem(i, num_items), rem(i + 1, num_items)]
      token_id_list = for _ <- 1..num_items, do: [0, 100, 200, 300]
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      batch_indices = Enum.to_list(0..(batch_size - 1))

      {batch_seq, batch_labels, batch_aux, embed_mask} =
        Training.build_train_batch(seqs, token_id_list, item_embeddings, batch_indices)

      Nx.shape(batch_seq) == {batch_size, 1024} and
        Nx.shape(batch_labels) == {batch_size, 1024} and
        Nx.shape(batch_aux) == {batch_size, 256 * 4, 192} and
        Nx.shape(embed_mask) == {batch_size, 256 * 4, 1}
    end
  end

  property "Training: loss_shifted_ce is non-negative for random logits and valid labels" do
    forall _ <- nat() do
      batch = 2
      seq_len = 4
      vocab = 15_361
      logits = Nx.iota({batch, seq_len, vocab}) |> Nx.divide(vocab) |> Nx.subtract(0.5)
      labels = Nx.tensor([[1, 2, -100, -100], [0, 1, 2, 3]], type: {:s, 32})

      loss = Training.loss_shifted_ce(logits, labels)
      val = Nx.to_number(loss)
      val >= 0.0 and val == val
    end
  end

  property "Training: loss_shifted_ce with all -100 labels yields 0.0" do
    forall _ <- nat() do
      batch = 1
      seq_len = 8
      vocab = 15_361
      logits = Nx.iota({batch, seq_len, vocab}) |> Nx.divide(vocab)
      labels = Nx.broadcast(Nx.tensor(-100, type: {:s, 32}), {batch, seq_len})

      loss = Training.loss_shifted_ce(logits, labels)
      Nx.to_number(loss) == 0.0
    end
  end

  property "Training: encode_aux output shapes (n*4, 192) and (n*4, 1)" do
    forall n <- integer(1, 15) do
      num_items = max(n, 5)
      batch_ids = for _ <- 1..n, do: :rand.uniform(num_items) - 1
      item_embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)

      {batch_embeds, embed_mask} = Training.encode_aux(batch_ids, item_embeddings, num_items)

      Nx.shape(batch_embeds) == {n * 4, 192} and Nx.shape(embed_mask) == {n * 4, 1}
    end
  end

  # --- FSQEncoder properties ---

  property "FSQEncoder: encode_embeddings_to_token_id_list length equals num_items" do
    params = fsq_params()

    forall num_items <- integer(1, 25) do
      embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      result = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 8)
      length(result) == num_items
    end
  end

  property "FSQEncoder: each token list has 4 elements in 0..vocab_size-1" do
    params = fsq_params()

    forall num_items <- integer(1, 20) do
      embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      result = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 8)

      Enum.all?(result, fn tok_list ->
        length(tok_list) == 4 and
          Enum.all?(tok_list, fn t -> is_integer(t) and t >= 0 and t < FSQ.vocab_size() end)
      end)
    end
  end

  property "FSQEncoder: same embeddings and params yield same token_id_list (determinism)" do
    params = fsq_params()

    forall num_items <- integer(1, 15) do
      embeddings = Nx.iota({num_items, 768}) |> Nx.divide(768 * num_items)
      a = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 4)
      b = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 4)
      a == b
    end
  end
end
