# Pretrain: one batch with synthetic data and write export dir (same flow as mix recgpt.pretrain).
defmodule RecGPT.PretrainIntegrationTest do
  use ExUnit.Case, async: false

  alias RecGPT.AxonTrain
  alias RecGPT.CheckpointExport
  alias RecGPT.CheckpointLoader

  @tag :integration
  test "pretrain flow runs one batch and writes export dir" do
    base = Path.join(System.tmp_dir!(), "recgpt_pretrain_#{:erlang.unique_integer([:positive])}")
    File.mkdir_p!(base)
    ckpt_dir = Path.join(base, "ckpt")
    out_dir = Path.join(base, "out")

    try do
      params = %{
        "wte" => Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
        "pred_head.weight" =>
          Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
        "pred_head.bias" => Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
      }

      CheckpointExport.write_export(params, ckpt_dir)

      token_id_list = [[1, 2, 3, 4], [5, 6, 7, 8]]
      sequences = [[0, 1], [1, 0]]
      item_embeddings = Nx.iota({2, 768}) |> Nx.divide(768 * 2) |> Nx.as_type({:f, 32})

      stream =
        AxonTrain.stream_batches(sequences, token_id_list, item_embeddings,
          batch_size: 2,
          shuffle: false
        )

      trained = AxonTrain.run(stream, params, iterations: 1, log: 0)
      CheckpointExport.write_export(trained, out_dir)

      assert File.dir?(out_dir)
      assert File.regular?(Path.join(out_dir, "manifest.json"))
      loaded = CheckpointLoader.load_from_export(out_dir)
      assert Map.has_key?(loaded, "wte")
      assert Map.has_key?(loaded, "pred_head.weight")
    after
      File.rm_rf(base)
    end
  end
end
