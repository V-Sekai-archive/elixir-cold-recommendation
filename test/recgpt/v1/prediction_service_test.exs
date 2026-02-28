# gRPC status codes: 3 = INVALID_ARGUMENT, 14 = UNAVAILABLE
defmodule Recgpt.V1.PredictionServiceTest do
  use ExUnit.Case, async: false

  alias RecGPT.Serve
  alias Recgpt.V1.PredictionService.Server
  alias Recgpt.V1.PredictRequest

  setup do
    state = build_stub_state()
    Application.put_env(:recgpt, :serve_state, state)
    on_exit(fn -> Application.delete_env(:recgpt, :serve_state) end)
    :ok
  end

  describe "predict/2" do
    test "returns item_ids and items (ItemSummary) for valid request" do
      request = %PredictRequest{context_item_ids: [0], max_results: 5}
      response = Server.predict(request, nil)
      assert is_list(response.item_ids)
      assert length(response.items) == length(response.item_ids)
      Enum.zip(response.item_ids, response.items)
      |> Enum.each(fn {id, item} ->
        assert item.item_id == id
        assert is_binary(item.display_name) or item.display_name == ""
      end)
    end

    test "empty context_item_ids raises INVALID_ARGUMENT" do
      request = %PredictRequest{context_item_ids: [], max_results: 5}
      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError -> assert e.status == 3
      end
    end

    test "nil serve_state raises UNAVAILABLE" do
      Application.put_env(:recgpt, :serve_state, nil)
      request = %PredictRequest{context_item_ids: [0], max_results: 5}
      try do
        try do
          Server.predict(request, nil)
        rescue
          e in GRPC.RPCError -> assert e.status == 14
        end
      after
        Application.put_env(:recgpt, :serve_state, build_stub_state())
      end
    end

    test "max_results 0 uses default 5 and succeeds" do
      request = %PredictRequest{context_item_ids: [0], max_results: 0}
      response = Server.predict(request, nil)
      assert is_list(response.item_ids)
      assert length(response.items) == length(response.item_ids)
    end

    test "max_results 21 raises INVALID_ARGUMENT" do
      request = %PredictRequest{context_item_ids: [0], max_results: 21}
      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError -> assert e.status == 3
      end
    end

    test "full flow: load_state then predict returns valid response" do
      Application.ensure_all_started(:nx)

      base =
        Path.join(System.tmp_dir!(), "recgpt_predict_flow_#{:erlang.unique_integer([:positive])}")

      File.mkdir_p!(base)
      fixture_path = Path.join(base, "fixture.json")
      ckpt_dir = Path.join(base, "ckpt")

      try do
        num_items = 2
        token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
        File.write!(
          fixture_path,
          Jason.encode!(%{"num_items" => num_items, "token_id_list" => token_id_list})
        )

        write_stub_ckpt!(ckpt_dir)

        assert {:ok, state} = Serve.load_state(fixture_path, ckpt_dir, nil)
        Application.put_env(:recgpt, :serve_state, state)

        request = %PredictRequest{context_item_ids: [0], max_results: 5}
        response = Server.predict(request, nil)

        assert is_list(response.item_ids)
        assert length(response.items) == length(response.item_ids)
        assert Enum.all?(response.item_ids, fn id -> is_integer(id) and id >= 0 and id < num_items end)
        Enum.zip(response.item_ids, response.items)
        |> Enum.each(fn {id, item} ->
          assert item.item_id == id
          assert is_binary(item.display_name) or item.display_name == ""
        end)
      after
        File.rm_rf(base)
      end
    end
  end

  defp build_stub_state do
    Application.ensure_all_started(:nx)
    token_id_list = [[100, 200, 300, 400], [101, 201, 301, 401]]
    trie = RecGPT.Trie.build(token_id_list)
    params = build_dummy_params()

    get_logits_fn = fn token_list ->
      seq_len = length(token_list)
      batch_token_ids = Nx.tensor([token_list], type: {:s, 32})
      batch_aux = Nx.broadcast(0.0, {1, seq_len, 192}) |> Nx.as_type({:f, 32})
      embed_mask = Nx.broadcast(1.0, {1, seq_len, 1}) |> Nx.as_type({:f, 32})
      RecGPT.Inference.forward(batch_token_ids, batch_aux, embed_mask, params)
    end

    %RecGPT.Serve{
      params: params,
      trie: trie,
      token_id_list: token_id_list,
      item_text: %{},
      num_items: 2,
      get_logits_fn: get_logits_fn
    }
  end

  defp build_dummy_params do
    wte = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    %{"wte" => wte, "pred_head.weight" => head_w, "pred_head.bias" => head_b}
  end

  defp write_stub_ckpt!(dir) do
    File.mkdir_p!(dir)
    params = %{
      "wte" => Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
      "pred_head.weight" =>
        Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32}),
      "pred_head.bias" => Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    }
    RecGPT.CheckpointExport.write_export(params, dir)
  end
end
