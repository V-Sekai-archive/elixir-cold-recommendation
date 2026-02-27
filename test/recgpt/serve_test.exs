defmodule RecGPT.ServeTest do
  use ExUnit.Case, async: false

  import Plug.Conn
  import Plug.Test

  alias RecGPT.Serve
  alias RecGPT.Serve.Plug

  describe "item_ids_to_context_token_ids/3" do
    test "left-pads to seq_token_capacity 1024" do
      token_id_list = for _ <- 1..10, do: [1, 2, 3, 4]
      ids = [0, 1]
      context = Serve.item_ids_to_context_token_ids(ids, token_id_list)
      assert length(context) == 1024
      # First tokens should be padding (15360), then 8 tokens from 2 items
      padding_count = 1024 - 8
      assert Enum.take(context, padding_count) == List.duplicate(15_360, padding_count)
      assert Enum.drop(context, padding_count) == [1, 2, 3, 4, 1, 2, 3, 4]
    end

    test "truncates to max_length items (255)" do
      token_id_list = for _ <- 1..300, do: [0, 0, 0, 0]
      ids = Enum.to_list(0..299)
      context = Serve.item_ids_to_context_token_ids(ids, token_id_list)
      assert length(context) == 1024
    end
  end

  describe "recommend/3" do
    test "returns error when item_ids empty" do
      state = build_stub_state()
      assert Serve.recommend(state, [], 5) == {:error, "item_ids must be non-empty"}
    end

    test "returns up to top_k recommended item_ids (best first)" do
      state = build_stub_state()
      assert {:ok, list} = Serve.recommend(state, [0], 5)
      assert is_list(list)
      assert length(list) <= 5
      assert length(list) <= 2, "stub catalog has 2 items"
      Enum.each(list, fn id -> assert id in [0, 1] end)
    end

    test "top_k=1 returns at most one item" do
      state = build_stub_state()
      assert {:ok, list} = Serve.recommend(state, [0], 1)
      assert length(list) <= 1
    end

    test "top_k is capped at 20" do
      state = build_stub_state()
      assert {:ok, list} = Serve.recommend(state, [0], 100)
      assert length(list) <= 20
    end
  end

  describe "search/3" do
    test "returns empty when q empty or no catalog" do
      state = build_stub_state()
      assert Serve.search(state, "", 20) == []
      state_empty = %{state | item_text: %{}}
      assert Serve.search(state_empty, "foo", 20) == []
    end

    test "returns matches when catalog has text" do
      state = %{build_stub_state() | item_text: %{0 => "Action game", 1 => "Puzzle"}}
      assert length(Serve.search(state, "action", 20)) >= 1
      assert length(Serve.search(state, "puzzle", 20)) >= 1
      assert Serve.search(state, "nonexistent", 20) == []
    end
  end

  describe "load_state/3" do
    test "returns error when fixture missing" do
      assert {:error, _} =
               Serve.load_state("/nonexistent/fixture.json", "data/recgpt_ckpt_export", nil)
    end

    test "returns error when checkpoint dir has no manifest" do
      dir = Path.join(System.tmp_dir!(), "recgpt_no_manifest_#{:erlang.unique_integer([:positive])}")
      File.mkdir_p!(dir)
      fixture = Path.join(System.tmp_dir!(), "recgpt_fixture_#{:erlang.unique_integer([:positive])}.json")
      File.write!(fixture, Jason.encode!(%{"token_id_list" => [[1, 2, 3, 4]], "num_items" => 1}))

      try do
        assert {:error, _} = Serve.load_state(fixture, dir, nil)
      after
        File.rm_rf(dir)
        File.rm(fixture)
      end
    end

    @tag :integration
    test "loads state when fixture and checkpoint exist" do
      fixture = Path.expand("../data/serve_e2e_fixture.json", File.cwd!())
      ckpt = Path.expand("../data/recgpt_ckpt_export", File.cwd!())

      if File.regular?(fixture) and File.regular?(Path.join(ckpt, "manifest.json")) do
        assert {:ok, state} = Serve.load_state(fixture, ckpt, nil)
        assert state.num_items > 0
        assert is_list(state.token_id_list)
      else
        raise "Skip: need data/serve_e2e_fixture.json and data/recgpt_ckpt_export (or run Serve E2E from M:\\reflex-logic-other)"
      end
    end
  end

  describe "Plug" do
    test "returns 503 when serve_state not set" do
      Application.delete_env(:recgpt, :serve_state)
      conn = conn(:get, "/health") |> Plug.call([])
      assert conn.status == 503
    end

    test "GET /health returns 200 when state set" do
      Application.put_env(:recgpt, :serve_state, build_stub_state())
      conn = conn(:get, "/health") |> Plug.call([])
      assert conn.status == 200
      assert Jason.decode!(conn.resp_body)["status"] == "ok"
    end

    test "GET /search returns matches" do
      state = %{build_stub_state() | item_text: %{0 => "Test game"}}
      Application.put_env(:recgpt, :serve_state, state)
      conn = conn(:get, "/search?q=test") |> Plug.call([])
      assert conn.status == 200
      body = Jason.decode!(conn.resp_body)
      assert Map.has_key?(body, "matches")
    end

    test "POST /recommend returns item_ids and item_texts" do
      Application.put_env(:recgpt, :serve_state, build_stub_state())

      conn =
        conn(:post, "/recommend", Jason.encode!(%{"item_ids" => [0], "top_k" => 5}))
        |> put_req_header("content-type", "application/json")
        |> Plug.call([])

      assert conn.status == 200
      body = Jason.decode!(conn.resp_body)
      assert Map.has_key?(body, "item_ids")
      assert Map.has_key?(body, "item_texts")
      assert is_list(body["item_ids"])
      assert is_list(body["item_texts"])
    end

    test "POST /recommend returns 400 when item_ids missing" do
      Application.put_env(:recgpt, :serve_state, build_stub_state())

      conn =
        conn(:post, "/recommend", Jason.encode!(%{}))
        |> put_req_header("content-type", "application/json")
        |> Plug.call([])

      assert conn.status == 400
    end

    test "404 for unknown path" do
      Application.put_env(:recgpt, :serve_state, build_stub_state())
      conn = conn(:get, "/unknown") |> Plug.call([])
      assert conn.status == 404
    end
  end

  describe "safe_str/1" do
    test "returns empty string for nil" do
      assert Serve.safe_str(nil) == ""
    end

    test "returns binary as-is" do
      assert Serve.safe_str("hello") == "hello"
    end

    test "returns inspect for map" do
      assert Serve.safe_str(%{a: 1}) =~ "%{"
    end

    test "returns to_string for number" do
      assert Serve.safe_str(42) == "42"
    end
  end

  defp build_stub_state do
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

    %Serve{
      params: params,
      trie: trie,
      token_id_list: token_id_list,
      item_text: %{},
      num_items: 2,
      get_logits_fn: get_logits_fn
    }
  end

  defp build_dummy_params do
    wte = Nx.iota({15361, 768}) |> Nx.divide(15361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({15361, 768}) |> Nx.divide(15361 * 768) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15361}) |> Nx.as_type({:f, 32})

    %{
      "wte" => wte,
      "pred_head.weight" => head_w,
      "pred_head.bias" => head_b
    }
  end
end
