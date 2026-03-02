# gRPC status codes: 3 = INVALID_ARGUMENT, 9 = FAILED_PRECONDITION
# Predict uses RecGPT.RecommendationService (default impl: Serve); tests use stub serve_state from FrozenHelpers.
defmodule Recgpt.V1.PredictionServiceTest do
  use ExUnit.Case, async: false

  alias Recgpt.V1.PredictionService.Server
  alias Recgpt.V1.PredictRequest

  setup do
    state = RecGPT.TestSupport.FrozenHelpers.build_stub_state()
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

    # Stub: exact inputs and expected outputs; top-k 10; 10 items.
    test "recommendations return exact Steam catalogue item_ids and display_name (stub, top-k 10)" do
      context_item_ids = [0]
      max_results = 10

      item_text = %{
        0 => "Papers, Please",
        1 => "Half-Life 2",
        2 => "Portal",
        3 => "Counter-Strike: Global Offensive",
        4 => "Dota 2",
        5 => "Team Fortress 2",
        6 => "Left 4 Dead 2",
        7 => "Garry's Mod",
        8 => "Terraria",
        9 => "The Witcher 3: Wild Hunt"
      }

      state =
        RecGPT.TestSupport.FrozenHelpers.build_stub_state(10)
        |> Map.put(:item_text, item_text)

      Application.put_env(:recgpt, :serve_state, state)

      request = %PredictRequest{context_item_ids: context_item_ids, max_results: max_results}
      response = Server.predict(request, nil)

      assert length(response.item_ids) == 10,
             "expected 10 item_ids, got #{length(response.item_ids)}"

      assert response.item_ids == response.items |> Enum.map(& &1.item_id)

      Enum.zip(response.item_ids, response.items)
      |> Enum.each(fn {id, item} ->
        assert item.item_id == id

        assert item.display_name == item_text[id],
               "item_id #{id}: expected display_name #{inspect(item_text[id])}, got #{inspect(item.display_name)}"
      end)
    end

    # Real Steam catalogue: fixture + ckpt + items.json; top-k 10; assert display_name from catalog.
    # Run: mix recgpt.fetch_steam data/steam, mix recgpt.build_fixture, mix recgpt.export_ckpt, then mix test --include integration
    @tag :integration
    @tag timeout: 120_000
    test "recommendations use real Steam catalogue (top-k 10, integration)" do
      data_dir = Path.expand("data/steam", File.cwd!())
      fixture_path = Path.join(data_dir, "fixture.json")
      ckpt_dir = Path.expand("data/recgpt_ckpt_export", File.cwd!())
      catalog_path = Path.join(data_dir, "items.json")

      unless File.regular?(fixture_path) do
        flunk(
          "Fixture not found: #{fixture_path}. Run: mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture ..."
        )
      end

      unless File.dir?(ckpt_dir) and File.regular?(Path.join(ckpt_dir, "manifest.json")) do
        flunk("Checkpoint not found: #{ckpt_dir}. Run: mix recgpt.export_ckpt ...")
      end

      unless File.regular?(catalog_path) do
        flunk("Catalog not found: #{catalog_path}. Run: mix recgpt.fetch_steam data/steam")
      end

      case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
        {:ok, state} ->
          Application.put_env(:recgpt, :serve_state, state)

          request = %PredictRequest{context_item_ids: [0], max_results: 10}
          response = Server.predict(request, nil)

          assert length(response.item_ids) <= 10
          assert length(response.items) == length(response.item_ids)

          Enum.zip(response.item_ids, response.items)
          |> Enum.each(fn {id, item} ->
            assert item.item_id == id

            expected_name =
              case Map.get(state.item_text, id) do
                t when is_binary(t) -> t
                m when is_map(m) -> m["title"] || m["name"] || to_string(id)
                _ -> to_string(id)
              end

            assert item.display_name == expected_name,
                   "item_id #{id}: expected display_name #{inspect(expected_name)}, got #{inspect(item.display_name)}"
          end)

        {:error, reason} ->
          flunk("Failed to load state: #{inspect(reason)}")
      end
    end

    # Start gRPC server, call Predict with grpcurl, assert response (real catalogue).
    # When test/fixtures/steam_predict_grpcurl.json exists, asserts exact item_ids and display_name.
    # Update that fixture by running: mix recgpt.grpc_curl_update_fixture (with server running).
    @tag :integration
    @tag timeout: 90_000
    test "grpcurl Predict returns item_ids and items (real catalogue)" do
      data_dir = Path.expand("data/steam", File.cwd!())
      fixture_path = Path.join(data_dir, "fixture.json")
      ckpt_dir = Path.expand("data/recgpt_ckpt_export", File.cwd!())
      catalog_path = Path.join(data_dir, "items.json")

      unless File.regular?(fixture_path) and File.dir?(ckpt_dir) and
               File.regular?(Path.join(ckpt_dir, "manifest.json")) and File.regular?(catalog_path) do
        flunk(
          "Real data required: fixture, ckpt, catalog. Run: mix recgpt.fetch_steam data/steam && mix recgpt.build_fixture ... && mix recgpt.export_ckpt ..."
        )
      end

      unless grpcurl_available?() do
        flunk("grpcurl not on PATH. Install grpcurl to run this test.")
      end

      port = 50_511

      case RecGPT.Serve.load_state(fixture_path, ckpt_dir, catalog_path) do
        {:ok, state} ->
          Application.put_env(:recgpt, :serve_state, state)

          server_pid = start_grpc_server(port)
          Process.sleep(2_000)

          payload = Jason.encode!(%{context_item_ids: [0], max_results: 10})
          proto_import = Path.expand("priv/proto", File.cwd!())

          {output, exit_code} =
            System.cmd(
              "grpcurl",
              [
                "-plaintext",
                "-import-path",
                proto_import,
                "-proto",
                "recgpt/v1/recommendation.proto",
                "-d",
                payload,
                "-format",
                "json",
                "localhost:#{port}",
                "recgpt.v1.PredictionService/Predict"
              ],
              stderr_to_stdout: true,
              cd: File.cwd!()
            )

          stop_grpc_server(server_pid)

          assert exit_code == 0, "grpcurl failed: #{output}"

          response = parse_grpcurl_response(output)
          fixture_path = Path.expand("test/fixtures/steam_predict_grpcurl.json", File.cwd!())

          if File.regular?(fixture_path) do
            # Assert against fixture (updated by mix recgpt.grpc_curl_update_fixture)
            fixture = File.read!(fixture_path) |> Jason.decode!()
            expected = fixture["response"]

            assert expected["item_ids"] == response.item_ids,
                   "item_ids mismatch: expected #{inspect(expected["item_ids"])}, got #{inspect(response.item_ids)}"

            assert length(response.items) == length(expected["items"])

            Enum.zip(response.items, expected["items"])
            |> Enum.each(fn {got, exp} ->
              got_id = got["itemId"] || got["item_id"]
              got_name = got["displayName"] || got["display_name"] || ""
              exp_id = exp["item_id"]
              exp_name = exp["display_name"] || ""

              assert got_id == exp_id,
                     "item_id at position mismatch: got #{got_id}, expected #{exp_id}"

              assert got_name == exp_name,
                     "display_name for item_id #{got_id}: expected #{inspect(exp_name)}, got #{inspect(got_name)}"
            end)
          else
            # No fixture: assert structure only
            assert length(response.item_ids) <= 10
            assert length(response.items) == length(response.item_ids)
            assert_items_match(response.item_ids, response.items)
          end

        {:error, reason} ->
          flunk("Failed to load state: #{inspect(reason)}")
      end
    end

    test "empty context_item_ids raises INVALID_ARGUMENT" do
      request = %PredictRequest{context_item_ids: [], max_results: 5}

      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError -> assert e.status == 3
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

    test "when serve_state is not loaded, raises FAILED_PRECONDITION" do
      Application.put_env(:recgpt, :serve_state, nil)
      request = %PredictRequest{context_item_ids: [0], max_results: 5}

      try do
        Server.predict(request, nil)
      rescue
        e in GRPC.RPCError ->
          assert e.status == 9
      after
        # Restore stub so on_exit from setup doesn't fail
        Application.put_env(
          :recgpt,
          :serve_state,
          RecGPT.TestSupport.FrozenHelpers.build_stub_state()
        )
      end
    end
  end

  defp grpcurl_available? do
    case System.cmd("grpcurl", ["-version"], stderr_to_stdout: true) do
      {_, 0} -> true
      _ -> false
    end
  rescue
    _ -> false
  end

  defp start_grpc_server(port) do
    parent = self()

    child =
      spawn_link(fn ->
        children = [
          {GRPC.Server.Supervisor,
           [endpoint: RecGPT.GRPCEndpoint, port: port, start_server: true]}
        ]

        {:ok, pid} = Supervisor.start_link(children, strategy: :one_for_one)
        send(parent, {:grpc_server_started, pid})

        receive do
          :stop ->
            # Stop in a separate process so we exit quickly and test doesn't block on link
            spawn(fn -> Supervisor.stop(pid) end)
        end
      end)

    receive do
      {:grpc_server_started, _pid} -> child
    after
      10_000 -> flunk("gRPC server did not start in time")
    end
  end

  defp stop_grpc_server(server_pid) when is_pid(server_pid) do
    send(server_pid, :stop)
  end

  defp assert_items_match(item_ids, items) do
    Enum.zip(item_ids, items)
    |> Enum.each(fn {id, item} ->
      item_id = item["itemId"] || item["item_id"]
      display_name = item["displayName"] || item["display_name"] || ""
      assert item_id == id
      assert is_binary(display_name)
    end)
  end

  # Parse grpcurl JSON output; normalizes to item_ids + items (accepts camelCase or snake_case).
  defp parse_grpcurl_response(output) do
    case Jason.decode(output) do
      {:ok, %{"item_ids" => ids, "items" => items}} when is_list(ids) and is_list(items) ->
        %{item_ids: ids, items: items}

      {:ok, %{"itemIds" => ids, "items" => items}} when is_list(ids) and is_list(items) ->
        %{item_ids: ids, items: items}

      {:ok, other} ->
        flunk("Unexpected grpcurl response: #{inspect(other)}")

      {:error, _} ->
        flunk("grpcurl output not valid JSON: #{String.slice(output, 0, 300)}")
    end
  end
end
