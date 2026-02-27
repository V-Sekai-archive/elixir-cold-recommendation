defmodule RecGPT.CompareTest do
  @moduledoc """
  Compare Python vs Elixir RecGPT FSQ encode.

  Generate fixtures from repo root:
    uv run python scripts/compare_recgpt_fsq.py --output-dir data/recgpt_compare

  Run: mix test test/recgpt/compare_test.exs
  """
  use ExUnit.Case, async: false

  alias RecGPT.FSQEncoder

  defp fixture_dir do
    cwd = File.cwd!()
    from_recgpt = Path.expand("../data/recgpt_compare", cwd)
    from_repo = Path.join(cwd, "data/recgpt_compare")

    from_env = System.get_env("RECGPT_COMPARE_FIXTURES")
    if from_env != nil and from_env != "" and File.exists?(Path.expand(from_env)) do
      Path.expand(from_env)
    else
      cond do
        File.exists?(from_recgpt) -> from_recgpt
        File.exists?(from_repo) -> from_repo
        true -> Path.join(cwd, "data/recgpt_compare")
      end
    end
  end

  defp load_fixture(name) do
    path = Path.join(fixture_dir(), name)

    if File.regular?(path) do
      path |> File.read!() |> Jason.decode!()
    else
      nil
    end
  end

  defp json_to_params(json) do
    pi = json["project_in"]
    po = json["project_out"]

    %{
      "project_in" => %{
        "kernel" => tensor(pi["kernel"]),
        "bias" => if(pi["bias"] == [] || is_nil(pi["bias"]), do: nil, else: tensor(pi["bias"]))
      },
      "project_out" => %{
        "kernel" => tensor(po["kernel"]),
        "bias" => if(po["bias"] == [] || is_nil(po["bias"]), do: nil, else: tensor(po["bias"]))
      }
    }
  end

  defp tensor(list) when is_list(list) do
    Nx.tensor(list, type: {:f, 32})
  end

  describe "FSQ encode vs Python (fixtures from compare_recgpt_fsq.py)" do
    @describetag :compare_python
    test "Elixir token_id_list matches Python expected_tokens when fixtures exist" do
      embeddings_json = load_fixture("embeddings.json")
      params_json = load_fixture("params.json")
      expected_tokens_json = load_fixture("expected_tokens.json")

      if is_nil(embeddings_json) or is_nil(params_json) or is_nil(expected_tokens_json) do
        raise """
        Fixtures missing. From repo root run:
          uv run python scripts/compare_recgpt_fsq.py --output-dir data/recgpt_compare
        Then run this test again.
        """
      end

      embeddings = Nx.tensor(embeddings_json, type: {:f, 32})
      params = json_to_params(params_json)
      expected_tokens = Enum.map(expected_tokens_json, &List.wrap/1)

      token_id_list = FSQEncoder.encode_embeddings_to_token_id_list(embeddings, params, 64)

      assert length(token_id_list) == length(expected_tokens),
             "length mismatch: got #{length(token_id_list)}, expected #{length(expected_tokens)}"

      Enum.zip(token_id_list, expected_tokens)
      |> Enum.with_index()
      |> Enum.each(fn {{got, expected}, idx} ->
        assert got == expected,
               "item #{idx}: Elixir got #{inspect(got)}, Python expected #{inspect(expected)}"
      end)
    end
  end
end
