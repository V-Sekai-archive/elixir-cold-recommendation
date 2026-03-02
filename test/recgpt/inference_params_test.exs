# RecGPT.InferenceParams: full defn params (stub and full).
defmodule RecGPT.InferenceParamsTest do
  use ExUnit.Case, async: true

  alias RecGPT.InferenceParams

  defp stub_params_map do
    wte = Nx.iota({15_361, 768}) |> Nx.divide(15_361 * 768) |> Nx.as_type({:f, 32})
    head_w = Nx.iota({768, 15_361}) |> Nx.divide(768 * 15_361) |> Nx.as_type({:f, 32})
    head_b = Nx.broadcast(0.0, {15_361}) |> Nx.as_type({:f, 32})
    %{
      "wte" => wte,
      "pred_head.weight" => head_w,
      "pred_head.bias" => head_b
    }
  end

  test "build_defn_params with n_layers 0 returns full structure with identity layers" do
    params_map = stub_params_map()
    full = InferenceParams.build_defn_params(params_map, 0)
    assert map_size(full) > 12
    assert Map.has_key?(full, :wte)
    assert Map.has_key?(full, :wpe)
    assert Map.has_key?(full, :ln_f_weight)
    assert Map.has_key?(full, :ln_f_bias)
    assert Map.has_key?(full, :pred_head_weight)
    assert Map.has_key?(full, :pred_head_bias)
    assert Map.has_key?(full, :layer_0_ln_1_weight)
    assert Map.has_key?(full, :layer_11_mlp_c_proj_bias)
    assert Nx.shape(full[:wte]) == {15_361, 768}
    assert Nx.shape(full[:pred_head_weight]) == {768, 15_361}
    assert Nx.shape(full[:layer_0_ln_1_weight]) == {768}
    assert Nx.shape(full[:layer_0_attn_c_attn_weight]) == {2304, 768}
  end

  test "build_defn_params with n_layers 0 has identity LayerNorm (ones/zeros) for layers" do
    params_map = stub_params_map()
    full = InferenceParams.build_defn_params(params_map, 0)
    ln1_w = full[:layer_0_ln_1_weight]
    ln1_b = full[:layer_0_ln_1_bias]
    assert Nx.all_close(ln1_w, Nx.broadcast(1.0, {768})) |> Nx.to_number() == 1
    assert Nx.all_close(ln1_b, Nx.broadcast(0.0, {768})) |> Nx.to_number() == 1
  end
end
