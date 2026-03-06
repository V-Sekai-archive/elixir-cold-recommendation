defmodule RecGPT.VisionProjectorTest do
  use ExUnit.Case, async: true

  alias RecGPT.VisionProjector

  describe "init_params/0" do
    test "returns map with expected string keys" do
      params = VisionProjector.init_params()
      assert Map.has_key?(params, "vision_proj.fc1.weight")
      assert Map.has_key?(params, "vision_proj.fc1.bias")
      assert Map.has_key?(params, "vision_proj.fc2.weight")
      assert Map.has_key?(params, "vision_proj.fc2.bias")
      assert Nx.shape(params["vision_proj.fc1.weight"]) == {768, 768}
      assert Nx.shape(params["vision_proj.fc2.weight"]) == {768, 768}
      assert Nx.shape(params["vision_proj.fc1.bias"]) == {768}
      assert Nx.shape(params["vision_proj.fc2.bias"]) == {768}
    end
  end

  describe "forward/2" do
    test "output shape is {batch, 768} and L2-normalized" do
      params = VisionProjector.init_params()
      batch = 4
      x = Nx.iota({batch, 768}, type: {:f, 32}) |> Nx.divide(768)
      out = VisionProjector.forward(params, x)
      assert Nx.shape(out) == {batch, 768}
      norms = Nx.LinAlg.norm(out, axes: [1])
      assert Nx.all_close(norms, Nx.broadcast(1.0, {batch}), atol: 1.0e-5) |> Nx.to_number() == 1
    end
  end
end
