defmodule RecGPT.VisionContrastiveTest do
  use ExUnit.Case, async: true

  alias RecGPT.VisionContrastive
  alias RecGPT.VisionProjector

  describe "loss/4" do
    test "returns scalar loss" do
      params = VisionProjector.init_params()
      batch = 4
      v = Nx.iota({batch, 768}, type: {:f, 32}) |> Nx.divide(768)
      t = Nx.iota({batch, 768}, type: {:f, 32}) |> Nx.divide(768)
      t = Nx.LinAlg.norm(t, axes: [-1], keep_axes: true) |> then(&Nx.divide(t, Nx.max(&1, 1.0e-8)))
      loss = VisionContrastive.loss(params, v, t)
      assert Nx.shape(loss) == {}
      loss_num = Nx.to_number(loss)
      assert is_number(loss_num) and loss_num > 0 and loss_num < 100
    end
  end

  describe "run/2" do
    test "runs a few steps and returns updated params" do
      params = VisionProjector.init_params()
      trained = VisionContrastive.run(params, steps: 3, batch_size: 4, log_every: 1)
      assert is_map(trained)
      assert Map.has_key?(trained, "vision_proj.fc1.weight")
    end
  end
end
