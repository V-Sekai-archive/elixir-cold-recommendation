defmodule RecGPT.GRPCEndpoint do
  @moduledoc """
  gRPC endpoint for RecGPT. Runs PredictionService (Predict RPC).
  """
  use GRPC.Endpoint

  run(Recgpt.V1.PredictionService.Server)
end
