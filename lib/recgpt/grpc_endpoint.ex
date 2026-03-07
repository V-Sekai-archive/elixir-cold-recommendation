defmodule RecGPT.GRPCEndpoint do
  @moduledoc """
  gRPC endpoint for RecGPT. Runs PredictionService (Predict RPC).
  Used by GRPC.Server.Supervisor when starting the gRPC server (mix recgpt.serve, ReleaseTasks.serve).
  """
  use GRPC.Endpoint

  run Recgpt.V1.PredictionService.Server
end
