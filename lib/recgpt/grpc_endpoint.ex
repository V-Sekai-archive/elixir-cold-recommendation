defmodule RecGPT.GRPCEndpoint do
  @moduledoc """
  gRPC endpoint for RecGPT. Runs PredictionService (recommendations) and StaffService (catalogues, pretrain).
  """
  use GRPC.Endpoint

  run(Recgpt.V1.PredictionService.Server)
  run(Recgpt.V1.StaffService.Server)
end
