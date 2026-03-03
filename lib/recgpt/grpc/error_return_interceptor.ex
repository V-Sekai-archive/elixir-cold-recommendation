defmodule RecGPT.GRPC.ErrorReturnInterceptor do
  @moduledoc """
  Converts handler return `{:error, %GRPC.RPCError{}}` into the return type the
  adapter expects, so the adapter can send the error to the client without raising.
  """
  @behaviour GRPC.Server.Interceptor

  @impl true
  def init(_opts), do: []

  @impl true
  def call(request, stream, next, _opts) do
    result = next.(request, stream)

    case result do
      {:ok, _stream, {:error, %GRPC.RPCError{} = e}} ->
        {:error, e}

      other ->
        other
    end
  end
end
