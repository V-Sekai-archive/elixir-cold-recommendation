defmodule RecGPT.InferenceParams do
  @moduledoc """
  Builds defn-friendly param maps. Delegates to FuXiLinearInferenceParams.

  Ignores n_layers (FuXi-Linear infers from checkpoint keys).
  """

  @doc """
  Build full params for Defn from checkpoint string-key map.

  - `params_map`: from `RecGPT.CheckpointLoader.load_from_export/1`
  - `n_layers`: ignored (FuXi-Linear infers from keys)
  - `dtype`: optional, default `{:f, 32}`. Use `{:bf, 16}` for BF16.

  Returns atom-keyed map for forward_last_4_logits/4.
  """
  @spec build_defn_params(map(), term()) :: map()
  def build_defn_params(params_map, n_layers_or_dtype \\ {:f, 32})

  def build_defn_params(params_map, dtype) when is_tuple(dtype) do
    RecGPT.FuxiLinearInferenceParams.build_defn_params(params_map, dtype)
  end

  def build_defn_params(params_map, _n_layers) do
    RecGPT.FuxiLinearInferenceParams.build_defn_params(params_map, {:f, 32})
  end
end
