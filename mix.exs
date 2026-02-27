defmodule RecGPT.MixProject do
  @moduledoc "RecGPT Elixir library: FSQ, embeddings (MPNet/Bumblebee), training data pipeline. No GenServer; use from any app."
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :recgpt,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "RecGPT library: FSQ, text embeddings (MPNet), training batches and loss. Depends on Bumblebee.",
      test_coverage: [summary: [threshold: 85]]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.6", override: true},
      {:axon, "~> 0.7"},
      {:bumblebee, github: "elixir-nx/bumblebee", ref: "main"},
      {:jason, "~> 1.4"},
      {:npy, "~> 0.1.2"},
      {:torchx, "~> 0.11"},
      {:plug_cowboy, "~> 2.6"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:propcheck, "~> 1.5", only: [:dev, :test]}
    ]
  end
end
