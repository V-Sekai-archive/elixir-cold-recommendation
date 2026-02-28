defmodule Mix.Tasks.Recgpt.GeneratePtFixture do
  @shortdoc "Generate a minimal PyTorch .pt fixture for PtLoader tests"
  @moduledoc """
  Writes a minimal zip-format .pt file to `test/fixtures/sample.pt` for use by
  `RecGPT.PtLoader` tests (tag `:pt_fixture`). No Python or PyTorch required.

  ## Usage

      mix recgpt.generate_pt_fixture
      mix recgpt.generate_pt_fixture path/to/sample.pt

  Then run: `mix test --include pt_fixture test/recgpt/pt_loader_test.exs`
  """
  use Mix.Task

  @impl true
  def run(args) do
    path = List.first(args) || Path.join(["test", "fixtures", "sample.pt"])
    RecGPT.PtFixtureGenerator.generate_to_path(path)
    Mix.shell().info("Wrote #{path}. Run: mix test --include pt_fixture test/recgpt/pt_loader_test.exs")
  end
end
