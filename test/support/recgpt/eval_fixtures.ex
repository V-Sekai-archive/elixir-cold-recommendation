defmodule RecGPT.EvalFixtures do
  @moduledoc """
  Synthetic generators for eval data shapes (see docs/06_eval_data_shapes.md).

  Use in tests to generate test_cases and test_sequences JSON without real datasets.
  All item IDs are in 0 .. num_items-1.
  """

  @doc """
  Generates a list of test case maps compatible with RecGPT.Eval.load_test_cases/1
  and RecGPT.Eval.evaluate/3.

  Options:
  - `:min_context` - minimum context length (default 1)
  - `:max_context` - maximum context length (default 64)
  """
  def generate_test_cases(num_items, n_cases, opts \\ [])

  def generate_test_cases(_num_items, 0, _opts), do: []

  def generate_test_cases(num_items, n_cases, opts) when num_items >= 1 and n_cases >= 1 do
    min_ctx = Keyword.get(opts, :min_context, 1)
    max_ctx = Keyword.get(opts, :max_context, 64)
    range = 0..(num_items - 1)

    for _ <- 1..n_cases do
      len = min(max_ctx, max(min_ctx, :rand.uniform(max(max_ctx - min_ctx + 1, 1))))
      context = for _ <- 1..len, do: Enum.random(range)
      next_item = Enum.random(range)
      %{"context" => context, "next_item" => next_item}
    end
  end

  @doc """
  Returns a map that can be Jason.encode!/1'd and written to a file; that file
  can be passed to RecGPT.Eval.load_test_cases/1.

  Same options as generate_test_cases/3.
  """
  def generate_test_sequences_json(num_items, n_cases, opts \\ []) do
    cases = generate_test_cases(num_items, n_cases, opts)
    %{"num_items" => num_items, "test_cases" => cases}
  end

  @doc """
  Generates an items list compatible with items.json shape: list of %{"id" => i, "title" => "item N"}.
  """
  def generate_items(num_items) when num_items >= 0 do
    for i <- 0..(num_items - 1)//1 do
      %{"id" => i, "title" => "item #{i}"}
    end
  end

  @doc """
  Returns a map compatible with items.json: %{"num_items" => n, "items" => [...]}.
  """
  def generate_items_json(num_items) do
    items = generate_items(num_items)
    %{"num_items" => num_items, "items" => items}
  end
end
