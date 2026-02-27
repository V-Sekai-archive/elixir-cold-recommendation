# RecGPT.EvalFixtures: generators for synthetic eval data shapes.
defmodule RecGPT.EvalFixturesTest do
  use ExUnit.Case, async: true

  alias RecGPT.EvalFixtures

  describe "generate_test_cases/3" do
    test "returns empty list when n_cases is 0" do
      assert EvalFixtures.generate_test_cases(10, 0) == []
    end

    test "returns n_cases maps with context and next_item" do
      cases = EvalFixtures.generate_test_cases(5, 20)
      assert length(cases) == 20

      for tc <- cases do
        assert is_list(tc["context"])
        assert is_integer(tc["next_item"])
        assert tc["next_item"] >= 0 and tc["next_item"] < 5
      end
    end

    test "respects min_context and max_context opts" do
      cases = EvalFixtures.generate_test_cases(10, 50, min_context: 2, max_context: 5)
      assert length(cases) == 50

      for tc <- cases do
        len = length(tc["context"])
        assert len >= 2 and len <= 5
      end
    end
  end

  describe "generate_test_sequences_json/3" do
    test "returns map with num_items and test_cases" do
      json = EvalFixtures.generate_test_sequences_json(8, 5)
      assert json["num_items"] == 8
      assert length(json["test_cases"]) == 5
    end
  end

  describe "generate_items/1" do
    test "returns empty list for num_items 0" do
      assert EvalFixtures.generate_items(0) == []
    end

    test "returns num_items maps with id and title" do
      items = EvalFixtures.generate_items(3)
      assert length(items) == 3
      assert Enum.at(items, 0) == %{"id" => 0, "title" => "item 0"}
      assert Enum.at(items, 2) == %{"id" => 2, "title" => "item 2"}
    end
  end

  describe "generate_items_json/1" do
    test "returns map with num_items and items list" do
      json = EvalFixtures.generate_items_json(4)
      assert json["num_items"] == 4
      assert length(json["items"]) == 4
    end
  end
end
