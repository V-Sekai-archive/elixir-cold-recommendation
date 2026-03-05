#!/usr/bin/env elixir

Application.ensure_all_started(:recgpt)

alias RecGPT.Repo
import Ecto.Query

# Update canonical texts for Figgie
texts = [
  {0, "Spades suit in Figgie card game"},
  {1, "Clubs suit in Figgie card game"},
  {2, "Hearts suit in Figgie card game"},
  {3, "Diamonds suit in Figgie card game"}
]

Enum.each(texts, fn {id, text} ->
  Repo.update_all(
    from(c in RecGPT.Catalog.CanonicalItemText, where: c.item_id == ^id),
    set: [text: text]
  )
end)

IO.puts("Updated canonical texts for Figgie items")