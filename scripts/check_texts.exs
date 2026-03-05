import Ecto.Query
texts = RecGPT.Repo.all(from c in RecGPT.Catalog.CanonicalItemText, limit: 5, select: c.text)
Enum.each(texts, &IO.puts/1)
