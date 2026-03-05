Application.ensure_all_started(:recgpt)
Application.ensure_all_started(:bumblebee)

# Load FSQ params
ckpt_dir = "priv/fuxi_checkpoint"
fsq_params = RecGPT.FSQ.load_params_from_vae_pt("thirdparty/checkpoints/vae/vae_len4_fsq88865_ep90.pt")

# Item texts
item_texts = %{
  0 => RecGPT.Embedding.recgpt_item_text(%{title: "Spades suit in Figgie card game"}),
  1 => RecGPT.Embedding.recgpt_item_text(%{title: "Clubs suit in Figgie card game"}),
  2 => RecGPT.Embedding.recgpt_item_text(%{title: "Hearts suit in Figgie card game"}),
  3 => RecGPT.Embedding.recgpt_item_text(%{title: "Diamonds suit in Figgie card game"})
}

# Encode embeddings
embeddings = RecGPT.Embedding.encode_item_text_dict(item_texts)

# Encode to tokens
tokens = RecGPT.FSQEncoder.encode_embeddings_to_token_id_list(embeddings, fsq_params)

# Print
Enum.each(0..3, fn i ->
  IO.puts("Item #{i}: #{inspect(Enum.at(tokens, i))}")
end)
