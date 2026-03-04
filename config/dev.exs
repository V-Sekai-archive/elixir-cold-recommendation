# Dev: Torchx uses device :cuda when available. Set config :nx, default_backend to override.
import Config

# BF16 (Tensor Cores) for faster inference.
config :recgpt, :inference_dtype, {:bf, 16}
