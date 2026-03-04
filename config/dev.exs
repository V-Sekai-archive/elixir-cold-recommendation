# Dev: EXLA uses config :exla, :default_client (:cuda or :host). GPU is used when client is :cuda.
import Config

# BF16 (Tensor Cores) for faster inference.
config :recgpt, :inference_dtype, {:bf, 16}
