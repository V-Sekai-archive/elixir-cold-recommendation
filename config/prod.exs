# Production config. Override as needed (e.g. RECGPT_CKPT_SHA256, RECGPT_FIXTURE).
import Config

# BF16 inference (default): 1.3–2× faster on Tensor Cores. Use {:f, 32} for FP32 if needed.
config :recgpt, :inference_dtype, {:bf, 16}
