#!/usr/bin/env python3
"""
Generate a minimal PyTorch .pt fixture (zip format) for RecGPT.PtLoader tests.

Creates test/fixtures/sample.pt with a small state_dict. Run once (requires torch):

    python scripts/generate_pt_fixture.py

Or from repo root:
    python -c "import torch; torch.save({'wte': torch.randn(4, 8), 'pred_head.weight': torch.randn(8, 4), 'pred_head.bias': torch.zeros(4)}, 'test/fixtures/sample.pt')"
"""
import os
import sys

try:
    import torch
except ImportError:
    print("PyTorch required: pip install torch", file=sys.stderr)
    sys.exit(1)

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "test", "fixtures")
    out_path = os.path.join(out_dir, "sample.pt")

    os.makedirs(out_dir, exist_ok=True)

    # Minimal state_dict: small tensors so fixture stays tiny
    state = {
        "wte": torch.randn(4, 8),
        "pred_head.weight": torch.randn(8, 4),
        "pred_head.bias": torch.zeros(4),
    }

    torch.save(state, out_path)
    print(out_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
