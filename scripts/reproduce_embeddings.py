#!/usr/bin/env python3
"""
Reproduce embeddings in Python to verify pipelines independently.

1. RecGPT official: SentenceTransformer.encode (same as dataset .npy pipeline).
   Compare Python ST output to dataset item_text_embeddings.npy.
2. Ours (Bumblebee): HF AutoModel + tokenizer, mean pooling with attention mask, L2 norm.
   Compare Python HF+pool output to Elixir dump (item0_elixir.raw).

Prereq for "ours": dump from Elixir first:
  mix recgpt.compare_embeddings --steam-dir data/steam --limit 1 --dump-row 0

Usage:
  uv run python scripts/reproduce_embeddings.py --official   # compare to dataset .npy
  uv run python scripts/reproduce_embeddings.py --ours      # compare to Elixir dump
  uv run python scripts/reproduce_embeddings.py --both      # run both (default)
"""
import argparse
import numpy as np

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TEXT = "'title': 'Papers, Please'"
MAX_LENGTH = 384


def reproduce_official(text: str, npy_path: str, row: int = 0) -> np.ndarray:
    """Encode with sentence-transformers (RecGPT dataset pipeline). Returns embedding."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID)
    vec = model.encode(text, normalize_embeddings=True)
    vec = vec.astype(np.float32).reshape(1, 768)
    print(f"[Official] Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Official] Shape: {vec.shape}  Norm: {np.linalg.norm(vec):.6f}")

    if npy_path:
        try:
            ref = np.load(npy_path)
            if ref.ndim == 3:
                ref = ref.squeeze(1)
            ref_row = ref[row : row + 1].astype(np.float32)
            cos = (vec @ ref_row.T).item() / (
                np.linalg.norm(vec) * np.linalg.norm(ref_row) + 1e-12
            )
            print(f"[Official] cos(Python ST, dataset .npy row {row}): {cos:.6f}")
            if cos >= 0.99:
                print("[Official] OK: reproduced RecGPT official embedding in Python.")
            else:
                print("[Official] Mismatch: Python ST vs dataset .npy differ.")
        except FileNotFoundError:
            print(f"[Official] Dataset .npy not found: {npy_path}")
    return vec


def reproduce_ours(text: str, elixir_dump_path: str | None) -> np.ndarray:
    """Replicate Bumblebee: HF model + mean pool (with attention_mask) + L2 norm."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_token_type_ids=False,
    )
    with torch.no_grad():
        out = model(**inputs)

    hidden = out.last_hidden_state  # (1, seq_len, 768)
    mask = inputs["attention_mask"]  # (1, seq_len)
    mask_expanded = mask.unsqueeze(-1).float()
    sum_masked = (hidden * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    pooled = (sum_masked / sum_mask).squeeze(0).numpy().astype(np.float32)
    norm = np.linalg.norm(pooled) + 1e-12
    pooled = (pooled / norm).astype(np.float32).reshape(1, 768)

    print(f"[Ours] Text: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Ours] Shape: {pooled.shape}  Norm: {np.linalg.norm(pooled):.6f}")

    if elixir_dump_path:
        try:
            elixir = np.fromfile(elixir_dump_path, dtype=np.float32).reshape(1, 768)
            cos = (pooled @ elixir.T).item() / (
                np.linalg.norm(pooled) * np.linalg.norm(elixir) + 1e-12
            )
            print(f"[Ours] cos(Python HF+pool, Elixir dump): {cos:.6f}")
            if cos >= 0.99:
                print("[Ours] OK: reproduced our (Bumblebee) embedding in Python.")
            else:
                print("[Ours] Mismatch: Python HF+pool vs Elixir dump differ.")
        except FileNotFoundError:
            print(f"[Ours] Elixir dump not found at {elixir_dump_path}")
            print("  Run: mix recgpt.compare_embeddings --steam-dir data/steam --limit 1 --dump-row 0")
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce RecGPT official and/or our (Bumblebee) embeddings in Python."
    )
    ap.add_argument(
        "--official",
        action="store_true",
        help="Reproduce RecGPT official (sentence-transformers) and compare to dataset .npy",
    )
    ap.add_argument(
        "--ours",
        action="store_true",
        help="Reproduce our pipeline (HF + mean pool + L2) and compare to Elixir dump",
    )
    ap.add_argument(
        "--both",
        action="store_true",
        help="Run both checks (default if no --official/--ours)",
    )
    ap.add_argument(
        "--npy",
        default="data/steam/item_text_embeddings.npy",
        metavar="PATH",
        help="Path to dataset .npy for --official (default: data/steam/item_text_embeddings.npy)",
    )
    ap.add_argument(
        "--elixir-dump",
        default="item0_elixir.raw",
        metavar="PATH",
        help="Path to Elixir dump for --ours (default: item0_elixir.raw)",
    )
    ap.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        metavar="STR",
        help=f"Input text to encode (default: {DEFAULT_TEXT!r})",
    )
    ap.add_argument(
        "--row",
        type=int,
        default=0,
        help="Row index in .npy to compare for --official (default: 0)",
    )
    args = ap.parse_args()

    run_official = args.official or args.both or (not args.official and not args.ours)
    run_ours = args.ours or args.both or (not args.official and not args.ours)

    if run_official:
        reproduce_official(args.text, args.npy, args.row)
        if run_ours:
            print()

    if run_ours:
        reproduce_ours(args.text, args.elixir_dump)


if __name__ == "__main__":
    main()
