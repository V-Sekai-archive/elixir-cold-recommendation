#!/usr/bin/env python3
"""
Python-side sanity check: encode the same string as Elixir (item 0) with
sentence-transformers and optionally compare with our dumped row.

Usage:
  pip install sentence-transformers numpy

  # 1. Dump our row 0 from Elixir:
  mix recgpt.compare_embeddings --limit 1 --dump-row 0

  # 2. Encode with sentence-transformers and save:
  python scripts/embed_one.py --save item0_sentence_transformers.npy

  # 3. Compare with our dump (if item0_elixir.raw exists):
  python scripts/embed_one.py --compare item0_elixir.raw
"""
import argparse
import numpy as np


def encode_and_save(path: str, normalize: bool = True) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    text = "'title': 'Papers, Please'"  # same as our first item
    vec = model.encode(text, normalize_embeddings=normalize)
    arr = vec.astype(np.float32)
    np.save(path, arr)
    print(f"Shape: {arr.shape}  Norm: {np.linalg.norm(arr):.4f}")
    print(f"Saved to {path}")


def compare_with_elixir(raw_path: str) -> None:
    st_vec = encode_in_memory()
    elixir_vec = np.fromfile(raw_path, dtype=np.float32).reshape(1, 768)
    cos = (elixir_vec @ st_vec.T).item() / (
        np.linalg.norm(elixir_vec) * np.linalg.norm(st_vec) + 1e-12
    )
    print(f"Cosine(Elixir vs sentence-transformers): {cos:.4f}")
    print(f"Norms - Elixir: {np.linalg.norm(elixir_vec):.4f}  ST: {np.linalg.norm(st_vec):.4f}")


def encode_in_memory(normalize: bool = True) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    text = "'title': 'Papers, Please'"
    vec = model.encode(text, normalize_embeddings=normalize)
    return vec.astype(np.float32).reshape(1, 768)


def main():
    ap = argparse.ArgumentParser(description="Encode item 0 string with sentence-transformers; optional compare with Elixir dump.")
    ap.add_argument("--save", metavar="PATH", help="Save ST embedding to .npy file")
    ap.add_argument("--compare", metavar="RAW_PATH", help="Compare with Elixir raw dump (e.g. item0_elixir.raw)")
    ap.add_argument("--no-normalize", action="store_true", help="Do not L2-normalize (match normalize_embeddings=False)")
    args = ap.parse_args()

    normalize = not args.no_normalize
    if args.save:
        encode_and_save(args.save, normalize=normalize)
    elif args.compare:
        compare_with_elixir(args.compare)
    else:
        encode_and_save("item0_sentence_transformers.npy", normalize=normalize)
        print("Run with --compare item0_elixir.raw after dumping from Elixir.")


if __name__ == "__main__":
    main()
