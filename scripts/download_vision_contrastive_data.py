#!/usr/bin/env python3
"""
Download an image-caption dataset and compute DINOv2 (vision) and MPNet (text) embeddings.
Saves vision_768.npy and text_768.npy for Elixir RecGPT vision contrastive training.

Usage:
  uv run python scripts/download_vision_contrastive_data.py [--dataset flickr30k|anigame] [--limit N] [--out DIR]
  Default: lmms-lab/flickr30k (Parquet), limit 5000, out data/vision_contrastive
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from torch import no_grad
from transformers import AutoImageProcessor, AutoModel


def get_caption(example, dataset_name: str, label_names: list | None = None) -> str:
    if dataset_name == "food101" and "label" in example:
        idx = int(example["label"])
        if label_names and 0 <= idx < len(label_names):
            return label_names[idx]
        return f"food class {idx}"
    if dataset_name == "flickr30k":
        # lmms-lab/flickr30k or similar: "caption" / "captions" / "caption_list"
        cap = example.get("caption") or example.get("captions") or example.get("caption_list") or example.get("text")
        if isinstance(cap, list):
            return cap[0] if cap else ""
        return str(cap) if cap else ""
    if dataset_name == "anigame":
        # mrzjy/AniGamePersonaCaps: title, description, caption.appearance, caption.personality
        parts = []
        if example.get("title"):
            parts.append(str(example["title"]))
        if example.get("description"):
            parts.append(str(example["description"]))
        app = example.get("caption", {}).get("appearance") if isinstance(example.get("caption"), dict) else None
        if app:
            parts.append(str(app))
        return " ".join(parts) if parts else "unknown"
    return str(example.get("caption", example.get("text", "")))


def main() -> None:
    p = argparse.ArgumentParser(description="Download dataset and compute vision/text embeddings")
    p.add_argument("--dataset", choices=["flickr30k", "anigame", "food101"], default="food101")
    p.add_argument("--limit", type=int, default=5000, help="Max samples (0 = all)")
    p.add_argument("--out", default="data/vision_contrastive", help="Output directory")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    if args.dataset == "flickr30k":
        dataset_id = "lmms-lab/flickr30k"
    elif args.dataset == "anigame":
        dataset_id = "mrzjy/AniGamePersonaCaps"
    else:
        dataset_id = "ethz/food101"  # small, no script; image + label
    split = "train"
    print(f"Loading dataset {dataset_id} ({split})...")
    kwargs = {}
    if args.dataset == "anigame":
        kwargs["trust_remote_code"] = True
    ds = load_dataset(dataset_id, split=split, **kwargs)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"Selected {n} samples.")

    # Image and caption column names
    image_col = "image" if "image" in ds.column_names else "img"
    if image_col not in ds.column_names:
        print("No image column found; columns:", ds.column_names, file=sys.stderr)
        sys.exit(1)
    # For food101 we use label -> string as caption
    label_names = None
    if args.dataset == "food101" and "label" in ds.column_names:
        try:
            label_names = ds.features["label"].names
        except Exception:
            label_names = [f"food class {i}" for i in range(101)]

    # DINOv2
    print("Loading DINOv2 (facebook/dinov2-base)...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.eval()

    # MPNet (match Elixir RecGPT.Embedding: sentence-transformers/all-mpnet-base-v2)
    print("Loading MPNet (sentence-transformers/all-mpnet-base-v2)...")
    text_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    vision_embs = []
    text_embs = []
    batch_size = args.batch_size

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = ds[start:end]

        # Vision: load images and run DINOv2
        images = batch[image_col]
        if not images:
            continue
        # Handle single image vs list
        if not isinstance(images, list):
            images = [images]
        # Convert to RGB PIL if needed
        from PIL import Image
        pil_images = []
        for im in images:
            if im is None:
                # placeholder: small gray image
                pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
            elif hasattr(im, "convert") and callable(im.convert):
                pil_images.append(im.convert("RGB") if im.mode != "RGB" else im)
            else:
                pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))
        inputs = processor(images=pil_images, return_tensors="pt")
        with no_grad():
            out = model(**inputs)
        # [CLS] token (index 0), shape (batch, 768)
        cls_emb = out.last_hidden_state[:, 0, :].float().numpy()
        vision_embs.append(cls_emb)

        # Text: one caption per sample
        captions = [
            get_caption({k: batch[k][i] for k in batch}, args.dataset, label_names)
            for i in range(len(pil_images))
        ]
        # sentence-transformers returns L2-normalized by default
        text_batch = text_encoder.encode(captions, convert_to_numpy=True, normalize_embeddings=True)
        text_embs.append(text_batch.astype(np.float32))

        if (start // batch_size + 1) % 50 == 0 or end == n:
            print(f"  {end}/{n}")

    vision_768 = np.concatenate(vision_embs, axis=0)
    text_768 = np.concatenate(text_embs, axis=0)
    assert vision_768.shape[0] == text_768.shape[0] == n, (vision_768.shape, text_768.shape, n)
    assert vision_768.shape[1] == text_768.shape[1] == 768

    os.makedirs(args.out, exist_ok=True)
    vision_path = os.path.join(args.out, "vision_768.npy")
    text_path = os.path.join(args.out, "text_768.npy")
    np.save(vision_path, vision_768)
    np.save(text_path, text_768)
    print(f"Saved {vision_768.shape} to {vision_path}")
    print(f"Saved {text_768.shape} to {text_path}")


if __name__ == "__main__":
    main()
