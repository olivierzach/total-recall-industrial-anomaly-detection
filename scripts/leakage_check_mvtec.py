#!/usr/bin/env python3
"""Leakage checks for MVTec-style splits.

We check for:
- exact duplicate files across train/test (sha256)
- near-duplicate images across train/test (dHash + Hamming threshold)

This is meant to catch accidental overlap/copying that would inflate AUROC.

Example:
  python3 scripts/leakage_check_mvtec.py --mvtec-root data/mvtec --category bottle --out outputs/leakage/mvtec_bottle.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(p: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dhash64(p: Path, *, size: int = 9) -> int:
    """Difference hash (dHash) 8x8 -> 64-bit integer.

    We resize to (9, 8) grayscale and compare adjacent pixels horizontally.
    """
    img = Image.open(p).convert("L").resize((size, size - 1))
    a = np.asarray(img, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    bits = diff.flatten().astype(np.uint8)
    h = 0
    for b in bits.tolist():
        h = (h << 1) | int(b)
    return int(h)


def hamming64(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def list_images(root: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--near-threshold", type=int, default=2, help="dHash Hamming distance <= this counts as near-duplicate")
    ap.add_argument("--max-near", type=int, default=50, help="Max near-duplicate pairs to record")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.mvtec_root)
    train_root = root / args.category / "train"
    test_root = root / args.category / "test"

    train_paths = list_images(train_root)
    test_paths = list_images(test_root)

    # Exact overlap by sha256
    train_sha = {}
    for p in train_paths:
        train_sha[str(p)] = sha256_file(p)
    test_sha = {}
    for p in test_paths:
        test_sha[str(p)] = sha256_file(p)

    inv_train = {}
    for path, h in train_sha.items():
        inv_train.setdefault(h, []).append(path)

    exact_overlaps = []
    for path, h in test_sha.items():
        if h in inv_train:
            exact_overlaps.append({"sha256": h, "test": path, "train": inv_train[h]})

    # Near duplicates by dhash (bruteforce; OK for MVTec scale)
    train_dh = {str(p): dhash64(p) for p in train_paths}
    test_dh = {str(p): dhash64(p) for p in test_paths}

    near = []
    thr = int(args.near_threshold)
    for tp, th in test_dh.items():
        for rp, rh in train_dh.items():
            d = hamming64(th, rh)
            if d <= thr:
                near.append({"distance": d, "test": tp, "train": rp})
                if len(near) >= int(args.max_near):
                    break
        if len(near) >= int(args.max_near):
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "mvtec_root": str(root.resolve()),
                "category": args.category,
                "counts": {"train_images": len(train_paths), "test_images": len(test_paths)},
                "exact_overlaps": exact_overlaps,
                "near_duplicates": sorted(near, key=lambda r: (r["distance"], r["test"])),
                "near_threshold": thr,
            },
            indent=2,
        )
    )
    print(str(out))


if __name__ == "__main__":
    main()
