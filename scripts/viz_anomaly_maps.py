#!/usr/bin/env python3
"""Visualize anomaly maps produced by score_images.py.

Inputs:
- original images
- patch-grid anomaly maps saved as .npy

Outputs:
- per-image overlay PNGs

Example:
  python3 scripts/viz_anomaly_maps.py \
    --images /path/to/images \
    --maps outputs/maps \
    --out outputs/overlays
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    if root.is_file() and root.suffix.lower() in exts:
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def upsample(amap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # Simple bilinear upsample via PIL.
    im = Image.fromarray(amap.astype(np.float32))
    im = im.resize(size, resample=Image.BILINEAR)
    return np.array(im).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--maps", required=True, help="Directory containing *.anomaly_patchgrid.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.45)
    args = ap.parse_args()

    images_root = Path(args.images)
    maps_dir = Path(args.maps)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in iter_images(images_root):
        m = maps_dir / (p.stem + ".anomaly_patchgrid.npy")
        if not m.exists():
            continue
        amap = np.load(m)
        img = Image.open(p).convert("RGB")
        amap_up = upsample(amap, img.size)

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.imshow(amap_up, cmap="inferno", alpha=float(args.alpha))
        plt.axis("off")
        fig.tight_layout(pad=0)
        out_path = out_dir / (p.stem + ".overlay.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    print(f"Wrote overlays to {out_dir}")


if __name__ == "__main__":
    main()
