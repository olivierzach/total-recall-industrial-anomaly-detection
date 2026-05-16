#!/usr/bin/env python3
"""Score arbitrary images using a fitted PatchCore memory bank.

This is the "QA pipeline" entrypoint: given a directory of images, output an anomaly score per image.

Optionally also write anomaly maps (upsampled heatmaps) for triage.

Example:
  python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/new_images --out outputs/scores.jsonl

  python3 scripts/score_images.py --model outputs/models/bottle --images /path/to/new_images \
    --out outputs/scores.jsonl --save-maps outputs/maps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.embedding import patch_embeddings
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.utils.io import load_patchcore


def iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    if root.is_file() and root.suffix.lower() in exts:
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model directory created by fit_mvtec_patchcore.py")
    ap.add_argument("--images", required=True, help="Directory (or file) of images to score")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/scores.jsonl")
    ap.add_argument("--save-maps", default=None, help="If set, write per-image anomaly maps (.npy) to this directory")
    args = ap.parse_args()

    cfg, memory = load_patchcore(args.model)

    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = FeatureHooks(backbone, list(cfg.layers))

    model = PatchCoreModel.fit(cfg, memory)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maps_dir = Path(args.save_maps) if args.save_maps else None
    if maps_dir:
        maps_dir.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        with torch.no_grad():
            for p in iter_images(Path(args.images)):
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                _ = backbone(x)
                feats = hooks.pop()
                emb, (H, W) = patch_embeddings(feats, cfg.layers, l2_normalize=cfg.l2_normalize, return_hw=True)  # type: ignore
                emb0 = to_numpy(emb[0])
                s = model.score_image(emb0)
                rec = {"path": str(p), "score": float(s)}
                f.write(json.dumps(rec) + "\n")

                if maps_dir:
                    amap = model.score_map(emb0, (H, W))
                    # Save patch-grid map; visualization script can upsample.
                    npy = maps_dir / (p.stem + ".anomaly_patchgrid.npy")
                    import numpy as np

                    np.save(npy, amap)

    print(f"Wrote {out_path}")
    if maps_dir:
        print(f"Wrote anomaly maps to {maps_dir}")


if __name__ == "__main__":
    main()
