#!/usr/bin/env python3
"""Fit PatchCore on a directory of nominal (good) images.

This is the recommended entrypoint for real QA usage.

Example:
  python3 scripts/fit_nominal_patchcore.py \
    --nominal /data/nominal_widgets_camA \
    --out outputs/models/widgets_camA \
    --device cpu --image-size 256 --coreset-ratio 0.02
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_batch
from src.patchcore import PatchCoreConfig
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.patchcore import to_numpy
from src.utils.io import save_patchcore


class NominalFolder(Dataset):
    def __init__(self, root: str | Path, transform):
        self.root = Path(root)
        self.transform = transform
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        self.paths = [p for p in sorted(self.root.rglob("*")) if p.suffix.lower() in exts]
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)

        class Item:
            pass

        it = Item()
        it.image = x
        it.label = 0
        it.mask = None
        it.path = str(p)
        return it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nominal", required=True, help="Directory of nominal images")
    ap.add_argument("--out", required=True, help="Output model directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.02)
    ap.add_argument("--backbone", type=str, default="wide_resnet50_2", help="torchvision backbone (e.g. wide_resnet50_2, vit_b_16)")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"], help="feature layers to hook (CNN only; for ViT we will default later)")
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()

    cfg = PatchCoreConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
        image_size=int(args.image_size),
        coreset_ratio=float(args.coreset_ratio),
    )
    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = NominalFolder(args.nominal, transform=tfm)
    dl = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    nominal_patches = []
    with torch.no_grad():
        for batch in dl:
            x = batch.image.to(device)
            emb = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=False,
            )
            if isinstance(emb, tuple):
                emb = emb[0]
            nominal_patches.append(to_numpy(emb.reshape(-1, emb.shape[-1])))

    X = np.concatenate(nominal_patches, axis=0)

    idx = KCenterGreedy().select(X, ratio=cfg.coreset_ratio)
    Xc = X[idx]

    save_patchcore(args.out, cfg, Xc)

    print(
        {
            "nominal_dir": str(Path(args.nominal).resolve()),
            "n_images": len(ds),
            "nominal_patches": int(X.shape[0]),
            "coreset": int(Xc.shape[0]),
            "cfg": asdict(cfg),
            "out": str(Path(args.out).resolve()),
        }
    )


if __name__ == "__main__":
    main()
