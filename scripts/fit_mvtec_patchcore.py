#!/usr/bin/env python3
"""Fit PatchCore memory bank for a single MVTec category.

Writes a self-contained model directory containing:
- config.json
- memory_bank.npy (coreset)

Example:
  python3 scripts/fit_mvtec_patchcore.py --mvtec-root data/mvtec --category bottle --device cpu --out outputs/models/bottle
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.mvtec import MVTecADDataset
from src.data.collate import collate_batch
from src.patchcore import PatchCoreConfig
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.embedding import patch_embeddings
from src.patchcore.patchcore import to_numpy
from src.utils.io import save_patchcore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", type=str, required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = PatchCoreConfig(layers=tuple(args.layers), image_size=int(args.image_size), coreset_ratio=float(args.coreset_ratio))
    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = MVTecADDataset(args.mvtec_root, args.category, "train", transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = FeatureHooks(backbone, list(cfg.layers))

    nominal_patches = []
    with torch.no_grad():
        for batch in train_dl:
            x = batch.image.to(device)  # type: ignore
            _ = backbone(x)
            feats = hooks.pop()
            emb = patch_embeddings(feats, cfg.layers, l2_normalize=cfg.l2_normalize)  # [B,P,D]
            nominal_patches.append(to_numpy(emb.reshape(-1, emb.shape[-1])))

    X = np.concatenate(nominal_patches, axis=0)

    selector = KCenterGreedy()
    idx = selector.select(X, ratio=cfg.coreset_ratio)
    Xc = X[idx]

    save_patchcore(args.out, cfg, Xc)

    out = {
        "category": args.category,
        "n_train": len(train_ds),
        "nominal_patches": int(X.shape[0]),
        "coreset": int(Xc.shape[0]),
        "cfg": asdict(cfg),
        "out": str(Path(args.out).resolve()),
    }
    print(out)


if __name__ == "__main__":
    main()
