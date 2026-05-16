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
import time
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
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.memory_bank import flatten_embeddings_with_metadata
from src.patchcore.patchcore import to_numpy
from src.utils.io import save_patchcore
from src.utils.random import make_numpy_rng, set_global_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", type=str, required=True)
    ap.add_argument("--category", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--backbone", type=str, default="wide_resnet50_2", help="torchvision backbone (e.g. wide_resnet50_2, vit_b_16)")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="If set, cap number of train images for a smoke run")
    ap.add_argument("--log-every", type=int, default=25, help="Progress logging cadence (batches)")
    args = ap.parse_args()
    set_global_seed(int(args.seed))

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

    train_ds = MVTecADDataset(args.mvtec_root, args.category, "train", transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    nominal_patches = []
    memory_metadata = []
    n_train_seen = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(train_dl):
            x = batch.image.to(device)  # type: ignore
            emb, hw = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)
            flat, meta = flatten_embeddings_with_metadata(emb_np, list(batch.path), hw)
            nominal_patches.append(flat)
            memory_metadata.extend(meta)
            n_train_seen += int(x.shape[0])
            if args.log_every and (bi + 1) % int(args.log_every) == 0:
                print(f"[fit/train] batches={bi+1} images={n_train_seen}", flush=True)
            if args.max_train and n_train_seen >= int(args.max_train):
                print(f"[fit/train] stopping early at images={n_train_seen} due to --max-train", flush=True)
                break

    X = np.concatenate(nominal_patches, axis=0)

    selector = KCenterGreedy()
    idx = selector.select(X, ratio=cfg.coreset_ratio, rng=make_numpy_rng(args.seed))
    Xc = X[idx]
    meta_c = [memory_metadata[int(i)] for i in idx]

    save_patchcore(
        args.out,
        cfg,
        Xc,
        backbone_state=backbone.state_dict(),
        memory_metadata=meta_c,
        seed=int(args.seed),
    )

    out = {
        "category": args.category,
        "n_train": n_train_seen if args.max_train else len(train_ds),
        "nominal_patches": int(X.shape[0]),
        "coreset": int(Xc.shape[0]),
        "cfg": asdict(cfg),
        "seed": int(args.seed),
        "fit_s": time.time() - t0,
        "out": str(Path(args.out).resolve()),
    }
    print(out)


if __name__ == "__main__":
    main()
