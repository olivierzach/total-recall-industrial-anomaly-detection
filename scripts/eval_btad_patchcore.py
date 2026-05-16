#!/usr/bin/env python3
"""Evaluate PatchCore on BTAD (DatasetNinja/Supervisely export).

This is meant as an easier-access dataset than MVTec.

Outputs:
- image AUROC
- pixel AUROC (if masks available)

Example:
  python3 scripts/eval_btad_patchcore.py --btad-root data/btad --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.btad import BTADDataset
from src.data.collate import collate_batch
from src.patchcore import PatchCoreConfig
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.metrics import auroc
from src.patchcore.patchcore import PatchCoreModel, to_numpy


def pixel_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    # Need both classes
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btad-root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--backbone", type=str, default="wide_resnet50_2", help="torchvision backbone (e.g. wide_resnet50_2, vit_b_16)")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"], help="CNN feature layers; ignored for ViT")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="outputs/btad_patchcore_result.json")
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
    mtfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
        ]
    )

    train_ds = BTADDataset(args.btad_root, "train", transform=tfm, mask_transform=mtfm)
    test_ds = BTADDataset(args.btad_root, "test", transform=tfm, mask_transform=mtfm)

    train_dl = DataLoader(
        train_ds,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    nominal_patches = []
    with torch.no_grad():
        for batch in train_dl:
            x = batch.image.to(device)  # type: ignore
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

    model = PatchCoreModel.fit(cfg, Xc)

    y_true = []
    y_score = []

    px_true = []
    px_score = []

    with torch.no_grad():
        for batch in test_dl:
            x = batch.image.to(device)  # type: ignore
            labels = batch.label.numpy().astype(np.int64)  # type: ignore
            masks = batch.mask  # type: ignore

            emb, (H, W) = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)

            for i in range(emb_np.shape[0]):
                score = model.score_image(emb_np[i])
                y_score.append(score)
            y_true.extend(list(labels))

            # Pixel metrics
            if masks is not None:
                for i in range(emb_np.shape[0]):
                    m = masks[i]
                    if m is None:
                        continue
                    # m: [1, Himg, Wimg] in {0,1}
                    amap = model.score_map(emb_np[i], (H, W))  # [H,W]
                    amap_t = torch.from_numpy(amap)[None, None, ...].to(device)
                    amap_up = F.interpolate(amap_t, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)[0, 0]

                    px_true.append((m[0] > 0.5).cpu().numpy().astype(np.uint8))
                    px_score.append(amap_up.cpu().numpy().astype(np.float32))

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    image_auroc = auroc(y_true, y_score)

    pixel_auc = float("nan")
    if px_true:
        pixel_auc = pixel_auroc(np.stack(px_true), np.stack(px_score))

    out = {
        "cfg": asdict(cfg),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "memory_bank": {"nominal_patches": int(X.shape[0]), "coreset": int(Xc.shape[0]), "ratio": cfg.coreset_ratio},
        "metrics": {"image_auroc": image_auroc, "pixel_auroc": pixel_auc},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
