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
import time
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
from src.patchcore.pro import compute_pro_auc
from src.utils.random import make_numpy_rng, set_global_seed


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
    ap.add_argument("--num-neighbors", type=int, default=1, help="k for kNN distance (PatchCore scoring)")
    ap.add_argument("--image-score", type=str, default="max", choices=["max", "mean"], help="aggregate patch scores into image score")
    ap.add_argument(
        "--l2-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize patch embeddings before NN",
    )

    ap.add_argument("--backbone", type=str, default="wide_resnet50_2", help="torchvision backbone (e.g. wide_resnet50_2, vit_b_16)")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"], help="CNN feature layers; ignored for ViT")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="outputs/btad_patchcore_result.json")
    ap.add_argument("--max-train", type=int, default=0, help="If set, cap number of nominal train images (fast smoke)")
    ap.add_argument("--max-test", type=int, default=0, help="If set, cap number of test images (fast smoke)")
    ap.add_argument("--log-every", type=int, default=50, help="Progress logging cadence (batches)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(int(args.seed))

    cfg = PatchCoreConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
        image_size=int(args.image_size),
        l2_normalize=bool(args.l2_normalize),
        coreset_ratio=float(args.coreset_ratio),
        num_neighbors=int(args.num_neighbors),
        image_score=str(args.image_score),
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

    timing = {
        "feature_train_s": 0.0,
        "feature_test_s": 0.0,
        "coreset_s": 0.0,
        "knn_fit_s": 0.0,
        "metric_s": 0.0,
        "total_s": 0.0,
    }

    t_total0 = time.time()

    nominal_patches = []
    n_train_seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(train_dl):
            x = batch.image.to(device)  # type: ignore
            t0 = time.time()
            emb = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=False,
            )
            # Best-effort device sync for accurate timing.
            if device.type == "mps":
                try:
                    torch.mps.synchronize()  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            timing["feature_train_s"] += time.time() - t0

            if isinstance(emb, tuple):
                emb = emb[0]
            nominal_patches.append(to_numpy(emb.reshape(-1, emb.shape[-1])))
            n_train_seen += int(x.shape[0])
            if args.log_every and (bi + 1) % int(args.log_every) == 0:
                print(f"[train] batches={bi+1} images={n_train_seen}")
            if args.max_train and n_train_seen >= int(args.max_train):
                break

    X = np.concatenate(nominal_patches, axis=0)

    t0 = time.time()
    idx = KCenterGreedy().select(X, ratio=cfg.coreset_ratio, rng=make_numpy_rng(args.seed))
    timing["coreset_s"] = time.time() - t0
    Xc = X[idx]

    t0 = time.time()
    model = PatchCoreModel.fit(cfg, Xc)
    timing["knn_fit_s"] = time.time() - t0

    y_true = []
    y_score = []

    px_true = []
    px_score = []

    n_test_seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(test_dl):
            x = batch.image.to(device)  # type: ignore
            labels = batch.label.numpy().astype(np.int64)  # type: ignore
            masks = batch.mask  # type: ignore

            t0 = time.time()
            emb, (H, W) = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            if device.type == "mps":
                try:
                    torch.mps.synchronize()  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif device.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            timing["feature_test_s"] += time.time() - t0

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
                    amap = model.score_map(emb_np[i], (H, W))  # [H,W]
                    amap_t = torch.from_numpy(amap)[None, None, ...].to(device)
                    amap_up = F.interpolate(amap_t, size=(cfg.image_size, cfg.image_size), mode="bilinear", align_corners=False)[0, 0]

                    px_true.append((m[0] > 0.5).cpu().numpy().astype(np.uint8))
                    px_score.append(amap_up.cpu().numpy().astype(np.float32))

            n_test_seen += int(x.shape[0])
            if args.log_every and (bi + 1) % int(args.log_every) == 0:
                print(f"[test] batches={bi+1} images={n_test_seen}")
            if args.max_test and n_test_seen >= int(args.max_test):
                break

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    image_auroc = auroc(y_true, y_score)

    pixel_auc = float("nan")
    pro_auc = float("nan")
    t0 = time.time()
    if px_true:
        px_true_arr = np.stack(px_true)
        px_score_arr = np.stack(px_score)
        pixel_auc = pixel_auroc(px_true_arr, px_score_arr)

        # PRO expects per-image 2D arrays.
        scores_2d = [px_score_arr[i] for i in range(px_score_arr.shape[0])]
        masks_2d = [px_true_arr[i] for i in range(px_true_arr.shape[0])]
        pro_auc = compute_pro_auc(scores_2d, masks_2d, fpr_limit=0.3, n_thresholds=200).pro_auc
    timing["metric_s"] = time.time() - t0

    timing["total_s"] = time.time() - t_total0

    out = {
        "cfg": asdict(cfg),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "memory_bank": {"nominal_patches": int(X.shape[0]), "coreset": int(Xc.shape[0]), "ratio": cfg.coreset_ratio},
        "metrics": {"image_auroc": image_auroc, "pixel_auroc": pixel_auc, "pro_auc": pro_auc},
        "timing": timing,
        "seed": int(args.seed),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
