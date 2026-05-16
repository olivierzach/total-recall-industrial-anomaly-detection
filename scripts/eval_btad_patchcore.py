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
from src.patchcore.metrics import auroc, classification_metrics
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.patchcore.preprocess import fit_pca
from src.patchcore.pro import compute_pro_auc
from src.utils.io import load_threshold_artifact
from src.utils.provenance import (
    RUN_RESULT_SCHEMA_VERSION,
    add_outputs,
    artifact_reference,
    build_run_manifest,
    dataset_reference,
    manifest_path_for_target,
    state_dict_sha256,
    write_run_manifest,
)
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
    ap.add_argument(
        "--coreset-method",
        type=str,
        default="kcenter",
        choices=["kcenter", "random", "kmeans"],
        help="kcenter is expensive; random is fast baseline; kmeans uses prototype centroids",
    )
    ap.add_argument("--kmeans-iters", type=int, default=50, help="Only used for --coreset-method kmeans")
    ap.add_argument("--cache-memory", action="store_true", help="Cache computed coreset memory bank to speed up ablations")
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
    ap.add_argument("--threshold", type=str, default=None, help="Optional threshold artifact JSON for operational metrics")
    ap.add_argument("--target-fpr", type=float, default=0.01, help="If no --threshold is provided, calibrate threshold to this FPR on held-out nominal")
    ap.add_argument("--calib-fraction", type=float, default=0.2, help="Fraction of nominal train images held out for threshold calibration")
    ap.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--pca-dim", type=int, default=0)
    ap.add_argument("--no-pca-whiten", action="store_true")
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
        distance_metric=str(args.distance_metric),
        pca_dim=int(args.pca_dim),
        pca_whiten=not bool(args.no_pca_whiten),
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

    train_ds_all = BTADDataset(args.btad_root, "train", transform=tfm, mask_transform=mtfm)
    test_ds = BTADDataset(args.btad_root, "test", transform=tfm, mask_transform=mtfm)

    # Deterministic split of nominal train into memory-build and calibration.
    n_train = len(train_ds_all)
    frac = float(args.calib_fraction)
    if not (0.0 <= frac < 1.0):
        raise ValueError("--calib-fraction must be in [0,1)")
    n_calib = int(round(frac * n_train))
    n_mem = max(1, n_train - n_calib)

    mem_indices = list(range(0, n_mem))
    calib_indices = list(range(n_mem, n_train))

    from torch.utils.data import Subset

    train_ds = Subset(train_ds_all, mem_indices)
    calib_ds = Subset(train_ds_all, calib_indices) if n_calib > 0 else None

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
    calib_dl = (
        DataLoader(calib_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)
        if calib_ds is not None
        else None
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    backbone_hash = state_dict_sha256(backbone.state_dict())
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

    # Optional PCA whitening (fit on nominal patches).
    pca = None
    X_for_nn = X
    if cfg.pca_dim and int(cfg.pca_dim) > 0:
        pca = fit_pca(X, int(cfg.pca_dim), whiten=bool(cfg.pca_whiten))
        X_for_nn = pca.transform(X)

    # Optionally load cached memory bank (post-PCA) for faster ablations.
    cache_path = None
    if args.cache_memory:
        outp = Path(args.out)
        cache_path = outp.with_suffix("").with_suffix(".memory_cache.npz")
        if cache_path.exists():
            data = dict(np.load(cache_path))
            Xc = data["memory_bank"].astype(np.float32)
            timing["coreset_s"] = 0.0
        else:
            Xc = None
    else:
        Xc = None

    if Xc is None:
        t0 = time.time()
        if args.coreset_method == "random":
            rng = make_numpy_rng(args.seed)
            N = X_for_nn.shape[0]
            k = max(1, int(np.ceil(float(cfg.coreset_ratio) * N)))
            idx = rng.choice(N, size=k, replace=False)
            Xc = X_for_nn[idx]
        elif args.coreset_method == "kmeans":
            # Prototype memory via MiniBatchKMeans centroids.
            from sklearn.cluster import MiniBatchKMeans

            N = X_for_nn.shape[0]
            k = max(1, int(np.ceil(float(cfg.coreset_ratio) * N)))
            km = MiniBatchKMeans(n_clusters=int(k), batch_size=4096, n_init=1, max_iter=int(args.kmeans_iters), random_state=int(args.seed))
            km.fit(X_for_nn)
            Xc = km.cluster_centers_.astype(np.float32, copy=False)
        else:
            idx = KCenterGreedy().select(X_for_nn, ratio=cfg.coreset_ratio, rng=make_numpy_rng(args.seed))
            Xc = X_for_nn[idx]

        timing["coreset_s"] = time.time() - t0
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, memory_bank=Xc.astype(np.float32, copy=False))

    t0 = time.time()
    model = PatchCoreModel.fit(cfg, Xc, pca=pca)
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
    threshold_eval = None

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
    # Threshold calibration for operational metrics.
    input_artifacts = []

    def _calibrate_threshold_from_nominal(calib_scores: np.ndarray, target_fpr: float) -> float:
        q = float(1.0 - float(target_fpr))
        q = min(max(q, 0.0), 1.0)
        return float(np.quantile(calib_scores, q))

    if args.threshold:
        thr = load_threshold_artifact(args.threshold)
        threshold_eval = classification_metrics(y_true, y_score, thr.threshold)
        threshold_eval["source"] = str(Path(args.threshold).resolve())
        if thr.source_run_id is not None:
            threshold_eval["source_run_id"] = thr.source_run_id
        input_artifacts.append(artifact_reference(args.threshold, role="threshold", kind="threshold_artifact"))
    else:
        if calib_dl is None:
            raise ValueError("No calibration split available; set --calib-fraction > 0 or provide --threshold")
        calib_scores = []
        with torch.no_grad():
            for batch in calib_dl:
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
                emb = to_numpy(emb)
                for i in range(emb.shape[0]):
                    calib_scores.append(model.score_image(emb[i]))
        calib_scores = np.asarray(calib_scores, dtype=np.float64)
        thr = _calibrate_threshold_from_nominal(calib_scores, float(args.target_fpr))
        threshold_eval = classification_metrics(y_true, y_score, thr)
        threshold_eval["source"] = "calibrated_from_nominal_train_holdout"
        threshold_eval["target_fpr"] = float(args.target_fpr)
        threshold_eval["calib_fraction"] = float(args.calib_fraction)
        threshold_eval["calib_n"] = int(calib_scores.shape[0])

    timing["total_s"] = time.time() - t_total0

    # Files for provenance: use underlying BTAD dataset items + deterministic split indices.
    train_items_all = train_ds_all._items  # type: ignore[attr-defined]
    mem_items = [train_items_all[i] for i in mem_indices[:n_train_seen]]
    calib_items = [train_items_all[i] for i in calib_indices] if calib_indices else []
    test_items = test_ds._items[:n_test_seen]  # type: ignore[attr-defined]

    train_files = [Path(p) for (p, _, ann_path) in mem_items]
    train_files.extend(Path(ann_path) for (_, _, ann_path) in mem_items if ann_path is not None)
    if calib_items:
        train_files.extend(Path(p) for (p, _, ann_path) in calib_items)
        train_files.extend(Path(ann_path) for (_, _, ann_path) in calib_items if ann_path is not None)

    test_files = [Path(p) for (p, _, ann_path) in test_items]
    test_files.extend(Path(ann_path) for (_, _, ann_path) in test_items if ann_path is not None)
    manifest_path = manifest_path_for_target(args.out)
    manifest = build_run_manifest(
        kind="eval_btad_patchcore",
        entrypoint="scripts/eval_btad_patchcore.py",
        requested_device=str(args.device),
        seed=int(args.seed),
        config=asdict(cfg),
        args=dict(vars(args)),
        datasets=[
            dataset_reference(
                dataset="btad",
                role="train",
                root=args.btad_root,
                files=train_files,
                split="train",
                selection={"n_images": n_train_seen, "max_train": int(args.max_train)},
            ),
            dataset_reference(
                dataset="btad",
                role="test",
                root=args.btad_root,
                files=test_files,
                split="test",
                selection={"n_images": n_test_seen, "max_test": int(args.max_test)},
            ),
        ],
        inputs=input_artifacts,
        extra={
            "nominal_patches": int(X.shape[0]),
            "coreset": int(Xc.shape[0]),
            "metrics": {"image_auroc": image_auroc, "pixel_auroc": pixel_auc, "pro_auc": pro_auc},
            "backbone_sha256": backbone_hash,
        },
    )

    out = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path.resolve()),
        "dataset": "btad",
        "kind": "eval",
        "cfg": asdict(cfg),
        "device": str(args.device),
        "n_train": n_train_seen if args.max_train else len(train_ds),
        "n_test": n_test_seen if args.max_test else len(test_ds),
        "memory_bank": {"nominal_patches": int(X.shape[0]), "coreset": int(Xc.shape[0]), "ratio": cfg.coreset_ratio},
        "metrics": {"image_auroc": image_auroc, "pixel_auroc": pixel_auc, "pro_auc": pro_auc},
        "timing": timing,
        "seed": int(args.seed),
    }
    if threshold_eval is not None:
        out["threshold_eval"] = threshold_eval

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    add_outputs(
        manifest,
        [
            artifact_reference(
                out_path,
                role="evaluation",
                kind="eval_result",
                known_run_id=manifest["run_id"],
                known_schema_version=RUN_RESULT_SCHEMA_VERSION,
                known_manifest_path=str(manifest_path.resolve()),
            )
        ],
    )
    write_run_manifest(out_path, manifest)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
