#!/usr/bin/env python3
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
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.mvtec import MVTecADDataset
from src.data.collate import collate_batch
from src.patchcore import PatchCoreConfig
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.metrics import auroc, classification_metrics
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.patchcore.preprocess import fit_pca
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", type=str, required=True)
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--backbone", type=str, default="wide_resnet50_2", help="torchvision backbone (e.g. wide_resnet50_2, vit_b_16)")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"], help="CNN feature layers; ignored for ViT")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="outputs/mvtec_patchcore_result.json")
    ap.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--pca-dim", type=int, default=0)
    ap.add_argument("--no-pca-whiten", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=str, default=None, help="Optional threshold artifact JSON for operational metrics")
    ap.add_argument("--target-fpr", type=float, default=0.01, help="If no --threshold is provided, calibrate threshold to this FPR on held-out nominal")
    ap.add_argument("--calib-fraction", type=float, default=0.2, help="Fraction of nominal train images held out for threshold calibration")
    ap.add_argument("--max-train", type=int, default=0, help="If set, cap number of nominal train images for a smoke run")
    ap.add_argument("--max-test", type=int, default=0, help="If set, cap number of test images for a smoke run")
    ap.add_argument("--log-every", type=int, default=25, help="Progress logging cadence (batches)")
    args = ap.parse_args()
    set_global_seed(int(args.seed))

    cfg = PatchCoreConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
        image_size=int(args.image_size),
        coreset_ratio=float(args.coreset_ratio),
        distance_metric=str(getattr(args, "distance_metric", "euclidean")),
        pca_dim=int(getattr(args, "pca_dim", 0)),
        pca_whiten=not bool(getattr(args, "no_pca_whiten", False)),
    )

    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds_all = MVTecADDataset(args.mvtec_root, args.category, "train", transform=tfm)
    test_ds = MVTecADDataset(args.mvtec_root, args.category, "test", transform=tfm)

    # Deterministic split of nominal train into memory-build and calibration sets.
    # Calibration set is used only to set an operating threshold at a target FPR.
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

    train_dl = DataLoader(train_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)
    test_dl = DataLoader(test_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)
    calib_dl = (
        DataLoader(calib_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)
        if calib_ds is not None
        else None
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    backbone_hash = state_dict_sha256(backbone.state_dict())
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))
    t_total0 = time.time()

    # Build memory bank from nominal patches.
    nominal_patches = []
    n_train_seen = 0
    with torch.no_grad():
        for bi, batch in enumerate(train_dl):
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
            n_train_seen += int(x.shape[0])
            if args.log_every and (bi + 1) % int(args.log_every) == 0:
                print(f"[eval/train] batches={bi+1} images={n_train_seen}", flush=True)
            if args.max_train and n_train_seen >= int(args.max_train):
                print(f"[eval/train] stopping early at images={n_train_seen} due to --max-train", flush=True)
                break

    X = np.concatenate(nominal_patches, axis=0)

    # Optional PCA whitening (fit on nominal patches).
    pca = None
    X_for_nn = X
    if cfg.pca_dim and int(cfg.pca_dim) > 0:
        pca = fit_pca(X, int(cfg.pca_dim), whiten=bool(cfg.pca_whiten))
        X_for_nn = pca.transform(X)

    # Coreset selection.
    selector = KCenterGreedy()
    idx = selector.select(X_for_nn, ratio=cfg.coreset_ratio, rng=make_numpy_rng(args.seed))
    Xc = X_for_nn[idx]

    model = PatchCoreModel.fit(cfg, Xc, pca=pca)

    # Evaluate image-level AUROC.
    y_true = []
    y_score = []
    n_test_seen = 0

    with torch.no_grad():
        for bi, batch in enumerate(test_dl):
            x = batch.image.to(device)  # type: ignore
            labels = batch.label.numpy().astype(np.int64)  # type: ignore
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
                score = model.score_image(emb[i])
                y_score.append(score)
            y_true.extend(list(labels))
            n_test_seen += int(x.shape[0])
            if args.log_every and (bi + 1) % int(args.log_every) == 0:
                print(f"[eval/test] batches={bi+1} images={n_test_seen}", flush=True)
            if args.max_test and n_test_seen >= int(args.max_test):
                print(f"[eval/test] stopping early at images={n_test_seen} due to --max-test", flush=True)
                break

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    image_auroc = auroc(y_true, y_score)
    # Threshold calibration for operational metrics.
    threshold_eval = None
    input_artifacts = []

    def _calibrate_threshold_from_nominal(calib_scores: np.ndarray, target_fpr: float) -> float:
        # Choose threshold at (1 - target_fpr) quantile of nominal scores.
        # This yields approximately target_fpr false positives on nominal.
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
        # Calibrate using held-out nominal images from the train split.
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

    # Files for provenance: use the underlying MVTec dataset items + our deterministic split indices.
    train_items_all = train_ds_all._items  # type: ignore[attr-defined]
    mem_items = [train_items_all[i] for i in mem_indices[:n_train_seen]]
    calib_items = [train_items_all[i] for i in calib_indices] if calib_indices else []
    test_items = test_ds._items[:n_test_seen]  # type: ignore[attr-defined]

    train_files = [Path(p) for (p, _, _) in mem_items]
    if calib_items:
        train_files.extend(Path(p) for (p, _, _) in calib_items)

    test_files = [Path(p) for (p, _, mask_path) in test_items]
    test_files.extend(Path(mask_path) for (_, _, mask_path) in test_items if mask_path is not None)
    manifest_path = manifest_path_for_target(args.out)
    manifest = build_run_manifest(
        kind="eval_mvtec_patchcore",
        entrypoint="scripts/eval_mvtec_patchcore.py",
        requested_device=str(args.device),
        seed=int(args.seed),
        config=asdict(cfg),
        args=dict(vars(args)),
        datasets=[
            dataset_reference(
                dataset="mvtec",
                role="train",
                root=args.mvtec_root,
                files=train_files,
                split="train",
                category=args.category,
                selection={"n_images": n_train_seen, "max_train": int(args.max_train)},
            ),
            dataset_reference(
                dataset="mvtec",
                role="test",
                root=args.mvtec_root,
                files=test_files,
                split="test",
                category=args.category,
                selection={"n_images": n_test_seen, "max_test": int(args.max_test)},
            ),
        ],
        inputs=input_artifacts,
        extra={
            "nominal_patches": int(X.shape[0]),
            "coreset": int(Xc.shape[0]),
            "metrics": {"image_auroc": image_auroc},
            "backbone_sha256": backbone_hash,
        },
    )

    out = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path.resolve()),
        "dataset": "mvtec",
        "kind": "eval",
        "cfg": asdict(cfg),
        "category": args.category,
        "device": str(args.device),
        "n_train": n_train_seen if args.max_train else len(train_ds),
        "n_test": n_test_seen if args.max_test else len(test_ds),
        "memory_bank": {"nominal_patches": int(X.shape[0]), "coreset": int(Xc.shape[0]), "ratio": cfg.coreset_ratio},
        "metrics": {"image_auroc": image_auroc},
        "seed": int(args.seed),
        "timing": {"total_s": time.time() - t_total0},
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
