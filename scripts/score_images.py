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
from dataclasses import asdict
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.explain import build_patch_explanations
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.patchcore.preprocess import pca_from_state
from src.utils.io import load_patchcore, load_threshold_artifact
from src.utils.paths import derived_output_path
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
    ap.add_argument("--model", required=True, help="Model directory created by fit_*_patchcore.py")
    ap.add_argument("--images", required=True, help="Directory (or file) of images to score")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/scores.jsonl")
    ap.add_argument("--save-maps", default=None, help="If set, write per-image anomaly maps (.npy) to this directory")
    ap.add_argument("--threshold", default=None, help="Optional threshold artifact JSON from calibrate_threshold.py")
    ap.add_argument("--top-k-neighbors", type=int, default=0, help="If >0, include nearest nominal matches for top anomaly patches")
    ap.add_argument("--top-k-patches", type=int, default=3, help="How many highest-scoring patches to explain per image")
    args = ap.parse_args()

    artifact = load_patchcore(args.model)
    cfg = artifact.cfg

    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained and artifact.backbone_state is None)
    if artifact.backbone_state is not None:
        backbone.load_state_dict(artifact.backbone_state)
    backbone = backbone.to(device)
    backbone_hash = state_dict_sha256(backbone.state_dict())
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    pca = pca_from_state(artifact.pca_state) if artifact.pca_state is not None else None
    model = PatchCoreModel.fit(cfg, artifact.memory_bank, pca=pca)
    images_root = Path(args.images)
    threshold = load_threshold_artifact(args.threshold) if args.threshold else None
    image_paths = list(iter_images(images_root))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maps_dir = Path(args.save_maps) if args.save_maps else None
    if maps_dir:
        maps_dir.mkdir(parents=True, exist_ok=True)

    input_artifacts = [artifact_reference(args.model, role="model", kind="patchcore_model")]
    if args.threshold:
        input_artifacts.append(artifact_reference(args.threshold, role="threshold", kind="threshold_artifact"))
    dataset_root = images_root if images_root.is_dir() else images_root.parent
    manifest_path = manifest_path_for_target(out_path)
    manifest = build_run_manifest(
        kind="score_images",
        entrypoint="scripts/score_images.py",
        requested_device=str(args.device),
        seed=artifact.seed,
        config=asdict(cfg),
        args=dict(vars(args)),
        datasets=[
            dataset_reference(
                dataset="image_folder",
                role="score_input",
                root=dataset_root,
                files=image_paths,
                selection={"n_images": len(image_paths)},
            )
        ],
        inputs=input_artifacts,
        extra={
            "model_run_id": artifact.artifact_info.get("run_id") if artifact.artifact_info else None,
            "backbone_sha256": backbone_hash,
        },
    )

    with out_path.open("w") as f:
        with torch.no_grad():
            for p in image_paths:
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                emb, (H, W) = extract_patch_embeddings(
                    backbone_name=cfg.backbone,
                    model=backbone,
                    hooks=hooks,
                    x=x,
                    layers=cfg.layers,
                    l2_normalize=cfg.l2_normalize,
                    return_hw=True,
                )
                emb0 = to_numpy(emb[0])
                nn_dists = None
                nn_inds = None
                if args.top_k_neighbors > 0:
                    nn_dists, nn_inds = model.query(emb0, n_neighbors=int(args.top_k_neighbors))
                    patch_scores = nn_dists[:, 0]
                else:
                    patch_scores = model.score_patches(emb0)

                if cfg.image_score == "max":
                    s = float(np.max(patch_scores))
                elif cfg.image_score == "mean":
                    s = float(np.mean(patch_scores))
                else:
                    raise ValueError(f"unknown image_score={cfg.image_score}")

                rec = {
                    "path": str(p),
                    "score": float(s),
                    "run_id": manifest["run_id"],
                    "schema_version": RUN_RESULT_SCHEMA_VERSION,
                    "model_run_id": artifact.artifact_info.get("run_id") if artifact.artifact_info else None,
                }
                if threshold is not None:
                    rec["threshold"] = float(threshold.threshold)
                    rec["is_anomaly"] = bool(s >= threshold.threshold)
                if args.top_k_neighbors > 0 and nn_dists is not None and nn_inds is not None:
                    rec["explanations"] = build_patch_explanations(
                        nn_dists,
                        nn_inds,
                        (H, W),
                        memory_metadata=artifact.memory_metadata,
                        top_k_patches=int(args.top_k_patches),
                    )
                f.write(json.dumps(rec) + "\n")

                if maps_dir:
                    amap = patch_scores.reshape(H, W).astype(np.float32, copy=False)
                    # Save patch-grid map; visualization script can upsample.
                    npy = maps_dir / derived_output_path(images_root, p, ".anomaly_patchgrid.npy")
                    npy.parent.mkdir(parents=True, exist_ok=True)
                    np.save(npy, amap)

    outputs = [
        artifact_reference(
            out_path,
            role="scores",
            kind="score_jsonl",
            known_run_id=manifest["run_id"],
            known_schema_version=RUN_RESULT_SCHEMA_VERSION,
            known_manifest_path=str(manifest_path.resolve()),
        )
    ]
    if maps_dir is not None:
        outputs.append(
            artifact_reference(
                maps_dir,
                role="anomaly_maps",
                kind="score_maps",
                known_run_id=manifest["run_id"],
                known_schema_version=RUN_RESULT_SCHEMA_VERSION,
                known_manifest_path=str(manifest_path.resolve()),
            )
        )
    add_outputs(manifest, outputs)
    write_run_manifest(out_path, manifest)
    print(f"Wrote {out_path}")
    if maps_dir:
        print(f"Wrote anomaly maps to {maps_dir}")


if __name__ == "__main__":
    main()
