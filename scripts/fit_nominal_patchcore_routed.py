#!/usr/bin/env python3
"""Fit a *query-adaptive* PatchCore model.

This builds a full nominal patch memory (optionally PCA-whitened), plus two routing indices:

- patch routing: k-means over patch embeddings
- image routing: k-means over per-image embeddings (mean of patch embeddings)

At inference time you can choose either routing mode:
- per-patch: route each query patch to candidate clusters
- per-image: route once per image, then search within patches from candidate image clusters

Output is compatible with the existing PatchCore artifact layout, plus routing files.

Example:
  python3 scripts/fit_nominal_patchcore_routed.py \
    --nominal /data/nominal_widgets_camA \
    --out outputs/models/widgets_camA_routed \
    --device cpu --image-size 256 \
    --pca-dim 256 \
    --patch-clusters 128 --image-clusters 32
"""

from __future__ import annotations

import argparse
import json
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
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.memory_bank import flatten_embeddings_with_metadata
from src.patchcore.patchcore import to_numpy
from src.patchcore.preprocess import fit_pca, pca_state
from src.patchcore.routing import build_image_routing, build_patch_routing
from src.utils.io import save_patchcore
from src.utils.provenance import (
    RUN_MANIFEST_SCHEMA_VERSION,
    add_outputs,
    artifact_reference,
    build_run_manifest,
    dataset_reference,
    manifest_path_for_target,
    state_dict_sha256,
    write_run_manifest,
)
from src.utils.random import make_numpy_rng, set_global_seed


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
    ap.add_argument("--nominal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--backbone", type=str, default="wide_resnet50_2")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"])
    ap.add_argument("--image-size", type=int, default=256)

    ap.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])

    ap.add_argument("--pca-dim", type=int, default=0)
    ap.add_argument("--no-pca-whiten", action="store_true")

    ap.add_argument("--patch-clusters", type=int, default=128)
    ap.add_argument("--image-clusters", type=int, default=32)
    ap.add_argument("--kmeans-iters", type=int, default=15)

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_global_seed(int(args.seed))

    cfg = PatchCoreConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
        image_size=int(args.image_size),
        coreset_ratio=1.0,  # routed models typically keep the full bank; routing does the subsetting
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

    ds = NominalFolder(args.nominal, transform=tfm)
    dl = DataLoader(ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers), collate_fn=collate_batch)

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    nominal_patches = []
    memory_metadata = []
    image_to_patch_indices: list[np.ndarray] = []
    image_embeddings: list[np.ndarray] = []

    offset = 0
    with torch.no_grad():
        for batch in dl:
            x = batch.image.to(device)
            emb, hw = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)  # [B, P, D]
            flat, meta = flatten_embeddings_with_metadata(emb_np, list(batch.path), hw)
            nominal_patches.append(flat)
            memory_metadata.extend(meta)

            # image embeddings: mean of patch embeddings per image
            B = emb_np.shape[0]
            P = emb_np.shape[1]
            for bi in range(B):
                img_emb = emb_np[bi].mean(axis=0)
                image_embeddings.append(img_emb)
                idxs = np.arange(offset + bi * P, offset + (bi + 1) * P, dtype=np.int64)
                image_to_patch_indices.append(idxs)
            offset += B * P

    X = np.concatenate(nominal_patches, axis=0).astype(np.float32, copy=False)
    imgX = np.stack(image_embeddings, axis=0).astype(np.float32, copy=False)

    pca = None
    X_for_routing = X
    imgX_for_routing = imgX
    if cfg.pca_dim and int(cfg.pca_dim) > 0:
        pca = fit_pca(X, int(cfg.pca_dim), whiten=bool(cfg.pca_whiten))
        X_for_routing = pca.transform(X)
        imgX_for_routing = pca.transform(imgX)

    rng = make_numpy_rng(args.seed)

    patch_routing = build_patch_routing(X_for_routing, n_clusters=int(args.patch_clusters), iters=int(args.kmeans_iters), rng=rng)
    image_routing = build_image_routing(imgX_for_routing, image_to_patch_indices, n_clusters=int(args.image_clusters), iters=int(args.kmeans_iters), rng=rng)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save routing state.
    np.savez(
        out_dir / "routing_state.npz",
        patch_centroids=patch_routing.centroids,
        image_centroids=image_routing.centroids,
        # store members as object arrays (lists of arrays)
        patch_members=np.array(patch_routing.members, dtype=object),
        image_members=np.array(image_routing.members, dtype=object),
    )
    (out_dir / "routing_state.json").write_text(
        json.dumps(
            {
                "patch_clusters": int(patch_routing.centroids.shape[0]),
                "image_clusters": int(image_routing.centroids.shape[0]),
                "metric": cfg.distance_metric,
                "pca": {"enabled": bool(pca is not None), "dim": int(cfg.pca_dim), "whiten": bool(cfg.pca_whiten)},
            },
            indent=2,
            sort_keys=True,
        )
    )

    # Persist a standard PatchCore artifact (memory bank in routed space).
    backbone_hash = state_dict_sha256(backbone.state_dict())
    manifest_path = manifest_path_for_target(args.out)
    manifest = build_run_manifest(
        kind="fit_nominal_patchcore_routed",
        entrypoint="scripts/fit_nominal_patchcore_routed.py",
        requested_device=str(args.device),
        seed=int(args.seed),
        config=asdict(cfg),
        args=dict(vars(args)),
        datasets=[
            dataset_reference(
                dataset="nominal_folder",
                role="train",
                root=args.nominal,
                files=ds.paths,
                selection={"n_images": len(ds)},
            )
        ],
        extra={
            "n_images": len(ds),
            "nominal_patches": int(X.shape[0]),
            "patch_clusters": int(args.patch_clusters),
            "image_clusters": int(args.image_clusters),
            "backbone_sha256": backbone_hash,
        },
    )

    save_patchcore(
        args.out,
        cfg,
        X_for_routing,
        backbone_state=backbone.state_dict(),
        memory_metadata=memory_metadata,
        seed=int(args.seed),
        pca_state=(pca_state(pca) if pca is not None else None),
        artifact_info={
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "manifest_path": str(manifest_path.resolve()),
            "seed": int(args.seed),
            "kind": "patchcore_routed_model",
            "nominal_dir": str(Path(args.nominal).resolve()),
            "backbone_sha256": backbone_hash,
        },
    )

    add_outputs(
        manifest,
        [
            artifact_reference(
                args.out,
                role="model",
                kind="patchcore_routed_model",
                known_run_id=manifest["run_id"],
                known_schema_version=RUN_MANIFEST_SCHEMA_VERSION,
                known_manifest_path=str(manifest_path.resolve()),
            )
        ],
    )
    write_run_manifest(args.out, manifest)

    print(
        {
            "out": str(out_dir.resolve()),
            "n_images": len(ds),
            "nominal_patches": int(X.shape[0]),
            "routing": {
                "patch_clusters": int(args.patch_clusters),
                "image_clusters": int(args.image_clusters),
                "kmeans_iters": int(args.kmeans_iters),
            },
        }
    )


if __name__ == "__main__":
    main()
