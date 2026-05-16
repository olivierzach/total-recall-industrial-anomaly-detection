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
from src.patchcore.memory_bank import flatten_embeddings_with_metadata
from src.patchcore.patchcore import to_numpy
from src.patchcore.preprocess import fit_pca, pca_state
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
    ap.add_argument("--nominal", required=True, help="Directory of nominal images")
    ap.add_argument("--out", required=True, help="Output model directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.02)
    ap.add_argument(
        "--coreset-method",
        type=str,
        default="kcenter",
        choices=["kcenter", "random", "kmeans"],
        help="kcenter is expensive; random is fast baseline; kmeans uses prototype centroids",
    )
    ap.add_argument("--kmeans-iters", type=int, default=50, help="Only used for --coreset-method kmeans")
    ap.add_argument("--cache-memory", action="store_true", help="Cache computed coreset memory bank to speed up retries")

    ap.add_argument("--backbone", type=str, default="wide_resnet50_2", help="torchvision backbone (e.g. wide_resnet50_2, vit_b_16)")
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"], help="feature layers to hook (CNN only; for ViT we will default later)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--distance-metric", type=str, default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--pca-dim", type=int, default=0, help="If >0, apply PCA (and optional whitening) to patch embeddings before kNN")
    ap.add_argument("--no-pca-whiten", action="store_true", help="Disable PCA whitening (projection only)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(int(args.seed))

    cfg = PatchCoreConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
        image_size=int(args.image_size),
        coreset_ratio=float(args.coreset_ratio),
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
            emb_np = to_numpy(emb)
            flat, meta = flatten_embeddings_with_metadata(emb_np, list(batch.path), hw)
            nominal_patches.append(flat)
            memory_metadata.extend(meta)

    X = np.concatenate(nominal_patches, axis=0)

    pca = None
    X_for_nn = X
    if cfg.pca_dim and int(cfg.pca_dim) > 0:
        pca = fit_pca(X, int(cfg.pca_dim), whiten=bool(cfg.pca_whiten))
        X_for_nn = pca.transform(X)

    # Optionally load cached memory bank (post-PCA) for speed/retries.
    cache_path = None
    if args.cache_memory:
        out_dir = Path(args.out)
        cache_path = out_dir / "memory_cache.npz"
        if cache_path.exists():
            data = dict(np.load(cache_path))
            Xc = data["memory_bank"].astype(np.float32)
            idx = None
        else:
            Xc = None
            idx = None
    else:
        Xc = None
        idx = None

    if Xc is None:
        if args.coreset_method == "random":
            rng = make_numpy_rng(args.seed)
            N = X_for_nn.shape[0]
            k = max(1, int(np.ceil(float(cfg.coreset_ratio) * N)))
            idx = rng.choice(N, size=k, replace=False)
            Xc = X_for_nn[idx]
        elif args.coreset_method == "kmeans":
            from sklearn.cluster import MiniBatchKMeans

            N = X_for_nn.shape[0]
            k = max(1, int(np.ceil(float(cfg.coreset_ratio) * N)))
            km = MiniBatchKMeans(n_clusters=int(k), batch_size=4096, n_init=1, max_iter=int(args.kmeans_iters), random_state=int(args.seed))
            km.fit(X_for_nn)
            Xc = km.cluster_centers_.astype(np.float32, copy=False)
            idx = None
        else:
            idx = KCenterGreedy().select(X_for_nn, ratio=cfg.coreset_ratio, rng=make_numpy_rng(args.seed))
            Xc = X_for_nn[idx]

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, memory_bank=Xc.astype(np.float32, copy=False), selected_indices=(idx.astype(np.int64) if idx is not None else np.array([], dtype=np.int64)))

    meta_c = [memory_metadata[int(i)] for i in idx] if idx is not None else None
    backbone_hash = state_dict_sha256(backbone.state_dict())
    manifest_path = manifest_path_for_target(args.out)
    manifest = build_run_manifest(
        kind="fit_nominal_patchcore",
        entrypoint="scripts/fit_nominal_patchcore.py",
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
        extra={"n_images": len(ds), "nominal_patches": int(X.shape[0]), "coreset": int(Xc.shape[0]), "backbone_sha256": backbone_hash},
    )

    save_patchcore(
        args.out,
        cfg,
        Xc,
        backbone_state=backbone.state_dict(),
        memory_metadata=meta_c,
        seed=int(args.seed),
        pca_state=(pca_state(pca) if pca is not None else None),
        artifact_info={
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": manifest["run_id"],
            "manifest_path": str(manifest_path.resolve()),
            "seed": int(args.seed),
            "kind": "patchcore_model",
            "nominal_dir": str(Path(args.nominal).resolve()),
            "backbone_sha256": backbone_hash,
            "coreset_method": str(args.coreset_method),
        },
    )
    add_outputs(
        manifest,
        [
            artifact_reference(
                args.out,
                role="model",
                kind="patchcore_model",
                known_run_id=manifest["run_id"],
                known_schema_version=RUN_MANIFEST_SCHEMA_VERSION,
                known_manifest_path=str(manifest_path.resolve()),
            )
        ],
    )
    write_run_manifest(args.out, manifest)

    print(
        {
            "nominal_dir": str(Path(args.nominal).resolve()),
            "n_images": len(ds),
            "nominal_patches": int(X.shape[0]),
            "coreset": int(Xc.shape[0]),
            "cfg": asdict(cfg),
            "seed": int(args.seed),
            "run_id": manifest["run_id"],
            "manifest_path": str(manifest_path.resolve()),
            "out": str(Path(args.out).resolve()),
        }
    )


if __name__ == "__main__":
    main()
