#!/usr/bin/env python3
"""Build an *unsupervised* defect memory bank from existing anomaly images.

This is a demo implementation for the "known failure mode lookup" idea when you don't yet
have defect labels.

We:
1) load a PatchCore model
2) embed test images
3) score patches vs nominal memory
4) collect top-k highest-scoring patch embeddings from anomalous images
5) cluster those embeddings by similarity (k-means)
6) write a defect bank artifact + a human-reviewable cluster report

Important:
- Clusters are NOT ground-truth defect types. They are "prototype groups".
- The output is meant to bootstrap a taxonomy: you review clusters and assign names.

Example (smoke MVTec bottle):
  python3 scripts/build_defect_bank_unsupervised_mvtec.py \
    --mvtec-root data/mvtec_smoke \
    --category bottle \
    --model outputs/models/bottle \
    --out outputs/defect_bank/mvtec_bottle_unsup \
    --top-k-patches 5 \
    --clusters 8
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
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import collate_batch
from src.data.mvtec import MVTecADDataset
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.learned_router import fit_linear_router
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.patchcore.preprocess import pca_from_state
from src.patchcore.routing import build_patch_routing
from src.utils.io import load_patchcore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--model", required=True, help="PatchCore model directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--top-k-patches", type=int, default=5, help="How many top anomaly patches per anomalous image to add")
    ap.add_argument("--clusters", type=int, default=8, help="k-means clusters over defect patches")
    ap.add_argument("--kmeans-iters", type=int, default=20)

    ap.add_argument("--learned-router", action="store_true", help="Also train a tiny linear router over defect clusters")

    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_patchcore(args.model)
    cfg = artifact.cfg
    pca = pca_from_state(artifact.pca_state) if artifact.pca_state is not None else None

    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_ds = MVTecADDataset(args.mvtec_root, args.category, "test", transform=tfm)
    # We will iterate directly to keep metadata access simple.

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained and artifact.backbone_state is None)
    if artifact.backbone_state is not None:
        backbone.load_state_dict(artifact.backbone_state)
    backbone = backbone.to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    model = PatchCoreModel.fit(cfg, artifact.memory_bank, pca=pca)

    top_k = int(args.top_k_patches)

    defect_embs: list[np.ndarray] = []
    defect_meta: list[dict] = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            it = test_ds[i]
            # MVTec dataset encodes label: 0=good, 1=anomaly
            if int(it.label) == 0:
                continue
            img = Image.open(it.path).convert("RGB")
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
            emb0 = to_numpy(emb[0])  # [P, D]

            # Score patches.
            scores = model.score_patches(emb0)  # [P]
            k = min(top_k, scores.shape[0])
            idx = np.argpartition(-scores, kth=k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]

            for pi in idx.tolist():
                e = emb0[int(pi)]
                if pca is not None:
                    e = pca.transform(e[None, :])[0]
                defect_embs.append(e.astype(np.float32, copy=False))
                r = int(pi) // int(W)
                c = int(pi) % int(W)
                defect_meta.append(
                    {
                        "source_path": str(it.path),
                        "patch_index": int(pi),
                        "row": int(r),
                        "col": int(c),
                        "grid_h": int(H),
                        "grid_w": int(W),
                        "patch_score": float(scores[int(pi)]),
                        "mvtec_category": str(args.category),
                    }
                )

    if not defect_embs:
        raise SystemExit("No anomalous patches collected. Check dataset path/category.")

    X = np.stack(defect_embs, axis=0)

    # Cluster defect patches.
    routing = build_patch_routing(X, n_clusters=int(args.clusters), iters=int(args.kmeans_iters), rng=np.random.default_rng(int(args.seed)))

    # Build a basic cluster report: top exemplars per cluster by score.
    assign = []
    # assignment by centroid
    # (reuse distance assignment by brute-force)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(routing.centroids * routing.centroids, axis=1, keepdims=True).T
    d2 = x2 - 2.0 * (X @ routing.centroids.T) + c2
    assign = np.argmin(d2, axis=1).astype(np.int64)

    clusters: dict[int, list[int]] = {}
    for i, c in enumerate(assign.tolist()):
        clusters.setdefault(int(c), []).append(i)

    report = {"n_patches": int(X.shape[0]), "n_clusters": int(routing.centroids.shape[0]), "clusters": []}
    for c in range(int(routing.centroids.shape[0])):
        idxs = clusters.get(c, [])
        # sort by patch_score desc
        idxs_sorted = sorted(idxs, key=lambda j: -float(defect_meta[j]["patch_score"]))
        exemplars = []
        for j in idxs_sorted[:10]:
            exemplars.append(defect_meta[j])
        report["clusters"].append({"cluster": int(c), "size": int(len(idxs)), "top_exemplars": exemplars})

    # Optionally fit a learned router over defect clusters.
    learned = None
    if bool(args.learned_router):
        learned = fit_linear_router(X, assign, iters=200, lr=0.1, l2=1.0, rng=np.random.default_rng(int(args.seed)))

    # Save artifact.
    np.save(out_dir / "defect_embeddings.npy", X.astype(np.float32, copy=False))
    (out_dir / "defect_metadata.json").write_text(json.dumps(defect_meta, indent=2))
    np.savez(
        out_dir / "defect_routing.npz",
        centroids=routing.centroids.astype(np.float32),
        members=np.array(routing.members, dtype=object),
        router_W=(learned.W.astype(np.float32) if learned is not None else None),
        router_b=(learned.b.astype(np.float32) if learned is not None else None),
    )
    (out_dir / "defect_cluster_report.json").write_text(json.dumps(report, indent=2))

    (out_dir / "defect_bank_info.json").write_text(
        json.dumps(
            {
                "kind": "defect_bank_unsupervised",
                "source": {"dataset": "mvtec", "root": str(Path(args.mvtec_root).resolve()), "category": str(args.category)},
                "model": str(Path(args.model).resolve()),
                "config": asdict(cfg),
                "top_k_patches": int(top_k),
                "clusters": int(args.clusters),
                "learned_router": bool(learned is not None),
                "note": "Cluster IDs are not labels; intended for human review + naming.",
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(str(out_dir))


if __name__ == "__main__":
    main()
