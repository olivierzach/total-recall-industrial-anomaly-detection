#!/usr/bin/env python3
"""Score images with a query-adaptive PatchCore model.

Requires a model produced by `fit_nominal_patchcore_routed.py`.

Supports two routing modes:
- patch: route each patch independently
- image: route once per image, then use that candidate patch set for all patches

Example:
  python3 scripts/score_images_routed.py \
    --model outputs/models/widgets_camA_routed \
    --images /data/new_images \
    --routing patch --probes 2 \
    --out outputs/routed_scores.jsonl
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

from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.patchcore import to_numpy
from src.patchcore.preprocess import pca_from_state
from src.patchcore.query_adaptive import QueryAdaptivePatchCore
from src.patchcore.learned_router import router_from_state
from src.patchcore.routing import RoutingIndex
from src.utils.io import load_patchcore
from src.utils.provenance import RUN_RESULT_SCHEMA_VERSION, build_run_manifest, dataset_reference, manifest_path_for_target, state_dict_sha256, write_run_manifest


def iter_images(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    if root.is_file() and root.suffix.lower() in exts:
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def load_routing(model_dir: Path) -> tuple[RoutingIndex, RoutingIndex, dict | None]:
    npz = np.load(model_dir / "routing_state.npz", allow_pickle=True)
    patch = RoutingIndex(mode="patch", centroids=npz["patch_centroids"].astype(np.float32), members=list(npz["patch_members"]))
    image = RoutingIndex(mode="image", centroids=npz["image_centroids"].astype(np.float32), members=list(npz["image_members"]))
    learned = None
    if "patch_router_W" in npz.files:
        learned = {
            "patch": {"W": npz["patch_router_W"].astype(np.float32), "b": npz["patch_router_b"].astype(np.float32)},
            "image": {"W": npz["image_router_W"].astype(np.float32), "b": npz["image_router_b"].astype(np.float32)},
        }
    return patch, image, learned


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--routing", choices=["patch", "image"], default="patch")
    ap.add_argument("--router", choices=["kmeans", "learned"], default="kmeans", help="How to choose clusters: distance-to-centroid or learned linear router")
    ap.add_argument("--probes", type=int, default=1)
    ap.add_argument("--out", default="outputs/routed_scores.jsonl")
    args = ap.parse_args()

    model_dir = Path(args.model)
    artifact = load_patchcore(model_dir)
    cfg = artifact.cfg

    pca = pca_from_state(artifact.pca_state) if artifact.pca_state is not None else None
    patch_routing, image_routing, learned_state = load_routing(model_dir)
    routing = patch_routing if args.routing == "patch" else image_routing

    learned_router = None
    if args.router == "learned":
        if learned_state is None:
            raise SystemExit("Model has no learned router; refit with --learned-router")
        learned_router = router_from_state(learned_state[args.routing])

    qapc = QueryAdaptivePatchCore(memory=artifact.memory_bank, routing=routing, metric=cfg.distance_metric, pca=pca)

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
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))

    images_root = Path(args.images)
    image_paths = list(iter_images(images_root))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_path_for_target(out_path)
    manifest = build_run_manifest(
        kind="score_images_routed",
        entrypoint="scripts/score_images_routed.py",
        requested_device=str(args.device),
        seed=artifact.seed,
        config=asdict(cfg),
        args=dict(vars(args)),
        datasets=[dataset_reference(dataset="image_folder", role="score_input", root=images_root if images_root.is_dir() else images_root.parent, files=image_paths, selection={"n_images": len(image_paths)})],
        inputs=[{"path": str(model_dir.resolve()), "role": "model", "kind": "patchcore_routed_model"}],
        extra={"routing": args.routing, "probes": int(args.probes), "backbone_sha256": state_dict_sha256(backbone.state_dict())},
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
                emb0 = to_numpy(emb[0])  # [P, D]

                if args.routing == "image":
                    img_emb = emb0.mean(axis=0, keepdims=True)
                    # candidates computed once; apply to all patches
                    if learned_router is None:
                        cand = qapc.candidate_indices_for_queries(img_emb, probes=int(args.probes))[0]
                    else:
                        # learned router chooses clusters directly
                        Xr = img_emb if pca is None else pca.transform(img_emb)
                        clusters = learned_router.topk(Xr, int(args.probes))[0]
                        cand = np.unique(np.concatenate([routing.members[int(c)] for c in clusters.tolist()]))
                    # brute force within this candidate set for every patch
                    # (compute distances patch-by-candidate)
                    # For now we do per patch knn using same qapc routing by temporarily overriding members.
                    # Simpler: call knn with probes=1 but routing members replaced.
                    # We'll approximate by calling qapc.knn on each patch (still okay for modest sizes).
                    # Use a temporary routing index containing only this candidate set.
                    tmp = QueryAdaptivePatchCore(memory=qapc.memory, routing=RoutingIndex(mode="patch", centroids=qapc.routing.centroids[:1], members=[cand]), metric=qapc.metric, pca=qapc.pca)
                    scores = tmp.score_patches(emb0, probes=1)
                else:
                    if learned_router is None:
                        scores = qapc.score_patches(emb0, probes=int(args.probes))
                    else:
                        # route each patch via learned router
                        Xr = emb0 if pca is None else pca.transform(emb0)
                        clusters = learned_router.topk(Xr, int(args.probes))  # [P, probes]
                        # build candidate sets per patch
                        dists = np.empty((Xr.shape[0],), dtype=np.float32)
                        for pi in range(Xr.shape[0]):
                            cand = np.unique(np.concatenate([routing.members[int(c)] for c in clusters[pi].tolist()]))
                            if cand.size == 0:
                                dists[pi] = np.inf
                                continue
                            # brute-force distances to candidates
                            Ym = artifact.memory_bank[cand]
                            # memory already in routed space if PCA enabled
                            diff = Ym - Xr[pi : pi + 1]
                            d2 = np.sum(diff * diff, axis=1)
                            dists[pi] = float(np.sqrt(max(0.0, float(np.min(d2)))))
                        scores = dists

                s_img = float(np.max(scores)) if cfg.image_score == "max" else float(np.mean(scores))

                rec = {
                    "path": str(p),
                    "score": s_img,
                    "routing": args.routing,
                    "router": args.router,
                    "probes": int(args.probes),
                    "run_id": manifest["run_id"],
                    "schema_version": RUN_RESULT_SCHEMA_VERSION,
                    "model_run_id": (artifact.artifact_info or {}).get("run_id"),
                }
                f.write(json.dumps(rec) + "\n")

    write_run_manifest(out_path, manifest)
    print(str(out_path))


if __name__ == "__main__":
    main()
