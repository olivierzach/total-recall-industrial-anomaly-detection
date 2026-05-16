#!/usr/bin/env python3
"""Score images by routing to one of multiple nominal PatchCore banks.

Use case: multi-product / multi-station deployments.

We:
- load multiple PatchCore models (each with its own memory bank)
- compute a lightweight *image embedding* (mean of patch embeddings)
- route the image to the closest bank centroid (cosine)
- score with that bank

This is a minimal demonstration. In production you'd likely route by explicit product ID
or maintain per-station banks.

Example (toy):
  .venv/bin/python scripts/score_images_multibank.py \
    --models bottle=outputs/models/bottle btad=outputs/models/btad_nominal \
    --images data/mvtec/bottle/test data/btad/test/img \
    --device mps \
    --out outputs/multibank_scores.jsonl
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
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.patchcore.preprocess import pca_from_state
from src.utils.io import load_patchcore


def iter_images(roots: list[Path]):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    for root in roots:
        if root.is_file() and root.suffix.lower() in exts:
            yield root
        else:
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() in exts:
                    yield p


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of NAME=PATH model dirs",
    )
    ap.add_argument("--images", nargs="+", required=True, help="One or more image roots")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/multibank_scores.jsonl")
    ap.add_argument("--routing", choices=["cosine"], default="cosine")
    args = ap.parse_args()

    banks = []
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"Bad --models entry: {spec}")
        name, path = spec.split("=", 1)
        art = load_patchcore(path)
        pca = pca_from_state(art.pca_state) if art.pca_state is not None else None
        model = PatchCoreModel.fit(art.cfg, art.memory_bank, pca=pca)
        # bank centroid: mean of memory vectors (already in same space as model)
        centroid = art.memory_bank.mean(axis=0)
        centroid = l2norm(centroid[None, :])[0]
        banks.append({"name": name, "artifact": art, "model": model, "centroid": centroid})

    # require all banks share backbone + layers + image_size for this simple demo
    cfg0 = banks[0]["artifact"].cfg
    for b in banks[1:]:
        cfg = b["artifact"].cfg
        if (cfg.backbone, cfg.layers, cfg.image_size) != (cfg0.backbone, cfg0.layers, cfg0.image_size):
            raise SystemExit("All banks must share backbone/layers/image_size for this demo")

    device = torch.device(args.device)
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg0.image_size, cfg0.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Use the backbone from the first bank (weights may differ; for this demo we assume pretrained frozen)
    backbone = load_backbone(cfg0.backbone, pretrained=cfg0.pretrained and banks[0]["artifact"].backbone_state is None)
    if banks[0]["artifact"].backbone_state is not None:
        backbone.load_state_dict(banks[0]["artifact"].backbone_state)
    backbone = backbone.to(device)
    hooks = None if is_vit_backbone(cfg0.backbone) else FeatureHooks(backbone, list(cfg0.layers))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img_roots = [Path(p) for p in args.images]
    paths = list(iter_images(img_roots))

    with out_path.open("w") as f:
        with torch.no_grad():
            for p in paths:
                img = Image.open(p).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                emb = extract_patch_embeddings(
                    backbone_name=cfg0.backbone,
                    model=backbone,
                    hooks=hooks,
                    x=x,
                    layers=cfg0.layers,
                    l2_normalize=cfg0.l2_normalize,
                    return_hw=False,
                )
                if isinstance(emb, tuple):
                    emb = emb[0]
                emb0 = to_numpy(emb[0])  # [P,D]
                img_emb = l2norm(emb0.mean(axis=0, keepdims=True))[0]

                # route
                sims = [float(np.dot(img_emb, b["centroid"])) for b in banks]
                bi = int(np.argmax(sims))
                bank = banks[bi]

                score = bank["model"].score_image(emb0)
                rec = {
                    "path": str(p),
                    "score": float(score),
                    "routed_bank": bank["name"],
                    "routing_sims": {banks[i]["name"]: sims[i] for i in range(len(banks))},
                }
                f.write(json.dumps(rec) + "\n")

    print(str(out_path))


if __name__ == "__main__":
    main()
