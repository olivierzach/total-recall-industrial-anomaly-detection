#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.mvtec import MVTecADDataset
from src.patchcore import PatchCoreConfig
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.coreset import KCenterGreedy
from src.patchcore.embedding import patch_embeddings
from src.patchcore.metrics import auroc
from src.patchcore.patchcore import PatchCoreModel, to_numpy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", type=str, required=True)
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--out", type=str, default="outputs/mvtec_patchcore_result.json")
    args = ap.parse_args()

    cfg = PatchCoreConfig(layers=tuple(args.layers), image_size=int(args.image_size), coreset_ratio=float(args.coreset_ratio))

    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = MVTecADDataset(args.mvtec_root, args.category, "train", transform=tfm)
    test_ds = MVTecADDataset(args.mvtec_root, args.category, "test", transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers))
    test_dl = DataLoader(test_ds, batch_size=int(args.batch), shuffle=False, num_workers=int(args.num_workers))

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained).to(device)
    hooks = FeatureHooks(backbone, list(cfg.layers))

    # Build memory bank from nominal patches.
    nominal_patches = []
    with torch.no_grad():
        for batch in train_dl:
            x = batch.image.to(device)  # type: ignore
            _ = backbone(x)
            feats = hooks.pop()
            emb = patch_embeddings(feats, cfg.layers, l2_normalize=cfg.l2_normalize)  # [B,P,D]
            nominal_patches.append(to_numpy(emb.reshape(-1, emb.shape[-1])))

    X = np.concatenate(nominal_patches, axis=0)

    # Coreset selection.
    selector = KCenterGreedy()
    idx = selector.select(X, ratio=cfg.coreset_ratio)
    Xc = X[idx]

    model = PatchCoreModel.fit(cfg, Xc)

    # Evaluate image-level AUROC.
    y_true = []
    y_score = []

    with torch.no_grad():
        for batch in test_dl:
            x = batch.image.to(device)  # type: ignore
            labels = batch.label.numpy().astype(np.int64)  # type: ignore
            _ = backbone(x)
            feats = hooks.pop()
            emb = patch_embeddings(feats, cfg.layers, l2_normalize=cfg.l2_normalize)  # [B,P,D]
            emb = to_numpy(emb)
            for i in range(emb.shape[0]):
                score = model.score_image(emb[i])
                y_score.append(score)
            y_true.extend(list(labels))

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    image_auroc = auroc(y_true, y_score)

    out = {
        "cfg": asdict(cfg),
        "category": args.category,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "memory_bank": {"nominal_patches": int(X.shape[0]), "coreset": int(Xc.shape[0]), "ratio": cfg.coreset_ratio},
        "metrics": {"image_auroc": image_auroc},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
