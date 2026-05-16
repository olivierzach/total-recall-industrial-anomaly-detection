#!/usr/bin/env python3
"""Render qualitative examples from a labeled dataset split.

Examples:
  python3 scripts/review_dataset_examples.py \
    --dataset mvtec --root data/mvtec --category bottle \
    --model outputs/models/bottle --threshold outputs/threshold.json \
    --select fp --top-n 12 --outdir outputs/review/mvtec_bottle_fp

  python3 scripts/review_dataset_examples.py \
    --dataset btad --root data/btad \
    --model outputs/models/btad --select top_score --top-n 20 \
    --outdir outputs/review/btad_top
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.btad import BTADDataset
from src.data.collate import collate_batch
from src.data.mvtec import MVTecADDataset
from src.patchcore.backbone import FeatureHooks, load_backbone
from src.patchcore.extract import extract_patch_embeddings, is_vit_backbone
from src.patchcore.patchcore import PatchCoreModel, to_numpy
from src.utils.image_viz import bbox_from_binary_mask, bbox_from_score_map, draw_bboxes, overlay_heatmap, upsample_score_map
from src.utils.io import load_patchcore, load_threshold_artifact


def _build_dataset(args, cfg):
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
    if args.dataset == "mvtec":
        if not args.category:
            raise SystemExit("--category is required for mvtec")
        return MVTecADDataset(args.root, args.category, "test", transform=tfm, mask_transform=mtfm)
    if args.dataset == "btad":
        return BTADDataset(args.root, "test", transform=tfm, mask_transform=mtfm)
    raise ValueError(args.dataset)


def _select_records(records: list[dict[str, Any]], select: str, top_n: int) -> list[dict[str, Any]]:
    if select == "top_score":
        filtered = records
    else:
        filtered = [r for r in records if r.get("outcome") == select]
    filtered = sorted(filtered, key=lambda r: r["score"], reverse=True)
    return filtered[:top_n]


def _outcome(label: int, pred: int | None) -> str | None:
    if pred is None:
        return None
    if label == 1 and pred == 1:
        return "tp"
    if label == 0 and pred == 1:
        return "fp"
    if label == 1 and pred == 0:
        return "fn"
    return "tn"


def _render_panel(
    image: Image.Image,
    score_map: np.ndarray,
    *,
    path: str,
    score: float,
    label: int,
    pred: int | None,
    threshold: float | None,
    predicted_bbox: tuple[int, int, int, int] | None,
    gt_bbox: tuple[int, int, int, int] | None,
    gt_mask: np.ndarray | None,
    out_path: Path,
    alpha: float,
) -> None:
    overlay = overlay_heatmap(image, score_map, alpha=alpha)
    boxed = draw_bboxes(overlay, predicted_bbox=predicted_bbox, gt_bbox=gt_bbox)

    cols = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
    if cols == 2:
        axes = np.array(axes)

    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(boxed)
    axes[1].set_title("overlay + boxes")
    if gt_mask is not None:
        axes[2].imshow(gt_mask, cmap="gray")
        axes[2].set_title("gt mask")

    title = f"{Path(path).name} | score={score:.4f} | label={label}"
    if threshold is not None and pred is not None:
        title += f" | pred={pred} @ {threshold:.4f}"
    fig.suptitle(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvtec", "btad"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--category", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--threshold", default=None)
    ap.add_argument("--select", choices=["top_score", "tp", "fp", "fn", "tn"], default="top_score")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--bbox-quantile", type=float, default=0.97)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    artifact = load_patchcore(args.model)
    cfg = artifact.cfg
    threshold = load_threshold_artifact(args.threshold) if args.threshold else None

    device = torch.device(args.device)
    dataset = _build_dataset(args, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_batch,
    )

    backbone = load_backbone(cfg.backbone, pretrained=cfg.pretrained and artifact.backbone_state is None)
    if artifact.backbone_state is not None:
        backbone.load_state_dict(artifact.backbone_state)
    backbone = backbone.to(device)
    hooks = None if is_vit_backbone(cfg.backbone) else FeatureHooks(backbone, list(cfg.layers))
    model = PatchCoreModel.fit(cfg, artifact.memory_bank)

    records: list[dict[str, Any]] = []
    dataset_index = 0
    with torch.no_grad():
        for batch in dataloader:
            x = batch.image.to(device)  # type: ignore[attr-defined]
            labels = batch.label.numpy().astype(np.int64)  # type: ignore[attr-defined]
            emb, (H, W) = extract_patch_embeddings(
                backbone_name=cfg.backbone,
                model=backbone,
                hooks=hooks,
                x=x,
                layers=cfg.layers,
                l2_normalize=cfg.l2_normalize,
                return_hw=True,
            )
            emb_np = to_numpy(emb)
            for i in range(emb_np.shape[0]):
                score = model.score_image(emb_np[i])
                pred = None
                if threshold is not None:
                    pred = int(score >= threshold.threshold)
                records.append(
                    {
                        "dataset_index": dataset_index,
                        "path": batch.path[i],  # type: ignore[attr-defined]
                        "label": int(labels[i]),
                        "score": float(score),
                        "pred": pred,
                        "outcome": _outcome(int(labels[i]), pred),
                    }
                )
                dataset_index += 1

    selected = _select_records(records, args.select, int(args.top_n))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for rank, rec in enumerate(selected, start=1):
        item = dataset[rec["dataset_index"]]
        x = item.image.unsqueeze(0).to(device)
        with torch.no_grad():
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
        amap = model.score_map(emb0, (H, W))
        score_map = upsample_score_map(amap, (cfg.image_size, cfg.image_size))

        image = Image.open(rec["path"]).convert("RGB").resize((cfg.image_size, cfg.image_size), resample=Image.BILINEAR)
        gt_mask = None
        gt_bbox = None
        if item.mask is not None:
            gt_mask = (item.mask[0].cpu().numpy() > 0.5).astype(np.uint8)
            gt_bbox = bbox_from_binary_mask(gt_mask)
        pred_bbox = bbox_from_score_map(score_map, quantile=float(args.bbox_quantile))

        out_path = outdir / f"{rank:03d}_{Path(rec['path']).stem}.png"
        _render_panel(
            image,
            score_map,
            path=rec["path"],
            score=float(rec["score"]),
            label=int(rec["label"]),
            pred=rec["pred"],
            threshold=None if threshold is None else float(threshold.threshold),
            predicted_bbox=pred_bbox,
            gt_bbox=gt_bbox,
            gt_mask=gt_mask,
            out_path=out_path,
            alpha=float(args.alpha),
        )

        rec_out = dict(rec)
        rec_out["rendered"] = str(out_path)
        rec_out["predicted_bbox"] = pred_bbox
        rec_out["gt_bbox"] = gt_bbox
        manifest.append(rec_out)

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(manifest)} examples to {outdir}")


if __name__ == "__main__":
    main()
