from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import numpy as np
from PIL import Image, ImageDraw


def upsample_score_map(amap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray(amap.astype(np.float32))
    im = im.resize(size, resample=Image.BILINEAR)
    return np.array(im).astype(np.float32)


def bbox_from_binary_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_from_score_map(score_map: np.ndarray, quantile: float = 0.97) -> tuple[int, int, int, int] | None:
    if score_map.size == 0:
        return None
    q = float(np.clip(quantile, 0.0, 1.0))
    thr = float(np.quantile(score_map, q))
    return bbox_from_binary_mask((score_map >= thr).astype(np.uint8))


def overlay_heatmap(image: Image.Image, score_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    score = score_map.astype(np.float32)
    if score.size == 0:
        return image.copy()
    smin = float(np.min(score))
    smax = float(np.max(score))
    if smax > smin:
        norm = (score - smin) / (smax - smin)
    else:
        norm = np.zeros_like(score, dtype=np.float32)
    rgba = colormaps["inferno"](norm)
    heat = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8), mode="RGB")
    return Image.blend(image.convert("RGB"), heat, alpha=float(alpha))


def draw_bboxes(
    image: Image.Image,
    *,
    predicted_bbox: tuple[int, int, int, int] | None = None,
    gt_bbox: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    if predicted_bbox is not None:
        draw.rectangle(predicted_bbox, outline=(255, 64, 64), width=4)
    if gt_bbox is not None:
        draw.rectangle(gt_bbox, outline=(64, 255, 64), width=4)
    return out
