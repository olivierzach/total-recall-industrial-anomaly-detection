from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


def _connected_components_4(mask: np.ndarray) -> List[np.ndarray]:
    """Return list of boolean masks for 4-connected components in a binary mask.

    mask: [H,W] uint8/bool, nonzero treated as foreground.

    This is a tiny dependency-free alternative to skimage/scipy.
    """

    H, W = mask.shape
    fg = mask.astype(bool)
    seen = np.zeros((H, W), dtype=bool)
    comps: List[np.ndarray] = []

    # BFS using Python list stack; OK for moderate masks.
    for y in range(H):
        for x in range(W):
            if not fg[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                if cy > 0 and fg[cy - 1, cx] and not seen[cy - 1, cx]:
                    seen[cy - 1, cx] = True
                    stack.append((cy - 1, cx))
                if cy + 1 < H and fg[cy + 1, cx] and not seen[cy + 1, cx]:
                    seen[cy + 1, cx] = True
                    stack.append((cy + 1, cx))
                if cx > 0 and fg[cy, cx - 1] and not seen[cy, cx - 1]:
                    seen[cy, cx - 1] = True
                    stack.append((cy, cx - 1))
                if cx + 1 < W and fg[cy, cx + 1] and not seen[cy, cx + 1]:
                    seen[cy, cx + 1] = True
                    stack.append((cy, cx + 1))

            cm = np.zeros((H, W), dtype=bool)
            yy, xx = zip(*pts)
            cm[np.array(yy), np.array(xx)] = True
            comps.append(cm)

    return comps


@dataclass(frozen=True)
class PROResult:
    pro_auc: float
    fpr: np.ndarray
    pro: np.ndarray


def compute_pro_auc(
    scores: List[np.ndarray],
    masks: List[np.ndarray],
    *,
    fpr_limit: float = 0.3,
    n_thresholds: int = 200,
) -> PROResult:
    """Compute PRO AUC for anomaly localization.

    This follows the spirit of MVTec-style PRO:
    - For each threshold, compute per-region overlap (PRO) on connected components of GT.
    - Compute false positive rate (FPR) on normal pixels (GT==0) across all images.
    - Return area under PRO vs FPR curve for FPR in [0, fpr_limit], normalized by fpr_limit.

    Inputs:
    - scores: list of float arrays [H,W] (anomaly score per pixel)
    - masks: list of uint8/bool arrays [H,W] (GT anomaly regions; 1=anomaly)

    Notes:
    - Images with empty masks contribute only to FPR.
    - This implementation is dependency-free and optimized for correctness/clarity.
    """

    assert len(scores) == len(masks)
    if not scores:
        return PROResult(pro_auc=float("nan"), fpr=np.array([]), pro=np.array([]))

    # Collect thresholds from score quantiles.
    all_scores = np.concatenate([s.reshape(-1) for s in scores]).astype(np.float64)
    qs = np.linspace(0.0, 1.0, n_thresholds)
    thr = np.quantile(all_scores, qs)

    # Precompute connected components for GT masks.
    regions: List[List[np.ndarray]] = []
    for m in masks:
        regions.append(_connected_components_4(m))

    fprs = []
    pros = []

    # Total normal pixels for FPR.
    total_normal = float(sum(int((m == 0).sum()) for m in masks))
    if total_normal <= 0:
        return PROResult(pro_auc=float("nan"), fpr=np.array([]), pro=np.array([]))

    for t in thr:
        fp = 0
        pro_vals = []
        for s, m, regs in zip(scores, masks, regions):
            pred = s >= t
            fp += int((pred & (m == 0)).sum())
            for r in regs:
                # overlap within region
                denom = int(r.sum())
                if denom == 0:
                    continue
                pro_vals.append(float((pred & r).sum()) / float(denom))

        fpr = fp / total_normal
        pro = float(np.mean(pro_vals)) if pro_vals else 0.0
        fprs.append(fpr)
        pros.append(pro)

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)

    # Sort by FPR.
    idx = np.argsort(fprs)
    fprs = fprs[idx]
    pros = pros[idx]

    # PRO curves often have repeated FPR values (many thresholds yield same FP count).
    # Compress by taking the *max* PRO achieved at each unique FPR.
    uniq_fprs = []
    uniq_pros = []
    i = 0
    while i < len(fprs):
        j = i
        f = fprs[i]
        pmax = pros[i]
        while j < len(fprs) and fprs[j] == f:
            if pros[j] > pmax:
                pmax = pros[j]
            j += 1
        uniq_fprs.append(float(f))
        uniq_pros.append(float(pmax))
        i = j

    fprs_u = np.asarray(uniq_fprs, dtype=np.float64)
    pros_u = np.asarray(uniq_pros, dtype=np.float64)

    # Clip to [0, fpr_limit].
    m = fprs_u <= float(fpr_limit)
    if not np.any(m):
        return PROResult(pro_auc=0.0, fpr=fprs_u, pro=pros_u)

    fpr_c = fprs_u[m]
    pro_c = pros_u[m]

    # Ensure starts at fpr=0.
    if fpr_c[0] > 0:
        fpr_c = np.concatenate([[0.0], fpr_c])
        pro_c = np.concatenate([[pro_c[0]], pro_c])

    # Ensure ends at fpr_limit by extending the last value (right-continuous step).
    if fpr_c[-1] < float(fpr_limit):
        fpr_c = np.concatenate([fpr_c, [float(fpr_limit)]])
        pro_c = np.concatenate([pro_c, [pro_c[-1]]])

    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    auc = float(integrate(pro_c, fpr_c)) / float(fpr_limit)
    return PROResult(pro_auc=auc, fpr=fprs_u, pro=pros_u)
