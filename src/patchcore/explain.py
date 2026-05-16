from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.io import MemoryBankEntry


def build_patch_explanations(
    distances: np.ndarray,
    indices: np.ndarray,
    hw: tuple[int, int],
    *,
    memory_metadata: list[MemoryBankEntry] | None,
    top_k_patches: int,
) -> list[dict[str, Any]]:
    if distances.ndim != 2 or indices.ndim != 2:
        raise ValueError("distances and indices must be rank-2 arrays")
    if distances.shape != indices.shape:
        raise ValueError("distances and indices must have the same shape")
    if top_k_patches <= 0 or distances.shape[0] == 0:
        return []

    _, W = hw
    patch_scores = distances[:, 0]
    patch_order = np.argsort(-patch_scores)[:top_k_patches]

    out: list[dict[str, Any]] = []
    for rank, patch_idx in enumerate(patch_order, start=1):
        row, col = divmod(int(patch_idx), W)
        neighbors = []
        for nn_rank, (dist, mem_idx) in enumerate(zip(distances[patch_idx], indices[patch_idx]), start=1):
            rec: dict[str, Any] = {
                "rank": nn_rank,
                "memory_index": int(mem_idx),
                "distance": float(dist),
            }
            if memory_metadata is not None and 0 <= int(mem_idx) < len(memory_metadata):
                meta = memory_metadata[int(mem_idx)]
                rec.update(
                    {
                        "source_path": meta.source_path,
                        "source_patch_index": meta.patch_index,
                        "source_patch_row": meta.row,
                        "source_patch_col": meta.col,
                        "source_grid_h": meta.grid_h,
                        "source_grid_w": meta.grid_w,
                    }
                )
            neighbors.append(rec)

        out.append(
            {
                "rank": rank,
                "patch_index": int(patch_idx),
                "patch_row": row,
                "patch_col": col,
                "score": float(patch_scores[patch_idx]),
                "neighbors": neighbors,
            }
        )

    return out
