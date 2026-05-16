from __future__ import annotations

import numpy as np

from src.utils.io import MemoryBankEntry


def flatten_embeddings_with_metadata(
    emb: np.ndarray,
    paths: list[str],
    hw: tuple[int, int],
) -> tuple[np.ndarray, list[MemoryBankEntry]]:
    if emb.ndim != 3:
        raise ValueError(f"Expected [B,P,D] embeddings, got shape {emb.shape}")

    H, W = hw
    B, P, D = emb.shape
    if P != H * W:
        raise ValueError(f"Patches {P} != H*W {H*W}")
    if len(paths) != B:
        raise ValueError(f"Paths {len(paths)} != batch size {B}")

    meta: list[MemoryBankEntry] = []
    for batch_idx, path in enumerate(paths):
        for patch_index in range(P):
            row, col = divmod(patch_index, W)
            meta.append(
                MemoryBankEntry(
                    source_path=str(path),
                    patch_index=int(patch_index),
                    row=int(row),
                    col=int(col),
                    grid_h=int(H),
                    grid_w=int(W),
                )
            )

    return emb.reshape(B * P, D), meta
