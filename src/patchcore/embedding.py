from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


def _resize_to(feat: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    # feat: [B,C,H,W]
    if feat.shape[-2:] == hw:
        return feat
    return F.interpolate(feat, size=hw, mode="bilinear", align_corners=False)


def patch_embeddings(
    features: Dict[str, torch.Tensor],
    layer_order: Iterable[str],
    *,
    l2_normalize: bool = True,
    return_hw: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[int, int]]:
    """Create patch embeddings by resizing features to a common grid and concatenating channels.

    Returns: embeddings [B, H*W, D]
    """

    feats: List[torch.Tensor] = []
    target_hw = None
    for name in layer_order:
        f = features[name]
        if target_hw is None:
            target_hw = f.shape[-2:]
        feats.append(f)

    assert target_hw is not None

    resized = [_resize_to(f, target_hw) for f in feats]
    cat = torch.cat(resized, dim=1)  # [B, sumC, H, W]

    B, C, H, W = cat.shape
    emb = cat.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, HW, C]

    if l2_normalize:
        emb = F.normalize(emb, p=2, dim=-1)

    if return_hw:
        return emb, (H, W)
    return emb
