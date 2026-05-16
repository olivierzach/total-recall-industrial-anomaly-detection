from __future__ import annotations

from typing import Tuple

import torch

from .backbone import FeatureHooks
from .embedding import patch_embeddings
from .vit_embedding import ViTEmbedder


def is_vit_backbone(name: str) -> bool:
    return name.startswith("vit_")


@torch.no_grad()
def extract_patch_embeddings(
    *,
    backbone_name: str,
    model,
    hooks: FeatureHooks | None,
    x: torch.Tensor,
    layers,
    l2_normalize: bool,
    return_hw: bool = False,
):
    """Unified embedding extraction for CNN and ViT backbones."""

    if is_vit_backbone(backbone_name):
        emb = ViTEmbedder(model).embed(x, l2_normalize=l2_normalize, return_hw=return_hw)
        return emb

    # CNN path
    assert hooks is not None
    _ = model(x)
    feats = hooks.pop()
    return patch_embeddings(feats, layers, l2_normalize=l2_normalize, return_hw=return_hw)
