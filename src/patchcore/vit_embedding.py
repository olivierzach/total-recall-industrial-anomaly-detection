from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTEmbedder:
    """Extract patch-token embeddings from a torchvision ViT.

    Torchvision ViT returns a classification output by default. Internally it produces
    token embeddings of shape [B, 1+N, D] (class token + patch tokens).

    This helper runs the forward pass up through the transformer encoder and returns
    patch tokens reshaped into a patch grid.

    Notes:
    - This is intended for `torchvision.models.vit_*`.
    - We currently return the **final encoder output** (post LayerNorm) tokens.
    - Future extension: capture intermediate block outputs.
    """

    model: nn.Module

    def __post_init__(self):
        if not hasattr(self.model, "encoder") or not hasattr(self.model, "conv_proj"):
            raise TypeError("ViTEmbedder expects a torchvision ViT model with .encoder and .conv_proj")

    def patch_grid_hw(self, image_size: int) -> tuple[int, int]:
        # torchvision ViT patch embed uses conv_proj with kernel_size=stride=patch_size.
        conv: nn.Conv2d = getattr(self.model, "conv_proj")
        ps = conv.kernel_size[0]
        if image_size % ps != 0:
            raise ValueError(f"image_size={image_size} not divisible by patch_size={ps}")
        h = image_size // ps
        w = image_size // ps
        return h, w

    @torch.no_grad()
    def embed(self, x: torch.Tensor, *, l2_normalize: bool = True, return_hw: bool = False):
        """Return patch embeddings for a batch.

        x: [B,3,H,W] where H=W=image_size
        returns: [B, P, D] (and optionally (Hpatch, Wpatch))
        """

        # Access torchvision internal method if present.
        if hasattr(self.model, "_process_input"):
            # Produces patch embeddings [B, N, D]
            x_p = self.model._process_input(x)  # type: ignore[attr-defined]
        else:
            # Fallback: conv_proj + reshape.
            x_p = self.model.conv_proj(x)  # [B, D, H', W']
            x_p = x_p.flatten(2).transpose(1, 2)  # [B, N, D]

        B, N, D = x_p.shape

        # Add class token.
        cls_token = self.model.class_token.expand(B, -1, -1)  # type: ignore[attr-defined]
        x_t = torch.cat([cls_token, x_p], dim=1)  # [B, 1+N, D]

        # Position embedding.
        x_t = x_t + self.model.encoder.pos_embedding  # type: ignore[attr-defined]
        x_t = self.model.encoder.dropout(x_t)  # type: ignore[attr-defined]

        # Encoder blocks.
        x_t = self.model.encoder.layers(x_t)  # type: ignore[attr-defined]
        x_t = self.model.encoder.ln(x_t)  # type: ignore[attr-defined]

        patch_tokens = x_t[:, 1:, :]  # [B, N, D]

        if l2_normalize:
            patch_tokens = F.normalize(patch_tokens, p=2, dim=-1)

        if return_hw:
            # infer patch grid from conv_proj
            conv: nn.Conv2d = getattr(self.model, "conv_proj")
            ps = conv.kernel_size[0]
            H, W = x.shape[-2:]
            return patch_tokens, (H // ps, W // ps)

        return patch_tokens
