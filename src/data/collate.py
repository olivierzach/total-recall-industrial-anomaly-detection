from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, List, Tuple

import torch


def collate_batch(items: List[Any]) -> Any:
    """Collate function that supports our dataclass dataset items.

    Returns a simple object with attributes: image (Tensor[B,C,H,W]), label (Tensor[B]), mask (list[Tensor|None]), path (list[str]).
    """

    if not items:
        raise ValueError("empty batch")

    # Support either dicts or dataclasses with attributes.
    def get(it, k):
        if isinstance(it, dict):
            return it.get(k)
        return getattr(it, k)

    images = torch.stack([get(it, "image") for it in items], dim=0)
    labels = torch.tensor([int(get(it, "label")) for it in items], dtype=torch.int64)
    masks = [get(it, "mask") for it in items]
    paths = [str(get(it, "path")) for it in items]

    class Batch:
        pass

    b = Batch()
    b.image = images
    b.label = labels
    b.mask = masks
    b.path = paths
    return b
