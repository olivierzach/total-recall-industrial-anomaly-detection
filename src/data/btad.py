from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class BTADItem:
    image: torch.Tensor
    label: int  # 0 ok, 1 ko
    mask: Optional[torch.Tensor]
    path: str


def _decode_supervisely_bitmap(obj_bitmap: dict) -> Image.Image:
    """Decode Supervisely bitmap to a PIL image.

    Supervisely format used by DatasetNinja stores `bitmap.data` as base64(zlib(png_bytes)).
    """

    raw = base64.b64decode(obj_bitmap["data"])
    png_bytes = zlib.decompress(raw)
    from io import BytesIO

    return Image.open(BytesIO(png_bytes))


def _render_mask(ann: dict) -> np.ndarray:
    """Render a binary mask from a Supervisely annotation JSON."""

    H = int(ann["size"]["height"])
    W = int(ann["size"]["width"])
    mask = np.zeros((H, W), dtype=np.uint8)

    for obj in ann.get("objects", []):
        bmp = obj.get("bitmap")
        if not bmp:
            continue
        origin = bmp.get("origin", [0, 0])
        oy, ox = int(origin[0]), int(origin[1])
        im = _decode_supervisely_bitmap(bmp).convert("L")
        arr = np.array(im)
        h, w = arr.shape[:2]

        # Any nonzero pixels are considered anomaly.
        y0, y1 = max(0, oy), min(H, oy + h)
        x0, x1 = max(0, ox), min(W, ox + w)
        if y1 <= y0 or x1 <= x0:
            continue
        sub = arr[(y0 - oy) : (y1 - oy), (x0 - ox) : (x1 - ox)]
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], (sub > 0).astype(np.uint8) * 255)

    return mask


class BTADDataset(Dataset):
    """BTAD loader for DatasetNinja/Supervisely export.

    Expected layout (after extraction):
      <root>/train/img/*
      <root>/train/ann/*.json
      <root>/test/img/*
      <root>/test/ann/*.json

    Filenames include `_ok_` and `_ko_`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        assert split in {"train", "test"}
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        img_dir = self.root / split / "img"
        ann_dir = self.root / split / "ann"
        if not img_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(f"BTAD missing dirs: {img_dir} {ann_dir}")

        self._items = []
        for p in sorted(img_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}:
                continue
            label = 1 if "_ko_" in p.name else 0
            ann = ann_dir / (p.name + ".json")
            self._items.append((p, label, ann if ann.exists() else None))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> BTADItem:
        p, label, ann_path = self._items[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)

        mask_t = None
        if label == 1 and ann_path is not None and self.mask_transform is not None:
            ann = json.loads(ann_path.read_text())
            mask = _render_mask(ann)
            mask_img = Image.fromarray(mask)
            mask_t = self.mask_transform(mask_img)

        return BTADItem(image=x, label=int(label), mask=mask_t, path=str(p))
