from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MVTecItem:
    image: torch.Tensor
    label: int  # 0 normal, 1 anomalous
    mask: Optional[torch.Tensor]  # None for normal images or when masks unavailable
    path: str


class MVTecADDataset(Dataset):
    """Minimal MVTec AD loader.

    Expects MVTec directory structure:
      <root>/<category>/train/good/*.png
      <root>/<category>/test/<defect>/*.png
      <root>/<category>/ground_truth/<defect>/*.png

    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        transform: Callable[[Image.Image], torch.Tensor],
        mask_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        assert split in {"train", "test"}
        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        base = self.root / category / split
        if not base.exists():
            raise FileNotFoundError(f"MVTec path not found: {base}")

        items = []
        if split == "train":
            # nominal only (good)
            for p in sorted((base / "good").glob("*.png")):
                items.append((p, 0, None))
        else:
            gt_root = self.root / category / "ground_truth"
            for defect_dir in sorted(base.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect = defect_dir.name
                for p in sorted(defect_dir.glob("*.png")):
                    if defect == "good":
                        items.append((p, 0, None))
                    else:
                        # ground truth masks are pngs with same stem
                        m = gt_root / defect / (p.stem + "_mask.png")
                        if not m.exists():
                            # some versions use same stem .png
                            m = gt_root / defect / (p.stem + ".png")
                        items.append((p, 1, m if m.exists() else None))

        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> MVTecItem:
        p, label, mask_path = self._items[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        mask = None
        if mask_path is not None and self.mask_transform is not None:
            mimg = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mimg)
        return MVTecItem(image=x, label=int(label), mask=mask, path=str(p))
