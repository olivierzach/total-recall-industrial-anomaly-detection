from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data.mvtec import MVTecADDataset


def _write_png(p: Path, size=(32, 32), value=0):
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(p)


def _write_mask(p: Path, size=(32, 32), value=0):
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.full((size[1], size[0]), value, dtype=np.uint8)
    Image.fromarray(arr).save(p)


def test_mvtec_minimal_structure(tmp_path: Path):
    root = tmp_path / "mvtec"
    cat = "bottle"

    # train good
    _write_png(root / cat / "train" / "good" / "000.png")
    _write_png(root / cat / "train" / "good" / "001.png")

    # test good + defect
    _write_png(root / cat / "test" / "good" / "000.png")
    _write_png(root / cat / "test" / "crack" / "000.png")
    _write_mask(root / cat / "ground_truth" / "crack" / "000_mask.png", value=255)

    tfm = lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mtfm = lambda img: torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

    train = MVTecADDataset(root, cat, "train", transform=tfm)
    assert len(train) == 2
    x0 = train[0]
    assert x0.label == 0
    assert x0.mask is None

    test = MVTecADDataset(root, cat, "test", transform=tfm, mask_transform=mtfm)
    assert len(test) == 2
    # Find defect
    defect = [it for it in test if it.label == 1][0]
    assert defect.mask is not None
    assert defect.mask.shape[0] == 1
