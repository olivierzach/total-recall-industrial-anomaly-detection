from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_memory_demo.layout import collect_failure_folders, load_folder_hybrid_layout


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def test_collect_failure_folders(tmp_path: Path):
    _write_rgb(tmp_path / "failures" / "crack" / "a.png")
    _write_rgb(tmp_path / "failures" / "scratch" / "b.png")

    out = collect_failure_folders(tmp_path / "failures")

    assert sorted(out) == ["crack", "scratch"]
    assert len(out["crack"]) == 1


def test_load_folder_hybrid_layout(tmp_path: Path):
    _write_rgb(tmp_path / "nominal" / "train" / "000.png")
    _write_rgb(tmp_path / "nominal" / "calibration" / "001.png")
    _write_rgb(tmp_path / "failures" / "dent" / "a.png")

    layout = load_folder_hybrid_layout(tmp_path)

    assert len(layout.nominal_train) == 1
    assert len(layout.nominal_calibration) == 1
    assert list(layout.labeled_failures) == ["dent"]
