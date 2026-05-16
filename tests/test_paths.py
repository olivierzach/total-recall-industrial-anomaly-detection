from __future__ import annotations

from pathlib import Path

from src.utils.paths import derived_output_path


def test_derived_output_path_preserves_relative_structure():
    root = Path("/tmp/images")
    a = root / "lot_a" / "000.png"
    b = root / "lot_b" / "000.png"

    out_a = derived_output_path(root, a, ".anomaly_patchgrid.npy")
    out_b = derived_output_path(root, b, ".anomaly_patchgrid.npy")

    assert out_a == Path("lot_a") / "000.anomaly_patchgrid.npy"
    assert out_b == Path("lot_b") / "000.anomaly_patchgrid.npy"
    assert out_a != out_b


def test_derived_output_path_handles_single_file_root():
    image = Path("/tmp/images/000.png")
    out = derived_output_path(image, image, ".overlay.png")
    assert out == Path("000.overlay.png")
