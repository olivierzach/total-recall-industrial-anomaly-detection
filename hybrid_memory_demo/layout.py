from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hybrid_memory_demo.pipeline import iter_image_files


@dataclass(frozen=True)
class FolderHybridLayout:
    root: Path
    nominal_train: list[Path]
    nominal_calibration: list[Path]
    labeled_failures: dict[str, list[Path]]


def collect_failure_folders(root: str | Path) -> dict[str, list[Path]]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Failure root not found: {root_path}")

    out: dict[str, list[Path]] = {}
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        paths = iter_image_files(child)
        if paths:
            out[child.name] = paths
    return out


def load_folder_hybrid_layout(root: str | Path) -> FolderHybridLayout:
    root_path = Path(root)
    nominal_train = iter_image_files(root_path / "nominal" / "train")
    nominal_calibration = iter_image_files(root_path / "nominal" / "calibration")
    labeled_failures = collect_failure_folders(root_path / "failures")

    if not nominal_train:
        raise FileNotFoundError(f"No nominal training images found under {root_path / 'nominal' / 'train'}")
    if not nominal_calibration:
        raise FileNotFoundError(f"No nominal calibration images found under {root_path / 'nominal' / 'calibration'}")
    if not labeled_failures:
        raise FileNotFoundError(f"No labeled failure folders found under {root_path / 'failures'}")

    return FolderHybridLayout(
        root=root_path,
        nominal_train=nominal_train,
        nominal_calibration=nominal_calibration,
        labeled_failures=labeled_failures,
    )
