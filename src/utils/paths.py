from __future__ import annotations

from pathlib import Path


def derived_output_path(images_root: Path, image_path: Path, artifact_suffix: str) -> Path:
    if images_root == image_path or images_root.suffix:
        rel = Path(image_path.name)
    else:
        rel = image_path.relative_to(images_root)
    return rel.with_name(rel.stem + artifact_suffix)
