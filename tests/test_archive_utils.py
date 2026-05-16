from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from src.utils.archive import safe_extract_tar, safe_extract_zip


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "nope")

    with zipfile.ZipFile(archive, "r") as zf:
        with pytest.raises(ValueError):
            safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path):
    archive = tmp_path / "bad.tar"
    data = b"nope"
    with tarfile.open(archive, "w") as tf:
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    with tarfile.open(archive, "r") as tf:
        with pytest.raises(ValueError):
            safe_extract_tar(tf, tmp_path / "out")
