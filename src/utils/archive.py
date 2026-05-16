from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def _resolved_member_path(out_dir: Path, member_name: str) -> Path:
    target = (out_dir / member_name).resolve()
    try:
        target.relative_to(out_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive member escapes output directory: {member_name}") from exc
    return target


def safe_extract_tar(tar: tarfile.TarFile, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for member in tar.getmembers():
        _resolved_member_path(out_dir, member.name)
        if member.issym() or member.islnk():
            raise ValueError(f"Refusing to extract link from archive: {member.name}")
    tar.extractall(out_dir)


def safe_extract_zip(zip_file: zipfile.ZipFile, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for member in zip_file.infolist():
        _resolved_member_path(out_dir, member.filename)
    zip_file.extractall(out_dir)
