#!/usr/bin/env python3
"""Download / unpack BTAD (BeanTech Anomaly Detection Dataset).

BTAD is often easier to access than MVTec. DatasetNinja provides a direct download
link in Supervisely project format.

This helper supports:
- default download from DatasetNinja/Supervisely hosting (no login), or
- user-provided URL/archive.

Example:
  python3 scripts/btad_get.py --out data/btad

  python3 scripts/btad_get.py --url <your_mirror_url> --out data/btad

"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.archive import safe_extract_tar

DEFAULT_URL = "https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjUyOV9CVEFEL2J0YWQtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAieUtGa2FWN2RRa3RRdzZIWEN5b0lEV3dDaGNNTGZpbzdRZG16ZW5pQ1dsVT0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22btad-DatasetNinja.tar%22"


def download(url: str, out_path: Path) -> None:
    """Download with resume support via curl if available."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer curl for resumable downloads.
    import subprocess

    curl = shutil.which("curl")
    if curl:
        # -L follow redirects, -C - resume, --fail for non-200.
        subprocess.check_call([curl, "-L", "-C", "-", "--fail", url, "-o", str(out_path)])
        return

    # Fallback to urllib (no resume).
    import urllib.request

    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as t:
        safe_extract_tar(t, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--url", default=None, help="Optional URL for BTAD tar")
    ap.add_argument("--archive", default=None, help="Optional local tar path")
    ap.add_argument("--download-to", default="data/raw/btad-DatasetNinja.tar", help="Where to save downloaded tar")
    args = ap.parse_args()

    out_dir = Path(args.out)
    archive = Path(args.archive) if args.archive else None

    if args.url is not None or archive is None:
        url = args.url or DEFAULT_URL
        archive = Path(args.download_to)
        print(f"Downloading BTAD from {url}\n -> {archive}")
        download(url, archive)

    assert archive is not None
    print(f"Extracting {archive} -> {out_dir}")
    extract_tar(archive, out_dir)
    print("Done")


if __name__ == "__main__":
    main()
