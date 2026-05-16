#!/usr/bin/env python3
"""Fetch / unpack MVTec AD.

MVTec AD is commonly distributed under terms that may require manual download.

This helper supports:
- downloading from a provided URL (e.g., internal mirror / signed URL), and
- extracting a local archive (.zip / .tar.*) into a target directory.

Examples:
  # Extract a locally downloaded archive
  python3 scripts/mvtec_get.py --archive data/raw/mvtec_ad.zip --out data/mvtec

  # Download from a URL then extract
  python3 scripts/mvtec_get.py --url "$MVTEC_URL" --out data/mvtec
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.archive import safe_extract_tar, safe_extract_zip


def download(url: str, out_path: Path) -> None:
    import urllib.request

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(out_path, "wb") as f:
        shutil.copyfileobj(r, f)


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(archive).endswith(".zip"):
        with zipfile.ZipFile(archive, "r") as z:
            safe_extract_zip(z, out_dir)
        return
    if any(str(archive).endswith(s) for s in [".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz"]):
        with tarfile.open(archive, "r:*") as t:
            safe_extract_tar(t, out_dir)
        return
    raise ValueError(f"Unknown archive type: {archive}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for extracted dataset")
    ap.add_argument("--url", default=None, help="Optional URL to download dataset archive from")
    ap.add_argument("--archive", default=None, help="Optional path to existing downloaded archive")
    ap.add_argument("--download-to", default="data/raw/mvtec_ad.zip", help="Where to save downloaded file")
    args = ap.parse_args()

    out_dir = Path(args.out)

    archive_path = Path(args.archive) if args.archive else None
    if args.url:
        archive_path = Path(args.download_to)
        print(f"Downloading to {archive_path} ...")
        download(str(args.url), archive_path)

    if archive_path is None:
        raise SystemExit("Provide --url or --archive")

    print(f"Extracting {archive_path} -> {out_dir}")
    extract(archive_path, out_dir)
    print("Done")


if __name__ == "__main__":
    main()
