#!/usr/bin/env python3
"""Fetch nominal images from the public web for a query.

⚠️ Important:
- Only use this for datasets you have the right to download/use.
- Many sources (including Google Images) have terms that restrict automated scraping.
- Prefer permissive sources (Wikimedia Commons, manufacturer press kits, Unsplash/Pexels APIs).

This script intentionally does NOT scrape Google Images HTML.
Instead it:
- searches the web for pages using Brave Search (via the `brave` HTTP API if key provided), OR
- accepts a list of page URLs, then
- extracts candidate <img src> URLs from those pages and downloads them.

Because OpenClaw already has a web_search tool in-chat, the intended workflow is:
1) Use web_search in chat to find a few relevant pages.
2) Run this script with --pages pointing at those URLs.

Example:
  python3 scripts/fetch_nominal_web_images.py \
    --pages https://example.com/product-gallery https://example.com/blog \
    --out data/web_nominal/my_product --limit 300

"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


class ImgParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.img_srcs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "img":
            return
        attrs = dict(attrs)
        src = attrs.get("src") or attrs.get("data-src") or attrs.get("data-original")
        if src:
            self.img_srcs.append(src)


def fetch_url(url: str, timeout: int = 20) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def normalize_url(base: str, src: str) -> str:
    return urllib.parse.urljoin(base, src)


def is_image_url(u: str) -> bool:
    u = u.split("?")[0].lower()
    return any(u.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"])


def download_image(url: str, out_dir: Path) -> Path | None:
    try:
        data = fetch_url(url, timeout=30)
    except Exception:
        return None

    # Dedup by hash.
    h = hashlib.sha256(data).hexdigest()[:16]

    # Guess extension.
    path_part = urllib.parse.urlparse(url).path
    ext = Path(path_part).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        ext = ".jpg"

    out = out_dir / f"{h}{ext}"
    if out.exists():
        return out
    out.write_bytes(data)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", nargs="*", default=[], help="Web pages to scrape <img> tags from")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--min-bytes", type=int, default=10_000, help="Skip tiny images")
    ap.add_argument("--sleep-ms", type=int, default=100, help="Politeness delay between downloads")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidate image URLs.
    img_urls: list[str] = []
    for page in args.pages:
        try:
            html = fetch_url(page).decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"WARN failed to fetch page {page}: {e}")
            continue
        p = ImgParser()
        p.feed(html)
        for src in p.img_srcs:
            u = normalize_url(page, src)
            if u.startswith("data:"):
                continue
            img_urls.append(u)

    # Prefer direct image URLs.
    direct = [u for u in img_urls if is_image_url(u)]
    others = [u for u in img_urls if u not in direct]
    img_urls = direct + others

    seen = set()
    kept = 0
    for u in img_urls:
        if kept >= int(args.limit):
            break
        if u in seen:
            continue
        seen.add(u)
        p = download_image(u, out_dir)
        if p is None:
            continue
        if p.stat().st_size < int(args.min_bytes):
            try:
                p.unlink()
            except Exception:
                pass
            continue
        kept += 1
        if args.sleep_ms:
            time.sleep(int(args.sleep_ms) / 1000.0)

    print(f"Downloaded {kept} images to {out_dir}")


if __name__ == "__main__":
    main()
