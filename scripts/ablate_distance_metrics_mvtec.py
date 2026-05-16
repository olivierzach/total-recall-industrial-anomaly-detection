#!/usr/bin/env python3
"""Ablation: distance metric + PCA whitening for PatchCore on MVTec.

Runs a small set of configurations and writes a JSON summary.

Example:
  python3 scripts/ablate_distance_metrics_mvtec.py \
    --mvtec-root /path/to/mvtec \
    --category bottle \
    --device mps \
    --coreset-ratio 0.1 \
    --pca-dim 256 \
    --out outputs/ablations/distance_mvtec_bottle.json

Notes:
- This script shells out to `scripts/eval_mvtec_patchcore.py` for simplicity and parity.
- It is meant for quick comparisons, not perfect benchmarking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout
    # eval script prints a JSON-ish dict at end; we also write --out JSON.
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out[-4000:]}")
    return {"stdout_tail": out[-2000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    ap.add_argument("--backbone", type=str, default="wide_resnet50_2")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--layers", type=str, nargs="*", default=["layer2", "layer3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pca-dim", type=int, default=256, help="PCA dim used for the PCA-whiten variant")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import sys as _sys
    base = [
        _sys.executable,
        "scripts/eval_mvtec_patchcore.py",
        "--mvtec-root",
        str(args.mvtec_root),
        "--category",
        str(args.category),
        "--device",
        str(args.device),
        "--batch",
        str(args.batch),
        "--num-workers",
        str(args.num_workers),
        "--coreset-ratio",
        str(args.coreset_ratio),
        "--backbone",
        str(args.backbone),
        "--image-size",
        str(args.image_size),
        "--seed",
        str(args.seed),
    ]
    for layer in args.layers:
        base.extend(["--layers", layer])
    if args.max_train:
        base.extend(["--max-train", str(args.max_train)])
    if args.max_test:
        base.extend(["--max-test", str(args.max_test)])

    variants = [
        {
            "name": "knn_euclidean",
            "flags": ["--distance-metric", "euclidean"],
        },
        {
            "name": "knn_cosine",
            "flags": ["--distance-metric", "cosine"],
        },
        {
            "name": "knn_pca_whiten_euclidean",
            "flags": ["--distance-metric", "euclidean", "--pca-dim", str(args.pca_dim)],
        },
    ]

    results = []
    for v in variants:
        tmp = out_path.parent / f"{out_path.stem}.{v['name']}.json"
        cmd = base + ["--out", str(tmp)] + v["flags"]
        run_meta = run(cmd)
        rec = json.loads(tmp.read_text())
        rec["variant"] = v["name"]
        rec["_run"] = run_meta
        results.append(rec)

    out_path.write_text(json.dumps({"variants": results}, indent=2))
    print(str(out_path))


if __name__ == "__main__":
    main()
