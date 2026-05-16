#!/usr/bin/env python3
"""Run a small PatchCore sweep on BTAD and write a summary table.

This is intentionally simple and reliable: it shells out to eval_btad_patchcore.py
so each run is isolated and produces a single JSON artifact.

Example:
  python3 scripts/sweep_btad.py --btad-root data/btad --device mps --outdir outputs/sweeps/btad_20260311

Tip: set --max-train/--max-test for fast iteration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    backbone: str
    image_size: int
    coreset_ratio: float
    num_neighbors: int
    image_score: str
    l2_normalize: bool


def run_one(btad_root: str, device: str, batch: int, num_workers: int, max_train: int, max_test: int, outdir: Path, spec: RunSpec) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"btad_{spec.backbone}_img{spec.image_size}"
        f"_coreset{spec.coreset_ratio:g}_k{spec.num_neighbors}"
        f"_{spec.image_score}_l2{int(spec.l2_normalize)}_{device}"
    )
    out_path = outdir / f"{tag}.json"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    cmd = [
        sys.executable,
        "scripts/eval_btad_patchcore.py",
        "--btad-root",
        btad_root,
        "--device",
        device,
        "--batch",
        str(batch),
        "--num-workers",
        str(num_workers),
        "--coreset-ratio",
        str(spec.coreset_ratio),
        "--backbone",
        spec.backbone,
        "--image-size",
        str(spec.image_size),
        "--num-neighbors",
        str(spec.num_neighbors),
        "--image-score",
        spec.image_score,
        "--out",
        str(out_path),
    ]
    if not spec.l2_normalize:
        cmd.append("--no-l2-normalize")

    if max_train:
        cmd += ["--max-train", str(max_train)]
    if max_test:
        cmd += ["--max-test", str(max_test)]

    t0 = time.time()
    subprocess.run(cmd, check=True)
    dt = time.time() - t0

    # Stamp wallclock into the json (non-authoritative; eval script also has timing breakdown).
    try:
        o = json.loads(out_path.read_text())
        o.setdefault("sweep", {})
        o["sweep"]["wall_s"] = dt
        out_path.write_text(json.dumps(o, indent=2, sort_keys=True) + "\n")
    except Exception:
        pass

    return out_path


def load_metrics(p: Path):
    o = json.loads(p.read_text())
    m = o.get("metrics", {})
    t = o.get("timing", {})
    cfg = o.get("cfg", {})
    return {
        "file": p.name,
        "backbone": cfg.get("backbone"),
        "image_size": cfg.get("image_size"),
        "coreset_ratio": cfg.get("coreset_ratio"),
        "num_neighbors": cfg.get("num_neighbors"),
        "image_score": cfg.get("image_score"),
        "l2_normalize": cfg.get("l2_normalize"),
        "image_auroc": m.get("image_auroc"),
        "pixel_auroc": m.get("pixel_auroc"),
        "pro_auc": m.get("pro_auc"),
        "total_s": t.get("total_s"),
        "feature_train_s": t.get("feature_train_s"),
        "feature_test_s": t.get("feature_test_s"),
        "metric_s": t.get("metric_s"),
    }


def write_markdown(rows: list[dict], out_path: Path) -> None:
    cols = [
        "backbone",
        "image_size",
        "coreset_ratio",
        "num_neighbors",
        "image_score",
        "l2_normalize",
        "image_auroc",
        "pixel_auroc",
        "pro_auc",
        "total_s",
    ]

    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            return f"{v:.4g}" if abs(v) >= 1 else f"{v:.4f}"
        return str(v)

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btad-root", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0)

    ap.add_argument("--backbone", action="append", default=[], help="repeatable (if omitted, uses a small default set)")
    ap.add_argument("--image-size", action="append", type=int, default=[], help="repeatable (if omitted, uses a small default set)")
    ap.add_argument("--coreset-ratio", action="append", type=float, default=[], help="repeatable (if omitted, uses a small default set)")
    ap.add_argument("--num-neighbors", action="append", type=int, default=[1, 5], help="repeatable")
    ap.add_argument("--image-score", action="append", default=["max", "mean"], help="repeatable")
    ap.add_argument("--l2-normalize", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    backbones = args.backbone or ["vit_b_16", "wide_resnet50_2"]
    image_sizes = args.image_size or [224, 256]
    coreset_ratios = args.coreset_ratio or [0.0005, 0.001]

    # If user specified multiple l2 settings, they'd run separately by invoking script twice.
    specs = [
        RunSpec(
            backbone=bb,
            image_size=sz,
            coreset_ratio=cr,
            num_neighbors=k,
            image_score=im,
            l2_normalize=bool(args.l2_normalize),
        )
        for (bb, sz, cr, k, im) in itertools.product(
            backbones,
            image_sizes,
            coreset_ratios,
            args.num_neighbors,
            args.image_score,
        )
    ]

    results: list[Path] = []
    for i, spec in enumerate(specs, start=1):
        print(f"[{i}/{len(specs)}] {spec}")
        p = run_one(
            btad_root=args.btad_root,
            device=args.device,
            batch=int(args.batch),
            num_workers=int(args.num_workers),
            max_train=int(args.max_train),
            max_test=int(args.max_test),
            outdir=outdir,
            spec=spec,
        )
        results.append(p)

    rows = [load_metrics(p) for p in results]
    # Sort primarily by PRO (then image AUROC).
    rows.sort(key=lambda r: (-(r.get("pro_auc") or float("-inf")), -(r.get("image_auroc") or float("-inf"))))

    (outdir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_markdown(rows, outdir / "summary.md")
    print(f"Wrote {outdir/'summary.md'}")


if __name__ == "__main__":
    main()
