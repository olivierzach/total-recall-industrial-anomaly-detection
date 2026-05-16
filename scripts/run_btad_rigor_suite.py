#!/usr/bin/env python3
"""Run a rigor-oriented BTAD suite.

Runs a small ablation matrix and writes a markdown + JSON report.

We calibrate thresholds using a held-out subset of nominal train images (no test leakage).

Example:
  .venv/bin/python scripts/run_btad_rigor_suite.py \
    --btad-root data/btad \
    --device mps \
    --layers layer3 \
    --coreset-ratio 0.01 \
    --outdir outputs/rigor/btad_suite
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(cmd: list[str]) -> dict:
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-4000:]}")
    return {"wall_s": time.time() - t0, "stdout_tail": p.stdout[-2000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btad-root", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--layers", nargs="+", default=["layer3"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--coreset-ratio", type=float, default=0.01)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--calib-fraction", type=float, default=0.2)
    ap.add_argument("--pca-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exe = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    if not Path(exe).exists():
        import sys

        exe = sys.executable

    variants = [
        ("knn_euclidean", ["--distance-metric", "euclidean"]),
        ("knn_cosine", ["--distance-metric", "cosine"]),
        ("knn_pca256_euclidean", ["--distance-metric", "euclidean", "--pca-dim", str(int(args.pca_dim))]),
    ]

    suite = {"btad_root": str(Path(args.btad_root).resolve()), "device": args.device, "variants": [v[0] for v in variants], "results": {}}

    for name, extra_flags in variants:
        out_path = outdir / f"{name}.json"
        cmd = [
            exe,
            "scripts/eval_btad_patchcore.py",
            "--btad-root",
            args.btad_root,
            "--device",
            args.device,
            "--batch",
            str(int(args.batch)),
            "--num-workers",
            str(int(args.num_workers)),
            "--coreset-ratio",
            str(float(args.coreset_ratio)),
            "--backbone",
            args.backbone,
            "--image-size",
            str(int(args.image_size)),
            "--seed",
            str(int(args.seed)),
            "--target-fpr",
            str(float(args.target_fpr)),
            "--calib-fraction",
            str(float(args.calib_fraction)),
            "--out",
            str(out_path),
        ]
        for layer in args.layers:
            cmd.extend(["--layers", layer])
        cmd.extend(extra_flags)
        meta = run(cmd)
        res = json.loads(out_path.read_text())
        res["_run"] = meta
        suite["results"][name] = res

    # markdown
    md = []
    md.append("# BTAD rigor suite\n")
    md.append(f"- btad_root: `{suite['btad_root']}`")
    md.append(f"- device: `{args.device}`")
    md.append(f"- backbone: `{args.backbone}`")
    md.append(f"- layers: `{args.layers}`")
    md.append(f"- coreset_ratio: `{args.coreset_ratio}`")
    md.append(f"- threshold: calibrated on held-out nominal train fraction `{args.calib_fraction}` to target FPR `{args.target_fpr}` (no test leakage).\n")

    md.append("## Results\n")
    md.append("Variant | Image AUROC | Pixel AUROC | PRO AUC | Recall@targetFPR | FPR@thr | Threshold | Total_s")
    md.append("---|---:|---:|---:|---:|---:|---:|---:")

    for v in suite["variants"]:
        r = suite["results"][v]
        m = r["metrics"]
        te = r.get("threshold_eval") or {}
        md.append(
            f"{v} | {m.get('image_auroc')} | {m.get('pixel_auroc')} | {m.get('pro_auc')} | {te.get('recall')} | {te.get('fpr')} | {te.get('threshold')} | {r.get('timing',{}).get('total_s')}"
        )

    report_md = outdir / "REPORT.md"
    report_json = outdir / "suite.json"
    report_md.write_text("\n".join(md) + "\n")
    report_json.write_text(json.dumps(suite, indent=2))
    print(str(report_md))


if __name__ == "__main__":
    main()
