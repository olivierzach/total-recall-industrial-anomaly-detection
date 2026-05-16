#!/usr/bin/env python3
"""Run a BTAD ablation matrix sequentially (stability-first).

This script exists because long parallel runs tend to get SIGKILL'd.
We run variants one-by-one and write a markdown summary.

Ablations covered:
- coreset_method: kcenter | random | kmeans
- distance_metric: euclidean | cosine
- pca: off | pca256

Example:
  .venv/bin/python scripts/run_btad_ablation_matrix.py \
    --btad-root data/btad --device mps --layers layer3 --coreset-ratio 0.01 \
    --outdir outputs/rigor/btad_matrix_layer3_c01
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout[-4000:]}")


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exe = str(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    if not Path(exe).exists():
        import sys

        exe = sys.executable

    coreset_methods = ["random", "kmeans", "kcenter"]
    distance_metrics = ["euclidean", "cosine"]
    pca_dims = [0, 256]

    results = []
    for cm in coreset_methods:
        for dm in distance_metrics:
            for pca_dim in pca_dims:
                name = f"cm={cm}__dm={dm}__pca={pca_dim if pca_dim else 'off'}"
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
                    "--coreset-method",
                    cm,
                    "--distance-metric",
                    dm,
                    "--target-fpr",
                    str(float(args.target_fpr)),
                    "--calib-fraction",
                    str(float(args.calib_fraction)),
                    "--seed",
                    str(int(args.seed)),
                    "--backbone",
                    args.backbone,
                    "--image-size",
                    str(int(args.image_size)),
                    "--out",
                    str(out_path),
                    "--log-every",
                    "0",
                ]
                for layer in args.layers:
                    cmd.extend(["--layers", layer])
                if pca_dim:
                    cmd.extend(["--pca-dim", str(int(pca_dim))])

                t0 = time.time()
                run(cmd)
                res = json.loads(out_path.read_text())
                res["_variant"] = {"coreset_method": cm, "distance_metric": dm, "pca_dim": int(pca_dim)}
                res["_wall_s"] = time.time() - t0
                results.append(res)

    # Write suite.json
    (outdir / "suite.json").write_text(json.dumps({"results": results}, indent=2))

    # Markdown report
    md = []
    md.append("# BTAD ablation matrix\n")
    md.append(f"- btad_root: `{Path(args.btad_root).resolve()}`")
    md.append(f"- device: `{args.device}`")
    md.append(f"- backbone: `{args.backbone}`")
    md.append(f"- layers: `{args.layers}`")
    md.append(f"- threshold calibration: holdout fraction `{args.calib_fraction}` to target FPR `{args.target_fpr}`\n")

    md.append("Variant | Image AUROC | Pixel AUROC | PRO AUC | Recall@targetFPR | Total_s | Coreset_s | Notes")
    md.append("---|---:|---:|---:|---:|---:|---:|---")
    for r in results:
        v = r["_variant"]
        m = r["metrics"]
        te = r.get("threshold_eval") or {}
        md.append(
            f"cm={v['coreset_method']} dm={v['distance_metric']} pca={v['pca_dim'] or 'off'} | "
            f"{m.get('image_auroc')} | {m.get('pixel_auroc')} | {m.get('pro_auc')} | {te.get('recall')} | "
            f"{r.get('timing',{}).get('total_s')} | {r.get('timing',{}).get('coreset_s')} | "
            + ("(kcenter is baseline; expect slow)" if v["coreset_method"] == "kcenter" else "")
        )

    (outdir / "REPORT.md").write_text("\n".join(md) + "\n")
    print(str(outdir / "REPORT.md"))


if __name__ == "__main__":
    main()
