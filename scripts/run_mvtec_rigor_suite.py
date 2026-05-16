#!/usr/bin/env python3
"""Run a rigor-oriented MVTec suite.

Includes:
1) leakage checks (exact sha256 overlap + dHash near-duplicates)
2) evaluation runs for a small ablation matrix
3) a markdown report summarizing results + caveats

This is intentionally conservative: we calibrate thresholds using a held-out subset
of nominal train images (no test leakage).

Example:
  .venv/bin/python scripts/run_mvtec_rigor_suite.py \
    --mvtec-root data/mvtec \
    --categories hazelnut cable screw zipper \
    --device mps \
    --coreset-ratio 0.01 \
    --layers layer3 \
    --outdir outputs/rigor/mvtec_suite
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
    ap.add_argument("--mvtec-root", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
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
    # fall back to current interpreter if venv doesn't exist
    if not Path(exe).exists():
        import sys

        exe = sys.executable

    variants = [
        ("knn_euclidean", ["--distance-metric", "euclidean"]),
        ("knn_cosine", ["--distance-metric", "cosine"]),
        ("knn_pca256_euclidean", ["--distance-metric", "euclidean", "--pca-dim", str(int(args.pca_dim))]),
    ]

    suite = {"mvtec_root": str(Path(args.mvtec_root).resolve()), "device": args.device, "categories": {}, "variants": [v[0] for v in variants]}

    for cat in args.categories:
        cat_dir = outdir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 1) leakage check
        leak_out = cat_dir / "leakage.json"
        leak_cmd = [exe, "scripts/leakage_check_mvtec.py", "--mvtec-root", args.mvtec_root, "--category", cat, "--out", str(leak_out)]
        leak_meta = run(leak_cmd)
        leak = json.loads(leak_out.read_text())

        # 2) eval variants
        results = {}
        for name, extra_flags in variants:
            out_path = cat_dir / f"{name}.json"
            cmd = [
                exe,
                "scripts/eval_mvtec_patchcore.py",
                "--mvtec-root",
                args.mvtec_root,
                "--category",
                cat,
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
            results[name] = res

        suite["categories"][cat] = {
            "leakage": leak,
            "leakage_run": leak_meta,
            "results": results,
        }

    # markdown report
    md = []
    md.append(f"# MVTec rigor suite\n")
    md.append(f"- mvtec_root: `{suite['mvtec_root']}`")
    md.append(f"- device: `{args.device}`")
    md.append(f"- backbone: `{args.backbone}`")
    md.append(f"- layers: `{args.layers}`")
    md.append(f"- coreset_ratio: `{args.coreset_ratio}`")
    md.append(f"- threshold: calibrated on held-out nominal train fraction `{args.calib_fraction}` to target FPR `{args.target_fpr}` (no test leakage).\n")

    md.append("## Summary table\n")
    md.append("Category | Variant | Image AUROC | Recall@targetFPR | Threshold | Notes")
    md.append("---|---|---:|---:|---:|---")

    for cat, rec in suite["categories"].items():
        for v in suite["variants"]:
            r = rec["results"][v]
            au = r["metrics"]["image_auroc"]
            te = r.get("threshold_eval")
            recall = te.get("recall") if te else None
            thr = te.get("threshold") if te else None
            md.append(f"{cat} | {v} | {au:.4f} | {recall if recall is not None else 'NA'} | {thr if thr is not None else 'NA'} | ")

    md.append("\n## Leakage checks\n")
    md.append("We check exact overlap by sha256 (should be empty) and near-duplicates by a coarse dHash. dHash can over-flag very similar images; treat it as a smoke test, not a proof of leakage.\n")

    for cat, rec in suite["categories"].items():
        leak = rec["leakage"]
        md.append(f"### {cat}\n")
        md.append(f"- train_images: {leak['counts']['train_images']}, test_images: {leak['counts']['test_images']}")
        md.append(f"- exact_overlaps: {len(leak['exact_overlaps'])}")
        md.append(f"- near_duplicates_recorded (dHash<= {leak['near_threshold']}): {len(leak['near_duplicates'])} (record cap)\n")

    report_md = outdir / "REPORT.md"
    report_json = outdir / "suite.json"
    report_md.write_text("\n".join(md) + "\n")
    report_json.write_text(json.dumps(suite, indent=2))
    print(str(report_md))


if __name__ == "__main__":
    main()
