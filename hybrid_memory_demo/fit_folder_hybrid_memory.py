from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_memory_demo.layout import load_folder_hybrid_layout
from hybrid_memory_demo.model import HybridMemoryConfig
from hybrid_memory_demo.pipeline import fit_hybrid_memory, save_artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Root with nominal/train, nominal/calibration, failures/<label>")
    ap.add_argument("--out", required=True, help="Output artifact directory")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--coreset-ratio", type=float, default=0.05)
    ap.add_argument("--anomaly-quantile", type=float, default=0.99)
    ap.add_argument("--known-failure-quantile", type=float, default=0.95)
    ap.add_argument("--classifier-neighbors", type=int, default=3)
    ap.add_argument("--failure-top-k-patches", type=int, default=8)
    args = ap.parse_args()

    layout = load_folder_hybrid_layout(args.data_root)
    cfg = HybridMemoryConfig(
        backbone=str(args.backbone),
        image_size=int(args.image_size),
        coreset_ratio=float(args.coreset_ratio),
        anomaly_quantile=float(args.anomaly_quantile),
        known_failure_quantile=float(args.known_failure_quantile),
        classifier_neighbors=int(args.classifier_neighbors),
        failure_top_k_patches=int(args.failure_top_k_patches),
    )

    artifact = fit_hybrid_memory(
        nominal_train_paths=layout.nominal_train,
        nominal_calibration_paths=layout.nominal_calibration,
        labeled_failure_paths=layout.labeled_failures,
        cfg=cfg,
        device=args.device,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        artifact_info={
            "dataset": "folder_hybrid_memory",
            "data_root": str(layout.root.resolve()),
            "known_failure_labels": sorted(layout.labeled_failures),
        },
    )
    save_artifact(args.out, artifact)

    summary = {
        "data_root": str(layout.root.resolve()),
        "out": str(Path(args.out).resolve()),
        "nominal_train_images": len(layout.nominal_train),
        "nominal_calibration_images": len(layout.nominal_calibration),
        "known_failure_labels": sorted(layout.labeled_failures),
        "known_failure_counts": {label: len(paths) for label, paths in sorted(layout.labeled_failures.items())},
        "anomaly_threshold": artifact.anomaly_threshold,
        "known_failure_threshold": artifact.known_failure_threshold,
        "margin_threshold": artifact.margin_threshold,
    }
    (Path(args.out) / "fit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
