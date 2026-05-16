from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_memory_demo.benchmark import (
    aggregate_results,
    build_fold,
    build_cached_predictor,
    evaluate_cached_predictor,
    extract_embedding_cache,
    load_mvtec_category_protocol,
    save_benchmark_report,
)
from hybrid_memory_demo.model import HybridMemoryConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", default="data/mvtec")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--out", default="outputs/hybrid_benchmark/mvtec_bottle")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration-good", type=int, default=40)
    ap.add_argument("--support-sizes", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--image-size", type=int, default=160)
    ap.add_argument("--coreset-ratio", type=float, default=0.05)
    args = ap.parse_args()

    nominal_train, nominal_calibration, defect_paths, good_test = load_mvtec_category_protocol(
        mvtec_root=args.mvtec_root,
        category=args.category,
        calibration_good=int(args.calibration_good),
        seed=int(args.seed),
    )

    cfg = HybridMemoryConfig(
        backbone=str(args.backbone),
        image_size=int(args.image_size),
        coreset_ratio=float(args.coreset_ratio),
    )

    all_paths = list(nominal_train) + list(nominal_calibration) + list(good_test)
    for paths in defect_paths.values():
        all_paths.extend(paths)
    unique_paths = sorted({str(path) for path in all_paths})

    cache = extract_embedding_cache(
        paths=unique_paths,
        cfg=cfg,
        device=args.device,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
    )

    baseline_predictor = build_cached_predictor(
        cache=cache,
        nominal_train=nominal_train,
        nominal_calibration=nominal_calibration,
        support_paths={},
        cfg=cfg,
        seed=int(args.seed),
    )

    runs = []
    unknown_classes = sorted(defect_paths)
    for unknown_class in unknown_classes:
        baseline_fold = build_fold(
            defect_paths=defect_paths,
            good_test=good_test,
            unknown_class=unknown_class,
            support_per_class=0,
        )
        baseline_metrics = evaluate_cached_predictor(baseline_predictor, cache, baseline_fold.eval_rows)
        runs.append(
            {
                "method": "baseline_nominal_only",
                "support_per_class": 0,
                "unknown_class": unknown_class,
                "known_classes": list(baseline_fold.known_classes),
                "metrics": baseline_metrics,
            }
        )

        for support_size in args.support_sizes:
            fold = build_fold(
                defect_paths=defect_paths,
                good_test=good_test,
                unknown_class=unknown_class,
                support_per_class=int(support_size),
            )
            predictor = build_cached_predictor(
                cache=cache,
                nominal_train=nominal_train,
                nominal_calibration=nominal_calibration,
                support_paths=fold.support_paths,
                cfg=cfg,
                seed=int(args.seed),
            )
            metrics = evaluate_cached_predictor(predictor, cache, fold.eval_rows)
            runs.append(
                {
                    "method": "hybrid_known_failure_bank",
                    "support_per_class": int(support_size),
                    "unknown_class": unknown_class,
                    "known_classes": list(fold.known_classes),
                    "metrics": metrics,
                }
            )

    summary = aggregate_results(runs)
    protocol = {
        "dataset": "mvtec",
        "category": args.category,
        "evaluation": "leave-one-defect-family-out",
        "unknown_classes": unknown_classes,
        "support_sizes": [int(x) for x in args.support_sizes],
        "nominal_train_images": len(nominal_train),
        "nominal_calibration_images": len(nominal_calibration),
        "good_test_images": len(good_test),
        "defect_class_sizes": {label: len(paths) for label, paths in sorted(defect_paths.items())},
        "config": {
            "backbone": cfg.backbone,
            "image_size": cfg.image_size,
            "coreset_ratio": cfg.coreset_ratio,
        },
    }

    save_benchmark_report(args.out, protocol=protocol, runs=runs, summary=summary)
    print(json.dumps({"out": str(Path(args.out).resolve()), "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
