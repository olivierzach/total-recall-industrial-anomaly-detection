from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_memory_demo.demo_common import (
    build_examples,
    compute_open_set_metrics,
    sample_paths,
    split_nominal_paths,
    write_demo_outputs,
)
from hybrid_memory_demo.model import HybridMemoryConfig
from hybrid_memory_demo.pipeline import HybridMemoryRuntime, fit_hybrid_memory, iter_image_files, save_artifact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec-root", default="data/mvtec")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--out", default="outputs/hybrid_memory_demo/mvtec_bottle")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support-per-class", type=int, default=4)
    ap.add_argument("--calibration-good", type=int, default=40)
    ap.add_argument("--known-classes", nargs="*", default=["broken_large", "broken_small"])
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--coreset-ratio", type=float, default=0.05)
    ap.add_argument("--example-count-per-group", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.mvtec_root) / args.category
    nominal_train = iter_image_files(root / "train" / "good")
    try:
        nominal_fit, nominal_calibration = split_nominal_paths(
            list(nominal_train),
            calibration_count=int(args.calibration_good),
            seed=int(args.seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    defect_dirs = sorted([p for p in (root / "test").iterdir() if p.is_dir() and p.name != "good"])
    known_classes = set(args.known_classes)
    support_paths: dict[str, list[Path]] = {}
    eval_rows: list[dict] = []

    for defect_dir in defect_dirs:
        images = iter_image_files(defect_dir)
        support = sample_paths(images, int(args.support_per_class), int(args.seed))
        support_set = {str(path) for path in support}
        if defect_dir.name in known_classes:
            query = [path for path in images if str(path) not in support_set]
            support_paths[defect_dir.name] = support
            for path in query:
                eval_rows.append({"path": str(path), "ground_truth_status": "known_failure", "ground_truth_label": defect_dir.name})
        else:
            for path in images:
                eval_rows.append({"path": str(path), "ground_truth_status": "unknown_anomaly", "ground_truth_label": defect_dir.name})

    for path in iter_image_files(root / "test" / "good"):
        eval_rows.append({"path": str(path), "ground_truth_status": "normal", "ground_truth_label": None})

    cfg = HybridMemoryConfig(
        backbone=str(args.backbone),
        image_size=int(args.image_size),
        coreset_ratio=float(args.coreset_ratio),
    )
    artifact = fit_hybrid_memory(
        nominal_train_paths=nominal_fit,
        nominal_calibration_paths=nominal_calibration,
        labeled_failure_paths=support_paths,
        cfg=cfg,
        device=args.device,
        batch_size=int(args.batch),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        artifact_info={
            "dataset": "mvtec",
            "category": args.category,
            "known_classes": sorted(known_classes),
            "held_out_unknown_classes": sorted([p.name for p in defect_dirs if p.name not in known_classes]),
        },
    )

    out_dir = Path(args.out)
    artifact_dir = out_dir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_artifact(artifact_dir, artifact)

    runtime = HybridMemoryRuntime(artifact, device=args.device)
    predictions = []
    for row in eval_rows:
        prediction, _ = runtime.predict_path(row["path"])
        predictions.append({**row, "prediction": prediction.__dict__})

    metrics = compute_open_set_metrics(predictions)
    examples = build_examples(
        predictions,
        example_count_per_group=int(args.example_count_per_group),
        seed=int(args.seed),
    )

    report = {
        "dataset": "mvtec",
        "category": args.category,
        "known_classes": sorted(known_classes),
        "held_out_unknown_classes": sorted([p.name for p in defect_dirs if p.name not in known_classes]),
        "task_type": "open_set_named_failure_retrieval",
        "primary_interpretation": "MVTec label accuracy is meaningful because the stored failure classes correspond to named defect families.",
        "n_eval": metrics["n_eval"],
        "status_accuracy": metrics["status_accuracy"],
        "normal_recall": metrics["normal_recall"],
        "known_failure_recall": metrics["known_failure_recall"],
        "unknown_anomaly_recall": metrics["unknown_anomaly_recall"],
        "known_label_accuracy": metrics["known_label_accuracy"],
        "known_label_accuracy_when_predicted_known": metrics["known_label_accuracy_when_predicted_known"],
        "novel_as_known_rate": metrics["novel_as_known_rate"],
        "normal_false_alarm_rate": metrics["normal_false_alarm_rate"],
        "confusion": metrics["confusion"],
        "artifact_dir": str(artifact_dir.resolve()),
    }
    write_demo_outputs(
        out_dir=out_dir,
        report=report,
        predictions=predictions,
        examples=examples,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print()
    print("Start the browser demo with:")
    print(
        f"python hybrid_memory_demo/web.py --artifact-dir {artifact_dir} "
        f"--examples-json {out_dir / 'examples.json'} --device {args.device}"
    )


if __name__ == "__main__":
    main()
