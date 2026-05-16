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


def _parse_btad_name(path: str | Path) -> tuple[str, str]:
    stem = Path(path).stem
    tokens = stem.split("_")
    if len(tokens) < 2:
        raise ValueError(f"Unexpected BTAD filename: {path}")
    return tokens[0], tokens[1]


def _cap_nominal_train(paths: list[Path], *, cap: int, seed: int) -> list[Path]:
    if int(cap) <= 0 or len(paths) <= int(cap):
        return list(paths)

    by_component: dict[str, list[Path]] = {}
    for path in paths:
        component, _ = _parse_btad_name(path)
        by_component.setdefault(component, []).append(path)

    total = sum(len(rows) for rows in by_component.values())
    selected: list[Path] = []
    leftovers: list[Path] = []
    for idx, component in enumerate(sorted(by_component)):
        rows = list(by_component[component])
        target = max(1, round(int(cap) * len(rows) / max(total, 1)))
        chosen = sample_paths(rows, min(target, len(rows)), int(seed) + idx)
        chosen_set = {str(path) for path in chosen}
        selected.extend(chosen)
        leftovers.extend([path for path in rows if str(path) not in chosen_set])

    if len(selected) > int(cap):
        return sample_paths(selected, int(cap), int(seed))
    if len(selected) < int(cap):
        fill = sample_paths(leftovers, int(cap) - len(selected), int(seed) + 97)
        selected.extend(fill)
    return selected


def _collect_btad_splits(
    *,
    btad_root: str | Path,
    known_components: set[str],
    support_per_class: int,
    nominal_train_cap: int,
    seed: int,
) -> tuple[list[Path], dict[str, list[Path]], list[dict], list[str]]:
    root = Path(btad_root)
    nominal_train = [path for path in iter_image_files(root / "train" / "img") if _parse_btad_name(path)[1] == "ok"]
    nominal_train = _cap_nominal_train(nominal_train, cap=int(nominal_train_cap), seed=int(seed))
    test_paths = iter_image_files(root / "test" / "img")

    support_paths: dict[str, list[Path]] = {}
    eval_rows: list[dict] = []
    unknown_components: set[str] = set()

    by_component: dict[str, list[Path]] = {}
    for path in test_paths:
        component, status = _parse_btad_name(path)
        if status != "ko":
            eval_rows.append({"path": str(path), "ground_truth_status": "normal", "ground_truth_label": None})
            continue
        by_component.setdefault(component, []).append(path)

    for component, paths in sorted(by_component.items()):
        label = f"component_{component}"
        if component in known_components:
            support = sample_paths(paths, int(support_per_class), int(seed))
            support_paths[label] = support
            support_set = {str(path) for path in support}
            for path in paths:
                if str(path) in support_set:
                    continue
                eval_rows.append({"path": str(path), "ground_truth_status": "known_failure", "ground_truth_label": label})
        else:
            unknown_components.add(component)
            for path in paths:
                eval_rows.append({"path": str(path), "ground_truth_status": "unknown_anomaly", "ground_truth_label": label})

    return nominal_train, support_paths, eval_rows, sorted(unknown_components)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--btad-root", default="data/btad")
    ap.add_argument("--out", default="outputs/hybrid_memory_demo/btad_components_demo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support-per-class", type=int, default=8)
    ap.add_argument("--calibration-good", type=int, default=72)
    ap.add_argument("--nominal-train-cap", type=int, default=360)
    ap.add_argument("--known-components", nargs="*", default=["01", "02"])
    ap.add_argument("--backbone", default="wide_resnet50_2")
    ap.add_argument("--layers", nargs="*", default=["layer3"])
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--coreset-ratio", type=float, default=0.01)
    ap.add_argument("--example-count-per-group", type=int, default=12)
    args = ap.parse_args()

    known_components = set(args.known_components)
    nominal_train, support_paths, eval_rows, unknown_components = _collect_btad_splits(
        btad_root=args.btad_root,
        known_components=known_components,
        support_per_class=int(args.support_per_class),
        nominal_train_cap=int(args.nominal_train_cap),
        seed=int(args.seed),
    )
    try:
        nominal_fit, nominal_calibration = split_nominal_paths(
            nominal_train,
            calibration_count=int(args.calibration_good),
            seed=int(args.seed),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = HybridMemoryConfig(
        backbone=str(args.backbone),
        layers=tuple(args.layers),
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
            "dataset": "btad",
            "category": "components",
            "known_components": sorted(known_components),
            "held_out_unknown_components": unknown_components,
            "label_scheme": "component_conditioned_failure",
            "note": "BTAD exposes ok/ko labels rather than fine defect families. This demo uses component-conditioned failure memory labels.",
            "nominal_train_cap": int(args.nominal_train_cap),
            "anomaly_backbone_profile": "wrn50_layer3_btad_best_local_family",
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
        "dataset": "btad",
        "category": "components",
        "known_components": sorted(known_components),
        "held_out_unknown_components": unknown_components,
        "label_scheme": "component_conditioned_failure",
        "task_type": "open_set_support_family_retrieval",
        "primary_interpretation": "Use status and open-set rejection metrics as the main BTAD signal. Component labels are support families, not named defect semantics.",
        "n_eval": metrics["n_eval"],
        "status_accuracy": metrics["status_accuracy"],
        "normal_recall": metrics["normal_recall"],
        "known_failure_recall": metrics["known_failure_recall"],
        "unknown_anomaly_recall": metrics["unknown_anomaly_recall"],
        "known_label_accuracy": metrics["known_label_accuracy"],
        "known_label_accuracy_when_predicted_known": metrics["known_label_accuracy_when_predicted_known"],
        "novel_as_known_rate": metrics["novel_as_known_rate"],
        "normal_false_alarm_rate": metrics["normal_false_alarm_rate"],
        "nominal_train_cap": int(args.nominal_train_cap),
        "backbone": str(args.backbone),
        "layers": list(args.layers),
        "image_size": int(args.image_size),
        "coreset_ratio": float(args.coreset_ratio),
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
        "python hybrid_memory_demo/web.py "
        f"--manifest-json hybrid_memory_demo/demo_manifest.json "
        f"--device {args.device}"
    )


if __name__ == "__main__":
    main()
