from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


def sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    rng = random.Random(int(seed))
    rows = list(paths)
    rng.shuffle(rows)
    return rows[: min(int(count), len(rows))]


def split_nominal_paths(paths: list[Path], calibration_count: int, seed: int) -> tuple[list[Path], list[Path]]:
    if len(paths) <= int(calibration_count):
        raise ValueError("Not enough nominal images for the requested calibration split")
    rng = random.Random(int(seed))
    rows = list(paths)
    rng.shuffle(rows)
    calibration = rows[: int(calibration_count)]
    fit = rows[int(calibration_count) :]
    return fit, calibration


def select_demo_examples(rows: list[dict], *, status: str, count: int, seed: int) -> list[dict]:
    if status == "known_failure":
        matched = [
            row
            for row in rows
            if row["prediction"]["status"] == status and row["prediction"]["predicted_label"] == row["ground_truth_label"]
        ]
    else:
        matched = [row for row in rows if row["prediction"]["status"] == status]
    pool = matched if len(matched) >= int(count) else rows
    sampled_paths = sample_paths([Path(item["path"]) for item in pool], count, seed)
    sampled_set = {str(path) for path in sampled_paths}
    return [row for row in pool if row["path"] in sampled_set]


def compute_prediction_summary(predictions: list[dict]) -> tuple[float, float, list[dict]]:
    metrics = compute_open_set_metrics(predictions)
    return metrics["status_accuracy"], metrics["known_label_accuracy"], metrics["confusion"]


def compute_open_set_metrics(predictions: list[dict]) -> dict:
    confusion = Counter((row["ground_truth_status"], row["prediction"]["status"]) for row in predictions)
    status_accuracy = sum(int(row["ground_truth_status"] == row["prediction"]["status"]) for row in predictions) / max(len(predictions), 1)

    def _recall(status: str) -> float:
        subset = [row for row in predictions if row["ground_truth_status"] == status]
        return sum(int(row["prediction"]["status"] == status) for row in subset) / max(len(subset), 1)

    known_rows = [row for row in predictions if row["ground_truth_status"] == "known_failure"]
    predicted_known_rows = [row for row in known_rows if row["prediction"]["status"] == "known_failure"]
    known_label_accuracy = (
        sum(int(row["ground_truth_label"] == row["prediction"]["predicted_label"]) for row in known_rows) / len(known_rows)
        if known_rows
        else 0.0
    )
    known_label_accuracy_when_predicted_known = (
        sum(int(row["ground_truth_label"] == row["prediction"]["predicted_label"]) for row in predicted_known_rows)
        / max(len(predicted_known_rows), 1)
    )
    novel_as_known_rate = sum(
        int(row["prediction"]["status"] == "known_failure")
        for row in predictions
        if row["ground_truth_status"] == "unknown_anomaly"
    ) / max(sum(int(row["ground_truth_status"] == "unknown_anomaly") for row in predictions), 1)
    normal_false_alarm_rate = sum(
        int(row["prediction"]["status"] != "normal")
        for row in predictions
        if row["ground_truth_status"] == "normal"
    ) / max(sum(int(row["ground_truth_status"] == "normal") for row in predictions), 1)

    confusion_rows = [
        {
            "ground_truth_status": gt,
            "predicted_status": pred,
            "count": count,
        }
        for (gt, pred), count in sorted(confusion.items())
    ]
    return {
        "n_eval": len(predictions),
        "status_accuracy": status_accuracy,
        "normal_recall": _recall("normal"),
        "known_failure_recall": _recall("known_failure"),
        "unknown_anomaly_recall": _recall("unknown_anomaly"),
        "known_label_accuracy": known_label_accuracy,
        "known_label_accuracy_when_predicted_known": known_label_accuracy_when_predicted_known,
        "novel_as_known_rate": novel_as_known_rate,
        "normal_false_alarm_rate": normal_false_alarm_rate,
        "confusion": confusion_rows,
    }


def build_examples(predictions: list[dict], *, example_count_per_group: int, seed: int) -> list[dict]:
    examples = []
    grouped: dict[str, list[dict]] = {"normal": [], "known_failure": [], "unknown_anomaly": []}
    for row in predictions:
        grouped[row["ground_truth_status"]].append(row)
    for group_name, rows in grouped.items():
        selected_rows = select_demo_examples(
            rows,
            status=group_name,
            count=int(example_count_per_group),
            seed=int(seed),
        )
        for idx, row in enumerate(selected_rows):
            examples.append(
                {
                    "id": f"{group_name}_{idx}",
                    "path": row["path"],
                    "ground_truth_status": row["ground_truth_status"],
                    "ground_truth_label": row["ground_truth_label"],
                    "predicted_status": row["prediction"]["status"],
                    "predicted_label": row["prediction"]["predicted_label"],
                }
            )
    return examples


def write_demo_outputs(
    *,
    out_dir: str | Path,
    report: dict,
    predictions: list[dict],
    examples: list[dict],
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out_path / "predictions.json").write_text(json.dumps(predictions, indent=2, sort_keys=True))
    (out_path / "examples.json").write_text(json.dumps(examples, indent=2, sort_keys=True))
