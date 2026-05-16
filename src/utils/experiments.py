from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _iter_json_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.rglob("*.json")))
        elif p.suffix.lower() == ".json":
            paths.append(p)
    return paths


def _flatten_eval_run(obj: dict[str, Any], source: Path) -> dict[str, Any]:
    cfg = obj.get("cfg", {})
    metrics = obj.get("metrics", {})
    timing = obj.get("timing", {})
    memory_bank = obj.get("memory_bank", {})
    threshold_eval = obj.get("threshold_eval", {})
    row = {
        "source": str(source),
        "file": source.name,
        "dataset": obj.get("dataset"),
        "category": obj.get("category"),
        "seed": obj.get("seed"),
        "backbone": cfg.get("backbone"),
        "image_size": cfg.get("image_size"),
        "coreset_ratio": cfg.get("coreset_ratio"),
        "num_neighbors": cfg.get("num_neighbors"),
        "image_score": cfg.get("image_score"),
        "l2_normalize": cfg.get("l2_normalize"),
        "image_auroc": metrics.get("image_auroc"),
        "pixel_auroc": metrics.get("pixel_auroc"),
        "pro_auc": metrics.get("pro_auc"),
        "nominal_patches": memory_bank.get("nominal_patches"),
        "coreset": memory_bank.get("coreset"),
        "total_s": timing.get("total_s"),
        "feature_train_s": timing.get("feature_train_s"),
        "feature_test_s": timing.get("feature_test_s"),
        "metric_s": timing.get("metric_s"),
    }
    for key, value in threshold_eval.items():
        row[f"threshold_{key}"] = value
    return row


def load_experiment_rows(inputs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _iter_json_paths(inputs):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue

        if isinstance(obj, dict) and "metrics" in obj:
            rows.append(_flatten_eval_run(obj, path))
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            for rec in obj:
                rec = dict(rec)
                rec.setdefault("source", str(path))
                rec.setdefault("file", path.name)
                rows.append(rec)
    return rows


def choose_primary_metric(rows: list[dict[str, Any]]) -> str:
    for key in ["pro_auc", "pixel_auroc", "image_auroc", "threshold_recall"]:
        if any(row.get(key) is not None for row in rows):
            return key
    raise ValueError("No supported metric columns found")
