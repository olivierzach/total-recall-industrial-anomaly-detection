from __future__ import annotations

import json
from pathlib import Path

from src.utils.experiments import choose_primary_metric, load_experiment_rows


def test_load_experiment_rows_handles_eval_and_summary(tmp_path: Path):
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "dataset": "btad",
                "cfg": {"backbone": "resnet18", "image_size": 256, "coreset_ratio": 0.001},
                "metrics": {"image_auroc": 0.9, "pixel_auroc": 0.8, "pro_auc": 0.3},
                "timing": {"total_s": 12.5},
                "threshold_eval": {"recall": 0.7},
            }
        )
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps([{"backbone": "vit_b_16", "pro_auc": 0.2, "total_s": 8.0}]))

    rows = load_experiment_rows([str(tmp_path)])

    assert len(rows) == 2
    assert any(row.get("dataset") == "btad" and row.get("threshold_recall") == 0.7 for row in rows)
    assert any(row.get("backbone") == "vit_b_16" for row in rows)


def test_choose_primary_metric_prefers_pro():
    metric = choose_primary_metric([{"image_auroc": 0.9}, {"pro_auc": 0.1}])
    assert metric == "pro_auc"
