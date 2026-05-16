import pytest

from hybrid_memory_demo.benchmark import EvalRow, aggregate_results, summary_to_markdown


def test_aggregate_results_groups_by_method_and_support():
    rows = [
        {
            "method": "baseline_nominal_only",
            "support_per_class": 0,
            "metrics": {
                "image_auroc": 0.9,
                "status_accuracy": 0.8,
                "normal_recall": 0.95,
                "known_failure_recall": 0.0,
                "unknown_anomaly_recall": 1.0,
                "known_label_accuracy": 0.0,
                "known_label_accuracy_when_predicted_known": 0.0,
                "novel_as_known_rate": 0.0,
                "normal_false_alarm_rate": 0.05,
                "confusion": [{"ground_truth_status": "normal", "predicted_status": "normal", "count": 10}],
            },
        },
        {
            "method": "baseline_nominal_only",
            "support_per_class": 0,
            "metrics": {
                "image_auroc": 1.0,
                "status_accuracy": 0.9,
                "normal_recall": 1.0,
                "known_failure_recall": 0.0,
                "unknown_anomaly_recall": 1.0,
                "known_label_accuracy": 0.0,
                "known_label_accuracy_when_predicted_known": 0.0,
                "novel_as_known_rate": 0.0,
                "normal_false_alarm_rate": 0.0,
                "confusion": [{"ground_truth_status": "normal", "predicted_status": "normal", "count": 9}],
            },
        },
    ]

    summary = aggregate_results(rows)

    assert len(summary) == 1
    assert summary[0]["support_per_class"] == 0
    assert summary[0]["image_auroc_mean"] == pytest.approx(0.95)
    assert summary[0]["status_accuracy_mean"] == pytest.approx(0.85)


def test_summary_to_markdown_contains_expected_headers():
    text = summary_to_markdown(
        [
            {
                "method": "hybrid_known_failure_bank",
                "support_per_class": 4,
                "image_auroc_mean": 0.95,
                "status_accuracy_mean": 0.8,
                "normal_recall_mean": 1.0,
                "known_failure_recall_mean": 0.7,
                "unknown_anomaly_recall_mean": 0.8,
                "known_label_accuracy_mean": 0.6,
                "novel_as_known_rate_mean": 0.2,
                "normal_false_alarm_rate_mean": 0.05,
            }
        ]
    )

    assert "| Method | Support/Class | Image AUROC |" in text
    assert "hybrid_known_failure_bank" in text
