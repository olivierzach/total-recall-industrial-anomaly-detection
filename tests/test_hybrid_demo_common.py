from hybrid_memory_demo.demo_common import compute_open_set_metrics


def test_compute_open_set_metrics_reports_open_set_recalls_and_rejection_rates():
    predictions = [
        {
            "ground_truth_status": "normal",
            "ground_truth_label": None,
            "prediction": {"status": "normal", "predicted_label": None},
        },
        {
            "ground_truth_status": "normal",
            "ground_truth_label": None,
            "prediction": {"status": "unknown_anomaly", "predicted_label": None},
        },
        {
            "ground_truth_status": "known_failure",
            "ground_truth_label": "family_a",
            "prediction": {"status": "known_failure", "predicted_label": "family_a"},
        },
        {
            "ground_truth_status": "known_failure",
            "ground_truth_label": "family_b",
            "prediction": {"status": "normal", "predicted_label": None},
        },
        {
            "ground_truth_status": "unknown_anomaly",
            "ground_truth_label": "family_c",
            "prediction": {"status": "known_failure", "predicted_label": "family_a"},
        },
        {
            "ground_truth_status": "unknown_anomaly",
            "ground_truth_label": "family_c",
            "prediction": {"status": "unknown_anomaly", "predicted_label": None},
        },
    ]

    metrics = compute_open_set_metrics(predictions)

    assert metrics["status_accuracy"] == 0.5
    assert metrics["normal_recall"] == 0.5
    assert metrics["known_failure_recall"] == 0.5
    assert metrics["unknown_anomaly_recall"] == 0.5
    assert metrics["known_label_accuracy"] == 0.5
    assert metrics["known_label_accuracy_when_predicted_known"] == 1.0
    assert metrics["novel_as_known_rate"] == 0.5
    assert metrics["normal_false_alarm_rate"] == 0.5
    assert metrics["n_eval"] == 6
