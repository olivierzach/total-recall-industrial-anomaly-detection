from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    # roc_auc_score expects both classes present
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)
    y_pred = (y_score >= float(threshold)).astype(np.int64)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    pos = tp + fn
    neg = tn + fp
    pred_pos = tp + fp
    total = len(y_true)

    recall = float(tp / pos) if pos else float("nan")
    precision = float(tp / pred_pos) if pred_pos else float("nan")
    fpr = float(fp / neg) if neg else float("nan")
    specificity = float(tn / neg) if neg else float("nan")
    accuracy = float((tp + tn) / total) if total else float("nan")
    alert_rate = float(pred_pos / total) if total else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "specificity": specificity,
        "accuracy": accuracy,
        "alert_rate": alert_rate,
    }
