from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HybridMemoryConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 256
    l2_normalize: bool = True
    coreset_ratio: float = 0.05
    image_score: str = "max"
    num_neighbors: int = 1
    anomaly_quantile: float = 0.99
    failure_top_k_patches: int = 8
    classifier_neighbors: int = 3
    known_failure_quantile: float = 0.95
    min_margin_ratio: float = 1.05


@dataclass(frozen=True)
class SupportRecord:
    label: str
    path: str
    anomaly_score: float


@dataclass(frozen=True)
class FailureDecision:
    predicted_label: str | None
    best_distance: float | None
    second_best_distance: float | None
    margin_ratio: float | None
    is_known_failure: bool
    neighbors: list[dict[str, Any]]


@dataclass(frozen=True)
class HybridPrediction:
    status: str
    anomaly_score: float
    anomaly_threshold: float
    predicted_label: str | None
    best_failure_distance: float | None
    failure_margin_ratio: float | None
    is_known_failure: bool
    nearest_failures: list[dict[str, Any]]


@dataclass(frozen=True)
class HybridMemoryArtifact:
    cfg: HybridMemoryConfig
    nominal_memory: np.ndarray
    failure_descriptors: np.ndarray
    support_records: list[SupportRecord]
    anomaly_threshold: float
    known_failure_threshold: float
    margin_threshold: float
    backbone_state: dict[str, Any] | None
    artifact_info: dict[str, Any] | None


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        return vector.astype(np.float32, copy=False)
    return (vector / norm).astype(np.float32, copy=False)


def build_failure_descriptor(
    patch_embeddings: np.ndarray,
    patch_scores: np.ndarray,
    *,
    top_k_patches: int,
) -> np.ndarray:
    if patch_embeddings.ndim != 2:
        raise ValueError(f"Expected [P,D] patch embeddings, got shape {patch_embeddings.shape}")
    if patch_scores.ndim != 1:
        raise ValueError(f"Expected [P] patch scores, got shape {patch_scores.shape}")
    if patch_embeddings.shape[0] != patch_scores.shape[0]:
        raise ValueError("patch_embeddings and patch_scores must have the same number of patches")
    if patch_embeddings.shape[0] == 0:
        raise ValueError("patch_embeddings is empty")

    k = max(1, min(int(top_k_patches), int(patch_embeddings.shape[0])))
    top_idx = np.argsort(-patch_scores)[:k]
    salient = patch_embeddings[top_idx].mean(axis=0)
    global_mean = patch_embeddings.mean(axis=0)
    descriptor = np.concatenate([salient, global_mean], axis=0).astype(np.float32, copy=False)
    return _l2_normalize(descriptor)


def calibrate_anomaly_threshold(scores: np.ndarray, quantile: float) -> float:
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty rank-1 array")
    q = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(scores.astype(np.float32, copy=False), q))


def _pairwise_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-2 arrays")
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def calibrate_known_failure_threshold(
    descriptors: np.ndarray,
    labels: list[str],
    *,
    quantile: float,
    min_margin_ratio: float,
) -> tuple[float, float, dict[str, float]]:
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("descriptors must be a non-empty rank-2 array")
    if descriptors.shape[0] != len(labels):
        raise ValueError("labels length must match descriptor count")

    distances = _pairwise_l2(descriptors.astype(np.float32, copy=False), descriptors.astype(np.float32, copy=False))
    np.fill_diagonal(distances, np.inf)

    positive_distances: list[float] = []
    margin_ratios: list[float] = []
    for idx, label in enumerate(labels):
        same_mask = np.array([other == label for other in labels], dtype=bool)
        same_mask[idx] = False
        diff_mask = np.array([other != label for other in labels], dtype=bool)

        if np.any(same_mask):
            best_same = float(np.min(distances[idx, same_mask]))
            positive_distances.append(best_same)
            if np.any(diff_mask):
                best_diff = float(np.min(distances[idx, diff_mask]))
                margin_ratios.append(best_diff / max(best_same, 1e-8))

    if positive_distances:
        known_threshold = float(np.quantile(np.array(positive_distances, dtype=np.float32), float(np.clip(quantile, 0.0, 1.0))))
    else:
        known_threshold = float("inf")

    if margin_ratios:
        margin_threshold = max(
            float(min_margin_ratio),
            float(np.quantile(np.array(margin_ratios, dtype=np.float32), 0.05)),
        )
    else:
        margin_threshold = float(min_margin_ratio)

    stats = {
        "positive_pairs": float(len(positive_distances)),
        "margin_pairs": float(len(margin_ratios)),
        "known_failure_threshold": known_threshold,
        "margin_threshold": margin_threshold,
    }
    return known_threshold, margin_threshold, stats


def classify_failure_descriptor(
    descriptor: np.ndarray,
    failure_descriptors: np.ndarray,
    support_records: list[SupportRecord],
    *,
    neighbors: int,
    known_failure_threshold: float,
    margin_threshold: float,
) -> FailureDecision:
    if failure_descriptors.ndim != 2:
        raise ValueError("failure_descriptors must be rank-2")
    if failure_descriptors.shape[0] != len(support_records):
        raise ValueError("support_records length must match failure descriptors")
    if failure_descriptors.shape[0] == 0:
        return FailureDecision(
            predicted_label=None,
            best_distance=None,
            second_best_distance=None,
            margin_ratio=None,
            is_known_failure=False,
            neighbors=[],
        )

    dists = np.linalg.norm(failure_descriptors - descriptor[None, :], axis=1)
    order = np.argsort(dists)[: max(1, min(int(neighbors), int(failure_descriptors.shape[0])))]

    by_label: dict[str, list[float]] = {}
    neighbor_rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        rec = support_records[int(index)]
        dist = float(dists[int(index)])
        by_label.setdefault(rec.label, []).append(dist)
        neighbor_rows.append(
            {
                "rank": rank,
                "support_index": int(index),
                "label": rec.label,
                "path": rec.path,
                "distance": dist,
                "support_anomaly_score": float(rec.anomaly_score),
            }
        )

    label_scores = sorted(
        ((label, float(np.mean(label_dists))) for label, label_dists in by_label.items()),
        key=lambda item: item[1],
    )
    predicted_label = label_scores[0][0] if label_scores else None
    best_distance = label_scores[0][1] if label_scores else None
    second_best_distance = label_scores[1][1] if len(label_scores) > 1 else None
    margin_ratio = None
    if best_distance is not None:
        if second_best_distance is None:
            margin_ratio = float("inf")
        else:
            margin_ratio = float(second_best_distance / max(best_distance, 1e-8))

    is_known_failure = bool(
        predicted_label is not None
        and best_distance is not None
        and best_distance <= float(known_failure_threshold)
        and (margin_ratio is None or margin_ratio >= float(margin_threshold))
    )

    return FailureDecision(
        predicted_label=predicted_label,
        best_distance=best_distance,
        second_best_distance=second_best_distance,
        margin_ratio=margin_ratio,
        is_known_failure=is_known_failure,
        neighbors=neighbor_rows,
    )
