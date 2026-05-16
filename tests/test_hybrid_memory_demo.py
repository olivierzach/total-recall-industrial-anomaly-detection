from pathlib import Path

import numpy as np

from hybrid_memory_demo.model import (
    SupportRecord,
    build_failure_descriptor,
    calibrate_anomaly_threshold,
    calibrate_known_failure_threshold,
    classify_failure_descriptor,
)
from hybrid_memory_demo.pipeline import compute_embedding_similarity_map, compute_patch_embedding_projection, load_artifact, save_artifact
from hybrid_memory_demo.model import HybridMemoryArtifact, HybridMemoryConfig


def test_build_failure_descriptor_is_normalized_and_uses_two_pools():
    patch_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    patch_scores = np.array([0.1, 0.9, 0.2], dtype=np.float32)
    descriptor = build_failure_descriptor(patch_embeddings, patch_scores, top_k_patches=1)

    assert descriptor.shape == (4,)
    assert np.isclose(np.linalg.norm(descriptor), 1.0, atol=1e-5)
    assert descriptor[1] > descriptor[0]


def test_threshold_calibration_prefers_same_class_distances():
    descriptors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ],
        dtype=np.float32,
    )
    labels = ["a", "a", "b", "b"]
    known_threshold, margin_threshold, stats = calibrate_known_failure_threshold(
        descriptors,
        labels,
        quantile=0.95,
        min_margin_ratio=1.05,
    )

    assert known_threshold < 0.2
    assert margin_threshold >= 1.05
    assert stats["positive_pairs"] == 4.0


def test_classifier_separates_known_and_unknown():
    failure_descriptors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 2.0],
            [2.1, 2.0],
        ],
        dtype=np.float32,
    )
    support_records = [
        SupportRecord(label="crack", path="a.png", anomaly_score=1.0),
        SupportRecord(label="crack", path="b.png", anomaly_score=1.0),
        SupportRecord(label="scratch", path="c.png", anomaly_score=1.0),
        SupportRecord(label="scratch", path="d.png", anomaly_score=1.0),
    ]

    known = classify_failure_descriptor(
        np.array([0.05, 0.0], dtype=np.float32),
        failure_descriptors,
        support_records,
        neighbors=3,
        known_failure_threshold=0.2,
        margin_threshold=1.05,
    )
    unknown = classify_failure_descriptor(
        np.array([1.0, 1.0], dtype=np.float32),
        failure_descriptors,
        support_records,
        neighbors=3,
        known_failure_threshold=0.2,
        margin_threshold=1.05,
    )

    assert known.is_known_failure is True
    assert known.predicted_label == "crack"
    assert unknown.is_known_failure is False


def test_anomaly_threshold_quantile():
    scores = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float32)
    threshold = calibrate_anomaly_threshold(scores, 0.75)
    assert np.isclose(threshold, 0.5)


def test_artifact_round_trip(tmp_path: Path):
    artifact = HybridMemoryArtifact(
        cfg=HybridMemoryConfig(),
        nominal_memory=np.ones((4, 3), dtype=np.float32),
        failure_descriptors=np.ones((2, 6), dtype=np.float32),
        support_records=[
            SupportRecord(label="known", path="support.png", anomaly_score=1.23),
        ],
        anomaly_threshold=0.3,
        known_failure_threshold=0.4,
        margin_threshold=1.1,
        backbone_state=None,
        artifact_info={"note": "round-trip"},
    )

    save_artifact(tmp_path, artifact)
    loaded = load_artifact(tmp_path)

    assert loaded.cfg.backbone == artifact.cfg.backbone
    assert loaded.support_records[0].label == "known"
    assert loaded.artifact_info["note"] == "round-trip"
    assert loaded.failure_descriptors.shape == (2, 6)


def test_compute_embedding_similarity_map_returns_spatial_field():
    patch_embeddings = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    reference_descriptor = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    projection = compute_embedding_similarity_map(patch_embeddings, (2, 2), reference_descriptor)

    assert projection.shape == (2, 2)
    assert np.isfinite(projection).all()


def test_compute_patch_embedding_projection_returns_nominal_and_query_points():
    patch_embeddings = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    patch_scores = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
    nominal_memory = np.array(
        [
            [0.1, 0.0, 0.9],
            [0.2, 0.1, 0.8],
            [0.8, 0.1, 0.1],
        ],
        dtype=np.float32,
    )

    projection = compute_patch_embedding_projection(
        patch_embeddings,
        patch_scores,
        nominal_memory,
        nominal_sample_size=2,
        top_k_patches=2,
        seed=0,
    )

    assert projection["nominal_projection"].shape == (2, 2)
    assert projection["query_projection"].shape == (4, 2)
    assert projection["top_query_mask"].shape == (4,)
    assert int(projection["top_query_mask"].sum()) == 2
