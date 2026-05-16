from pathlib import Path

import numpy as np

from hybrid_memory_demo.router import NominalRouterArtifact, NominalRouterConfig, load_nominal_router, save_nominal_router


def test_nominal_router_round_trip(tmp_path: Path):
    artifact = NominalRouterArtifact(
        cfg=NominalRouterConfig(),
        labels=["a", "b"],
        label_display_names={"a": "Alpha", "b": "Beta"},
        router_W=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        router_b=np.array([0.1, -0.1], dtype=np.float32),
        centroids=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        backbone_state=None,
        artifact_info={"train_accuracy": 1.0},
    )

    save_nominal_router(tmp_path, artifact)
    loaded = load_nominal_router(tmp_path)

    assert loaded.labels == ["a", "b"]
    assert loaded.label_display_names["a"] == "Alpha"
    assert loaded.router_W.shape == (2, 2)
    assert loaded.artifact_info["train_accuracy"] == 1.0
