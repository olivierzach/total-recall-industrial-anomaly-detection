from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.patchcore.backbone import load_backbone
from src.patchcore.config import PatchCoreConfig
from src.utils.io import MemoryBankEntry, ThresholdArtifact, load_patchcore, load_threshold_artifact, save_patchcore, save_threshold_artifact


def test_save_load_patchcore_round_trip_with_backbone_state(tmp_path: Path):
    cfg = PatchCoreConfig(backbone="resnet18", pretrained=False, layers=("layer1",))
    memory = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)
    meta = [
        MemoryBankEntry(source_path="nominal/a.png", patch_index=0, row=0, col=0, grid_h=4, grid_w=4),
        MemoryBankEntry(source_path="nominal/a.png", patch_index=1, row=0, col=1, grid_h=4, grid_w=4),
    ]

    backbone = load_backbone(cfg.backbone, pretrained=False)
    backbone.eval()

    model_dir = tmp_path / "artifact"
    save_patchcore(model_dir, cfg, memory, backbone_state=backbone.state_dict(), memory_metadata=meta, seed=17)
    artifact = load_patchcore(model_dir)

    assert artifact.backbone_state is not None
    assert artifact.memory_metadata == meta
    assert artifact.seed == 17
    np.testing.assert_allclose(artifact.memory_bank, memory)

    restored = load_backbone(cfg.backbone, pretrained=False)
    restored.load_state_dict(artifact.backbone_state)
    restored.eval()

    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y0 = backbone(x)
        y1 = restored(x)
    assert torch.allclose(y0, y1)


def test_save_load_patchcore_backward_compatible_without_backbone_state(tmp_path: Path):
    cfg = PatchCoreConfig()
    memory = np.random.default_rng(1).normal(size=(4, 8)).astype(np.float32)

    model_dir = tmp_path / "artifact"
    save_patchcore(model_dir, cfg, memory)
    artifact = load_patchcore(model_dir)

    assert artifact.backbone_state is None
    assert artifact.memory_metadata is None
    assert artifact.seed is None
    np.testing.assert_allclose(artifact.memory_bank, memory)


def test_threshold_artifact_round_trip(tmp_path: Path):
    artifact = ThresholdArtifact(
        n=10,
        target_fpr=0.001,
        quantile=0.999,
        threshold=1.23,
        score_min=0.1,
        score_med=0.3,
        score_p99=1.2,
        score_max=1.5,
        source_scores="scores.jsonl",
    )

    out = tmp_path / "threshold.json"
    save_threshold_artifact(out, artifact)
    loaded = load_threshold_artifact(out)
    assert loaded == artifact
