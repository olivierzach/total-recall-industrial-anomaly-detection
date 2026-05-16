from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.patchcore.backbone import load_backbone
from src.patchcore.config import PatchCoreConfig
from src.utils.io import load_patchcore, save_patchcore


def test_save_load_patchcore_round_trip_with_backbone_state(tmp_path: Path):
    cfg = PatchCoreConfig(backbone="resnet18", pretrained=False, layers=("layer1",))
    memory = np.random.default_rng(0).normal(size=(8, 16)).astype(np.float32)

    backbone = load_backbone(cfg.backbone, pretrained=False)
    backbone.eval()

    model_dir = tmp_path / "artifact"
    save_patchcore(model_dir, cfg, memory, backbone_state=backbone.state_dict())
    artifact = load_patchcore(model_dir)

    assert artifact.backbone_state is not None
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
    np.testing.assert_allclose(artifact.memory_bank, memory)
