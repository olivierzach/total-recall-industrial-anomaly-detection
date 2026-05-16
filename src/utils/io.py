from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.patchcore.config import PatchCoreConfig


@dataclass(frozen=True)
class MemoryBankEntry:
    source_path: str
    patch_index: int
    row: int
    col: int
    grid_h: int
    grid_w: int


@dataclass(frozen=True)
class PatchCoreArtifact:
    cfg: PatchCoreConfig
    memory_bank: np.ndarray
    backbone_state: dict[str, Any] | None
    memory_metadata: list[MemoryBankEntry] | None
    seed: int | None
    artifact_info: dict[str, Any] | None
    run_manifest: dict[str, Any] | None
    pca_state: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThresholdArtifact:
    n: int
    target_fpr: float
    quantile: float
    threshold: float
    score_min: float
    score_med: float
    score_p99: float
    score_max: float
    source_scores: str | None = None
    source_scores_manifest: str | None = None
    source_run_id: str | None = None
    run_id: str | None = None
    schema_version: str | None = None


def _state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cpu_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu()
        else:
            cpu_state[key] = value
    return cpu_state


def save_patchcore(
    out_dir: str | Path,
    cfg: PatchCoreConfig,
    memory_bank: np.ndarray,
    *,
    backbone_state: dict[str, Any] | None = None,
    memory_metadata: list[MemoryBankEntry] | None = None,
    seed: int | None = None,
    artifact_info: dict[str, Any] | None = None,
    run_manifest: dict[str, Any] | None = None,
    pca_state: dict[str, Any] | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    np.save(out_dir / "memory_bank.npy", memory_bank.astype(np.float32, copy=False))
    if backbone_state is not None:
        torch.save(_state_dict_to_cpu(backbone_state), out_dir / "backbone_state.pt")
    if memory_metadata is not None:
        (out_dir / "memory_metadata.json").write_text(json.dumps([asdict(m) for m in memory_metadata], indent=2, sort_keys=True))
    if pca_state is not None:
        # Store as NPZ (float arrays) + a small JSON for flags.
        np.savez(out_dir / "pca_state.npz", **{k: np.asarray(v) for k, v in pca_state.items() if k != "_meta"})
        meta = pca_state.get("_meta") or {}
        (out_dir / "pca_state.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    info = dict(artifact_info or {})
    if seed is not None:
        info.setdefault("seed", int(seed))
    if info:
        (out_dir / "artifact_info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    if run_manifest is not None:
        (out_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True))


def load_patchcore(model_dir: str | Path) -> PatchCoreArtifact:
    model_dir = Path(model_dir)
    cfg = PatchCoreConfig(**json.loads((model_dir / "config.json").read_text()))
    memory = np.load(model_dir / "memory_bank.npy")
    backbone_path = model_dir / "backbone_state.pt"
    backbone_state = None
    if backbone_path.exists():
        try:
            backbone_state = torch.load(backbone_path, map_location="cpu", weights_only=True)
        except TypeError:
            backbone_state = torch.load(backbone_path, map_location="cpu")
    meta_path = model_dir / "memory_metadata.json"
    memory_metadata = None
    if meta_path.exists():
        memory_metadata = [MemoryBankEntry(**rec) for rec in json.loads(meta_path.read_text())]
    info_path = model_dir / "artifact_info.json"
    artifact_info = None
    seed = None
    if info_path.exists():
        artifact_info = json.loads(info_path.read_text())
        seed = artifact_info.get("seed")
    manifest_path = model_dir / "run_manifest.json"
    run_manifest = None
    if manifest_path.exists():
        run_manifest = json.loads(manifest_path.read_text())

    pca_state = None
    pca_npz = model_dir / "pca_state.npz"
    if pca_npz.exists():
        arrays = dict(np.load(pca_npz))
        meta = {}
        meta_path2 = model_dir / "pca_state.json"
        if meta_path2.exists():
            meta = json.loads(meta_path2.read_text())
        arrays["_meta"] = meta
        pca_state = arrays

    return PatchCoreArtifact(
        cfg=cfg,
        memory_bank=memory,
        backbone_state=backbone_state,
        memory_metadata=memory_metadata,
        seed=seed,
        artifact_info=artifact_info,
        run_manifest=run_manifest,
        pca_state=pca_state,
    )


def save_threshold_artifact(path: str | Path, artifact: ThresholdArtifact) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(artifact), indent=2, sort_keys=True))


def load_threshold_artifact(path: str | Path) -> ThresholdArtifact:
    return ThresholdArtifact(**json.loads(Path(path).read_text()))
