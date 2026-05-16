from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.patchcore.config import PatchCoreConfig


def save_patchcore(out_dir: str | Path, cfg: PatchCoreConfig, memory_bank: np.ndarray) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True))
    np.save(out_dir / "memory_bank.npy", memory_bank.astype(np.float32, copy=False))


def load_patchcore(model_dir: str | Path) -> tuple[PatchCoreConfig, np.ndarray]:
    model_dir = Path(model_dir)
    cfg = PatchCoreConfig(**json.loads((model_dir / "config.json").read_text()))
    memory = np.load(model_dir / "memory_bank.npy")
    return cfg, memory
