from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .config import PatchCoreConfig


@dataclass
class PatchCoreModel:
    cfg: PatchCoreConfig
    memory: np.ndarray  # [N,D]
    nn: NearestNeighbors

    @classmethod
    def fit(cls, cfg: PatchCoreConfig, nominal_patch_embeddings: np.ndarray) -> "PatchCoreModel":
        # nominal_patch_embeddings: [N,D]
        nn = NearestNeighbors(n_neighbors=int(cfg.num_neighbors), algorithm="auto")
        nn.fit(nominal_patch_embeddings)
        return cls(cfg=cfg, memory=nominal_patch_embeddings, nn=nn)

    def score_patches(self, patch_embeddings: np.ndarray) -> np.ndarray:
        """Return per-patch anomaly scores for a single image.

        patch_embeddings: [P,D]
        returns: [P] distances to nearest nominal patch
        """
        dists, _ = self.nn.kneighbors(patch_embeddings, return_distance=True)
        # For k>1, could aggregate; for now use nearest.
        return dists[:, 0]

    def score_image(self, patch_embeddings: np.ndarray) -> float:
        s = self.score_patches(patch_embeddings)
        if self.cfg.image_score == "max":
            return float(np.max(s))
        if self.cfg.image_score == "mean":
            return float(np.mean(s))
        raise ValueError(f"unknown image_score={self.cfg.image_score}")

    def score_map(self, patch_embeddings: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        """Return an anomaly map in patch-grid space.

        patch_embeddings: [P,D]
        hw: (H,W) such that P == H*W
        returns: [H,W] float32
        """
        H, W = hw
        s = self.score_patches(patch_embeddings)
        if s.shape[0] != H * W:
            raise ValueError(f"Patches {s.shape[0]} != H*W {H*W}")
        return s.reshape(H, W).astype(np.float32, copy=False)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float32, copy=False)
