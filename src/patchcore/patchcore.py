from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .config import PatchCoreConfig
from .preprocess import PCATransform


@dataclass
class PatchCoreModel:
    cfg: PatchCoreConfig
    memory: np.ndarray  # [N,D]
    nn: NearestNeighbors
    pca: PCATransform | None = None

    @classmethod
    def fit(
        cls,
        cfg: PatchCoreConfig,
        nominal_patch_embeddings: np.ndarray,
        *,
        pca: PCATransform | None = None,
    ) -> "PatchCoreModel":
        """Fit the kNN index.

        Args:
            cfg: configuration
            nominal_patch_embeddings: [N, D] memory bank used for kNN search.
            pca: optional PCA transform applied to both query and memory.
        """
        nn = NearestNeighbors(n_neighbors=int(cfg.num_neighbors), algorithm="auto", metric=str(cfg.distance_metric))
        nn.fit(nominal_patch_embeddings)
        return cls(cfg=cfg, memory=nominal_patch_embeddings, nn=nn, pca=pca)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return X
        return self.pca.transform(X)

    def score_patches(self, patch_embeddings: np.ndarray) -> np.ndarray:
        """Return per-patch anomaly scores for a single image.

        patch_embeddings: [P,D]
        returns: [P] distances to nearest nominal patch
        """
        dists, _ = self.query(patch_embeddings)
        # For k>1, could aggregate; for now use nearest.
        return dists[:, 0]

    def query(self, patch_embeddings: np.ndarray, n_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if n_neighbors is None:
            n_neighbors = int(self.cfg.num_neighbors)
        Xq = self._transform(patch_embeddings)
        dists, inds = self.nn.kneighbors(Xq, n_neighbors=n_neighbors, return_distance=True)
        return dists, inds

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
