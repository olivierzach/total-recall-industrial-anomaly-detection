from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCATransform:
    """Lightweight PCA projection + optional whitening.

    We keep this intentionally minimal to avoid pulling sklearn into the core.

    X: [N, D]
    transform: (X - mean) @ components.T  -> [N, pca_dim]
    if whiten: divide by sqrt(explained_variance + eps)
    """

    mean: np.ndarray  # [D]
    components: np.ndarray  # [K, D]
    explained_variance: np.ndarray  # [K]
    whiten: bool = True
    eps: float = 1e-8

    @property
    def dim_in(self) -> int:
        return int(self.mean.shape[0])

    @property
    def dim_out(self) -> int:
        return int(self.components.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        Z = Xc @ self.components.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance + self.eps)
        return Z.astype(np.float32, copy=False)


def pca_from_state(state: dict) -> PCATransform:
    """Rehydrate a PCATransform from a state dict as saved by utils.io."""
    meta = state.get("_meta") or {}
    whiten = bool(meta.get("whiten", True))
    eps = float(meta.get("eps", 1e-8))
    return PCATransform(
        mean=np.asarray(state["mean"], dtype=np.float32),
        components=np.asarray(state["components"], dtype=np.float32),
        explained_variance=np.asarray(state["explained_variance"], dtype=np.float32),
        whiten=whiten,
        eps=eps,
    )


def pca_state(pca: PCATransform) -> dict:
    return {
        "mean": pca.mean,
        "components": pca.components,
        "explained_variance": pca.explained_variance,
        "_meta": {"whiten": bool(pca.whiten), "eps": float(pca.eps)},
    }


def fit_pca(X: np.ndarray, k: int, *, whiten: bool = True) -> PCATransform:
    """Fit PCA on X using SVD.

    This is deterministic given X.

    Args:
        X: [N, D]
        k: number of components
        whiten: whether to whiten (scale by inverse std along PCs)

    Returns:
        PCATransform
    """

    if k <= 0:
        raise ValueError("k must be > 0")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, D = X.shape
    k = int(min(k, D))
    mean = X.mean(axis=0)
    Xc = X - mean

    # Economy SVD: Xc = U S V^T, PCs are rows of V^T.
    # For numerical stability, use float64 here.
    Xc64 = Xc.astype(np.float64, copy=False)
    _, S, Vt = np.linalg.svd(Xc64, full_matrices=False)
    components = Vt[:k].astype(np.float32, copy=False)  # [k, D]

    # explained variance along PCs: (S^2) / (N-1)
    # Guard N==1.
    denom = max(1, N - 1)
    explained_variance = ((S[:k] ** 2) / denom).astype(np.float32, copy=False)

    return PCATransform(mean=mean.astype(np.float32, copy=False), components=components, explained_variance=explained_variance, whiten=bool(whiten))
