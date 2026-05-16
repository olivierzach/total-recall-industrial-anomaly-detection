from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .preprocess import PCATransform
from .routing import RoutingIndex, topk_centroids


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> np.ndarray:
    """Compute distances from X to Y.

    Returns [X.shape[0], Y.shape[0]].

    NOTE: brute-force; intended for small candidate sets.
    """
    if metric == "cosine":
        Xn = _l2_normalize(X)
        Yn = _l2_normalize(Y)
        sims = Xn @ Yn.T
        return (1.0 - sims).astype(np.float32, copy=False)

    # Euclidean
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    d2 = x2 - 2.0 * (X @ Y.T) + y2
    # Guard numerical negatives
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2, dtype=np.float32)


@dataclass
class QueryAdaptivePatchCore:
    """Query-adaptive PatchCore.

    Uses a routing index to choose a candidate subset of memory items, then runs brute-force
    kNN within that subset.

    This is an IVF-like approach: partition memory, then multi-probe at query time.
    """

    memory: np.ndarray  # [N, D]
    routing: RoutingIndex
    metric: Literal["euclidean", "cosine"] = "euclidean"
    pca: PCATransform | None = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            return X
        return self.pca.transform(X)

    def candidate_indices_for_queries(self, Xq: np.ndarray, *, probes: int = 1) -> list[np.ndarray]:
        Xq = self._transform(Xq)
        top = topk_centroids(Xq, self.routing.centroids, k=int(probes), metric=self.metric)
        out: list[np.ndarray] = []
        for i in range(top.shape[0]):
            idxs = []
            for c in top[i].tolist():
                idxs.append(self.routing.members[int(c)])
            if not idxs:
                out.append(np.array([], dtype=np.int64))
            else:
                out.append(np.unique(np.concatenate(idxs)))
        return out

    def knn(self, Xq: np.ndarray, *, k: int = 1, probes: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """kNN within routed candidates.

        Returns:
            dists: [Q, k]
            inds: [Q, k] indices into full memory
        """
        Xq_t = self._transform(Xq)
        cand = self.candidate_indices_for_queries(Xq, probes=int(probes))
        Q = Xq.shape[0]
        k = int(k)
        dists_out = np.full((Q, k), np.inf, dtype=np.float32)
        inds_out = np.full((Q, k), -1, dtype=np.int64)

        for i in range(Q):
            idx = cand[i]
            if idx.size == 0:
                continue
            Ym = self.memory[idx]
            # If PCA exists, memory is expected already in PCA space for routed runs.
            # If not, we still transform it here.
            Ym_t = Ym if self.pca is None else Ym
            di = pairwise_distances(Xq_t[i : i + 1], Ym_t, metric=self.metric)[0]  # [M]
            kk = min(k, di.shape[0])
            j = np.argpartition(di, kth=kk - 1)[:kk]
            j = j[np.argsort(di[j])]
            dists_out[i, :kk] = di[j]
            inds_out[i, :kk] = idx[j]

        return dists_out, inds_out

    def score_patches(self, Xq: np.ndarray, *, probes: int = 1) -> np.ndarray:
        d, _ = self.knn(Xq, k=1, probes=int(probes))
        return d[:, 0]
