from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class KCenterGreedy:
    """k-Center Greedy coreset selection.

    This is a standard approximation used by PatchCore to subsample a large memory bank.

    Implementation: iteratively select the point farthest from current centers.
    Complexity is O(kN) distance updates; for very large N you want batching/FAISS.
    """

    def select(self, X: np.ndarray, ratio: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
        assert X.ndim == 2
        N = X.shape[0]
        if N == 0:
            return np.array([], dtype=np.int64)

        k = max(1, int(math.ceil(ratio * N)))
        if k >= N:
            return np.arange(N, dtype=np.int64)

        rng = rng or np.random.default_rng(0)

        # Start from a random point.
        centers = [int(rng.integers(0, N))]
        # Track distance to nearest center.
        # Using squared L2.
        d = np.sum((X - X[centers[0]]) ** 2, axis=1)

        for _ in range(1, k):
            idx = int(np.argmax(d))
            centers.append(idx)
            d_new = np.sum((X - X[idx]) ** 2, axis=1)
            d = np.minimum(d, d_new)

        return np.array(centers, dtype=np.int64)
