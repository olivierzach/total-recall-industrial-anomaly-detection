from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


RoutingMode = Literal["patch", "image"]


@dataclass(frozen=True)
class RoutingIndex:
    """A simple k-means style routing index.

    This is an *IVF-like* partitioning: we assign points to clusters (lists of indices).

    - For patch routing, points are patch embeddings.
    - For image routing, points are image embeddings (one per nominal image), and each
      image maps to a set of patch indices.

    At query time, we select the top-`probes` closest clusters and only search within
    their member indices.
    """

    mode: RoutingMode
    centroids: np.ndarray  # [K, D]
    members: list[np.ndarray]  # len K, each [n_k] int64 indices into patch memory


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def assign_to_centroids(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> np.ndarray:
    """Return index of nearest centroid for each row in X."""
    if metric == "cosine":
        Xn = _l2_normalize(X)
        Cn = _l2_normalize(centroids)
        # maximize cosine -> minimize negative cosine
        sims = Xn @ Cn.T  # [N, K]
        return np.argmax(sims, axis=1).astype(np.int64)

    # Euclidean: argmin ||x-c||^2 = argmin (||x||^2 - 2x·c + ||c||^2)
    x2 = np.sum(X * X, axis=1, keepdims=True)  # [N,1]
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T  # [1,K]
    d2 = x2 - 2.0 * (X @ centroids.T) + c2
    return np.argmin(d2, axis=1).astype(np.int64)


def topk_centroids(
    X: np.ndarray,
    centroids: np.ndarray,
    *,
    k: int,
    metric: Literal["euclidean", "cosine"] = "euclidean",
) -> np.ndarray:
    """Return top-k centroid indices per query.

    Returns:
        [N, k] int64 indices.
    """
    k = int(k)
    if k <= 0:
        raise ValueError("k must be > 0")
    if metric == "cosine":
        Xn = _l2_normalize(X)
        Cn = _l2_normalize(centroids)
        sims = Xn @ Cn.T  # [N,K]
        # top-k by similarity
        idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        # sort within top-k
        row = np.arange(idx.shape[0])[:, None]
        order = np.argsort(-sims[row, idx], axis=1)
        return idx[row, order].astype(np.int64)

    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centroids * centroids, axis=1, keepdims=True).T
    d2 = x2 - 2.0 * (X @ centroids.T) + c2
    idx = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
    row = np.arange(idx.shape[0])[:, None]
    order = np.argsort(d2[row, idx], axis=1)
    return idx[row, order].astype(np.int64)


def build_members_from_assignments(assign: np.ndarray, k: int) -> list[np.ndarray]:
    """Build cluster membership lists from assignment vector."""
    k = int(k)
    members: list[list[int]] = [[] for _ in range(k)]
    for i, c in enumerate(assign.tolist()):
        if 0 <= c < k:
            members[c].append(i)
    return [np.asarray(m, dtype=np.int64) for m in members]


def kmeans_lloyd(
    X: np.ndarray,
    k: int,
    *,
    iters: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Very small, dependency-free k-means (Lloyd) for moderate-sized problems.

    Returns centroids [k, D].

    Note: for huge patch banks, use FAISS/sklearn MiniBatchKMeans instead.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, D = X.shape
    k = int(min(k, N))
    if k <= 0:
        raise ValueError("k must be > 0")
    rng = rng or np.random.default_rng(0)

    # k-means++ init (approx)
    centers = np.empty((k, D), dtype=np.float32)
    first = int(rng.integers(0, N))
    centers[0] = X[first]
    d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for ci in range(1, k):
        probs = d2 / max(1e-12, float(d2.sum()))
        idx = int(rng.choice(N, p=probs))
        centers[ci] = X[idx]
        d2 = np.minimum(d2, np.sum((X - centers[ci]) ** 2, axis=1))

    for _ in range(int(iters)):
        assign = assign_to_centroids(X, centers, metric="euclidean")
        for ci in range(k):
            mask = assign == ci
            if not np.any(mask):
                # reinit empty cluster
                centers[ci] = X[int(rng.integers(0, N))]
            else:
                centers[ci] = X[mask].mean(axis=0)

    return centers.astype(np.float32, copy=False)


def build_patch_routing(
    X: np.ndarray,
    *,
    n_clusters: int,
    iters: int = 20,
    rng: np.random.Generator | None = None,
) -> RoutingIndex:
    centroids = kmeans_lloyd(X, int(n_clusters), iters=int(iters), rng=rng)
    assign = assign_to_centroids(X, centroids, metric="euclidean")
    members = build_members_from_assignments(assign, centroids.shape[0])
    return RoutingIndex(mode="patch", centroids=centroids, members=members)


def build_image_routing(
    image_embeddings: np.ndarray,
    image_to_patch_indices: list[np.ndarray],
    *,
    n_clusters: int,
    iters: int = 20,
    rng: np.random.Generator | None = None,
) -> RoutingIndex:
    """Build routing over images.

    `members[c]` will be patch indices belonging to images assigned to cluster c.
    """
    centroids = kmeans_lloyd(image_embeddings, int(n_clusters), iters=int(iters), rng=rng)
    assign = assign_to_centroids(image_embeddings, centroids, metric="euclidean")

    K = centroids.shape[0]
    patch_members: list[list[int]] = [[] for _ in range(K)]
    for img_i, c in enumerate(assign.tolist()):
        if 0 <= c < K:
            patch_members[c].extend(image_to_patch_indices[img_i].tolist())

    members = [np.asarray(m, dtype=np.int64) for m in patch_members]
    return RoutingIndex(mode="image", centroids=centroids, members=members)
