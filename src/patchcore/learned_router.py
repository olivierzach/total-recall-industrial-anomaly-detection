from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearRouter:
    """A tiny learned router: multiclass linear classifier over embeddings.

    This is trained to predict routing cluster IDs (e.g., k-means assignments).

    We store it as pure numpy arrays so inference has no sklearn dependency.

    logits = X @ W.T + b
    probs = softmax(logits)
    """

    W: np.ndarray  # [K, D]
    b: np.ndarray  # [K]

    def topk(self, X: np.ndarray, k: int) -> np.ndarray:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be > 0")
        logits = X @ self.W.T + self.b
        # top-k by logits (monotonic with softmax probs)
        idx = np.argpartition(-logits, kth=min(k, logits.shape[1] - 1), axis=1)[:, :k]
        row = np.arange(idx.shape[0])[:, None]
        order = np.argsort(-logits[row, idx], axis=1)
        return idx[row, order].astype(np.int64)


def _softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.max(Z, axis=1, keepdims=True)
    e = np.exp(Z)
    return e / np.sum(e, axis=1, keepdims=True)


def fit_linear_router(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    iters: int = 200,
    lr: float = 0.1,
    rng: np.random.Generator | None = None,
) -> LinearRouter:
    """Fit a simple multinomial logistic regression with GD.

    This is intentionally minimal (no dependencies). It is *not* the most efficient
    optimizer, but is good enough for routing on moderate datasets.

    Args:
        X: [N, D] float32
        y: [N] int64 class ids in [0, K)
        l2: L2 regularization weight
        iters: gradient steps
        lr: learning rate

    Returns:
        LinearRouter
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    N, D = X.shape
    K = int(np.max(y)) + 1
    rng = rng or np.random.default_rng(0)

    W = (0.01 * rng.standard_normal((K, D))).astype(np.float32)
    b = np.zeros((K,), dtype=np.float32)

    # One-hot labels
    Y = np.zeros((N, K), dtype=np.float32)
    Y[np.arange(N), y.astype(np.int64)] = 1.0

    for _ in range(int(iters)):
        logits = X @ W.T + b
        P = _softmax(logits).astype(np.float32, copy=False)
        # gradients
        dZ = (P - Y) / max(1, N)  # [N,K]
        gW = dZ.T @ X + (l2 * W)
        gb = np.sum(dZ, axis=0)
        W -= lr * gW
        b -= lr * gb

    return LinearRouter(W=W.astype(np.float32, copy=False), b=b.astype(np.float32, copy=False))


def router_state(router: LinearRouter) -> dict:
    return {"W": router.W, "b": router.b}


def router_from_state(state: dict) -> LinearRouter:
    return LinearRouter(W=np.asarray(state["W"], dtype=np.float32), b=np.asarray(state["b"], dtype=np.float32))
