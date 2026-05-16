import numpy as np

from src.patchcore.coreset import KCenterGreedy


def test_kcenter_select_shapes_and_bounds():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 8)).astype(np.float32)
    idx = KCenterGreedy().select(X, ratio=0.1, rng=rng)
    assert idx.ndim == 1
    assert 1 <= len(idx) <= 100
    assert idx.min() >= 0
    assert idx.max() < 100
    assert len(np.unique(idx)) == len(idx)


def test_kcenter_select_ratio_one_returns_all():
    X = np.zeros((10, 2), dtype=np.float32)
    idx = KCenterGreedy().select(X, ratio=1.0)
    assert len(idx) == 10


def test_kcenter_select_same_seed_is_deterministic():
    X = np.random.default_rng(123).normal(size=(64, 4)).astype(np.float32)
    idx0 = KCenterGreedy().select(X, ratio=0.15, rng=np.random.default_rng(7))
    idx1 = KCenterGreedy().select(X, ratio=0.15, rng=np.random.default_rng(7))
    assert np.array_equal(idx0, idx1)
