import numpy as np

from src.patchcore.metrics import classification_metrics
from src.patchcore.pro import compute_pro_auc


def test_pro_auc_smoke():
    # Two images, one with a 2x2 anomaly region.
    s1 = np.zeros((4, 4), dtype=np.float32)
    m1 = np.zeros((4, 4), dtype=np.uint8)

    s2 = np.zeros((4, 4), dtype=np.float32)
    m2 = np.zeros((4, 4), dtype=np.uint8)
    m2[1:3, 1:3] = 1
    # Put high scores on the anomaly region.
    s2[1:3, 1:3] = 10.0

    res = compute_pro_auc([s1, s2], [m1, m2], fpr_limit=0.3, n_thresholds=50)
    assert 0.0 <= res.pro_auc <= 1.0
    # Should be reasonably high for this trivial case.
    assert res.pro_auc > 0.5


def test_classification_metrics_smoke():
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_score = np.array([0.1, 0.8, 0.7, 0.2], dtype=np.float32)
    res = classification_metrics(y_true, y_score, threshold=0.5)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["tn"] == 1
    assert res["fn"] == 1
