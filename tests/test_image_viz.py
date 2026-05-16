import numpy as np
from PIL import Image

from src.utils.image_viz import bbox_from_binary_mask, bbox_from_score_map, overlay_heatmap, upsample_score_map


def test_bbox_from_binary_mask():
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    assert bbox_from_binary_mask(mask) == (3, 2, 5, 4)


def test_bbox_from_score_map():
    score = np.zeros((4, 4), dtype=np.float32)
    score[1:3, 2:4] = 10.0
    assert bbox_from_score_map(score, quantile=0.9) == (2, 1, 3, 2)


def test_overlay_and_upsample_shapes():
    img = Image.new("RGB", (16, 16), color=(128, 128, 128))
    score = np.arange(16, dtype=np.float32).reshape(4, 4)
    up = upsample_score_map(score, img.size)
    over = overlay_heatmap(img, up)
    assert up.shape == (16, 16)
    assert over.size == (16, 16)
