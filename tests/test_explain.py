import numpy as np

from src.patchcore.explain import build_patch_explanations
from src.utils.io import MemoryBankEntry


def test_build_patch_explanations_uses_memory_metadata():
    dists = np.array(
        [
            [0.1, 0.2],
            [0.9, 1.1],
            [0.4, 0.5],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    inds = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int64,
    )
    meta = [
        MemoryBankEntry(source_path="nominal/a.png", patch_index=2, row=0, col=2, grid_h=2, grid_w=2),
        MemoryBankEntry(source_path="nominal/b.png", patch_index=3, row=1, col=1, grid_h=2, grid_w=2),
    ]

    out = build_patch_explanations(dists, inds, (2, 2), memory_metadata=meta, top_k_patches=2)

    assert len(out) == 2
    assert out[0]["patch_index"] == 1
    assert out[0]["neighbors"][0]["source_path"] == "nominal/b.png"
    assert out[1]["patch_index"] == 3
