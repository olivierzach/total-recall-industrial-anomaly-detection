import torch

from src.patchcore.embedding import patch_embeddings


def test_patch_embeddings_concat_and_normalize():
    # Two layers with different spatial sizes.
    feats = {
        "layer2": torch.randn(2, 8, 16, 16),
        "layer3": torch.randn(2, 4, 8, 8),
    }
    emb = patch_embeddings(feats, ["layer2", "layer3"], l2_normalize=True)
    assert emb.shape[0] == 2
    # target grid is first layer: 16x16
    assert emb.shape[1] == 16 * 16
    assert emb.shape[2] == 8 + 4

    # L2 norm ~1
    n = torch.linalg.norm(emb, dim=-1)
    assert torch.allclose(n.mean(), torch.ones_like(n.mean()), atol=1e-2, rtol=1e-2)
