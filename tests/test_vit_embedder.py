import torch
import torchvision

from src.patchcore.vit_embedding import ViTEmbedder


def test_vit_embedder_shapes_no_weights():
    # Instantiate without weights to keep the test lightweight.
    vit = torchvision.models.vit_b_16(weights=None)
    vit.eval()

    emb = ViTEmbedder(vit)
    x = torch.randn(2, 3, 224, 224)
    toks, (H, W) = emb.embed(x, l2_normalize=False, return_hw=True)
    assert toks.shape[0] == 2
    assert toks.shape[1] == H * W
    assert toks.shape[2] == vit.hidden_dim
    assert (H, W) == (224 // 16, 224 // 16)
