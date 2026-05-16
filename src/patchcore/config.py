from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchCoreConfig:
    # Backbone
    backbone: str = "wide_resnet50_2"  # torchvision model name (e.g. wide_resnet50_2, vit_b_16)
    pretrained: bool = True

    # Feature layers to hook (torchvision naming). For WRN50_2, these are safe defaults.
    # We'll concatenate features from these layers after resizing to a common spatial grid.
    layers: tuple[str, ...] = ("layer2", "layer3")

    # Input
    image_size: int = 256

    # Embedding
    # If true, L2-normalize patch embeddings before memory bank / NN.
    l2_normalize: bool = True

    # Distance / preprocessing
    # NearestNeighbors metric. In sklearn this can be e.g. "euclidean" or "cosine".
    # Note: if l2_normalize=True, euclidean and cosine are monotonic-equivalent:
    #   ||x-y||^2 = 2(1-cos(x,y)) for unit vectors.
    distance_metric: str = "euclidean"  # {euclidean, cosine}

    # Optional PCA whitening on nominal patch embeddings before kNN.
    # If set, we fit PCA on nominal patches, project to pca_dim and optionally whiten.
    # This can mitigate correlated/redundant embedding dimensions.
    pca_dim: int = 0  # 0 disables PCA
    pca_whiten: bool = True

    # Memory bank / coreset
    coreset_ratio: float = 0.1  # fraction of nominal patches to keep
    num_neighbors: int = 1

    # Scoring
    # Image score aggregation over patch scores.
    image_score: str = "max"  # {max, mean}
