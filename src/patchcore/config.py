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

    # Memory bank / coreset
    coreset_ratio: float = 0.1  # fraction of nominal patches to keep
    num_neighbors: int = 1

    # Scoring
    # Image score aggregation over patch scores.
    image_score: str = "max"  # {max, mean}
