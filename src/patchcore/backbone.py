from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision


@dataclass
class FeatureHooks:
    model: nn.Module
    layers: List[str]

    def __post_init__(self):
        self._features: Dict[str, torch.Tensor] = {}
        self._handles = []
        name_to_module = dict(self.model.named_modules())
        for name in self.layers:
            if name not in name_to_module:
                raise KeyError(f"Layer '{name}' not found in model. Available: {list(name_to_module)[:20]} ...")
            m = name_to_module[name]
            self._handles.append(m.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._features[name] = out

        return hook

    def pop(self) -> Dict[str, torch.Tensor]:
        feats = self._features
        self._features = {}
        return feats

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _weights_enum_name(model_name: str) -> str:
    # torchvision convention examples:
    #   wide_resnet50_2 -> Wide_ResNet50_2_Weights
    #   vit_b_16        -> ViT_B_16_Weights
    parts = model_name.split("_")
    nice = []
    for p in parts:
        if p.lower() == "vit":
            nice.append("ViT")
        elif len(p) == 1 and p.isalpha():
            nice.append(p.upper())
        else:
            # Handle common cases like "resnet" -> "ResNet".
            pl = p.lower()
            if pl.startswith("resnet") and len(p) > 6:
                nice.append("ResNet" + p[6:])
            else:
                nice.append(p[:1].upper() + p[1:])
    return "_".join(nice) + "_Weights"


def load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Load a torchvision backbone.

    Supports CNNs (e.g. wide_resnet50_2) and ViTs (e.g. vit_b_16).

    We keep this self-contained and use torchvision's Weights enums when available.
    """

    if not hasattr(torchvision.models, name):
        raise KeyError(f"torchvision.models has no attribute {name}")
    ctor = getattr(torchvision.models, name)

    kwargs = {}
    if pretrained:
        weights_enum = getattr(torchvision.models, _weights_enum_name(name), None)
        if weights_enum is not None:
            kwargs["weights"] = weights_enum.DEFAULT
        else:
            # Older torchvision
            kwargs["pretrained"] = True
    else:
        # Newer torchvision uses weights=None.
        kwargs["weights"] = None

    model = ctor(**kwargs)
    model.eval()
    return model
