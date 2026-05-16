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


def load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    # Torchvision model loader.
    if not hasattr(torchvision.models, name):
        raise KeyError(f"torchvision.models has no attribute {name}")
    ctor = getattr(torchvision.models, name)

    # Handle torchvision weights API.
    kwargs = {}
    try:
        if pretrained:
            # e.g. Wide_ResNet50_2_Weights.DEFAULT
            weights_enum = getattr(torchvision.models, f"{name.title().replace('_', '')}_Weights", None)
            if weights_enum is not None:
                kwargs["weights"] = weights_enum.DEFAULT
            else:
                kwargs["pretrained"] = True
        else:
            kwargs["weights"] = None
    except Exception:
        if pretrained:
            kwargs["pretrained"] = True

    model = ctor(**kwargs)
    model.eval()
    return model
