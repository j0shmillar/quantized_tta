from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models.quantization import (
    resnet18, ResNet18_QuantizedWeights,
    mobilenet_v2, MobileNet_V2_QuantizedWeights,
)


class QuantizedFC(nn.Module):
    def __init__(self, arch, num_classes):
        super().__init__()
        arch = arch.lower()
        self.arch = arch
        if arch == "resnet18":
            m = resnet18(weights=ResNet18_QuantizedWeights.DEFAULT, quantize=True)
            in_features = m.fc.in_features
            m.fc = nn.Identity()
        elif arch == "mobilenet_v2":
            m = mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True)
            in_features = m.classifier[1].in_features
            m.classifier = nn.Identity()
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.featurizer = m
        self.classifier = nn.Linear(in_features, num_classes)
        self.n_outputs = in_features

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.featurizer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(arch, num_classes, checkpoint = None, map_location="cpu") -> QuantizedFC:
    model = QuantizedFC(arch=arch, num_classes=num_classes)
    if checkpoint:
        payload = torch.load(checkpoint, map_location=map_location)
        state = payload.get("model", payload)
        model.load_state_dict(state, strict=True)
    return model
