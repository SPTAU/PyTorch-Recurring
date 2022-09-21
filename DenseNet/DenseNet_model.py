"""
DenseNet model vertion 1
Auther: SPTAU
Date: September 2022
"""
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["DenseNet121", "DenseNet169", "DenseNet201"]


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, input_features: Tensor) -> Tensor:
        concated_features = torch.cat(input_features, 1)

        new_features = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, growth_rate: int, num_layers: int, num_input_features: int, bn_size: int, drop_rate: int) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module("DenseLayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.Conv = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
        )
        self.AvgPool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        output = self.Conv(x)
        output = self.AvgPool(output)
        return output


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 32,
        num_layers: List[int] = [6, 12, 24, 16],
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "Convolution",
                        nn.Sequential(
                            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
                            nn.BatchNorm2d(num_init_features),
                            nn.ReLU(inplace=True),
                        ),
                    ),
                    ("MaxPool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layer in enumerate(num_layers):
            block = _DenseBlock(
                growth_rate=growth_rate,
                num_layers=num_layer,
                num_input_features=num_features,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.features.add_module("DenseBlock%d" % (i + 1), block)
            num_features = num_features + num_layer * growth_rate
            if i != len(num_layers) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("Transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        output = self.classifier(features)
        return output


def DenseNet121(num_classes: int = 1000) -> DenseNet:
    return DenseNet(32, [6, 12, 24, 16], 64, num_classes=num_classes)


def DenseNet169(num_classes: int = 1000) -> DenseNet:
    return DenseNet(32, [6, 12, 32, 32], 64, num_classes=num_classes)


def DenseNet201(num_classes: int = 1000) -> DenseNet:
    return DenseNet(32, [6, 12, 48, 32], 64, num_classes=num_classes)
