"""
DenseNet model vertion 1
Auther: SPTAU
Date: September 2022
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["DenseNet121", "DenseNet169", "DenseNet201"]


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bottleneck_size: int = 4, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bottleneck_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneck_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, input_features: Tensor) -> Tensor:
        # concated_features = torch.cat(input_features, 1)

        new_features = self.conv1(self.relu1(self.bn1(input_features)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        new_features = torch.cat((input_features, new_features), 1)

        return new_features


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
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_input_features = num_init_features
        self.DenseBlock1 = self._make_DenseBlock(growth_rate, num_layers[0], num_input_features, drop_rate)
        num_input_features += num_layers[0] * growth_rate

        self.Transition1 = _Transition(num_input_features, num_input_features // 2)
        num_input_features = num_input_features // 2

        self.DenseBlock2 = self._make_DenseBlock(growth_rate, num_layers[1], num_input_features, drop_rate)
        num_input_features += num_layers[1] * growth_rate

        self.Transition2 = _Transition(num_input_features, num_input_features // 2)
        num_input_features = num_input_features // 2

        self.DenseBlock3 = self._make_DenseBlock(growth_rate, num_layers[2], num_input_features, drop_rate)
        num_input_features += num_layers[2] * growth_rate

        self.Transition3 = _Transition(num_input_features, num_input_features // 2)
        num_input_features = num_input_features // 2

        self.DenseBlock4 = self._make_DenseBlock(growth_rate, num_layers[3], num_input_features, drop_rate)
        num_input_features += num_layers[3] * growth_rate

        self.GlobleAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.FC = nn.Linear(num_input_features, num_classes)

    def _make_DenseBlock(self, growth_rate: int, num_layers: int, num_input_features: int, drop_rate: int) -> nn.Sequential:
        layers = []
        for i in range(int(num_layers)):
            layers.append(_DenseLayer(num_input_features, growth_rate, drop_rate=drop_rate))
            num_input_features += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        output = self.Conv(x)
        output = self.MaxPool(output)
        output = self.Transition1(self.DenseBlock1(output))
        output = self.Transition2(self.DenseBlock2(output))
        output = self.Transition3(self.DenseBlock3(output))
        output = self.DenseBlock4(output)
        output = self.GlobleAvgPool(output)
        output = torch.flatten(output, 1)
        output = self.FC(output)
        return output


def DenseNet121() -> DenseNet:
    return DenseNet(32, [6, 12, 24, 16])


def DenseNet169() -> DenseNet:
    return DenseNet(32, [6, 12, 32, 32])


def DenseNet201() -> DenseNet:
    return DenseNet(32, [6, 12, 48, 32])
