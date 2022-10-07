"""
ResNet CIFAR10 model
Auther: SPTAU
Date: September 2022
"""
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ResNet20", "ResNet32", "ResNet44", "ResNet56", "ResNet110", "ResNet1202"]


class _Block(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # transforming (batch_size * x * x * input_channel) to (batch_size * x * x * output_channel)
        #                                                   or (batch_size * x/2 * x/2 * output_channel)
        # floor(((x - 3 + 2 * 1) / stride) + 1) => floor(x) stride = 1
        #                                       => floor(x/2) stride = 2
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x' * x' * output_channel) to (batch_size * x' * x' * output_channel)
        # floor(((x' - 3 + 2 * 1) / 1) + 1) => floor(x')
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block: _Block, num_blocks: List[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_channel = 16
        # transforming (batch_size * 32 * 32 * input_channel) to (batch_size * 32 * 32 * 16)
        # floor(((32 - 3 + 2 * 1) / 1) + 1) => floor(112.5) => floor(112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        # transforming (batch_size * 32 * 32 * 16) to (batch_size * 32 * 32 * 16)
        self.conv2_x = self._make_layer(block, 16, num_blocks, stride=1)
        # transforming (batch_size * 32 * 32 * 16) to (batch_size * 16 * 16 * 32)
        self.conv3_x = self._make_layer(block, 32, num_blocks, stride=2)
        # transforming (batch_size * 16 * 16 * 32) to (batch_size * 8 * 8 * 64)
        self.conv4_x = self._make_layer(block, 64, num_blocks, stride=2)
        # transforming (batch_size * 8 * 8 * 64) to (batch_size * 1 * 1 * 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # transforming (batch_size * 64) to (batch_size * num_classes)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: _Block, out_channel: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet20() -> ResNet:
    return ResNet(_Block, 3)


def ResNet32() -> ResNet:
    return ResNet(_Block, 5)


def ResNet44() -> ResNet:
    return ResNet(_Block, 7)


def ResNet56() -> ResNet:
    return ResNet(_Block, 9)


def ResNet110() -> ResNet:
    return ResNet(_Block, 18)


def ResNet1202() -> ResNet:
    return ResNet(_Block, 200)
