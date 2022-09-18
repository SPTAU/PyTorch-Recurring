"""
ResNet model
Auther: SPTAU
Date: September 2022
"""
from typing import List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


class BasicBlock(nn.Module):
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


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # transforming (batch_size * x * x * output_channel) to (batch_size * x * x * output_channel)
        # floor(((x - 3 + 2 * 1) / 1) + 1) => floor(x)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x * x * output_channel) to (batch_size * x * x * output_channel)
        #                                                    or (batch_size * x/2 * x/2 * output_channel)
        # floor(((x - 3 + 2 * 1) / stride) + 1) => floor(x) stride = 1
        #                                       => floor(x/2) stride = 2
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)  # 为什么 padding 为 1
        self.bn2 = nn.BatchNorm2d(out_channel)
        # transforming (batch_size * x' * x' * output_channel) to (batch_size * x' * x' * (output_channel* expansion))
        # floor(((x' - 3 + 2 * 1) / 1) + 1) => floor(x')
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
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
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], num_block: List[int], num_classes: int = 1000) -> None:
        super().__init__()
        self.in_channel = 64
        # transforming (batch_size * 224 * 224 * input_channel) to (batch_size * 112 * 112 * 64)
        # floor(((224 - 7 + 2 * 3) / 2) + 1) => floor(112.5) => floor(112)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False),  # bias参数的选取
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        # transforming (batch_size * 112 * 112 * 64) to (batch_size * 56 * 56 * 64)
        # floor(((112 - 3 + 2 * 1) / 2) + 1) => floor(56.5) => floor(56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # transforming (batch_size * 56 * 56 * 64) to (batch_size * 56 * 56 * (64 * block.expansion))
        self.conv2_x = self._make_layer(block, 64, num_block[0], stride=1)
        # transforming (batch_size * 56 * 56 * 256) to (batch_size * 28 * 28 * (256 * block.expansion))
        self.conv3_x = self._make_layer(block, 128, num_block[1], stride=2)
        # transforming (batch_size * 28 * 28 * 512) to (batch_size * 14 * 14 * (512 * block.expansion))
        self.conv4_x = self._make_layer(block, 256, num_block[2], stride=2)
        # transforming (batch_size * 14 * 14 * 1024) to (batch_size * 7 * 7 * (1024 * block.expansion))
        self.conv5_x = self._make_layer(block, 512, num_block[3], stride=2)
        # transforming (batch_size * 7 * 7 * 2048) to (batch_size * 1 * 1 * 2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 不知道为什么是这个东西  每个通道取最大值
        # transforming (batch_size * 2048) to (batch_size * num_classes)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block: Type[Union[BasicBlock, Bottleneck]], out_channel: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:  # 判断哪个layer
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, out_channel * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channel, out_channel, stride, downsample))
        self.in_channel = out_channel * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layers)  # (*)可以解压参数列表，即将列表解压为多个单参数
        # https://blog.csdn.net/weixin_40796925/article/details/107574267

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34() -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152() -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3])
