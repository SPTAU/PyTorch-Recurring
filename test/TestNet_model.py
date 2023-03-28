"""
TestNet model
Auther: SPTAU
Date: March 2023
"""
import torch.nn as nn
from torch import sigmoid

__all__ = ["TestNet_CIFAR10"]


class TestNet_CIFAR10(nn.Module):
    def __init__(self):
        super(TestNet_CIFAR10, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=32)
        self.fl = nn.Flatten()
        self.fc = nn.Linear(3, 10)

    def forward(self, x):
        x = sigmoid(self.conv(x))
        x = self.fl(x)
        x = self.fc(x)
        return x
