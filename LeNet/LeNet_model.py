"""
LeNet model
Auther: SPTAU
Date: March 2023
"""
import torch.nn as nn
from torch import sigmoid

__all__ = ["LeNet_CIFAR10"]


class LeNet_CIFAR10(nn.Module):
    def __init__(self):
        super(LeNet_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fl = nn.Flatten()
        self.fc1 = nn.Linear(1 * 1 * 120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = sigmoid(self.conv1(x))
        x = self.pool1(x)
        x = sigmoid(self.conv2(x))
        x = self.pool2(x)
        x = sigmoid(self.conv3(x))
        x = self.fl(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
