"""
AlexNet model
Auther: SPTAU
Date: September 2022
"""
import torch.nn as nn

__all__ = ["AlexNet", "AlexNet_Paper", "AlexNet_CIFAR10"]


class AlexNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            # transforming (batch_size * 224 * 224 * input_channel) to (batch_size * 55 * 55 * 64)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor(((224 - 11 + 2 * 2) / 4) + 1) => floor(55.25) => floor(55)
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 55 * 55 * 96) to (batch_size * 27 * 27 * 96)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((55 - 3) / 2) + 1) => floor(27)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            # transforming (batch_size * 27 * 27 * 64) to (batch_size * 27 * 27 * 192)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((27 - 5 + 2 * 2) + 1) => floor(27)
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 27 * 27 * 192) to (batch_size * 13 * 13 * 192)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((27 - 3) / 2) + 1) => floor(13)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            # transforming (batch_size * 13 * 13 * 192) to (batch_size * 13 * 13 * 384)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((13 - 3 + 2 * 1) + 1) => floor(13)
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            # transforming (batch_size * 13 * 13 * 384) to (batch_size * 13 * 13 * 256)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((13 - 3 + 2 * 1) + 1) => floor(13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            # transforming (batch_size * 13 * 13 * 256) to (batch_size * 13 * 13 * 256)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((13 - 3 + 2 * 1) + 1) => floor(13)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 13 * 13 * 256) to (batch_size * 6 * 6 * 256)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((13 - 3) / 2) + 1) => floor(6)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc1 = nn.Sequential(
            # transforming (batch_size * 6 * 6 * 256) to (batch_size * 9216)
            nn.Flatten(),
            # transforming (batch_size * 9216) to (batch_size * 4096)
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            # transforming (batch_size * 4096) to (batch_size * 4096)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Sequential(
            # transforming (batch_size * 4096) to (batch_size * num_classes)
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNet_Paper(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super(AlexNet_Paper, self).__init__()
        self.conv1 = nn.Sequential(
            # transforming (batch_size * 224 * 224 * input_channel) to (batch_size * 55 * 55 * 96)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor(((224 - 11 + 2 * 2) / 4) + 1) => floor(55.25) => floor(55)
            nn.Conv2d(in_channels=input_channel, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 55 * 55 * 96) to (batch_size * 27 * 27 * 96)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((55 - 3) / 2) + 1) => floor(27)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = nn.Sequential(
            # transforming (batch_size * 27 * 27 * 96) to (batch_size * 27 * 27 * 256)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((27 - 5 + 2 * 2) + 1) => floor(27)
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 27 * 27 * 256) to (batch_size * 13 * 13 * 256)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((27 - 3) / 2) + 1) => floor(13)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv3 = nn.Sequential(
            # transforming (batch_size * 13 * 13 * 256) to (batch_size * 13 * 13 * 384)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((13 - 3 + 2 * 1) + 1) => floor(13)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            # transforming (batch_size * 13 * 13 * 384) to (batch_size * 13 * 13 * 384)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((13 - 3 + 2 * 1) + 1) => floor(13)
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            # transforming (batch_size * 13 * 13 * 384) to (batch_size * 13 * 13 * 256)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((13 - 3 + 2 * 1) + 1) => floor(13)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 13 * 13 * 256) to (batch_size * 6 * 6 * 256)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((13 - 3) / 2) + 1) => floor(6)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.fc1 = nn.Sequential(
            # transforming (batch_size * 6 * 6 * 256) to (batch_size * 9216)
            nn.Flatten(),
            # transforming (batch_size * 9216) to (batch_size * 4096)
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            # transforming (batch_size * 4096) to (batch_size * 4096)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Sequential(
            # transforming (batch_size * 4096) to (batch_size * num_classes)
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class AlexNet_CIFAR10(nn.Module):
    def __init__(self):
        super(AlexNet_CIFAR10, self).__init__()
        self.conv1 = nn.Sequential(
            # transforming (batch_size * 32 * 32 * input_channel) to (batch_size * 30 * 30 * 64)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor((32 - 3 + 2 * 1) + 1) => floor(32) => floor(32)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 30 * 30 * 64) to (batch_size * 15 * 15 * 64)
            # floor(((input_size - padding_kernel_size + 2 * padding_padding)  / padding_stride + 1)
            # => floor(((32 - 2) / 2) + 1) => floor(16)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            # transforming (batch_size * 16 * 16 * 64) to (batch_size * 16 * 16 * 256)
            # floor((16 - 3 + 2 * 1) + 1) => floor(16)
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 16 * 16 * 256) to (batch_size * 8 * 8 * 256)
            # floor(((16 - 2) / 2) + 1) => floor(8)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            # transforming (batch_size * 8 * 8 * 256) to (batch_size * 8 * 8 * 384)
            # floor((8 - 3 + 2 * 1) + 1) => floor(8)
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            # transforming (batch_size * 8 * 8 * 384) to (batch_size * 8 * 8 * 256)
            # floor((8 - 3 + 2 * 1) + 1) => floor(8)
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            # transforming (batch_size * 8 * 8 * 256) to (batch_size * 8 * 8 * 256)
            # floor((8 - 3 + 2 * 1) + 1) => floor(8)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # transforming (batch_size * 8 * 8 * 256) to (batch_size * 4 * 4 * 256)
            # floor(((8 - 2) / 2) + 1) => floor(4)
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            # transforming (batch_size * 4 * 4 * 256) to (batch_size * 4096)
            nn.Flatten(),
            # transforming (batch_size * 4096) to (batch_size * 4096)
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc2 = nn.Sequential(
            # transforming (batch_size * 4096) to (batch_size * 4096)
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Sequential(
            # transforming (batch_size * 4096) to (batch_size * 10)
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
