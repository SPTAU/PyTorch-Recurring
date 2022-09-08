# ImageNet Classification with Deep Convolutional Neural Networks
# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
# https://github.com/mengjizhiyou/pytorch_model/blob/main/AlexNet.py
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            # transforming (batch_size * 224 * 224 * input_channel) to (batch_size * 55 * 55 * 64)
            # floor(((input_size - conv_kernel_size + 2 * conv_padding)  / conv_stride + 1)
            # => floor(((224 - 11 + 2 * 2) / 4) + 1) => floor(55.25) => floor(55)
            nn.Conv2d(in_channel=input_channel, out_channel=64, kernel_size=11, stride=4, padding=2),
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
            # transforming (batch_size * 27 * 27 * 256) to (batch_size * 13 * 13 * 256)
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
            nn.Softmax(),
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
            nn.Conv2d(in_channel=input_channel, out_channel=96, kernel_size=11, stride=4, padding=2),
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
            nn.Softmax(),
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
