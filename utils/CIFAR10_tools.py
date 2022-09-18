import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

__all__ = ["get_CIFAR10_mean_std"]


def get_CIFAR10_mean_std(dataset_dir) -> tuple(Tensor, Tensor):
    """


    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.

    Returns:
        mean, std

    """
    train_dataset = datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=T.ToTensor())
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in train_loader:  # 批量*通道*高*宽
        channels_sum += torch.mean(data, dim=[0, 2, 3])  # 剩下通道这个维度
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std
