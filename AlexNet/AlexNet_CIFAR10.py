"""
Auther: SPTAU
"""

import argparse
import os
import ssl

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm

from AlexNet import AlexNet_CIFAR10


ssl._create_default_https_context = ssl._create_unverified_context


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Training")
    parser.add_argument("--gpu", action="store_true", default=True, help="use gpu mode")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size in training")
    parser.add_argument("--num_epochs", type=int, default=100, help="epochs in training")
    return parser.parse_args()


def main():
    args = parse_args()

    ROOT_DIR = os.path.abspath(os.path.dirname(os.getcwd()))
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "CIFAR10")

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(root=DATASET_DIR, train=True, download=True, transform=transform)
    eval_dataset = datasets.CIFAR10(root=DATASET_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    model = AlexNet_CIFAR10()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # training epoch
    print("------Starting training------")
    for epoch in tqdm(range(args.num_epochs)):
        training_epoch_loss = 0.0
        training_correct, training_total = 0, 0
        training_loss = []
        training_acc = []

        for batch_idx, data in tqdm(enumerate(train_loader, start=0)):

            inputs, targets = data
            if args.gpu:
                inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            _, labels = torch.max(targets.data, dim=1)
            training_correct += (predicted == labels).sum().item()
            training_total += targets.size(0)

        training_epoch_loss = training_epoch_loss / batch_idx
        training_epoch_acc = 100 * training_correct / training_total
        training_loss.append(training_epoch_loss)
        training_acc.append(training_epoch_acc)

        print("[Epoch %3d] loss: %.3f acc: %.3f" % (epoch + 1, training_epoch_loss, training_epoch_acc))

    print("Training process has finished. Saving trained model.")
    SAVE_DIR = "./AlexNet_CIFAR10.pth"
    torch.save(model.state_dict(), SAVE_DIR)

    print("------Starting evaluating------")

    eval_loss = 0.0
    eval_correct, eval_total = 0, 0
    model.eval()  # 将模型调整为测试模型
    with torch.no_grad():  # 清空历史梯度，进行测试  与训练最大的区别是测试过程中取消了反向传播
        for batch_idx, data in tqdm(enumerate(eval_loader, start=0)):
            inputs, targets = data
            if args.gpu:
                inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            _, labels = torch.max(targets.data, dim=1)
            eval_correct += (predicted == labels).sum().item()
            eval_total += targets.size(0)

    eval_loss = eval_loss / batch_idx
    eval_acc = 100 * eval_correct / eval_total

    print("[Test    ] loss: %.3f acc: %.3f" % (eval_loss, eval_acc))

    fig, ax = plt.subplots(nrows=2, ncols=1)
    ax[0, 0].plot(np.arange(args.num_epochs), training_loss)
    ax[0, 0].set_ylabel("YLabel0")
    ax[0, 0].set_xlabel("XLabel0")
    ax[1, 0].plot(np.arange(args.num_epochs), training_acc)
    ax[1, 0].set_ylabel("YLabel0")
    ax[1, 0].set_xlabel("XLabel0")
    plt.show()


if __name__ == "__main__":
    main()
