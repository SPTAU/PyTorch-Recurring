"""
ResNet CIFAR10
Auther: SPTAU
Date: September 2022
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from ResNet_CIFAR10_model import ResNet110
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.CIFAR10_tools import get_CIFAR10_mean_std  # noqa: E402


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser(description="ResNet CIFAR10 Training")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size in training")
    parser.add_argument("--num_epochs", type=int, default=20, help="epochs in training")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    return parser.parse_args()


def main():
    args = parse_args()

    ROOT_DIR = os.getcwd()
    DATASET_DIR = os.path.join(ROOT_DIR, "dataset", "CIFAR10")

    writer = SummaryWriter("./ResNet/CIFAR10/runs")

    CIFAR10_mean, CIFAR10_std = get_CIFAR10_mean_std(DATASET_DIR)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=CIFAR10_mean, std=CIFAR10_std),
        ]
    )

    train_dataset = datasets.CIFAR10(root=DATASET_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=DATASET_DIR, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    test_size = len(test_dataset)

    train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = ResNet110()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # training epoch
    print("------Starting training------")
    train_loss, train_acc, eval_loss, eval_acc = [], [], [], []
    for epoch in tqdm(range(args.num_epochs)):

        training_epoch_loss, training_epoch_acc = 0.0, 0.0
        training_temp_loss, training_temp_correct = 0, 0
        model.train()
        for batch_idx, data in enumerate(train_loader, start=0):

            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_temp_loss += loss.item()
            predicted = torch.argmax(outputs.data, dim=1)
            training_temp_correct += (predicted == targets).sum().item()

        training_epoch_loss = training_temp_loss / batch_idx
        training_epoch_acc = 100 * training_temp_correct / train_size

        writer.add_scalar("LOSS/Train_loss", float(training_epoch_loss), (epoch + 1))
        writer.add_scalar("ACC/Train_acc", float(training_epoch_acc), (epoch + 1))

        train_loss.append(training_epoch_loss)
        train_acc.append(training_epoch_acc)

        evaling_epoch_loss, evaling_epoch_acc = 0.0, 0.0
        evaling_temp_loss, evaling_temp_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(eval_loader, start=0):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                evaling_temp_loss += loss.item()
                predicted = torch.argmax(outputs.data, dim=1)
                evaling_temp_correct += (predicted == targets).sum().item()

        evaling_epoch_loss = evaling_temp_loss / batch_idx
        evaling_epoch_acc = 100 * evaling_temp_correct / eval_size

        writer.add_scalar("LOSS/Valid_loss", float(evaling_epoch_loss), (epoch + 1))
        writer.add_scalar("ACC/Valid_acc", float(evaling_epoch_acc), (epoch + 1))

        eval_loss.append(evaling_epoch_loss)
        eval_acc.append(evaling_epoch_acc)

        print(
            "[Epoch {:3d}] train_loss: {:.3f} train_acc: {:.3f}% eval_loss: {:.3f} eval_acc: {:.3f}%".format(
                epoch + 1, training_epoch_loss, training_epoch_acc, evaling_epoch_loss, evaling_epoch_acc
            )
        )

    print("Training process has finished. Saving trained model.")
    torch.save(model.state_dict(), "./ResNet/CIFAR10/ResNet_CIFAR10.pth")

    print("------Starting testing------")
    testing_temp_loss, testing_temp_correct = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, start=0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            testing_temp_loss += loss.item()
            predicted = torch.argmax(outputs.data, dim=1)
            testing_temp_correct += (predicted == targets).sum().item()

    testing_loss = testing_temp_loss / batch_idx
    testing_acc = 100 * testing_temp_correct / test_size

    print("[Test     ] loss: {:.3f} acc: {:.3f}%%".format(testing_loss, testing_acc))

    fig, ax = plt.subplots()
    ax.plot(np.arange(args.num_epochs), train_loss, np.arange(args.num_epochs), eval_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(["train_loss", "eval_loss"])
    plt.savefig("./ResNet/CIFAR10/ResNet_CIFAR10_loss.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(np.arange(args.num_epochs), train_acc, np.arange(args.num_epochs), eval_acc)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("acc")
    ax.legend(["train_acc", "eval_acc"])
    plt.savefig("./ResNet/CIFAR10/ResNet_CIFAR10_acc.png")
    plt.show()

    fig, axs = plt.subplots(nrows=3, ncols=3)
    for col in range(3):
        for row in range(3):
            ax = axs[row, col]
            img_data, label_id = random.choice(list(zip(test_dataset.data, test_dataset.targets)))
            img = T.ToPILImage()(img_data)
            predict_id = torch.argmax(model(transform(img).unsqueeze(0).to(device)))
            predict = test_dataset.classes[predict_id]
            label = test_dataset.classes[label_id]
            ax.imshow(img)
            ax.set_title("truth:{}\npredict:{}".format(label, predict))
            ax.axis("off")
    plt.savefig("./ResNet/CIFAR10/ResNet_CIFAR10_test.png")
    plt.show()


if __name__ == "__main__":
    main()
