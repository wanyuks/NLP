import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os

root_dir = "../data"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)


def mnist_data():
    train_dataset = torchvision.datasets.MNIST(root=root_dir,
                                               transform=transforms.ToTensor(),
                                               download=True)
    test_dataset = torchvision.datasets.MNIST(root=root_dir,
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    return train_loader, test_loader

