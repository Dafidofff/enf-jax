# This file contains a CIFAR10 dataset that, in addition to the images, also returns the image ID (the index) of each image.

import torchvision
from torch.utils.data import Dataset


class CIFAR10WithID(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.cifar10 = torchvision.datasets.CIFAR10(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.cifar10[index]
        return img, target, index

    def __len__(self):
        return len(self.cifar10)