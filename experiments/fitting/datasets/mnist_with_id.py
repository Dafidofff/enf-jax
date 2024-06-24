# This file contains an MNIST dataset that, in addition to the images, also returns the image ID (the index) of each image.

import torchvision
from torch.utils.data import Dataset


class MNISTWithID(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.mnist = torchvision.datasets.MNIST(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

    def __getitem__(self, index):
        img, target = self.mnist[index]
        return img, target, index

    def __len__(self):
        return len(self.mnist)