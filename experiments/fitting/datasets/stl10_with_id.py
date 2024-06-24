# This file contains a STL10 dataset that, in addition to the images, also returns the image ID (the index) of each image.

import torchvision
from torch.utils.data import Dataset


class STL10WithID(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.stl10 = torchvision.datasets.STL10(
            root=root,
            split='train' if train else 'test',
            transform=transform,
            target_transform=target_transform,
            download=download
        )

    def __getitem__(self, index):
        img, target = self.stl10[index]
        return img, target, index

    def __len__(self):
        return len(self.stl10)
