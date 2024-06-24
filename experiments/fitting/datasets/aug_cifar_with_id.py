# This file contains a CIFAR10 dataset that, in addition to the images, also returns the image ID (the index) of each image.
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import os
import pathlib


def create_augmented_cifar10(root, save_path, train, transform, target_transform, download):
    # Load CIFAR10 dataset
    t = [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip()]
    if transform is not None:
        t.extend(transform)
    aug_transform = transforms.Compose(t)
    cifar10 = torchvision.datasets.CIFAR10(root, train, aug_transform, target_transform, download)

    # Augment the images
    augmented_images = []
    augmented_labels = []
    for _ in range(50):
        for img, label in cifar10:
            augmented_images.append(img)
            augmented_labels.append(label)

    # Create random permutation of the augmented images
    perm = np.random.permutation(len(augmented_images))

    # Convert to numpy arrays
    augmented_images = np.array(augmented_images)[perm]
    augmented_labels = np.array(augmented_labels)[perm]

    # Store the augmented dataset
    np.save(save_path / "cifar-10-augmented.npy", augmented_images)
    np.save(save_path / "cifar-10-augmented-labels.npy", augmented_labels)


class AugmentedCIFAR10WithID(Dataset):
    """This class is a wrapper around the CIFAR10 dataset that returns the image ID (the index) of each image.
    It additionally implements the augmentations used in functa:
    - RandomHorizontalFlip
    - RandomCrop (32x32 crop of 40x40 padded image)
    50 per image, leading to a total of 50000 * 50 = 2500000 images.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):

        # Store transform
        self.transform = transform

        # Create directory for augmented CIFAR10 dataset
        path = pathlib.Path(root) / "cifar-10-augmented"
        if not os.path.exists(path):
            os.makedirs(path)

        # Check if the agumented dataset exists
        if not os.path.exists(path / "cifar-10-augmented.npy"):
            print("Augmented CIFAR10 dataset not found. Creating it now...")
            create_augmented_cifar10(root, path, train, None, None, download)

        # Load the augmented dataset
        augmented_images = np.load(path / "cifar-10-augmented.npy")
        augmented_labels = np.load(path / "cifar-10-augmented-labels.npy")

        self.augmented_images = augmented_images
        self.augmented_labels = augmented_labels

    def __getitem__(self, index):
        img = self.augmented_images[index]
        target = self.augmented_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index

    def __len__(self):
        return len(self.augmented_images)


if __name__ == "__main__":
    # Example usage
    dataset = AugmentedCIFAR10WithID(root='./data', train=True, download=True)
