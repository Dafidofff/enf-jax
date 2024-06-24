from experiments.fitting.datasets.celebA_with_id import CelebAWithID
from experiments.fitting.datasets.stl10_with_id import STL10WithID
from experiments.fitting.datasets.cifar_with_id import CIFAR10WithID
from experiments.fitting.datasets.mnist_with_id import MNISTWithID
from experiments.fitting.datasets.shapenet_with_id import ShapeNet
from experiments.fitting.datasets.aug_cifar_with_id import AugmentedCIFAR10WithID
from experiments.fitting.datasets.shapenet_sdf_with_id import ShapeNetSDF
from experiments.fitting.datasets.dft_with_id import DFTWithID

from typing import Union, Any, Sequence

import numpy as np
from torch.utils import data
import torchvision


def image_to_numpy(image):
    return np.array(image) / 255


def add_channel_axis(image: np.ndarray):
    return image[..., np.newaxis]


def permute_image_channels(image: np.ndarray):
    if len(image.shape) == 3:
        return np.moveaxis(image, 2, 0)
    else:
        return image


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """
    TODO: this might be a repeat, maybe it's ok to make it special for shapes, but needs a check
    Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def get_dataloader(dataset_cfg):
    if dataset_cfg.name == "stl10":
        transforms = torchvision.transforms.Compose([image_to_numpy])

        train_dset = STL10WithID(
            root=dataset_cfg.path,
            train=True,
            transform=transforms,
            download=True,
        )

        test_dset = STL10WithID(
            root=dataset_cfg.path,
            train=False,
            transform=transforms,
            download=True,
        )

    elif dataset_cfg.name == "celeba":

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((64, 64)),
            # torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            image_to_numpy
        ])

        train_dset = CelebAWithID(
            root=dataset_cfg.path,
            train=True,
            transform=transforms,
            download=False,
        )

        test_dset = CelebAWithID(
            root=dataset_cfg.path,
            train=False,
            transform=transforms,
            download=False,
        )

    elif dataset_cfg.name == "cifar10":

        transforms = torchvision.transforms.Compose([image_to_numpy])

        train_dset = CIFAR10WithID(
            root=dataset_cfg.path,
            train=True,
            transform=transforms,
            download=True,
        )

        test_dset = CIFAR10WithID(
            root=dataset_cfg.path,
            train=False,
            transform=transforms,
            download=True,
        )

    elif dataset_cfg.name == "augmented_cifar10":

        transforms = torchvision.transforms.Compose([image_to_numpy])

        train_dset = AugmentedCIFAR10WithID(
            root=dataset_cfg.path,
            train=True,
            transform=transforms,
            download=True,
        )

        test_dset = CIFAR10WithID(
            root=dataset_cfg.path,
            train=False,
            transform=transforms,
            download=True,
        )

    elif dataset_cfg.name == "mnist":

        transforms = torchvision.transforms.Compose([image_to_numpy, add_channel_axis])

        train_dset = MNISTWithID(
            root=dataset_cfg.path,
            train=True,
            transform=transforms,
            download=True,
        )

        test_dset = MNISTWithID(
            root=dataset_cfg.path,
            train=False,
            transform=transforms,
            download=True,
        )

    elif dataset_cfg.name == "shapenet":

        full_dset = ShapeNet(
            root=dataset_cfg.path,
            num_points=(dataset_cfg.max_num_sampled_points//4, (dataset_cfg.max_num_sampled_points//4) * 3),
            seed=42,
        )

        total_num_samples = len(full_dset)

        if dataset_cfg.num_signals_train == -1:
            train_dset, test_dset = data.Subset(full_dset, np.arange(0, total_num_samples * 0.8)), data.Subset(full_dset, np.arange(total_num_samples * 0.8, total_num_samples))
        else:
            assert len(full_dset) >= dataset_cfg.num_signals_train + dataset_cfg.num_signals_test, \
                "Not enough samples in dataset for specified train and test sizes."

            train_dset, test_dset = (data.Subset(full_dset, np.arange(0, dataset_cfg.num_signals_train)),
                                     data.Subset(full_dset, np.arange(dataset_cfg.num_signals_train, dataset_cfg.num_signals_train + dataset_cfg.num_signals_test)))

    elif dataset_cfg.name == "shapenet_sdf":

        full_dset = ShapeNetSDF(
            root=dataset_cfg.path,
            num_points=(dataset_cfg.max_num_sampled_points//2, dataset_cfg.max_num_sampled_points//2),
            seed=42,
        )

        total_num_samples = len(full_dset)

        if dataset_cfg.num_signals_train == -1:
            train_dset, test_dset = data.Subset(full_dset, np.arange(0, total_num_samples * 0.8)), data.Subset(
                full_dset, np.arange(total_num_samples * 0.8, total_num_samples))
        else:
            assert len(full_dset) >= dataset_cfg.num_signals_train + dataset_cfg.num_signals_test, \
                "Not enough samples in dataset for specified train and test sizes."

            train_dset, test_dset = (data.Subset(full_dset, np.arange(0, dataset_cfg.num_signals_train)),
                                     data.Subset(full_dset, np.arange(dataset_cfg.num_signals_train,
                                                                      dataset_cfg.num_signals_train + dataset_cfg.num_signals_test)))
            
    elif dataset_cfg.name == "dft":
        full_dset = DFTWithID(
            root=dataset_cfg.path,
            train=True,
            transform=None,
            download=True,
        )

        total_num_samples = len(full_dset)

        if dataset_cfg.num_signals_train == -1:
            train_dset, test_dset = data.Subset(full_dset, np.arange(0, total_num_samples * 0.8)), data.Subset(
                full_dset, np.arange(total_num_samples * 0.8, total_num_samples))
        else:
            assert total_num_samples >= dataset_cfg.num_signals_train + dataset_cfg.num_signals_test, \
                "Not enough samples in dataset for specified train and test sizes."

            train_dset, test_dset = (data.Subset(full_dset, np.arange(0, dataset_cfg.num_signals_train)),
                                     data.Subset(full_dset, np.arange(dataset_cfg.num_signals_train,
                                                                      dataset_cfg.num_signals_train + dataset_cfg.num_signals_test)))

    else:
        raise ValueError(f"Unknown dataset name: {dataset_cfg.name}")

    if dataset_cfg.num_signals_train != -1:
        train_dset = data.Subset(train_dset, np.arange(0, dataset_cfg.num_signals_train))
    if dataset_cfg.num_signals_test != -1:
        test_dset = data.Subset(test_dset, np.arange(0, dataset_cfg.num_signals_test))

    train_loader = data.DataLoader(
        train_dset,
        batch_size=dataset_cfg.batch_size,
        shuffle=True,
        num_workers=dataset_cfg.num_workers,
        collate_fn=numpy_collate,
        persistent_workers=False,
        drop_last=True
    )

    test_loader = data.DataLoader(
        test_dset,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        collate_fn=numpy_collate,
        persistent_workers=False,
        drop_last=True
    )

    return train_loader, test_loader
