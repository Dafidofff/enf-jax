from experiments.downstream.latent_dataset.latent_dataset import (
    LatentDataset,
    perturb_positions,
    perturb_appearance,
    drop_latents
)
from experiments.downstream.latent_dataset.utils import get_or_create_latent_dataset_from_snef, get_latent_dataset

from torch.utils.data import DataLoader
import numpy as np
from typing import Any, Sequence, Union
import pathlib
from functools import partial


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


def get_augmentations(cfg):
    # Create augmentations
    train_transforms = []
    for k in cfg.latent_dataset.augmentations:
        if k == "perturb_pos":
            p_scale = cfg.latent_dataset.augmentations[k]
            train_transforms.append(partial(perturb_positions, perturbation_scale=p_scale))
        elif k == "perturb_a":
            p_scale = cfg.latent_dataset.augmentations[k]
            train_transforms.append(partial(perturb_positions, perturbation_scale=p_scale))
        elif k == "drop_a":
            p_drop = cfg.latent_dataset.augmentations[k]
            train_transforms.append(partial(drop_latents, drop_rate=p_drop))
    val_transforms = []
    return train_transforms, val_transforms


def get_latent_dataloader_from_snef(cfg, snef_state=None, snef_trainer=None):
    # Get dataset of latents
    train_latents, val_latents, train_labels, val_labels = get_or_create_latent_dataset_from_snef(
        cfg, snef_state, snef_trainer
    )

    # Get augmentations optionally
    train_transforms, val_transforms = get_augmentations(cfg)

    # Normalize the appearance latents
    if cfg.latent_dataset.normalize:
        # Normalize the appearance latents
        tp, ta, twindow = train_latents
        vp, va, vwindow = val_latents
        mean = np.mean(ta, axis=0)
        std = np.std(ta, axis=0)

        ta = (ta - mean) / std
        va = (va - mean) / std
        train_latents = (tp, ta, twindow)
        val_latents = (vp, va, vwindow)

    # Create the dataset
    train_dataset = LatentDataset(*train_latents, train_labels, train_transforms)
    val_dataset = LatentDataset(*val_latents, val_labels, val_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=numpy_collate)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=numpy_collate)
    return train_loader, val_loader


def get_latent_dataloader_from_path(cfg, path):
    # Get stored latents.
    train_latents, val_latents, train_labels, val_labels = get_latent_dataset(pathlib.Path(path))

    # Get augmentations optionally
    train_transforms, val_transforms = get_augmentations(cfg)

    # Normalize the appearance latents
    if cfg.latent_dataset.normalize:
        # Normalize the appearance latents
        tp, ta, twindow = train_latents
        vp, va, vwindow = val_latents
        mean = np.mean(ta, axis=0)
        std = np.std(ta, axis=0)

        ta = (ta - mean) / std
        va = (va - mean) / std
        train_latents = (tp, ta, twindow)
        val_latents = (vp, va, vwindow)

    # Create the dataset
    train_dataset = LatentDataset(*train_latents, train_labels, train_transforms)
    val_dataset = LatentDataset(*val_latents, val_labels, val_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=numpy_collate)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=numpy_collate)
    return train_loader, val_loader
