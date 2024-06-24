from typing import Union, Sequence, Any
from torch.utils.data import Dataset, DataLoader
from functools import partial

from experiments.fitting.trainers._base._ad_enf_trainer import AutoDecodingSNeFTrainer
from experiments.fitting.trainers._base._ad_snef_trainer_meta import MetaAutoDecodingSNeFTrainer
from experiments.fitting.trainers._base._ad_enf_trainer_meta_sgd import MetaSGDAutoDecodingSNeFTrainer

import numpy as np


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


def perturb_positions(p, a, window, label, perturbation_scale=0.2):
    """Perturb the positions.

    Args:
        p (np.ndarray): The positions.
        a (np.ndarray): The appearance latents.
        window (np.ndarray): The window.
        perturbation_scale (float, optional): The perturbation scale. Defaults to 0.1.

    Returns:
        np.ndarray: The perturbed positions.
    """
    p_perturbed = p + np.random.randn(*p.shape) * perturbation_scale
    return p_perturbed, a, window, label


def perturb_appearance(p, a, window, label, perturbation_scale=0.2):
    """Perturb the appearance latents.

    Args:
        p (np.ndarray): The positions.
        a (np.ndarray): The appearance latents.
        window (np.ndarray): The window.
        perturbation_scale (float, optional): The perturbation scale. Defaults to 0.1.

    Returns:
        np.ndarray: The perturbed appearance latents.
    """
    a_perturbed = a + np.random.randn(*a.shape) * perturbation_scale
    return p, a_perturbed, window, label


def drop_latents(p, a, window, label, drop_rate=0.2):
    """Drop latents. Only mask out appearance

    Args:
        p (np.ndarray): The positions.
        a (np.ndarray): The appearance latents.
        window (np.ndarray): The window.
        drop_rate (float, optional): The drop rate. Defaults to 0.1.

    Returns:
        np.ndarray: The dropped latents.
    """
    mask = np.random.rand(*a.shape[:-1]) > drop_rate
    a_dropped = a * mask[..., None]
    return p, a_dropped, window, label


class LatentDataset(Dataset):

    def __init__(self, p, a, window, labels, transforms=None):
        self.p = p
        self.a = a
        self.window = window
        self.labels = labels

        if transforms is not None:
            self.transforms = transforms

    def __len__(self):
        return len(self.p)

    def __getitem__(self, idx):
        p, a, window, labels = self.p[idx], self.a[idx], self.window[idx], self.labels[idx]
        if hasattr(self, 'transforms'):
            for transform in self.transforms:
                p, a, window, labels = transform(p, a, window, labels)
        return self.p[idx], self.a[idx], self.window[idx], self.labels[idx]


def create_dataloader_from_trainer(cfg, state, snef_trainer: Union[AutoDecodingSNeFTrainer, MetaAutoDecodingSNeFTrainer, MetaSGDAutoDecodingSNeFTrainer]):
    """Create a dataset from a trainer.

    Args:
        snef_trainer (Union[AutoDecodingSNeFTrainer, MetaAutoDecodingSNeFTrainer, MetaSGDAutoDecodingSNeFTrainer]): The trainer.

    Returns:
        Dataset: The dataset.
    """

    # Initialize transforms to be applied to latent dataset.
    # train_transforms = [perturb_positions, perturb_appearance, drop_latents]
    train_transforms = []
    val_transforms = []

    if isinstance(snef_trainer, MetaAutoDecodingSNeFTrainer):

        # Do inner loop for train set
        tp, ta, twindow, tlabels = [], [], [], []
        for batch_idx, batch in enumerate(snef_trainer.train_loader):

            # Unpack the batch
            _, label_i, img_idx = batch

            # Do inner loop
            _, last_inner_state = snef_trainer.inner_loop(state.params, state, batch)

            # Get the latents
            p_pos_i = last_inner_state.params['autodecoder']['params']['p_pos']
            a_i = last_inner_state.params['autodecoder']['params']['a']
            window_i = last_inner_state.params['autodecoder']['params']['gaussian_window']

            # Append to the dataset
            tp.append(p_pos_i)
            ta.append(a_i)
            twindow.append(window_i)
            tlabels.append(label_i)

        snef_trainer.visualize_batch(state, batch, name='train_sample')

        # Concatenate the dataset
        tp = np.concatenate(tp, axis=0)
        ta = np.concatenate(ta, axis=0)
        twindow = np.concatenate(twindow, axis=0)
        tlabels = np.concatenate(tlabels, axis=0)

        # Do inner loop for val set
        vp, va, vwindow, vlabels = [], [], [], []
        for batch_idx, batch in enumerate(snef_trainer.val_loader):

            # Unpack the batch
            _, label, img_idx = batch

            # Do inner loop
            _, last_inner_state = snef_trainer.inner_loop(state.params, state, batch)

            # Get the latents
            p_pos_i = last_inner_state.params['autodecoder']['params']['p_pos']
            a_i = last_inner_state.params['autodecoder']['params']['a']
            window_i = last_inner_state.params['autodecoder']['params']['gaussian_window']

            # Append to the dataset
            vp.append(p_pos_i)
            va.append(a_i)
            vwindow.append(window_i)
            vlabels.append(label)

        # Visualize validation batch
        snef_trainer.visualize_batch(state, batch, name='val_sample')

        # Concatenate the dataset
        vp = np.concatenate(vp, axis=0)
        va = np.concatenate(va, axis=0)
        vwindow = np.concatenate(vwindow, axis=0)
        vlabels = np.concatenate(vlabels, axis=0)

        # Normalize the appearance latents
        mean_a = np.mean(ta, axis=0)        # B, Num-latens, Latent-dim
        std_a = np.std(ta, axis=0)
        ta_min, ta_max = np.min(ta, axis=0), np.max(ta, axis=0)
        # ta = (2* (ta - ta_min) / (ta_max - ta_min)) - 1
        # va = (2* (va - ta_min) / (ta_max - ta_min)) - 1
        # ta = (ta - mean_a) / std_a
        # va = (va - mean_a) / std_a

        # Get mean and std positions
        # mean_pos = np.mean(tp, axis=(0), keepdims=True)
        # std_pos = np.std(tp, axis=0)
        # tp = (tp - mean_pos) / std_pos
        # vp = (vp - mean_pos) / std_pos

        # Create the dataset
        train_dataset = LatentDataset(tp, ta, twindow, tlabels, train_transforms)
        val_dataset = LatentDataset(vp, va, vwindow, vlabels, val_transforms)
        train_dataset.min_a = ta_min
        train_dataset.max_a = ta_max

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=numpy_collate)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=numpy_collate)

        return train_loader, val_loader
    elif isinstance(snef_trainer, AutoDecodingSNeFTrainer):

        # Fit training set
        val_loader = snef_trainer.val_loader
        val_autodecoder = snef_trainer.val_autodecoder
        num_signals_test = snef_trainer.config.dataset.num_signals_test

        snef_trainer.val_autodecoder = snef_trainer.train_autodecoder
        snef_trainer.config.dataset.num_signals_test = snef_trainer.config.dataset.num_signals_train
        autodecoder = snef_trainer.train_autodecoder
        snef_trainer.val_loader = snef_trainer.train_loader
        # snef_trainer.create_functions()

        # Init the train state for correct param sizes
        # snef_trainer.init_train_state()
        train_state = snef_trainer.validate_epoch(state)

        # Get the dataset
        tp, ta, twindow, tlabels = [], [], [], []
        for batch_idx, batch in enumerate(snef_trainer.train_loader):

            # Unpack the batch
            _, label_i, img_idx = batch

            # Get the latents
            p_i, a_i, window_i = autodecoder.apply(train_state.params['autodecoder'], img_idx)

            # Append to the dataset
            tp.append(p_i)
            ta.append(a_i)
            twindow.append(window_i)
            tlabels.append(label_i)

        # Visualize validation batch
        snef_trainer.visualize_batch(train_state, batch, name='train_sample')

        # Concatenate the dataset
        tp = np.concatenate(tp, axis=0)
        ta = np.concatenate(ta, axis=0)
        twindow = np.concatenate(twindow, axis=0)
        tlabels = np.concatenate(tlabels, axis=0)

        # Fit validation set
        snef_trainer.val_loader = val_loader
        snef_trainer.val_autodecoder = val_autodecoder
        snef_trainer.config.dataset.num_signals_test = num_signals_test
        snef_trainer.create_functions()
        snef_trainer.init_train_state()
        val_state = snef_trainer.validate_epoch(state)

        # Get the dataset
        vp, va, vwindow, vlabels = [], [], [], []
        for batch_idx, batch in enumerate(snef_trainer.val_loader):

            # Unpack the batch
            _, label, img_idx = batch

            # Get the latents
            p_i, a_i, window_i = val_autodecoder.apply(val_state.params['autodecoder'], img_idx)

            # Append to the dataset
            vp.append(p_i)
            va.append(a_i)
            vwindow.append(window_i)
            vlabels.append(label)

        # Visualize validation batch
        snef_trainer.visualize_batch(val_state, batch, name='val_sample', train=False)

        # Concatenate the dataset
        vp = np.concatenate(vp, axis=0)
        va = np.concatenate(va, axis=0)
        vwindow = np.concatenate(vwindow, axis=0)
        vlabels = np.concatenate(vlabels, axis=0)

        # Normalize the appearance latents
        mean = np.mean(ta, axis=0)
        std = np.std(ta, axis=0)
        ta = (ta - mean) / std
        va = (va - mean) / std

        # Create the dataset
        train_dataset = LatentDataset(tp, ta, twindow, tlabels, train_transforms)
        val_dataset = LatentDataset(vp, va, vwindow, vlabels, val_transforms)

        train_dataset.mean = mean
        train_dataset.std = std

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=numpy_collate)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, collate_fn=numpy_collate)

        return train_loader, val_loader
