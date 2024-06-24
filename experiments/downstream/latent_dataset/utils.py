import numpy as np
import os
import pathlib
import logging

from tqdm import tqdm

# For typing
from experiments.fitting.trainers._base._ad_enf_trainer import AutoDecodingENFTrainer
from experiments.fitting.trainers._base._ad_enf_trainer_meta_sgd import MetaSGDAutoDecodingENFTrainer


def fit_latents_meta(enf_state, enf_trainer):
    # Do inner loop for train set
    tp, tc, twindow, tlabels = [], [], [], []
    for batch_idx, batch in tqdm(enumerate(enf_trainer.train_loader), desc="Fitting latents for train set", total=len(enf_trainer.train_loader)):
        # Unpack the batch
        if 'shape' in enf_trainer.config.dataset.name:
            _, _, label_i, _ = batch
        else:
            _, label_i, _ = batch

        # Do inner loop
        _, last_inner_state = enf_trainer.inner_loop(enf_state.params, enf_state, batch)

        # Get the latents
        p_pos_i = last_inner_state.params['autodecoder']['params']['p_pos']
        c_i = last_inner_state.params['autodecoder']['params']['c']
        window_i = last_inner_state.params['autodecoder']['params']['gaussian_window']

        # Append to the dataset
        tp.append(p_pos_i)
        tc.append(c_i)
        twindow.append(window_i)
        tlabels.append(label_i)

    enf_trainer.visualize_batch(enf_state, batch, name='train_sample')

    # Concatenate the dataset
    tp = np.concatenate(tp, axis=0)
    tc = np.concatenate(tc, axis=0)
    twindow = np.concatenate(twindow, axis=0)
    tlabels = np.concatenate(tlabels, axis=0)

    # Do inner loop for val set
    vp, vc, vwindow, vlabels = [], [], [], []
    for batch_idx, batch in tqdm(enumerate(enf_trainer.val_loader), desc="Fitting latents for val set", total=len(enf_trainer.val_loader)):
        # Unpack the batch
        if 'shape' in enf_trainer.config.dataset.name:
            _, _, label, _ = batch
        else:
            _, label, _ = batch

        # Do inner loop
        _, last_inner_state = enf_trainer.inner_loop(enf_state.params, enf_state, batch)

        # Get the latents
        p_pos_i = last_inner_state.params['autodecoder']['params']['p_pos']
        c_i = last_inner_state.params['autodecoder']['params']['c']
        window_i = last_inner_state.params['autodecoder']['params']['gaussian_window']

        # Append to the dataset
        vp.append(p_pos_i)
        vc.append(c_i)
        vwindow.append(window_i)
        vlabels.append(label)

    # Visualize validation batch
    enf_trainer.visualize_batch(enf_state, batch, name='val_sample')

    # Concatenate the dataset
    vp = np.concatenate(vp, axis=0)
    vc = np.concatenate(vc, axis=0)
    vwindow = np.concatenate(vwindow, axis=0)
    vlabels = np.concatenate(vlabels, axis=0)

    return [tp, tc, twindow], [vp, vc, vwindow], tlabels, vlabels


def fit_latents(enf_state, enf_trainer):
    """ Function for fitting the latents for training and validation datasets attached to enf_trainers.

    Args:
        enf_state: The state of the enf model.
        enf_trainer: The autodecoding trainer.
    """
    # We're using the validation loop to fit the latents, so we swap out the validation set
    # for the training set.
    val_loader = enf_trainer.val_loader
    val_autodecoder = enf_trainer.val_autodecoder
    num_signals_test = enf_trainer.config.dataset.num_signals_test

    # Create training functions
    enf_trainer.val_autodecoder = enf_trainer.train_autodecoder
    enf_trainer.config.dataset.num_signals_test = enf_trainer.config.dataset.num_signals_train
    autodecoder = enf_trainer.train_autodecoder
    enf_trainer.val_loader = enf_trainer.train_loader
    enf_trainer.create_functions()

    # Init the train state for correct param sizes
    enf_trainer.init_train_state()
    train_state = enf_trainer.validate_epoch(enf_state)

    # Get the dataset
    tp, ta, twindow, tlabels = [], [], [], []
    for batch_idx, batch in tqdm(enumerate(enf_trainer.train_loader), desc="Fitting latents for train set", total=len(enf_trainer.train_loader)):
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
    enf_trainer.visualize_batch(train_state, batch, name='train_sample')

    # Concatenate the dataset
    tp = np.concatenate(tp, axis=0)
    ta = np.concatenate(ta, axis=0)
    twindow = np.concatenate(twindow, axis=0)
    tlabels = np.concatenate(tlabels, axis=0)

    # Fit validation set, restore the original validation set and decoder.
    enf_trainer.val_loader = val_loader
    enf_trainer.val_autodecoder = val_autodecoder
    enf_trainer.config.dataset.num_signals_test = num_signals_test
    enf_trainer.create_functions()
    enf_trainer.init_train_state()
    val_state = enf_trainer.validate_epoch(enf_state)

    # Get the dataset
    vp, va, vwindow, vlabels = [], [], [], []
    for batch_idx, batch in tqdm(enumerate(enf_trainer.val_loader), desc="Fitting latents for val set", total=len(enf_trainer.val_loader)):
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
    enf_trainer.visualize_batch(val_state, batch, name='val_sample', train=False)

    # Concatenate the dataset
    vp = np.concatenate(vp, axis=0)
    va = np.concatenate(va, axis=0)
    vwindow = np.concatenate(vwindow, axis=0)
    vlabels = np.concatenate(vlabels, axis=0)

    return [tp, ta, twindow], [vp, va, vwindow], tlabels, vlabels


def create_latent_dataset_from_enf(enf_trainer, enf_state):
    if isinstance(enf_trainer, MetaSGDAutoDecodingENFTrainer):
        train_latents, val_latents, train_labels, val_labels = fit_latents_meta(enf_state, enf_trainer)
    elif isinstance(enf_trainer, AutoDecodingENFTrainer):
        train_latents, val_latents, train_labels, val_labels = fit_latents(enf_state, enf_trainer)
    else:
        raise ValueError("Incorrect trainer specified.")
    return train_latents, val_latents, train_labels, val_labels


def get_or_create_latent_dataset_from_enf(cfg, enf_state, enf_trainer):
    """
    Get or create the latent dataset from the enf model.
    
    Args:
        cfg: The configuration for the latent dataset.
        enf_state: The state of the enf model.
        enf_trainer: The autodecoding trainer.
        
    Returns:
        train_latents (np.ndarray): The latents for the training dataset.
        val_latents (np.ndarray): The latents for the validation dataset.
        train_labels (np.ndarray): The labels for the training dataset.
        val_labels (np.ndarray): The labels for the validation dataset.
    """
    
    # Determine the latent dataset path.
    latent_dataset_path = pathlib.Path(cfg.checkpoint_dir) / "latent_dataset"

    # If the latent dataset is already created, load it and check whether the config matches.
    if cfg.latent_dataset.load and os.path.exists(latent_dataset_path):
        train_latents, val_latents, train_labels, val_labels = get_latent_dataset(latent_dataset_path)
    else:
        logging.info("Latent dataset not found... creating a new one.")
        train_latents, val_latents, train_labels, val_labels = create_latent_dataset_from_enf(enf_state, enf_trainer)

        # Save the latent dataset.
        if cfg.latent_dataset.store_if_new:
            logging.info("Storing latent dataset...")
            save_latent_dataset(latent_dataset_path, train_latents, val_latents, train_labels, val_labels)
    return train_latents, val_latents, train_labels, val_labels


def get_latent_dataset(path: pathlib.Path):
    """ Load the latent dataset from a given path.

    Args:
        path: The path to the latent dataset.

    Returns:
        train_latents (np.ndarray): The latents for the training dataset.
        val_latents (np.ndarray): The latents for the validation dataset.
        train_labels (np.ndarray): The labels for the training dataset.
        val_labels (np.ndarray): The labels for the validation dataset.
    """
    # Simply load in the latent dataset.
    train_p = np.load(path / "train_p.npy")
    train_a = np.load(path / "train_a.npy")
    train_window = np.load(path / "train_window.npy")
    train_latents = [train_p, train_a, train_window]
    train_labels = np.load(path / "train_labels.npy")

    val_p = np.load(path / "val_p.npy")
    val_a = np.load(path / "val_a.npy")
    val_window = np.load(path / "val_window.npy")
    val_latents = [val_p, val_a, val_window]
    val_labels = np.load(path / "val_labels.npy")
    return train_latents, val_latents, train_labels, val_labels


def save_latent_dataset(path: pathlib.Path, train_latents, val_latents, train_labels, val_labels):
    """ Store the latent dataset to a given path.

    Args:
        path: The path to the latent dataset.
    """
    # Save the latent dataset.
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "train_p.npy", train_latents[0])
    np.save(path / "train_a.npy", train_latents[1])
    np.save(path / "train_window.npy", train_latents[2])
    np.save(path / "train_labels.npy", train_labels)

    np.save(path / "val_p.npy", val_latents[0])
    np.save(path / "val_a.npy", val_latents[1])
    np.save(path / "val_window.npy", val_latents[2])
    np.save(path / "val_labels.npy", val_labels)
