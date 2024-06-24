import jax.numpy as jnp
from functools import partial

import optax
import jax

from experiments.fitting.trainers.image._ad_image_base_functions import AutodecodingImageBaseFunctions
from experiments.fitting.trainers._base._ad_enf_trainer import AutoDecodingENFTrainer


class AutoDecodingENFTrainerImage(AutodecodingImageBaseFunctions, AutoDecodingENFTrainer):

    def __init__(
            self,
            config,
            enf,
            train_autodecoder,
            val_autodecoder,
            train_loader,
            val_loader,
            coords,
            seed
    ):
        AutodecodingImageBaseFunctions.__init__(
            self=self,
            coords=coords
        )
        AutoDecodingENFTrainer.__init__(
            self=self,
            config=config,
            enf=enf,
            train_autodecoder=train_autodecoder,
            val_autodecoder=val_autodecoder,
            train_loader=train_loader,
            val_loader=val_loader,
            seed=seed
        )

    def step(self, state, batch, train=True):
        """Performs a single training step.

        Args:
            state (TrainState): The current training state.
            batch (dict): The current batch of data.
            train (bool): Whether we're training or validating. If training, we optimize both autodecoder and nef,
                otherwise only autodecoder.

        Returns:
            TrainState: The updated training state.
        """
        # Unpack batch, reshape images
        img, _, img_idx = batch
        img = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

        # Split random key
        rng, key = jax.random.split(state.rng)

        # Generate random mask of coordinates
        mask = jax.random.permutation(rng, jnp.arange(self.coords.shape[0]))[
               :self.config.training.max_num_sampled_points]

        # Mask the coordinates and labels
        coords = self.coords[mask]
        img = img[:, mask]

        # Broadcast the coordinates over the batch dimension
        coords = jnp.broadcast_to(coords, (img_idx.shape[0], *coords.shape))

        # Select correct autodecoder
        if train:
            apply_autodecoder = self.train_autodecoder.apply
        else:
            apply_autodecoder = self.val_autodecoder.apply

        def loss_fn(params):
            p, a, window = apply_autodecoder(params['autodecoder'], img_idx)
            out = self.enf.apply(params['enf'], coords, p, a, window)
            return jnp.mean((out - img) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        # Update enf if we are training. Otherwise, only update autodecoder.
        if train:
            enf_updates, enf_opt_state = self.enf_opt.update(grads['enf'], state.enf_opt_state)
            enf_params = optax.apply_updates(state.params['enf'], enf_updates)
        else:
            enf_params = state.params['enf']
            enf_opt_state = state.enf_opt_state

        # Update autodecoder, scale the gradients by the batch size, as we're averaging MSE over the batch
        grads['autodecoder'] = jax.tree_map(lambda x: x * x.shape[0], grads['autodecoder'])
        if not self.config.nef.optimize_gaussian_window:
            grads['autodecoder']['params']['gaussian_window'] = jnp.zeros_like(grads['autodecoder']['params']['gaussian_window'])
        autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(grads['autodecoder'],
                                                                                 state.autodecoder_opt_state)
        autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)

        return loss, state.replace(
            params={'enf': enf_params, 'autodecoder': autodecoder_params},
            enf_opt_state=enf_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
            rng=key
        )

    def visualize_batch(self, state, batch, name, train=True):
        """ Visualizes the current batch.

        Args:
            state: The current training state.
            batch: The current batch.
            name: The name of the visualization.
            train: Whether we are training or validating.
        """
        gt, _, sample_idx = batch
        if train:
            p, a, window = self.train_autodecoder.apply(state.params['autodecoder'], sample_idx)
        else:
            p, a, window = self.val_autodecoder.apply(state.params['autodecoder'], sample_idx)

        self.visualize_and_log(gt, state, p, a, window, name=name)
