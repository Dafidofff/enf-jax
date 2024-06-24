import jax.numpy as jnp
from functools import partial

import optax
import jax

from experiments.fitting.trainers.shape._ad_shape_base_functions import AutodecodingShapeBaseFunctions
from experiments.fitting.trainers._base._ad_enf_trainer import AutoDecodingENFTrainer


class AutoDecodingENFTrainerShape(AutodecodingShapeBaseFunctions, AutoDecodingENFTrainer):

    def __init__(
            self,
            config,
            nef,
            train_autodecoder,
            val_autodecoder,
            train_loader,
            val_loader,
            seed
    ):
        AutodecodingShapeBaseFunctions.__init__(
            self=self
        )
        AutoDecodingENFTrainer.__init__(
            self=self,
            config=config,
            nef=nef,
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
        coords, sdf, _, shape_idx = batch

        # Unsqueeze sdf
        sdf = jax.nn.tanh(jnp.expand_dims(self.config.training.surf_delta_clamp**-1 * sdf, axis=-1))

        # Select correct autodecoder
        if train:
            apply_autodecoder = self.train_autodecoder.apply
        else:
            apply_autodecoder = self.val_autodecoder.apply

        def loss_fn(params):
            p, a, window = apply_autodecoder(params['autodecoder'], shape_idx)
            out = self.nef.apply(params['nef'], coords, p, a, window)

            # We apply tanh to both sdf and network output, since values closer to the surface are more important
            out = jax.nn.tanh(self.config.training.surf_delta_clamp**-1 * out)

            # Calculate clamped L1 loss
            l1_loss = jnp.sum(jnp.abs(out - sdf))
            # l1_loss = jnp.sum(jnp.abs(out - sdf))
            # bce_loss = -jnp.mean(occ * jnp.log(p_out + 1e-6) + (1 - occ) * jnp.log(1 - jax.nn.sigmoid(p_out) + 1e-6))
            # return mse_loss + bce_loss
            return l1_loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)

        # Update nef if we are training. Otherwise, only update autodecoder.
        if train:
            nef_updates, nef_opt_state = self.nef_opt.update(grads['nef'], state.nef_opt_state)
            nef_params = optax.apply_updates(state.params['nef'], nef_updates)
        else:
            nef_params = state.params['nef']
            nef_opt_state = state.nef_opt_state

        # Update autodecoder
        # grads['autodecoder'] = jax.tree_map(lambda x: x * x.shape[0], grads['autodecoder'])
        autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(grads['autodecoder'],
                                                                                 state.autodecoder_opt_state)
        autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)

        return loss, state.replace(
            params={'nef': nef_params, 'autodecoder': autodecoder_params},
            nef_opt_state=nef_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
        )

    def visualize_batch(self, state, batch, name, train=True):
        """ Visualizes the current batch. Also calculates mean sdf.

        Args:
            state: The current training state.
            batch: The current batch.
            name: The name of the visualization.
            train: Whether we are training or validating.
        """
        # Obtain latents
        coords, sdf, _, shape_idx = batch

        # Select correct autodecoder
        if train:
            apply_autodecoder = self.train_autodecoder.apply
        else:
            apply_autodecoder = self.val_autodecoder.apply
        p, a, window = apply_autodecoder(state.params['autodecoder'], shape_idx)

        # Visualize and log this batch
        self.visualize_and_log(state, batch, p, a, window, name=name)
