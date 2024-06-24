import wandb
import jax.numpy as jnp
import jax
import optax
from functools import partial

from experiments.fitting.trainers.shape._ad_shape_base_functions import AutodecodingShapeBaseFunctions
from experiments.fitting.trainers._base._ad_enf_trainer_meta_sgd import MetaSGDAutoDecodingSNeFTrainer


class MetaSGDAutoDecodingSNeFTrainerShape(AutodecodingShapeBaseFunctions, MetaSGDAutoDecodingSNeFTrainer):

    def __init__(
            self,
            config,
            nef,
            inner_autodecoder,
            outer_autodecoder,
            train_loader,
            val_loader,
            seed
    ):
        AutodecodingShapeBaseFunctions.__init__(
            self=self,
        )
        MetaSGDAutoDecodingSNeFTrainer.__init__(
            self=self,
            config=config,
            nef=nef,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            train_loader=train_loader,
            val_loader=val_loader,
            seed=seed
        )

    @partial(jax.jit, static_argnums=(0,))
    def inner_loop(self, outer_params, outer_state, batch):
        # Unpack batch, reshape images
        coords, sdf, _, shape_idx = batch

        # Determine num points per inner step
        num_p_inner = sdf.shape[1] / self.config.training.num_inner_steps

        # Unsqueeze sdf [batch, num_points] -> [batch, num_points, 1], apply tanh
        sdf = jax.nn.tanh(jnp.expand_dims(self.config.training.surf_delta_clamp**-1 * sdf, axis=-1))

        # Broadcast autodecoder params over the batch dimension
        inner_autodecoder_params = jax.tree_map(
            lambda p: jnp.repeat(p, sdf.shape[0], axis=0), outer_params['autodecoder']
        )

        # Create inner state
        inner_state = outer_state.replace(
            params={'nef': outer_params['nef'], 'autodecoder': inner_autodecoder_params, 'meta_sgd_lrs': outer_params['meta_sgd_lrs']}
        )

        # Define loss function
        def loss_fn(params, masked_coords, masked_sdf):
            p, a, window = self.inner_autodecoder.apply(params['autodecoder'])
            out = self.nef.apply(params['nef'], masked_coords, p, a, window)

            # We apply tanh to both sdf and network output, since values closer to the surface are more important
            out = jax.nn.tanh(self.config.training.surf_delta_clamp**-1 * out)

            # Calculate clamped L1 loss
            l1_loss = jnp.sum(jnp.abs(out - masked_sdf))
            return l1_loss

        def apply_inner_lr(key, grad, lrs):
            # Get meta learning rates for p, a, and gaussian window
            if key[-1].key in lrs:
                return - lrs[key[-1].key] * grad
            else:
                raise ValueError(f'Unknown key: {key}')

        # Create inner grad fn
        inner_grad_fn = jax.grad(loss_fn)

        # Do inner loop
        for inner_step in range(self.config.training.num_inner_steps):
            # Mask the coordinates and labels
            masked_coords = coords[:, int(inner_step * num_p_inner):int((inner_step + 1) * num_p_inner)]
            masked_sdf = sdf[:, int(inner_step * num_p_inner):int((inner_step + 1) * num_p_inner)]
            
            # Get inner grads and update inner autodecoder params
            inner_grad = inner_grad_fn(
                inner_state.params,
                masked_coords=masked_coords,
                masked_sdf=masked_sdf,
            )

            # Optionally zero out gaussian window param updates.
            if not self.config.nef.optimize_gaussian_window:
                inner_grad['autodecoder']['params']['gaussian_window'] = jnp.zeros_like(
                    inner_grad['autodecoder']['params']['gaussian_window'])

            # Scale inner grads by the learning rates
            inner_autodecoder_updates = jax.tree_util.tree_map_with_path(
                jax.tree_util.Partial(apply_inner_lr, lrs=inner_state.params['meta_sgd_lrs']),
                inner_grad['autodecoder'])

            updated_inner_params = optax.apply_updates(inner_state.params['autodecoder'], inner_autodecoder_updates)
            inner_state = inner_state.replace(
                params={'nef': inner_state.params['nef'], 'autodecoder': updated_inner_params,
                        'meta_sgd_lrs': inner_state.params['meta_sgd_lrs']})

        # Get loss for resulting params, this is used to update the outer loop
        return loss_fn(
            inner_state.params,
            masked_coords=masked_coords,
            masked_sdf=masked_sdf), inner_state

    def visualize_batch(self, state, batch, name):
        """ Visualize the results of the model on a batch of data.

        Args:
            state: The current training state.
            batch: The current batch of data.
            name: The name of the plot.
            train: Whether we are training or validating.
        """
        _, inner_state = self.inner_loop(state.params, state, batch)
        p, a, window = self.inner_autodecoder.apply(inner_state.params['autodecoder'])
        self.visualize_and_log(state, batch, p, a, window, name=name)

        # Log gaussian windows
        wandb.log({f'{name}/gaussian_window': window})
        wandb.log({f'{name}/a': a})
        wandb.log({f'{name}/meta_sgd_lr_p': state.params['meta_sgd_lrs']['p_pos']})
        wandb.log({f'{name}/meta_sgd_lr_a': state.params['meta_sgd_lrs']['a']})
        wandb.log({f'{name}/meta_sgd_lr_gaussian_window': state.params['meta_sgd_lrs']['gaussian_window']})
