import wandb
import jax.numpy as jnp
import jax
import optax
from functools import partial

from experiments.fitting.trainers.image._ad_image_base_functions import AutodecodingImageBaseFunctions
from experiments.fitting.trainers._base._ad_enf_trainer_meta_sgd import MetaSGDAutoDecodingENFTrainer


class MetaSGDAutoDecodingENFTrainerImage(AutodecodingImageBaseFunctions, MetaSGDAutoDecodingENFTrainer):

    def __init__(
            self,
            config,
            enf,
            inner_autodecoder,
            outer_autodecoder,
            train_loader,
            val_loader,
            coords,
            seed
    ):
        AutodecodingImageBaseFunctions.__init__(
            self=self,
            coords=coords
        )
        MetaSGDAutoDecodingENFTrainer.__init__(
            self=self,
            config=config,
            enf=enf,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            train_loader=train_loader,
            val_loader=val_loader,
            seed=seed
        )

    @partial(jax.jit, static_argnums=(0,))
    def inner_loop(self, outer_params, outer_state, batch):
        # Unpack batch, reshape images
        img, _, img_idx = batch
        img = jnp.reshape(img, (img.shape[0], -1, img.shape[-1]))

        # Generate random mask of coordinates, one for every inner step
        mask = jax.random.permutation(
            outer_state.rng,
            jnp.broadcast_to(jnp.arange(self.coords.shape[0])[:, jnp.newaxis], (self.coords.shape[0], self.config.training.num_inner_steps + 1)),
            independent=True,
        )
        mask = mask[:self.config.training.max_num_sampled_points, :]

        # Broadcast autodecoder params over the batch dimension
        inner_autodecoder_params = jax.tree_map(
            lambda p: jnp.repeat(p, img.shape[0], axis=0), outer_params['autodecoder']
        )

        # Randomly sample positions
        if self.config.meta.noise_pos_inner_loop:
            inner_autodecoder_params['params']['p_pos'] = inner_autodecoder_params['params']['p_pos'] + (jax.random.normal(
                outer_state.rng,
                inner_autodecoder_params['params']['p_pos'].shape,
            ) * self.config.meta.noise_pos_inner_loop)

        # Create inner state
        inner_state = outer_state.replace(
            params={'enf': outer_params['enf'], 'autodecoder': inner_autodecoder_params, 'meta_sgd_lrs': outer_params['meta_sgd_lrs']}
        )

        # Define loss function
        def loss_fn(params, masked_coords, masked_img):
            """Loss function for the inner loop.

            Args:
                params: The current parameters of the model.
                masked_coords: The masked coordinates.
                masked_img: The masked image.
                img_idx: The index of the image.
            """
            p, c, window = self.inner_autodecoder.apply(params['autodecoder'])
            out = self.enf.apply(params['enf'], masked_coords, p, c, window)
            return jnp.mean((out - masked_img) ** 2)

        def apply_inner_lr(key, grad, lrs):
            # Get meta learning rates for p, c, and gaussian window
            if key[-1].key in lrs:
                return - lrs[key[-1].key] * grad
            else:
                raise ValueError(f'Unknown key: {key}')

        # Create inner grad fn
        inner_grad_fn = jax.grad(loss_fn)

        # Do inner loop
        for inner_step in range(self.config.training.num_inner_steps):
            # Mask the coordinates and labels
            masked_coords = self.coords[mask[:, inner_step]]
            masked_img = img[:, mask[:, inner_step]]

            # Broadcast the coordinates over the batch dimension
            masked_coords = jnp.broadcast_to(masked_coords, (img.shape[0], *masked_coords.shape))

            # Get inner grads and update inner autodecoder params
            inner_grad = inner_grad_fn(
                inner_state.params,
                masked_coords=masked_coords,
                masked_img=masked_img,
            )

            # Scale the gradient by the batch size, we need to do this since we're taking the mean of the loss.
            inner_grad['autodecoder'] = jax.tree_map(lambda x: x * img.shape[0], inner_grad['autodecoder'])

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
                params={'enf': inner_state.params['enf'], 'autodecoder': updated_inner_params,
                        'meta_sgd_lrs': inner_state.params['meta_sgd_lrs']})

        # Get loss for resulting params, this is used to update the outer loop
        masked_coords = self.coords[mask[:, inner_step + 1]]
        masked_coords = jnp.broadcast_to(masked_coords, (img.shape[0], *masked_coords.shape))
        masked_img = img[:, mask[:, inner_step + 1]]
        return loss_fn(
            inner_state.params,
            masked_coords=masked_coords,
            masked_img=masked_img), inner_state
    

    def validate_epoch(self, state):
        """ Validates the model. Since we're doing autodecoding, requires
            training a validation autodecoder from scratch.

        Args:
            state: The current training state.
        Returns:
            state: The updated training state.
        """
        # Initialize autodecoder
        key, init_key = jax.random.split(state.rng)
        autodecoder_params = state.params['autodecoder']

        # Create validation state
        val_state = state.replace(
            params={'enf': state.params['enf'], 'autodecoder': autodecoder_params,
                    'meta_sgd_lrs': state.params['meta_sgd_lrs']},
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            rng=key
        )

        self.total_val_epochs = max(self.epoch, self.config.test.min_num_epochs)
        self.global_val_step = 0
        # Loop over batches
        for epoch in range(1, self.total_val_epochs):
            losses = 0
            self.val_epoch = epoch

            for batch_idx, batch in enumerate(self.val_loader):
                loss, val_state = self.val_step(val_state, batch)
                losses += loss

                # Log every n steps
                if batch_idx % self.config.logging.log_every_n_steps == 0:
                    self.metrics['val_loss'] = loss
                    self.update_prog_bar(step=batch_idx, train=False)

                if self.global_val_step % self.config.logging.visualize_every_n_steps == 0:
                    self.visualize_batch(val_state, batch, name='val/recon-fitting', train=False)
                
                # Increment global val step
                self.global_val_step += 1

        # Reset val epoch
        self.val_epoch = 0
        self.total_val_epochs = 0
        self.global_val_step = 0

        # Visualize last batch
        self.visualize_batch(val_state, batch, name='val/recon-final', train=False)

        # Update epoch loss by last loss
        self.metrics['val_mse_epoch'] = losses / len(self.val_loader)
        wandb.log({'val_mse_epoch': self.metrics['val_mse_epoch']}, commit=False)
        return val_state
    

    def visualize_batch(self, state, batch, name):
        """ Visualize the results of the model on a batch of data.

        Args:
            state: The current training state.
            batch: The current batch of data.
            name: The name of the plot.
            train: Whether we are training or validating.
        """
        gt, _, _ = batch
        _, inner_state = self.inner_loop(state.params, state, batch)
        p, c, window = self.inner_autodecoder.apply(inner_state.params['autodecoder'])
        self.visualize_and_log(gt, state, p, c, window, name=name)

        # Save top val psnr for sweeping
        cur_val_metric = jnp.mean(jnp.asarray(self.cur_val_metric))
        self.top_val_metric = cur_val_metric if self.top_val_metric < cur_val_metric else self.top_val_metric
        wandb.log({'val/psnr': cur_val_metric})

        # Log gaussian windows
        wandb.log({f'{name}/gaussian_window': window})
        wandb.log({f'{name}/a': c})
        wandb.log({f'{name}/meta_sgd_lr_p': state.params['meta_sgd_lrs']['p_pos']})
        wandb.log({f'{name}/meta_sgd_lr_c': state.params['meta_sgd_lrs']['c']})
        wandb.log({f'{name}/meta_sgd_lr_gaussian_window': state.params['meta_sgd_lrs']['gaussian_window']})
