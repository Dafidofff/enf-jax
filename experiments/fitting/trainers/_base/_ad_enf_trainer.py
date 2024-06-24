import wandb
import jax.numpy as jnp

import optax
from flax import struct, core
import jax
from typing import Any

from experiments.fitting.trainers._base._trainer import ENFTrainer


class AutoDecodingENFTrainer(ENFTrainer):
    """AutoDecoding ENF trainer.

    This trainer is used to train the AutoDecoding ENF model. Some differences with the base ENFTrainer are:
    - We have an additional autodecoder that is trained alongside the enf model.
    - We have a train and val autodecoder.
    - During validation, we fit a new autodecoder for the test images.

    Some similarities with the base enf Trainer are:
    - The training and validation loops are the same, we still have a train_step and val_step.

    Inheriting classes should implement:
    - create_functions: This method should create the training functions.
    - visualize_batch: This method should visualize a batch of data.
    """

    class TrainState(struct.PyTreeNode):
        params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
        enf_opt_state: optax.OptState = struct.field(pytree_node=True)
        autodecoder_opt_state: optax.OptState = struct.field(pytree_node=True)
        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(
            self,
            config,
            enf,
            train_autodecoder,
            val_autodecoder,
            train_loader,
            val_loader,
            seed
    ):
        super().__init__(config, enf, train_loader, val_loader, seed)

        # Since we're optimizing during validation, keep track of the total number of validation epochs
        self.val_epoch = 0
        self.total_val_epochs = 0

        self.train_autodecoder = train_autodecoder
        self.val_autodecoder = val_autodecoder
        self.autodecoder_opt = None

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            TrainState: The training state.
        """
        # Initialize optimizer and scheduler
        self.enf_opt = optax.adam(self.config.optimizer.learning_rate_enf)
        self.autodecoder_opt = optax.adam(self.config.optimizer.learning_rate_codes)

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, enf_key = jax.random.split(key)
        key, autodecoder_key = jax.random.split(key)

        # Create a test batch to get the shape of the latent space
        autodecoder_params = self.train_autodecoder.init(autodecoder_key, jnp.ones(3, dtype=jnp.int32))
        p, a, window = self.train_autodecoder.apply(autodecoder_params, jnp.ones(3, dtype=jnp.int32))

        # Initialize enf
        sample_coords = jax.random.normal(enf_key, (3, 128, self.config.nef.num_in))
        enf_params = self.enf.init(enf_key, sample_coords[:, :self.config.training.max_num_sampled_points], p, a, window)

        train_state = self.TrainState(
            params={'enf': enf_params, 'autodecoder': autodecoder_params},
            enf_opt_state=self.enf_opt.init(enf_params),
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            rng=key
        )
        return train_state

    def create_functions(self):
        """ Create training functions. """
        self.apply_enf_jitted = jax.jit(self.enf.apply)
        def val_step(state, batch):
            return self.step(state, batch, train=False)
        def train_step(state, batch):
            return self.step(state, batch, train=True)

        self.train_step = jax.jit(train_step)
        self.val_step = jax.jit(val_step)

    def step(self, state, batch, train=True):
        """ Implements a single training/validation step. """
        raise NotImplementedError("This method should be implemented in a subclass.")

    def train_epoch(self, state):
        """ Trains the model for one epoch.

        Args:
            state: The current training state.
        Returns:
            state: The updated training state.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss, state = self.train_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.metrics['loss'] = loss
                wandb.log({'train_mse_step': loss})
                self.update_prog_bar(step=batch_idx)

            if self.global_step % self.config.logging.visualize_every_n_steps == 0:
                self.visualize_batch(state, batch, name='train/recon')

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.metrics['train_mse_epoch'] = losses / len(self.train_loader)
        wandb.log({'train_mse_epoch': self.metrics['train_mse_epoch']}, commit=False)
        return state

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
        autodecoder_params = self.val_autodecoder.init(init_key, jnp.ones(3, dtype=jnp.int32))

        # Create validation state
        val_state = state.replace(
            params={'enf': state.params['enf'], 'autodecoder': autodecoder_params},
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
