import wandb
from typing import Any

import tqdm

# For trainstate
from flax import struct, core
import optax
import jax.numpy as jnp

# Checkpointing
import orbax.checkpoint as ocp
from omegaconf import OmegaConf


class JaxTrainer:

    class TrainState(struct.PyTreeNode):
        params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
        time_params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
        opt_state: optax.OptState = struct.field(pytree_node=True)
        rng: jnp.ndarray = struct.field(pytree_node=True)

    def __init__(
            self,
            config,
            train_loader=None,
            val_loader=None,
            seed=42
    ):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.seed = seed

        # Placeholders for train and val steps
        self.train_step = None
        self.val_step = None

        # Keep track of training state
        self.global_step = 0
        self.epoch = 0

        # Keep track of state of validation
        self.global_val_step = 0

        # Keep track of metrics
        self.metrics = {}
        self.top_val_metric = -jnp.inf
        self.cur_val_metric = -jnp.inf

        # Description strings for train and val progress bars
        self.prog_bar_desc = """{state} :: epoch - {epoch}/{total_epochs} | step - {step}/{global_step} ::"""
        self.prog_bar = tqdm.tqdm(
            desc=self.prog_bar_desc.format(
                state='Training',
                epoch=self.epoch,
                total_epochs=self.config.training.num_epochs,
                step=0,
                global_step=self.global_step,
            ),
            total=len(self.train_loader)
        )

        # Set checkpoint options
        if self.config.logging.checkpoint:
            checkpoint_options = ocp.CheckpointManagerOptions(
                save_interval_steps=config.logging.checkpoint_every_n_epochs,
                max_to_keep=config.logging.keep_n_checkpoints,
            )
            self.checkpoint_manager = ocp.CheckpointManager(
                directory=str(config.logging.log_dir) + '/checkpoints',
                options=checkpoint_options,
                item_handlers={
                    'state': ocp.StandardCheckpointHandler(),
                    'config': ocp.JsonCheckpointHandler(),
                },
                item_names=['state', 'config']
            )

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            state: The training state.
        """
        raise NotImplementedError("init_train_state method must be implemented.")

    def create_functions(self):
        """Creates the functions for training and validation. Should implement train_step and val_step. """
        raise NotImplementedError("create_functions method must be implemented.")

    def train_model(self, num_epochs, state=None):
        """Trains the model for the given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train for.

        Returns:
            state: The final training state.
        """

        # Keep track of global step
        self.global_step = 0
        self.epoch = 0

        if state is None:
            state = self.init_train_state()

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch
            wandb.log({'epoch': epoch}, commit=False)
            state = self.train_epoch(state)

            # Save checkpoint (ckpt manager takes care of saving every n epochs)
            self.save_checkpoint(state)

            # Validate every n epochs
            if epoch % self.config.test.test_interval == 0:
                self.validate_epoch(state)
        return state

    def train_epoch(self, state):
        """ Train the model for one epoch.

        Args:
            state: The current training state.
            epoch: The current epoch.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss, state = self.train_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'train_mse_step': loss})
                self.update_prog_bar(step=batch_idx)

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.metrics['train_mse_epoch'] = losses / len(self.train_loader)
        wandb.log({'train_mse_epoch': self.metrics['train_mse_epoch']}, commit=False)
        return state

    def validate_epoch(self, state):
        """ Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss, _ = self.val_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'val_mse_step': loss})
                self.update_prog_bar(step=batch_idx, train=False)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.metrics['val_mse_epoch'] = losses / len(self.val_loader)
        wandb.log({'val_mse_epoch': self.metrics['val_mse_epoch']}, commit=False)

    def save_checkpoint(self, state):
        """ Save the current state to a checkpoint

        Args:
            state: The current training state.
        """
        if self.config.logging.checkpoint:
            self.checkpoint_manager.save(step=self.epoch, args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                config=ocp.args.JsonSave(OmegaConf.to_container(self.config))))

    def load_checkpoint(self):
        """ Load the latest checkpoint"""
        ckpt = self.checkpoint_manager.restore(self.checkpoint_manager.latest_step())
        return self.TrainState(**ckpt.state)

    def update_prog_bar(self, step, train=True):
        """ Update the progress bar.

        Args:
            desc: The description string.
            loss: The current loss.
            epoch: The current epoch.
            step: The current step.
        """
        # If we are at the beginning of the epoch, reset the progress bar
        if step == 0:
            # Depending on whether we are training or validating, set the total number of steps
            if train:
                self.prog_bar.total = len(self.train_loader)
            else:
                self.prog_bar.total = len(self.val_loader)
            self.prog_bar.reset()
        else:
            self.prog_bar.update(self.config.logging.log_every_n_steps)

        epoch = self.epoch
        total_epochs = self.config.training.num_epochs

        if train:
            global_step = self.global_step
        else:
            global_step = self.global_val_step

        # Update description string
        prog_bar_str = self.prog_bar_desc.format(
            state='Training' if train else 'Validation',
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            global_step=global_step,
        )

        # Append metrics to description string
        if self.metrics:
            for key, value in self.metrics.items():
                prog_bar_str += f" -- {key} {value:.4f}"

        self.prog_bar.set_description_str(prog_bar_str)


class ENFTrainer(JaxTrainer):

    def __init__(
            self,
            config,
            enf,
            train_loader,
            val_loader,
            seed
    ):
        super().__init__(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            seed=seed
        )

        # Store the NEF and the optimizer
        self.enf = enf
        self.nef_opt = None

        # Set the number of images to log
        self.num_logged_samples = min([config.logging.num_logged_samples, len(train_loader)])

    def update_prog_bar(self, step, train=True):
        """ Update the progress bar.

        Args:
            desc: The description string.
            loss: The current loss.
            epoch: The current epoch.
            step: The current step.
        """
        # If we are at the beginning of the epoch, reset the progress bar
        if step == 0:
            # Depending on whether we are training or validating, set the total number of steps
            if train:
                self.prog_bar.total = len(self.train_loader)
            else:
                self.prog_bar.total = len(self.val_loader)
            self.prog_bar.reset()
        else:
            self.prog_bar.update(self.config.logging.log_every_n_steps)

        if train:
            global_step = self.global_step
            epoch = self.epoch
            total_epochs = self.config.training.num_epochs
        else:
            global_step = self.global_val_step
            epoch = self.val_epoch
            total_epochs = self.total_val_epochs

        prog_bar_str = self.prog_bar_desc.format(
            state='Training' if train else 'Validation',
            epoch=epoch,
            total_epochs=total_epochs,
            step=step,
            global_step=global_step,
        )

        # append metrics to description string
        if self.metrics:
            for k, v in self.metrics.items():
                prog_bar_str += f" -- {k} {v:.4f}"

        self.prog_bar.set_description_str(prog_bar_str)

    def visualize_and_log(self, **kwargs):
        """ Visualize and log the results.
        """
        raise NotImplementedError("visualize_and_log functions should be implemented.")
