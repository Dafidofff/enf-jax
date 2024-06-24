import wandb

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

# Load base trainer
from experiments.fitting.trainers._base._trainer import JaxTrainer


class DownstreamClassifierTrainer(JaxTrainer):

    def __init__(
            self,
            classifier,
            config,
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

        # Store classifer
        self.classifier = classifier

        # Keep track of train and val accuracy
        self.train_accuracy_epoch = 0
        self.val_accuracy_epoch = 0

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            TrainState: The training state.
        """
        # Initialize optimizer and scheduler
        self.opt = optax.adamw(
            learning_rate=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay
        )

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, init_key = jax.random.split(key)

        # Take a sample from the train loader
        sample_batch = next(iter(self.train_loader))

        # Unpack
        p, a, window, _ = sample_batch

        # Initialize classifier
        classifier_params = self.classifier.init(init_key, (p, a, window))

        train_state = self.TrainState(
            params=classifier_params,
            time_params=None,
            opt_state=self.opt.init(classifier_params),
            rng=key
        )
        return train_state

    def create_functions(self):
        """Create the train and validation functions."""

        def step(state, batch):
            """Single optimization step."""

            def cross_entropy_loss(params, batch):
                latents, labels = batch[:-1], batch[-1]
                logits = self.classifier.apply(params, latents)
                out = nn.log_softmax(logits, axis=-1)
                one_hot_labels = jnp.reshape(jax.nn.one_hot(labels, num_classes=10), (-1, 10))
                return -jnp.mean(jnp.sum(one_hot_labels * out, axis=-1)), jnp.mean(jnp.argmax(out, axis=-1) == labels)

            # Compute loss and gradients
            (loss, acc), grads = jax.value_and_grad(cross_entropy_loss, has_aux=True)(state.params, batch)

            # Compute gradient updates
            updates, opt_state = self.opt.update(grads, state.opt_state, state.params)

            # Update parameters
            classifier_params = optax.apply_updates(state.params, updates)

            return loss, acc, state.replace(
                params=classifier_params,
                opt_state=opt_state,
                rng=jax.random.split(state.rng)[0]
            )

        # Create train step
        self.train_step = jax.jit(step)
        self.val_step = self.train_step

    def train_epoch(self, state):
        """ Train the model for one epoch.

        Args:
            state: The current training state.
            epoch: The current epoch.
        """
        # Loop over batches
        losses = 0
        accs = 0
        for batch_idx, batch in enumerate(self.train_loader):
            loss, acc, state = self.train_step(state, batch)
            losses += loss
            accs += acc

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({'train_mse_step': loss})
                self.metrics['loss'] = loss
                self.update_prog_bar(step=batch_idx)

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.metrics['train_mse_epoch'] = losses / len(self.train_loader)
        self.metrics['train_acc_epoch'] = accs / len(self.train_loader)

        wandb.log(self.metrics, commit=False)
        return state

    def validate_epoch(self, state):
        """ Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over batches
        losses = 0
        accs = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss, acc, _ = self.val_step(state, batch)
            losses += loss
            accs += acc

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.metrics['val_loss'] = loss
                self.update_prog_bar(step=batch_idx, train=False)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.metrics['val_mse_epoch'] = losses / len(self.val_loader)
        self.metrics['val_acc_epoch'] = accs / len(self.val_loader)
        wandb.log(self.metrics, commit=False)
