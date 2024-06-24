from functools import partial
import wandb

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from experiments.fitting.trainers._base._trainer import JaxTrainer
from experiments.downstream.utils.diffusion_utils import edm_sampler, edm_loss


class DownstreamDiffusionTrainer(JaxTrainer):

    def __init__(
            self,
            snef,
            snef_state,
            model,
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

        # Store backbone model
        self.snef = snef
        self.snef_state = snef_state
        self.model = model

        # Extract config
        self.fix_positions = config.diffusion.fix_positions

        # Store diffusion loss
        self.criterion = partial(edm_loss, sigma_data=self.config.diffusion.sigma_data, normalize_x_factor=self.config.diffusion.normalize_x_factor)

        # Keep track of train and val accuracy
        self.train_accuracy_epoch = 0
        self.val_accuracy_epoch = 0

    def init_train_state(self):
        """Initializes the training state.

        Returns:
            TrainState: The training state.
        """
        # Initialize optimizer and scheduler
        self.opt = optax.adam(self.config.optimizer.learning_rate)

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, init_key = jax.random.split(key)

        # Take a sample from the train loader
        sample_batch = next(iter(self.train_loader))

        # Unpack
        p, a, self.window, label = sample_batch

        # Set some variables
        self.batch_size = p.shape[0]
        self.num_latents = p.shape[1]
        self.latent_dim = a.shape[2]

        # If last
        a = jnp.concatenate([a, jnp.ones((self.batch_size, self.num_latents, 1))], axis=-1)

        # Initialize classifier
        model_params = self.model.init(init_key, (p, a, self.window))

        train_state = self.TrainState(
            params=model_params,
            opt_state=self.opt.init(model_params),
            rng=key
        )
        return train_state

    def create_functions(self):
        """Create the train and validation functions."""

        def step(state, batch):
            """Single optimization step."""

            key, new_key = jax.random.split(state.rng)

            def edm_diffusion_loss(params, batch):
                loss, (D_pos, D_x), error_pos, error_x = self.criterion(params, self.model, batch, key)
                return loss, (error_pos, error_x) #(loss, D_pos, D_x, error_pos, error_x)

            # Compute loss and gradients
            (loss, (error_pos, error_x)), grads = jax.value_and_grad(edm_diffusion_loss, has_aux=True)(state.params, batch)

            # Compute gradients
            updates, opt_state = self.opt.update(grads, state.opt_state)

            # Update parameters
            diffusion_params = optax.apply_updates(state.params, updates)

            return (loss, error_pos, error_x), state.replace(
                params=diffusion_params,
                opt_state=opt_state,
                rng=new_key
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
        for batch_idx, batch in enumerate(self.train_loader):
            (loss, error_pos, error_x), state = self.train_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                wandb.log({
                    'train_mse_step': loss,
                    # 'D_pos_hist': wandb.Histogram(D_pos),
                    # 'D_x_hist': wandb.Histogram(D_x),
                    'error_pos': error_pos,
                    'error_x': error_x,
                })
                self.metrics['loss'] = loss
                self.update_prog_bar(step=batch_idx)

            # Increment global step
            self.global_step += 1

        # Update epoch loss
        self.metrics['train_mse_epoch'] = losses / len(self.train_loader)
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
            (loss, error_pos, error_x), state = self.val_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.metrics['loss'] = loss
                self.update_prog_bar(step=batch_idx, train=False)
            
                # Sample from the model
                self.sample(state)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.metrics['val_mse_epoch'] = losses / len(self.val_loader)
        wandb.log(self.metrics, commit=False)

    def sample(self, state, num_samples=2):
        """Sample from the model.

        Args:
            num_samples: The number of samples to generate.
        """
        # Sample random point clouds
        pos_0 = jax.random.normal(state.rng, [num_samples, self.num_latents, 2])
        pos_0_mean = pos_0.mean(axis=(1), keepdims=True)
        pos_0 = pos_0 - pos_0_mean
        x_0 = jax.random.normal(state.rng, [num_samples, self.num_latents, self.latent_dim])

        # # Sample 2d uniform grid between -1 and 1 of self.num_latents/2 by self.num_latents/2
        # # Calculate the number of latents per dimension
        # num_dims = 2
        # num_latents_per_dim = int(round(self.num_latents ** (1. / num_dims)))

        # # Create an n-dimensional mesh grid [-1 to 1] for each dimension
        # grid_axes = jnp.linspace(-1 + 1 / num_latents_per_dim, 1 - 1 / num_latents_per_dim, num_latents_per_dim)
        # grids = jnp.meshgrid(*[grid_axes for _ in range(num_dims)], indexing='ij')

        # # Stack and reshape to create the positions matrix
        # positions = jnp.stack(grids, axis=-1).reshape(-1, num_dims)

        # # Repeat for the number of signals
        # pos_0 = jnp.repeat(positions[None, :, :], num_samples, axis=0)

        # Sample from the model
        D_pos, D_x = edm_sampler(
            model=self.model,
            state=state, 
            pos_0=pos_0,
            x_0=x_0, 
            S_churn=self.config.diffusion.S_churn, 
            num_steps=self.config.diffusion.num_steps, 
            sigma_max=self.config.diffusion.sigma_max
        )        

        # Unnormalize appearance
        D_x = ((D_x + 1)/2) * (self.train_loader.dataset.max_a - self.train_loader.dataset.min_a) + self.train_loader.dataset.min_a
        # D_pos = D_pos * (self.train_loader.dataset.max_pos - self.train_loader.dataset.min_pos) + self.train_loader.dataset.min_pos
        # D_x = D_x * self.train_loader.dataset.std_a + self.train_loader.dataset.mean_a
        # D_pos = D_pos / self.train_loader.dataset.mean_pos + self.train_loader.dataset.std_pos
        
        # Decode with snef
        self.snef.visualize_and_log(jnp.zeros((D_pos.shape[0], 28, 28, 1)), self.snef_state, D_pos, D_x, self.window[:num_samples], name='HIHI')
        return D_pos, D_x, self.window[:num_samples]
