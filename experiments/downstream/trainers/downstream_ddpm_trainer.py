from functools import partial
import wandb

import jax
import jax.numpy as jnp
import optax
from jax import random

from experiments.fitting.trainers._base._trainer import JaxTrainer
from experiments.downstream.utils.ddpm_utils import Diffuser, TimeEmbedding


class DownstreamDDPMTrainer(JaxTrainer):

    def __init__(
            self,
            enf,
            enf_state,
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
        self.enf = enf
        self.enf_state = enf_state
        self.model = model

        # Store diffusion loss
        self.diffuser = Diffuser(model, config.diffusion)
        self.time_embedding = TimeEmbedding(dim=self.config.ponita.time_embedding_dim, sinusoidal_embed_dim= self.config.ponita.num_hidden)

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
        a = jnp.concatenate([a, jnp.ones((self.batch_size, self.num_latents, self.config.ponita.time_embedding_dim))], axis=-1)

        # Initialize classifier
        model_params = self.model.init(init_key, (p, a, self.window))

        # Initialize the time embedding
        time_params = self.time_embedding.init(init_key, jnp.ones((self.batch_size,1)))

        # Create train state
        train_state = self.TrainState(
            params=model_params,
            time_params=time_params,
            opt_state=self.opt.init({"model_params": model_params, "time_params":time_params}),
            rng=key
        )
        return train_state

    def create_functions(self):
        """Create the train and validation functions."""

        def step(state, batch):
            """Single optimization step."""

            # Get random keys
            key, new_key = jax.random.split(state.rng)
            key1, key2 = jax.random.split(key)

            # Unpack batch and create input
            p_0, a_0, window, _ = batch
            x_0 = a_0
            pos_t, eps_pos, x_t, eps_x, t = self.diffuser.forward(p_0, x_0, key1)

            def ddpm_diffusion_loss(state, p_t, x_t, t, eps_pos, eps_x):
                # Embed time and repeat for number of latents
                t_emb = self.time_embedding.apply(state['time_params'], t)
                t_emb = jnp.repeat(t_emb[:,None,:], self.num_latents, axis=1)

                # Concatenate time embedding to input
                x_t = jnp.concatenate([x_t, t_emb], axis=-1)

                # Predict noise for timestep.
                e_hat_x, e_hat_pos = self.model.apply(state['model_params'], (p_t, x_t, None))
                loss_pos = optax.l2_loss(e_hat_pos.squeeze(-2), eps_pos).mean()
                loss_x = optax.l2_loss(e_hat_x, eps_x).mean()
                return loss_x + loss_pos

            # Compute loss and gradients
            loss, grads = jax.value_and_grad(ddpm_diffusion_loss)({"model_params": state.params, "time_params":state.time_params}, pos_t, x_t, t, eps_pos, eps_x)

            # Compute gradients
            updates, opt_state = self.opt.update(grads, state.opt_state)

            # Update parameters
            diffusion_params = optax.apply_updates({"model_params": state.params, "time_params":state.time_params}, updates)

            return loss, state.replace(
                params=diffusion_params['model_params'],
                time_params=diffusion_params['time_params'],
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
        wandb.log(self.metrics, commit=False)
        return state

    def validate_epoch(self, state):
        """ Validates the model.

        Args:
            state: The current training state.
        """
        # Loop over batches
        losses = 0
        for batch_idx, batch in enumerate(self.val_loader):
            loss, state = self.val_step(state, batch)
            losses += loss

            # Log every n steps
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                self.metrics['loss'] = loss
                self.update_prog_bar(step=batch_idx, train=False)
            
            # Sample from the model
            if batch_idx % self.config.logging.visualise_every_n_epoch == 0:
                self.sample(state)

            # Increment global step
            self.global_val_step += 1

        # Update epoch loss
        self.metrics['val_mse_epoch'] = losses / len(self.val_loader)
        wandb.log(self.metrics, commit=False)

    def sample(self, state, num_samples=2, num_steps=100):
        """Sample from the model.

        Args:
            num_samples: The number of samples to generate.
        """
        # Get rng key
        key, new_key = jax.random.split(state.rng)

        # Define shape
        pos_shape = (num_samples, self.num_latents, self.config.nef.num_in)
        appearance_shape = (num_samples, self.num_latents, self.latent_dim)
        image_shape = (num_samples, *self.config.latent_dataset.image_shape)

        # Sample random pointclouds and denoise. 
        p_T = random.normal(key, pos_shape, dtype=jnp.float32)
        x_T = random.normal(new_key, appearance_shape, dtype=jnp.float32)
        
        # Denoise sampled latent pointclouds.
        p_0, x_0 = self.diffuser.ddim_backward(state, p_T, x_T, num_steps, self.time_embedding)
        
        # Reconstruct and visualise the denoised latent pointclouds. 
        self.enf.visualize_and_log(jnp.zeros(image_shape), self.enf_state, p_0, x_0, self.window[:num_samples], name='generated_samples')
        return x_0
        
