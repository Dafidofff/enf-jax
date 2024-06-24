import jax.numpy as jnp
import jax
import optax
from flax import struct

from experiments.fitting.trainers._base._ad_enf_trainer import AutoDecodingENFTrainer


class MetaSGDAutoDecodingENFTrainer(AutoDecodingENFTrainer):
    """Meta-learning using meta-sgd trainer for AutoDecoding SNeF.

    This trainer is used to train the AutoDecoding SNeF model using meta-learning and meta-sgd. Some differences with
    the base MetaAutoDecodingSNeFTrainer are:
    - The train state is initialized differently; we additionally optimize inner learning rates.

    Some similarities with the base AutoDecodingSNeFTrainer are:
    - The training and validation loops are the same, we still have a train_step and val_step.

    Inheriting classes should implement:
    - inner_loop: This method should take care of the inner loop.
    - visualize_batch: This method should visualize a batch of data.
    """

    class TrainState(AutoDecodingENFTrainer.TrainState):
        meta_sgd_opt_state: optax.OptState = struct.field(pytree_node=True)

    def __init__(
            self,
            config,
            enf,
            inner_autodecoder,
            outer_autodecoder,
            train_loader,
            val_loader,
            seed
    ):
        super().__init__(
            config=config,
            enf=enf,
            train_loader=train_loader,
            val_loader=val_loader,
            train_autodecoder=None,
            val_autodecoder=None,
            seed=seed
        )

        # Set autodecoders
        self.outer_autodecoder = outer_autodecoder
        self.inner_autodecoder = inner_autodecoder

        # Set the meta sgd and autodecoder optimizer to None
        self.meta_sgd_opt = None
        self.inner_autodecoder_opt = None

    def init_train_state(self):
        # Initialize optimizer and scheduler
        self.enf_opt = optax.adam(self.config.optimizer.learning_rate_snef)
        self.autodecoder_opt = optax.adam(self.config.optimizer.learning_rate_codes)
        self.meta_sgd_opt = optax.adam(self.config.meta.learning_rate_meta_sgd)

        # Random key
        key = jax.random.PRNGKey(self.seed)

        # Split key
        key, enf_key = jax.random.split(key)
        key, autodecoder_key = jax.random.split(key)
        key, inner_autodecoder_key = jax.random.split(key)

        # Create a test batch to get the shape of the latent space
        autodecoder_params = self.outer_autodecoder.init(autodecoder_key)
        p, c, window = self.outer_autodecoder.apply(autodecoder_params)

        # Initialize learning rates for the autodecoder
        lr_pos = jnp.ones((1))  * self.config.meta.inner_learning_rate_p
        lr_c = jnp.ones((c.shape[-1])) * self.config.meta.inner_learning_rate_c
        lr_gaussian_window = jnp.ones((1)) * self.config.meta.inner_learning_rate_window

        # Put lrs in frozendict
        meta_sgd_lrs = {
            'p_pos': lr_pos,
            'c': lr_c,
            'gaussian_window': lr_gaussian_window
        }

        # Add orientation learning rate if we have orientation dimensions
        if self.outer_autodecoder.num_ori_dims > 0:
            lr_ori = jnp.ones((1)) * self.config.meta.inner_learning_rate_p
            meta_sgd_lrs['p_ori'] = lr_ori

        # Initialize enf
        sample_coords = jax.random.normal(enf_key, (p.shape[0], 128, self.config.nef.num_in))
        enf_params = self.enf.init(enf_key, sample_coords[:, :self.config.training.max_num_sampled_points], p, c, window)

        # Init but discard the params, we only care about the optimizer state
        _ = self.inner_autodecoder.init(inner_autodecoder_key)

        # Create train state
        train_state = self.TrainState(
            params={'enf': enf_params, 'autodecoder': autodecoder_params, 'meta_sgd_lrs': meta_sgd_lrs},
            enf_opt_state=self.enf_opt.init(enf_params),
            autodecoder_opt_state=self.autodecoder_opt.init(autodecoder_params),
            meta_sgd_opt_state=self.meta_sgd_opt.init(meta_sgd_lrs),
            rng=key
        )
        return train_state
    
    def create_functions(self):
        """Creates the training and validation functions."""
        # Jit functions
        self.apply_enf_jitted = jax.jit(self.enf.apply)

        # Train and val steps
        self.train_step = jax.jit(jax.tree_util.Partial(self.outer_step, train=True))
        self.val_step = jax.jit(jax.tree_util.Partial(self.outer_step, train=False))

    def inner_loop(self, outer_params, outer_state, batch):
        raise NotImplementedError

    def outer_step(self, state, batch, train=True):
        """Performs a single outer-loop training step.

        Args:
            state (TrainState): The current training state.
            batch (dict): The current batch of data.
            train (bool): Whether we're training or validating. If training, we optimize both autodecoder and enf,
                otherwise only autodecoder.

        Returns:
            TrainState: The updated training state.
        """
        # Split random key
        inner_key, new_outer_key = jax.random.split(state.rng)
        outer_state = state.replace(rng=new_outer_key)

        # Get gradients for the outer loop and update params
        # NOTE: the inner loop is called here, but is created in a subclass.
        (loss, _), grads = jax.value_and_grad(self.inner_loop, has_aux=True)(state.params, outer_state=outer_state,
                                                                             batch=batch)
        if train:
            enf_updates, enf_opt_state = self.enf_opt.update(grads['enf'], state.enf_opt_state)
            enf_params = optax.apply_updates(state.params['enf'], enf_updates)
            meta_sgd_lr_updates, meta_sgd_opt_state = self.meta_sgd_opt.update(grads['meta_sgd_lrs'],
                                                                               state.meta_sgd_opt_state)
            meta_sgd_lrs = optax.apply_updates(state.params['meta_sgd_lrs'], meta_sgd_lr_updates)

            # Clip meta_sgd_lrs between 1e-6 and 10
            meta_sgd_lrs = jax.tree_map(lambda x: jnp.clip(x, 1e-6, 10), meta_sgd_lrs)
        else:
            enf_params = state.params['enf']
            meta_sgd_lrs = state.params['meta_sgd_lrs']
            enf_opt_state = state.enf_opt_state
            meta_sgd_opt_state = state.meta_sgd_opt_state

        # Update autodecoder
        autodecoder_updates, autodecoder_opt_state = self.autodecoder_opt.update(grads['autodecoder'],
                                                                                 state.autodecoder_opt_state)
        autodecoder_params = optax.apply_updates(state.params['autodecoder'], autodecoder_updates)

        return loss, state.replace(
            params={'enf': enf_params, 'autodecoder': autodecoder_params, 'meta_sgd_lrs': meta_sgd_lrs},
            enf_opt_state=enf_opt_state,
            autodecoder_opt_state=autodecoder_opt_state,
            meta_sgd_opt_state=meta_sgd_opt_state,
            rng=new_outer_key
        )

