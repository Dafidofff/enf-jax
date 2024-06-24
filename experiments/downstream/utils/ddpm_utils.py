from typing import Union
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jax import random



class SinusoidalPosEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, pos):
        """Refer to https://arxiv.org/pdf/1706.03762.pdf#subsection.3.5"""
        batch_size = pos.shape[0]

        assert self.dim % 2 == 0, self.dim
        assert pos.shape == (batch_size, 1), pos.shape

        d_model = self.dim // 2
        i = jnp.arange(d_model)[None, :]

        pos_embedding = pos * jnp.exp(-(2 * i / d_model) * jnp.log(10000))
        pos_embedding = jnp.concatenate(
            (jnp.sin(pos_embedding), jnp.cos(pos_embedding)), axis=-1
        )

        assert pos_embedding.shape == (batch_size, self.dim), pos_embedding.shape

        return pos_embedding
    

class TimeEmbedding(nn.Module):
    dim: int
    sinusoidal_embed_dim: int

    @nn.compact
    def __call__(self, time):
        x = SinusoidalPosEmbedding(self.sinusoidal_embed_dim)(time)
        x = nn.Dense(self.dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x
    

class Diffuser:
    def __init__(self, eps_fn, diffusion_config):
        self.eps_fn = eps_fn
        self.config = diffusion_config
        self.betas = jnp.asarray(self._betas(**diffusion_config))
        self.alphas = jnp.asarray(self._alphas(self.betas))
        self.alpha_bars = jnp.asarray(self._alpha_bars(self.alphas))

    @property
    def steps(self) -> int:
        return self.config.T

    def timesteps(self, steps: int):
        timesteps = jnp.linspace(0, self.steps, steps + 1)
        timesteps = jnp.rint(timesteps).astype(jnp.int32)
        return timesteps[::-1]
    
    #################################
    ### DD forward formulation     ##
    #################################

    @partial(jax.jit, static_argnums=(0,))
    def forward(self, pos_0, x_0, rng):
        """See algorithm 1 in https://arxiv.org/pdf/2006.11239.pdf"""
        rng_t, rng_q = random.split(rng)
        
        # Sample random timestep for every sample in batch
        t = random.randint(rng_t, (len(x_0), 1), 0, self.steps)

        # Sample x_t given x_0 and t
        pos_t, eps_pos, x_t, eps_x = self.sample_q(pos_0, x_0, t, rng_q)
        t = t.astype(x_t.dtype)
        return pos_t, eps_pos, x_t, eps_x, t

    def sample_q(self, pos_0, x_0, t, rng):
        """Samples x_t given x_0 by the q(x_t|x_0) formula."""
        # Get one rng for pos and x
        rng_pos, rng_x = random.split(rng)

        # (bs, 1, 1)
        alpha_t_bar = self.alpha_bars[t][:,:,None]

        # Sample random noise for x and calculate x_t
        eps_x = random.normal(rng_x, shape=x_0.shape, dtype=x_0.dtype)
        x_t = (alpha_t_bar**0.5) * x_0 + ((1 - alpha_t_bar) ** 0.5) * eps_x

        # Sample randome noise for position and calculate pos_t
        eps_pos = random.normal(rng_pos, shape=pos_0.shape, dtype=pos_0.dtype) #/ 5
        # eps_pos_normalised = eps_pos - jnp.mean(eps_pos, axis=0, keepdims=True)
        pos_t = (alpha_t_bar**0.5) * pos_0 + ((1 - alpha_t_bar) ** 0.5) * eps_pos
        return pos_t, eps_pos, x_t, eps_x
    
    #################################
    ### DDPM Backward formulation  ##
    #################################

    @partial(jax.jit, static_argnums=(0,))
    def ddpm_backward_step(self, params, x_t, t, rng):
        """See algorithm 2 in https://arxiv.org/pdf/2006.11239.pdf"""
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alpha_bars[t]
        sigma_t = self.betas[t] ** 0.5

        z = (t > 0) * random.normal(rng, shape=x_t.shape, dtype=x_t.dtype)
        eps = self.eps_fn(params, x_t, t, train=False)

        x = (1 / alpha_t**0.5) * (
            x_t - ((1 - alpha_t) / (1 - alpha_t_bar) ** 0.5) * eps
        ) + sigma_t * z

        return x

    def ddpm_backward(self, params, x_T, rng):
        x = x_T

        for t in range(self.steps - 1, -1, -1):
            rng, rng_ = random.split(rng)
            x = self.ddpm_backward_step(params, x, t, rng_)

        return x

    #################################
    ### DDIM Backward formulation  ##
    #################################

    @partial(jax.jit, static_argnums=(0,6))
    def ddim_backward_step(self, params, p_t, x_t, t, t_next, latent_dim):
        """See section 4.1 and C.1 in https://arxiv.org/pdf/2010.02502.pdf

        Note: alpha in the DDIM paper is actually alpha_bar in DDPM paper
        """
        alpha_t = self.alpha_bars[t]
        alpha_t_next = self.alpha_bars[t_next]

        # Predict noise for timestep.
        eps_x, eps_p = self.eps_fn.apply(params, (p_t, x_t, None))
        eps_p = eps_p.squeeze(-2)

        # Denoise the appearances with scalar updates
        x_0 = (x_t[:,:,:latent_dim] - (1 - alpha_t) ** 0.5 * eps_x) / alpha_t**0.5
        x_t_direction = (1 - alpha_t_next) ** 0.5 * eps_x
        x_t_next = alpha_t_next**0.5 * x_0 + x_t_direction

        # Denoise the position with a vector update
        p_0 = (p_t - (1 - alpha_t) ** 0.5 * eps_p) / alpha_t**0.5
        p_t_direction = (1 - alpha_t_next) ** 0.5 * eps_p
        p_t_next = alpha_t_next**0.5 * p_0 + p_t_direction

        # Renormalize the positions
        p_t_next = p_t_next - jnp.mean(p_t_next, axis=(1), keepdims=True)

        return p_t_next, x_t_next

    def ddim_backward(self, state, p_t, x_t, steps, time_embedding_fn):
        p, x = p_t, x_t
        latent_dim = x_t.shape[-1]

        ts = self.timesteps(steps)
        for t, t_next in zip(ts[:-1], ts[1:]):
            # Embed time and repeat for number of latents
            t_emb = jnp.full((len(x), 1), t, dtype=x.dtype)
            t_emb = time_embedding_fn.apply(state.time_params, t_emb)
            t_emb = jnp.repeat(t_emb[:,None,:], x_t.shape[1], axis=1)

            # Concatenate time embedding to input
            x = jnp.concatenate([x, t_emb], axis=-1)

            # Backward diffusion step
            p, x = self.ddim_backward_step(state.params, p, x, t, t_next, latent_dim)

        return p, x
    
    #################################
    ### Auxilary                   ##
    #################################

    @classmethod
    def _betas(cls, beta_1, beta_T, T):
        return jnp.linspace(beta_1, beta_T, T, dtype=jnp.float32)

    @classmethod
    def _alphas(cls, betas):
        return 1 - betas

    @classmethod
    def _alpha_bars(cls, alphas):
        return jnp.cumprod(alphas)

    @staticmethod
    def expand_t(t, x):
        return jnp.full((len(x), 1), t, dtype=x.dtype)