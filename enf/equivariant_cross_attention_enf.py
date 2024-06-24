from typing import Union, Optional
from functools import partial

import jax.numpy as jnp
from jax.nn import gelu
from flax import linen as nn

from enf.steerable_attention.invariant._base_invariant import BaseInvariant
from enf.steerable_attention.equivariant_cross_attention import EquivariantCrossAttention, PointwiseFFN


class EquivariantCrossAttentionBlock(nn.Module):
    """ Cross attention layer for the latent points, conditioned on the poses.

        Args:
            num_hidden (int): The number of hidden units.
            num_heads (int): The number of attention heads.
            attn_operator (EquivariantCrossAttention): The attention operator to use.
        """
    num_hidden: int
    num_heads: int
    attn_operator: EquivariantCrossAttention
    residual: bool
    project_heads: bool
    top_k: Optional[int | None] = None

    def setup(self):
        # Layer normalization
        self.layer_norm_attn = nn.LayerNorm()

        # Attention layer
        self.attn = self.attn_operator(num_hidden=self.num_hidden, num_heads=self.num_heads,
                                       project_heads=self.project_heads, top_k=self.top_k)

        # Pointwise feedforward network
        if self.project_heads:
            self.pointwise_ffn = PointwiseFFN(num_in=self.num_hidden, num_hidden=self.num_hidden,
                                              num_out=self.num_hidden)
        else:
            self.pointwise_ffn = PointwiseFFN(num_in=self.num_heads * self.num_hidden,
                                              num_hidden=self.num_heads * self.num_hidden,
                                              num_out=self.num_heads * self.num_hidden)

    def __call__(self, x, p, c, x_h, window_size):
        """ Perform attention over the latent points, conditioned on the poses.

        Args:
            x (torch.Tensor): The poses. Shape [batch_size, num_coords, num_x_dimensions].
            p (torch.Tensor): The poses of the latent points. Shape [batch_size, num_latents, num_x_dimensions + num_ori_dimensions].
            c (torch.Tensor): The features or context corresponding to the latent poses. Shape [batch_size, num_latents, num_hidden].
            x_h (torch.Tensor): The conditioning factors (E.g. inputs former layers). Shape [batch_size, num_coords, num_hidden].
            window_size (float): The window size for the gaussian window.
        """

        # Apply layer normalization to 'a'
        c_norm = self.layer_norm_attn(c)

        # Apply attention
        c_attn = self.attn(x=x, p=p, c=c_norm, x_h=x_h, window_sigma=window_size)

        # Apply residual connection if specified
        if self.residual:
            a_res = c + c_attn
            a_out = self.pointwise_ffn(a_res)
        else:
            a_out = self.pointwise_ffn(c_attn)
        return a_out


class EquivariantCrossAttentionENF(nn.Module):
    """ Equivariant cross attention network for the latent points, conditioned on the poses.

    Args:
        num_hidden (int): The number of hidden units.
        num_heads (int): The number of attention heads.
        num_self_att_layers (int): The number of self-attention layers.
        num_out (int): The number of output coordinates.
        latent_dim (int): The dimensionality of the latent code.
        invariant (BaseInvariant): The invariant to use for the attention operation.
        embedding_type (str): The type of embedding to use. 'rff' or 'siren'.
        embedding_freq_multiplier (Union[float, float]): The frequency multiplier for the embedding.
        condition_value_transform (bool): Whether to condition the value transform on the invariant.
    """

    num_hidden: int
    num_heads: int
    num_self_att_layers: int
    num_out: int
    latent_dim: int
    cross_attn_invariant: BaseInvariant
    self_attn_invariant: BaseInvariant
    embedding_type: str
    embedding_freq_multiplier: Union[float, float]
    condition_value_transform: bool
    cross_attention_blocks = []
    top_k_latent_sampling: Optional[int | None] = None

    def setup(self):

        # Create a constructor for the equivariant cross attention operation.
        equivariant_cross_attn = partial(
            EquivariantCrossAttention,
            invariant=self.cross_attn_invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
            condition_value_transform=self.condition_value_transform,
            condition_invariant_embedding=False,
            top_k=self.top_k_latent_sampling,
        )
        equivariant_self_attn = partial(
            EquivariantCrossAttention,
            invariant=self.self_attn_invariant,
            embedding_type=self.embedding_type,
            embedding_freq_multiplier=self.embedding_freq_multiplier,
            condition_value_transform=self.condition_value_transform,
            condition_invariant_embedding=False,
            top_k=self.top_k_latent_sampling,
        )

        # Create network
        cross_attention_blocks = []
        self_attention_blocks = []
        self.activation = gelu

        # Add code->latent stem
        self.latent_stem = nn.Dense(self.num_hidden)

        # Add latent self-attention blocks
        for i in range(self.num_self_att_layers):
            self_attention_blocks.append(
                EquivariantCrossAttentionBlock(
                    num_hidden=self.num_hidden,
                    num_heads=self.num_heads,
                    attn_operator=equivariant_self_attn,
                    residual=True,
                    project_heads=True,
                    top_k=self.top_k_latent_sampling,
                )
            )

        # If there is only one layer, we need to add the final cross attention block that doesn't take conditioning
        cross_attention_blocks.append(
            EquivariantCrossAttentionBlock(
                num_hidden=self.num_hidden,
                num_heads=self.num_heads,
                attn_operator=equivariant_cross_attn,
                residual=False,
                project_heads=False,
                top_k=self.top_k_latent_sampling,
            )
        )

        self.cross_attention_blocks = cross_attention_blocks
        self.self_attention_blocks = self_attention_blocks

        # Output layer
        self.out_proj = nn.Sequential([
            nn.Dense(self.num_hidden),
            gelu,
            nn.Dense(self.num_hidden),
            gelu,
            nn.Dense(self.num_out)
        ])

    def __call__(self, x, p, a, gaussian_window_size):
        """ Sample from the model.

        Args:
            x (torch.Tensor): The pose of the input points. Shape (batch_size, num_coords, 2).
            p (torch.Tensor): The pose of the latent points. Shape (batch_size, num_latents, num_ori (1), 4).
            a (torch.Tensor): The latent features. Shape (batch_size, num_latents, num_hidden).
            gaussian_window_size (float or None): The window size for the gaussian window.
        """
        # p contains angles, so we need to embed them into a circle.
        if self.cross_attn_invariant.num_z_ori_dims > 0:
            p_pos, p_angles = p[:, :, :self.cross_attn_invariant.num_z_pos_dims], p[:, :,
                                                                                  self.cross_attn_invariant.num_z_pos_dims:]
            p = jnp.concatenate((p_pos, jnp.cos(p_angles), jnp.sin(p_angles)), axis=-1)

        # Map code to latent space.
        a = self.latent_stem(a)

        # Self attention layers.
        for i in range(0, self.num_self_att_layers):
            # Apply self attention between latents.
            a = a + self.self_attention_blocks[i](p, p, a, x_h=None, window_size=gaussian_window_size)
            a = self.activation(a)

        # Final cross attention block
        out = self.cross_attention_blocks[-1](x, p, a, x_h=None, window_size=gaussian_window_size)
        out = self.activation(out)

        # Output layer
        out = self.out_proj(out)

        return out