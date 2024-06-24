import math

from enf import EquivariantCrossAttentionENF
from enf.steerable_attention.invariant import get_sa_invariant, get_ca_invariant
from enf.latents.autodecoder import PositionOrientationFeatureAutodecoder
from enf.latents.autodecoder import PositionOrientationFeatureAutodecoderMeta


def get_model(cfg):
    """ Get autodecoders and snef based on the configuration. """

    # Determine whether we are doing meta-learning
    if "meta" not in cfg:
        # Init invariant
        self_attn_invariant = get_sa_invariant(cfg.nef)
        cross_attn_invariant = get_ca_invariant(cfg.nef)

        # Calculate initial gaussian window size
        assert math.sqrt(cfg.nef.num_latents)

        # Init autodecoder
        train_autodecoder = PositionOrientationFeatureAutodecoder(
            num_signals=cfg.dataset.num_signals_train,
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )

        val_autodecoder = PositionOrientationFeatureAutodecoder(
            num_signals=cfg.dataset.num_signals_test,
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )

        # Init model
        enf = EquivariantCrossAttentionENF(
            num_hidden=cfg.nef.num_hidden,
            num_heads=cfg.nef.num_heads,
            num_self_att_layers=cfg.nef.num_self_att_layers,
            num_out=cfg.nef.num_out,
            latent_dim=cfg.nef.latent_dim,
            self_attn_invariant=self_attn_invariant,
            cross_attn_invariant=cross_attn_invariant,
            embedding_type=cfg.nef.embedding_type,
            embedding_freq_multiplier=[cfg.nef.embedding_freq_multiplier_invariant,
                                       cfg.nef.embedding_freq_multiplier_value],
            condition_value_transform=cfg.nef.condition_value_transform,
            top_k_latent_sampling=cfg.nef.top_k,
        )
        return enf, train_autodecoder, val_autodecoder
    else:

        # Init invariant
        self_attn_invariant = get_sa_invariant(cfg.nef)
        cross_attn_invariant = get_ca_invariant(cfg.nef)

        # Calculate initial gaussian window size
        assert math.sqrt(cfg.nef.num_latents)

        # Init autodecoder
        inner_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=cfg.dataset.batch_size,
            # Since we're doing meta-learning, the inner and val autodecoders have batch_size as num signals
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )
        outer_autodecoder = PositionOrientationFeatureAutodecoderMeta(
            num_signals=1,  # Since we're doing meta-learning, we only optimize one set of latents
            num_latents=cfg.nef.num_latents,
            latent_dim=cfg.nef.latent_dim,
            num_pos_dims=cross_attn_invariant.num_z_pos_dims,
            num_ori_dims=cross_attn_invariant.num_z_ori_dims,
            gaussian_window_size=cfg.nef.gaussian_window,
        )

        # Init model
        enf = EquivariantCrossAttentionENF(
            num_hidden=cfg.nef.num_hidden,
            num_heads=cfg.nef.num_heads,
            num_self_att_layers=cfg.nef.num_self_att_layers,
            num_out=cfg.nef.num_out,
            latent_dim=cfg.nef.latent_dim,
            self_attn_invariant=self_attn_invariant,
            cross_attn_invariant=cross_attn_invariant,
            embedding_type=cfg.nef.embedding_type,
            embedding_freq_multiplier=[cfg.nef.embedding_freq_multiplier_invariant,
                                       cfg.nef.embedding_freq_multiplier_value],
            condition_value_transform=cfg.nef.condition_value_transform,
            top_k_latent_sampling=cfg.nef.top_k,
        )
        return enf, inner_autodecoder, outer_autodecoder
