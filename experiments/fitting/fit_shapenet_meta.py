import hydra
from omegaconf import DictConfig
import omegaconf
import wandb

import numpy as np
import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.shape.ad_enf_trainer_meta_shape import MetaAutoDecodingENFTrainerShape
from experiments.fitting.trainers.shape.ad_enf_trainer_meta_sgd_shape import MetaSGDAutoDecodingENFTrainerShape

from experiments.fitting import get_model


@hydra.main(version_base=None, config_path=".", config_name="ad_config_meta_shape")
def train(cfg: DictConfig):

    # Set log dir
    if not cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Set max num sampled points
    cfg.dataset.max_num_sampled_points = cfg.training.max_num_sampled_points

    # Create the dataset
    trainset, testset = get_dataloader(dataset_cfg=cfg.dataset)

    # Set dimensionality of input and output
    cfg.nef.num_in = 3
    cfg.nef.num_out = 1

    # Initialize wandb
    wandb.init(
        entity="equivariance",
        project="enf-jax",
        name=f"meta_{cfg.dataset.name}_ad",
        dir=cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(cfg),
        mode='disabled' if cfg.logging.debug else 'online',
    )

    # Get nef and autodecoders
    nef, inner_autodecoder, outer_autodecoder = get_model(cfg)

    if cfg.meta.meta_sgd:
        trainer = MetaSGDAutoDecodingENFTrainerShape(
            nef=nef,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            config=cfg,
            train_loader=trainset,
            val_loader=testset,
            seed=42,
        )
    else:
        trainer = MetaAutoDecodingENFTrainerShape(
            nef=nef,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            config=cfg,
            train_loader=trainset,
            val_loader=testset,
            seed=42,
        )

    trainer.create_functions()

    # Train model
    trainer.train_model(cfg.training.num_epochs)


if __name__ == "__main__":
    train()
