import hydra
import wandb
import omegaconf
from omegaconf import DictConfig

import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.image.ad_enf_trainer_meta_sgd_image import MetaSGDAutoDecodingENFTrainerImage
from experiments.fitting import get_model


@hydra.main(version_base=None, config_path="./configs/", config_name="ad_config_meta_image")
def train(cfg: DictConfig):

    # Set log dir
    if not cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Create the dataset
    trainset, testset = get_dataloader(dataset_cfg=cfg.dataset)

    # Get image shape
    sample_batch = next(iter(trainset))

    smp_image = sample_batch[0][0]
    image_shape = smp_image.shape
    cfg.dataset.image_shape = image_shape

    # Position grid over the image domain
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-1, 1, image_shape[0]), jnp.linspace(-1, 1, image_shape[1])), axis=-1).reshape(-1, 2)

    # Set dimensionality of input and output
    cfg.nef.num_in = 2
    cfg.nef.num_out = image_shape[-1]

    # Initialize wandb
    wandb.init(
        project="enf-jax",
        name=f"meta_{cfg.dataset.name}",
        dir=cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(cfg),
        mode='disabled' if cfg.logging.debug else 'online',
    )

    # Get nef and autodecoders
    enf, inner_autodecoder, outer_autodecoder = get_model(cfg)

    # Create trainer
    trainer = MetaSGDAutoDecodingENFTrainerImage(
        enf=enf,
        inner_autodecoder=inner_autodecoder,
        outer_autodecoder=outer_autodecoder,
        config=cfg,
        train_loader=trainset,
        val_loader=testset,
        coords=coords,
        seed=42,
    )

    trainer.create_functions()

    # Train model
    trainer.train_model(cfg.training.num_epochs)


if __name__ == "__main__":
    train()
