import hydra
import wandb
import omegaconf
from omegaconf import DictConfig
from pathlib import Path

import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.image.ad_enf_trainer_image import AutoDecodingSNeFTrainerImage
from experiments.fitting.trainers.image.ad_enf_trainer_meta_sgd_image import MetaSGDAutoDecodingENFTrainerImage

from experiments.fitting import get_model
from experiments.downstream.latent_dataset.utils import get_or_create_latent_dataset_from_snef


@hydra.main(version_base=None, config_path="./configs/", config_name="config_create_latent_dataset_from_checkpoint")
def create_latent_dataset_from_checkpoint(conf: DictConfig):
    # Check that log dir has been set, this is used to load config and checkpoints
    assert conf.checkpoint_dir
    conf.checkpoint_dir = Path(conf.checkpoint_dir).resolve()

    # Set log dir
    if not conf.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        conf.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Load config from log_dir
    snef_cfg = omegaconf.OmegaConf.load(f"{conf.checkpoint_dir}/.hydra/config.yaml")
    snef_cfg.logging.log_dir = conf.checkpoint_dir

    # Overwrite dataset specs as per the config
    snef_cfg.dataset.name = conf.dataset.name if conf.dataset.name else snef_cfg.dataset.name
    snef_cfg.dataset.num_signals_train = conf.dataset.num_signals_train
    snef_cfg.dataset.num_signals_test = conf.dataset.num_signals_test
    snef_cfg.dataset.batch_size = conf.training.batch_size
    snef_cfg.test.min_num_epochs = conf.training.fit_codes_num_epochs

    # Create the dataset
    trainset, testset = get_dataloader(dataset_cfg=snef_cfg.dataset)

    # Get image shape
    sample_batch = next(iter(trainset))
    smp_image = sample_batch[0][0]
    image_shape = smp_image.shape
    snef_cfg.dataset.image_shape = image_shape

    # Create position grid
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-1, 1, image_shape[0]), jnp.linspace(-1, 1, image_shape[1])), axis=-1).reshape(-1, 2)

    # Set dimensionality of input and output
    snef_cfg.nef.num_in = 2
    snef_cfg.nef.num_out = image_shape[-1]

    # Get the correct trainer
    if "meta" not in snef_cfg:
        nef, train_autodecoder, val_autodecoder = get_model(snef_cfg)
        snef_trainer = AutoDecodingENFTrainerImage(
            config=snef_cfg,
            enf=enf,
            train_autodecoder=train_autodecoder,
            val_autodecoder=val_autodecoder,
            train_loader=trainset,
            val_loader=testset,
            coords=coords,
            seed=42
        )
    else:
        enf, inner_autodecoder, outer_autodecoder = get_model(snef_cfg)

        snef_trainer = MetaSGDAutoDecodingENFTrainerImage(
            config=snef_cfg,
            enf=enf,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            train_loader=trainset,
            val_loader=testset,
            coords=coords,
            seed=42
        )

    snef_trainer.create_functions()

    # Load checkpoint
    snef_state = snef_trainer.load_checkpoint()

    # Close prog bar
    snef_trainer.prog_bar.close()

    # Check if config makes sense
    assert not conf.latent_dataset.load, "You are trying to load a latent dataset from a checkpoint using a script meant for fiting a latent dataset."
    assert conf.latent_dataset.store_if_new, "You are trying to create a latent dataset without storing it."

    # Initialize wandb
    wandb.init(
        entity="equivariance",
        project="enf-jax-dataset-creation",
        name=f"latent-dataset-{conf.dataset.name}",
        dir=conf.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(snef_cfg),
        mode='disabled' if conf.logging.debug else 'online',
    )

    # Create downstream dataset
    _ = get_or_create_latent_dataset_from_snef(conf, snef_trainer, snef_state)


if __name__ == "__main__":
    # Load config
    create_latent_dataset_from_checkpoint()
