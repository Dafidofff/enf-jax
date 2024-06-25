import omegaconf
import hydra
from omegaconf import DictConfig
import wandb

import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.shape.ad_enf_trainer_shape import AutoDecodingENFTrainerShape
from experiments.fitting.trainers.shape.ad_enf_trainer_meta_sgd_shape import MetaSGDAutoDecodingENFTrainerShape

from experiments.fitting import get_model
from experiments.downstream.latent_dataset.utils import get_or_create_latent_dataset_from_enf


@hydra.main(version_base=None, config_path="./configs/", config_name="config_create_latent_dataset_from_checkpoint")
def create_latent_dataset_from_checkpoint(conf: DictConfig):
    # Check that log dir has been set, this is used to load config and checkpoints
    assert conf.checkpoint_dir

    # Set log dir
    if not conf.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        conf.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Load config from log_dir
    enf_cfg = omegaconf.OmegaConf.load(f"{conf.checkpoint_dir}/.hydra/config.yaml")
    enf_cfg.logging.log_dir = conf.checkpoint_dir

    # Overwrite dataset specs as per the config
    enf_cfg.dataset.name = conf.dataset.name if conf.dataset.name else enf_cfg.dataset.name
    enf_cfg.dataset.path = conf.dataset.path if conf.dataset.path else enf_cfg.dataset.path
    enf_cfg.dataset.num_signals_train = conf.dataset.num_signals_train
    enf_cfg.dataset.num_signals_test = conf.dataset.num_signals_test
    enf_cfg.dataset.batch_size = conf.training.batch_size
    enf_cfg.test.min_num_epochs = conf.training.fit_codes_num_epochs

    # Create the dataset
    trainset, testset = get_dataloader(dataset_cfg=enf_cfg.dataset)

    # Get image shape
    sample_batch = next(iter(trainset))
    smp_image = sample_batch[0][0]
    image_shape = smp_image.shape
    enf_cfg.dataset.image_shape = image_shape

    # Set dimensionality of input and output
    enf_cfg.nef.num_in = 3
    enf_cfg.nef.num_out = 1

    # Get the correct trainer
    if "meta" not in enf_cfg:
        nef, train_autodecoder, val_autodecoder = get_model(enf_cfg)
        enf_trainer = AutoDecodingENFTrainerShape(
            nef=nef,
            train_autodecoder=train_autodecoder,
            val_autodecoder=val_autodecoder,
            config=enf_cfg,
            train_loader=trainset,
            val_loader=testset,
            seed=42,
        )
    else:
        # Get nef and autodecoders
        enf, inner_autodecoder, outer_autodecoder = get_model(enf_cfg)
        enf_trainer = MetaSGDAutoDecodingENFTrainerShape(
            enf=enf,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            config=enf_cfg,
            train_loader=trainset,
            val_loader=testset,
            seed=42,
        )

    enf_trainer.create_functions()

    # Load checkpoint
    enf_state = enf_trainer.load_checkpoint()

    # Close prog bar
    enf_trainer.prog_bar.close()

    # Check if config makes sense
    assert not conf.latent_dataset.load, "You are trying to load a latent dataset from a checkpoint using a script meant for fiting a latent dataset."
    assert conf.latent_dataset.store_if_new, "You are trying to create a latent dataset without storing it."

    # Initialize wandb
    wandb.init(
        entity="equivariance",
        project="enf-jax-dataset-creation",
        name=f"latent-dataset-{conf.dataset.name}",
        dir=conf.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(enf_cfg),
        mode='disabled' if conf.logging.debug else 'online',
    )

    # Create downstream dataset
    _ = get_or_create_latent_dataset_from_enf(conf, enf_trainer, enf_state)


if __name__ == "__main__":
    # Load config
    create_latent_dataset_from_checkpoint()
