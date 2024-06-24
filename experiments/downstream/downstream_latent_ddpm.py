import hydra
import wandb
import pathlib
import omegaconf
from omegaconf import DictConfig

import jax.numpy as jnp

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.image.ad_enf_trainer_image import AutoDecodingENFTrainerImage
from experiments.fitting.trainers.image.ad_enf_trainer_meta_sgd_image import MetaSGDAutoDecodingENFTrainerImage

from experiments.fitting import get_model
from experiments.downstream.latent_dataset import get_latent_dataloader_from_path
from experiments.downstream.utils.downstream_models.fc_ponita import PonitaFixedSize
from experiments.downstream.trainers.downstream_ddpm_trainer import DownstreamDDPMTrainer


@hydra.main(version_base=None, config_path=".", config_name="config_latent_ddpm")
def train(classifier_cfg: DictConfig):
    # Check that log dir has been set, this is used to load config and checkpoints
    assert classifier_cfg.checkpoint_dir

    # Set log dir
    if not classifier_cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        classifier_cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Load config from log_dir
    cfg = omegaconf.OmegaConf.load(f"{classifier_cfg.checkpoint_dir}/.hydra/config.yaml")
    cfg.logging.log_dir = classifier_cfg.checkpoint_dir

    # Set number of signals
    cfg.diffusion = classifier_cfg.diffusion
    cfg.training.batch_size = classifier_cfg.training.batch_size
    cfg.dataset.name = classifier_cfg.dataset.name if classifier_cfg.dataset.name != '' else cfg.dataset.name
    cfg.dataset.path = classifier_cfg.dataset.path if classifier_cfg.dataset.path != '' else cfg.dataset.path
    cfg.dataset.num_signals_train = classifier_cfg.dataset.num_signals_train
    cfg.dataset.num_signals_test = classifier_cfg.dataset.num_signals_test

    # Check for the existence of the latent dataset
    latent_dataset_path = pathlib.Path(classifier_cfg.checkpoint_dir).joinpath("latent_dataset")
    assert latent_dataset_path.exists(), f"Latent dataset does not exist, create it first using: 'python create_latent_dataset_from_checkpoint.py --checkpoint_dir={classifier_cfg.checkpoint_dir}'"
    train_loader, val_loader = get_latent_dataloader_from_path(classifier_cfg, latent_dataset_path)

    # Create the signal dataset
    trainset, testset = get_dataloader(dataset_cfg=cfg.dataset)

    # Get image shape
    sample_batch = next(iter(trainset))
    smp_image = sample_batch[0][0]
    image_shape = smp_image.shape
    cfg.dataset.image_shape = image_shape
    classifier_cfg.latent_dataset.image_shape = image_shape
    
    # Create position grid
    coords = jnp.stack(jnp.meshgrid(
        jnp.linspace(-1, 1, image_shape[0]), jnp.linspace(-1, 1, image_shape[1])), axis=-1).reshape(-1, 2)

    # Get the enf model
    if 'shape' in str(cfg.dataset.name).lower():
        cfg.nef.num_in = 3
        cfg.nef.num_out = 1
    else:
        cfg.nef.num_in = 2
        cfg.nef.num_out = image_shape[-1]
    
    # Get the correct trainer
    if "meta" not in cfg:
        nef, train_autodecoder, val_autodecoder = get_model(cfg)
        enf_trainer = AutoDecodingENFTrainerImage(
            config=cfg,
            nef=nef,
            train_autodecoder=train_autodecoder,
            val_autodecoder=val_autodecoder,
            train_loader=trainset,
            val_loader=testset,
            coords=coords,
            seed=42
        )
    else:
        nef, inner_autodecoder, outer_autodecoder = get_model(cfg)

        enf_trainer = MetaSGDAutoDecodingENFTrainerImage(
            config=cfg,
            nef=nef,
            inner_autodecoder=inner_autodecoder,
            outer_autodecoder=outer_autodecoder,
            train_loader=trainset,
            val_loader=testset,
            coords=coords,
            seed=42
        )

    enf_trainer.create_functions()
    enf_state = enf_trainer.load_checkpoint()

    # Initialize wandb
    wandb.init(
        entity="equivariance",
        project="enf-jax-diffusion",
        dir=cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(cfg),
        mode='disabled' if cfg.logging.debug else 'online',
    )

    # Get diffusion model
    diffuser = PonitaFixedSize(
        num_hidden=classifier_cfg.ponita.num_hidden+classifier_cfg.ponita.time_embedding_dim,
        num_layers=classifier_cfg.ponita.num_layers,
        scalar_num_out=cfg.nef.latent_dim,
        vec_num_out=1,
        spatial_dim=cfg.nef.num_in,
        num_ori=classifier_cfg.ponita.num_ori,
        basis_dim=classifier_cfg.ponita.basis_dim,
        degree=3,
        widening_factor=2,
        global_pool=False,
        kernel_size=classifier_cfg.ponita.kernel_size,
        last_feature_conditioning=True
    )

    # Setup the diffusion trainer
    diffusion_trainer = DownstreamDDPMTrainer(
        enf=enf_trainer,
        enf_state=enf_state,
        model=diffuser,
        config=classifier_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        seed=42
    )
    diffusion_trainer.create_functions()

    # Train the classifier
    diffusion_trainer.train_model(classifier_cfg.training.num_epochs)


if __name__ == "__main__":
    train()
