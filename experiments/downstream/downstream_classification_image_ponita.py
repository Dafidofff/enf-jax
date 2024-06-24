import hydra
import wandb
import omegaconf
from omegaconf import DictConfig
from pathlib import Path

import pathlib
from experiments.downstream.latent_dataset import get_latent_dataloader_from_path
from experiments.downstream.utils.downstream_models.fc_ponita import PonitaFixedSize
from experiments.downstream.trainers.downstream_classifier_trainer import DownstreamClassifierTrainer


@hydra.main(version_base=None, config_path="./configs/", config_name="config_classification_image_ponita")
def train(classifier_cfg: DictConfig):
    # Check that log dir has been set, this is used to load config and checkpoints
    assert classifier_cfg.checkpoint_dir
    classifier_cfg.checkpoint_dir = Path(classifier_cfg.checkpoint_dir)

    # Set log dir
    if not classifier_cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        classifier_cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Load config from log_dir
    enf_cfg = omegaconf.OmegaConf.load(f"{classifier_cfg.checkpoint_dir}/.hydra/config.yaml")

    # Check for the existence of the latent dataset
    latent_dataset_path = pathlib.Path(classifier_cfg.checkpoint_dir).joinpath("latent_dataset")
    assert latent_dataset_path.exists(), f"Latent dataset does not exist, create it first using: 'python create_latent_dataset_from_checkpoint.py --checkpoint_dir={classifier_cfg.checkpoint_dir}'"
    train_loader, val_loader = get_latent_dataloader_from_path(classifier_cfg, latent_dataset_path)

    # Get MLP
    classifier = PonitaFixedSize(
        num_hidden=classifier_cfg.classifier.num_hidden,
        num_layers=classifier_cfg.classifier.num_layers,
        scalar_num_out=10,
        vec_num_out=0,
        spatial_dim=2,
        num_ori=classifier_cfg.classifier.num_ori,
        basis_dim=classifier_cfg.classifier.basis_dim,
        degree=classifier_cfg.classifier.degree,
        widening_factor=classifier_cfg.classifier.widening_factor,
        global_pool=True,
        kernel_size=classifier_cfg.classifier.kernel_size
    )

    classifier_trainer = DownstreamClassifierTrainer(
        classifier=classifier,
        config=classifier_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        seed=42
    )

    classifier_trainer.create_functions()

    # Initialize wandb
    wandb.init(
        entity="equivariance",
        project="snef-jax-classification",
        name=f"classification_ponita_{enf_cfg.dataset.name}",
        dir=classifier_cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(classifier_cfg),
        mode='disabled' if classifier_cfg.logging.debug else 'online',
    )

    # Train the classifier
    classifier_trainer.train_model(classifier_cfg.training.num_epochs)


if __name__ == "__main__":
    train()
