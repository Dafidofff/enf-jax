import hydra
from omegaconf import DictConfig
import omegaconf
import wandb

import pathlib

from experiments.downstream.utils.downstream_models.equivariant_transformer import EquivariantTransformer
from enf.steerable_attention.invariant import get_sa_invariant

from experiments.downstream.latent_dataset import get_latent_dataloader_from_path
from experiments.downstream.trainers.downstream_classifier_trainer import DownstreamClassifierTrainer


@hydra.main(version_base=None, config_path=".", config_name="config_classification_image_transformer")
def train(classifier_cfg: DictConfig):
    # Check that log dir has been set, this is used to load config and checkpoints
    assert classifier_cfg.checkpoint_dir

    # Set log dir
    if not classifier_cfg.logging.log_dir:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        classifier_cfg.logging.log_dir = hydra_cfg['runtime']['output_dir']

    # Load config from log_dir
    snef_cfg = omegaconf.OmegaConf.load(f"{classifier_cfg.checkpoint_dir}/.hydra/config.yaml")

    # Check for the existence of the latent dataset
    latent_dataset_path = pathlib.Path(classifier_cfg.checkpoint_dir).joinpath("latent_dataset")
    assert latent_dataset_path.exists(), f"Latent dataset does not exist, create it first using: 'python create_latent_dataset_from_checkpoint.py --checkpoint_dir={classifier_cfg.checkpoint_dir}'"
    train_loader, val_loader = get_latent_dataloader_from_path(classifier_cfg, latent_dataset_path)

    # Set self attn invariant
    snef_cfg.nef.num_in = 2
    self_attn_invariant = get_sa_invariant(snef_cfg.nef)

    # Get MLP
    classifier = EquivariantTransformer(
        num_hidden=classifier_cfg.classifier.num_hidden,
        num_heads=classifier_cfg.classifier.num_heads,
        num_layers=classifier_cfg.classifier.num_layers,
        num_out=10,
        embedding_type='rff',
        embedding_freq_multiplier=(0.1, 0.1),
        condition_value_transform=True,
        self_attn_invariant=self_attn_invariant,
        global_pooling=True
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
        name=f"classification_ponita",
        dir=classifier_cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(classifier_cfg),
        mode='disabled' if classifier_cfg.logging.debug else 'online',
    )

    # Train the classifier
    classifier_trainer.train_model(classifier_cfg.training.num_epochs)


if __name__ == "__main__":
    train()
