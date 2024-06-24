import hydra
import wandb
import omegaconf
from omegaconf import DictConfig

from experiments.fitting.datasets import get_dataloader
from experiments.fitting.trainers.shape.ad_enf_trainer_shape import AutoDecodingENFTrainerShape

from experiments.fitting import get_model


@hydra.main(version_base=None, config_path=".", config_name="ad_config_shape")
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

    # Get nef and autodecoders
    nef, train_autodecoder, val_autodecoder = get_model(cfg)

    trainer = AutoDecodingENFTrainerShape(
        nef=nef,
        train_autodecoder=train_autodecoder,
        val_autodecoder=val_autodecoder,
        config=cfg,
        train_loader=trainset,
        val_loader=testset,
        seed=42,
    )

    trainer.create_functions()

    # Initialize wandb
    wandb.init(
        project="enf-jax",
        name=f"ad_{cfg.dataset.name}",
        dir=cfg.logging.log_dir,
        config=omegaconf.OmegaConf.to_container(cfg),
        mode='disabled' if cfg.logging.debug else 'online',
    )

    # Train model
    trainer.train_model(cfg.training.num_epochs)


if __name__ == "__main__":
    train()
