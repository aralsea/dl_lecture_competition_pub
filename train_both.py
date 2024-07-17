import os

import hydra
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.library.brain_module import BrainModule
from src.library.classification_module import MLPHead
from src.library.datasets import ThingsMEGDatasetWithImages
from src.library.image_module import (
    PRETRAINED_MODEL_TO_LATENT_DIMENSION,
    get_image_module,
)
from src.library.train_brain_module import train_brain_module
from src.library.utils import get_model_id, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config-baseline")
def run(args: DictConfig) -> None:
    set_seed(args.seed)
    model_id = get_model_id(args)
    logdir = f"outputs/{model_id + str(args.loss_weight)}"
    os.makedirs(logdir, exist_ok=True)

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    model_id = get_model_id(args)
    train_set = ThingsMEGDatasetWithImages(
        "train",
        args.data_dir,
        embedding_model_id=(model_id if args.use_cache else None),
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDatasetWithImages(
        "val",
        args.data_dir,
        embedding_model_id=(model_id if args.use_cache else None),
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    latent_dim = PRETRAINED_MODEL_TO_LATENT_DIMENSION[
        (args.image_module.repo, args.image_module.model)
    ]

    # ------------------
    #       Model
    # ------------------
    image_module = get_image_module(args)
    for param in image_module.parameters():
        param.requires_grad = False

    classifier = MLPHead(in_dim=latent_dim, num_classes=train_set.num_classes)
    classifier.load_state_dict(
        torch.load(
            os.path.join(logdir, f"classifier_best_{model_id}.pt"),
            map_location=args.device,
        )
    )
    print("Best classifier loaded")
    for param in classifier.parameters():
        param.requires_grad = False

    brain_module = BrainModule(out_dim=latent_dim)

    if args.start_from_best:
        try:
            brain_module.load_state_dict(
                torch.load(
                    os.path.join(logdir, f"brain_module_best_{model_id}.pt"),
                    map_location=args.device,
                )
            )
            print("Best brain module loaded")
        except FileNotFoundError:
            print("Best brain module not found, starting from scratch")
    # ------------------
    #     Optimizer
    # ------------------

    brain_optimizer = torch.optim.Adam(
        brain_module.parameters(), lr=args.brain_module_lr
    )

    # ------------------
    #   Start training
    # ------------------

    # train brain module
    train_brain_module(
        args,
        logdir,
        train_loader,
        val_loader,
        image_module=image_module,
        classifier=classifier,
        brain_module=brain_module,
        optimizer=brain_optimizer,
    )


if __name__ == "__main__":
    run()
