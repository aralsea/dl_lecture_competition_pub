import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.library.classification_module import MLPHead
from src.library.datasets import ThingsMEGDatasetWithImages
from src.library.image_module import PRETRAINED_MODEL_TO_LATENT_DIMENSION
from src.library.train_classifier import train_classifier
from src.library.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config-baseline")
def run(args: DictConfig) -> None:
    set_seed(args.seed)
    logdir = HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    train_set = ThingsMEGDatasetWithImages(
        "train",
        args.data_dir,
        # embedding_model_id=args.image_module.repo + "-" + args.image_module.model,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDatasetWithImages(
        "val",
        args.data_dir,
        # embedding_model_id=args.image_module.repo + "-" + args.image_module.model,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDatasetWithImages("test", args.data_dir)
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    latent_dim = PRETRAINED_MODEL_TO_LATENT_DIMENSION[
        (args.image_module.repo, args.image_module.model)
    ]

    # ------------------
    #       Model
    # ------------------
    image_module: nn.Module = torch.hub.load(
        args.image_module.repo, args.image_module.model
    )  # type: ignore
    for param in image_module.parameters():
        param.requires_grad = False
    classifier = MLPHead(in_dim=latent_dim, num_classes=train_set.num_classes)
    # brain_module = BrainModule(out_dim=latent_dim)

    # ------------------
    #     Optimizer
    # ------------------
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.classifier_lr
    )
    # brain_optimizer = torch.optim.Adam(
    #     brain_module.parameters(), lr=args.brain_module_lr
    # )

    # ------------------
    #   Start training
    # ------------------
    # train classifier
    train_classifier(
        args,
        logdir,
        train_loader,
        val_loader,
        image_module=image_module,
        classifier=classifier,
        optimizer=classifier_optimizer,
    )

    # train brain module
    # train_brain_module(
    #     args,
    #     logdir,
    #     train_loader,
    #     val_loader,
    #     image_module=image_module,
    #     brain_module=brain_module,
    #     optimizer=brain_optimizer,
    # )

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    classifier.load_state_dict(
        torch.load(os.path.join(logdir, "classifier_best.pt"), map_location=args.device)
    )
    # brain_module.load_state_dict(
    #     torch.load(
    #         os.path.join(logdir, "brain_module_best.pt"), map_location=args.device
    #     )
    # )
    return
    model = nn.Sequential(brain_module, classifier).to(args.device)
    preds = []
    model.eval()
    for brain_X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(
            model(brain_X.to(args.device), subject_idxs.to(args.device)).detach().cpu()
        )

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")  # type: ignore


if __name__ == "__main__":
    run()
