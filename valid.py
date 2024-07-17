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

from src.library.brain_module import BrainModule
from src.library.classification_module import MLPHead
from src.library.datasets import ThingsMEGDatasetWithImages
from src.library.image_module import PRETRAINED_MODEL_TO_LATENT_DIMENSION
from src.library.utils import get_model_id, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config-baseline")
def run(args: DictConfig) -> None:
    model_id = get_model_id(args)
    set_seed(args.seed)
    logdir = HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    test_set = ThingsMEGDatasetWithImages("test", args.data_dir)
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    latent_dim = PRETRAINED_MODEL_TO_LATENT_DIMENSION[
        (args.image_module.repo, args.image_module.model)
    ]

    # ------------------
    #       Model
    # ------------------
    classifier = MLPHead(in_dim=latent_dim, num_classes=test_set.num_classes)
    brain_module = BrainModule(out_dim=latent_dim)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    classifier.load_state_dict(
        torch.load(
            os.path.join(logdir, f"classifier_best_{model_id}.pt"),
            map_location=args.device,
        )
    )
    brain_module.load_state_dict(
        torch.load(
            os.path.join(logdir, f"brain_module_best_{model_id}.pt"),
            map_location=args.device,
        )
    )

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
