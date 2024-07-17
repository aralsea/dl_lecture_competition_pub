import os

import hydra
import numpy as np
import torch
from library.brain_module import BrainModule
from library.classification_module import MLPHead
from library.datasets import ThingsMEGDatasetWithImages
from library.image_module import PRETRAINED_MODEL_TO_LATENT_DIMENSION
from library.utils import get_model_id, set_seed
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig) -> None:
    set_seed(args.seed)
    model_id = get_model_id(args)
    logdir = f"outputs/{model_id}"

    latent_dim = PRETRAINED_MODEL_TO_LATENT_DIMENSION[
        (args.image_module.repo, args.image_module.model)
    ]

    # ------------------
    #    Dataloader
    # ------------------
    test_set = ThingsMEGDatasetWithImages("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ------------------
    #       Model
    # ------------------
    classifier = MLPHead(in_dim=latent_dim, num_classes=test_set.num_classes)
    classifier.load_state_dict(
        torch.load(os.path.join(logdir, f"classifier_best_{model_id}.pt"))
    )
    brain_module = BrainModule(out_dim=latent_dim).to(args.device)
    brain_module.load_state_dict(
        torch.load(os.path.join(logdir, f"brain_module_best_{model_id}.pt"))
    )

    # ------------------
    #  Start evaluation
    # ------------------
    preds = []
    classifier.eval()
    brain_module.eval()
    for brain_X, subject_idx in tqdm(test_loader, desc="test"):
        clip_latent_vector, mse_latent_vector = brain_module(
            brain_X.to(args.device), subject_idx.to(args.device)
        )
        preds.append(classifier(clip_latent_vector)).detach().cpu()  # type: ignore

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, f"submission_{model_id}"), preds)
    cprint(
        f"Submission {preds.shape} saved at {os.path.join(logdir, f"submission_{model_id}")}",
        "cyan",
    )  # type: ignore


if __name__ == "__main__":
    run()
