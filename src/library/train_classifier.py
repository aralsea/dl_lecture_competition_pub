import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from src.library.utils import get_model_id


def train_classifier(
    args: DictConfig,
    logdir: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    image_module: nn.Module,
    classifier: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    model_id = get_model_id(args)

    image_module.to(args.device)
    classifier.to(args.device)
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass",
        num_classes=train_loader.dataset.num_classes,  # type: ignore
        top_k=10,
    ).to(args.device)

    for epoch in range(args.classifier_epochs):
        print(f"Epoch {epoch+1}/{args.classifier_epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        classifier.train()
        for image_X, brain_X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            image_X, y = image_X.to(args.device), y.to(args.device)

            if not train_loader.dataset.use_embedded_images:  # type:ignore
                image_X = image_module(image_X)
            y_pred = classifier(image_X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        classifier.eval()
        for image_X, brain_X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            image_X, y = image_X.to(args.device), y.to(args.device)

            with torch.no_grad():
                if not val_loader.dataset.use_embedded_images:  # type:ignore
                    image_X = image_module(image_X)
                y_pred = classifier(image_X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch+1}/{args.classifier_epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(
            classifier.state_dict(),
            os.path.join(logdir, f"classifier_last_{model_id}.pt"),
        )
        if args.use_wandb:
            wandb.log(
                {
                    "classifier_train_loss": np.mean(train_loss),
                    "classifier_train_acc": np.mean(train_acc),
                    "classifier_val_loss": np.mean(val_loss),
                    "classifier_val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(
                classifier.state_dict(),
                os.path.join(logdir, f"classifier_best_{model_id}.pt"),
            )
            max_val_acc = np.mean(val_acc)
