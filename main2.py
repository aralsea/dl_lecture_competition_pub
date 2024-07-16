import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from termcolor import cprint
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

from src.library.brain_module import BrainModule
from src.library.classification_module import MLPHead
from src.library.constants import PRETRAINED_MODEL_TO_LATENT_DIMENSION
from src.library.datasets import ThingsMEGDatasetWithImages
from src.library.function import MSE_loss
from src.library.utils import set_seed


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
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass",
        num_classes=train_loader.dataset.num_classes,  # type: ignore
        top_k=10,
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        classifier.train()
        for emb_image_X, brain_X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            emb_image_X, y = emb_image_X.to(args.device), y.to(args.device)

            y_pred = classifier(emb_image_X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()  # type: ignore
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        classifier.eval()
        for emb_image_X, brain_X, y, subject_idxs in tqdm(
            val_loader, desc="Validation"
        ):
            emb_image_X, y = emb_image_X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = classifier(emb_image_X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(classifier.state_dict(), os.path.join(logdir, "classifier_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(
                classifier.state_dict(), os.path.join(logdir, "classifier_best.pt")
            )
            max_val_acc = np.mean(val_acc)


def train_brain_module(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    n_epochs: int,
    loss_weight: float,
    device: torch.device,
    image_module: nn.Module,
    brain_module: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    image_module.to(device)
    brain_module.to(device)
    train_loss_list = []
    valid_loss_list = []

    for epoch in range(n_epochs):
        clip_losses_train = []  # 訓練誤差を格納しておくリスト
        clip_losses_valid = []  # 検証データの誤差を格納しておくリスト
        mse_losses_train = []  # 訓練誤差を格納しておくリスト
        mse_losses_valid = []  # 検証データの誤差を格納しておくリスト

        brain_module.train()  # 訓練モードにする
        for image_X, brain_X, y, subject_idx in tqdm(train_loader):
            optimizer.zero_grad()  # 勾配の初期化

            z = image_module(image_X.to(device))
            pred_z = brain_module(brain_X.to(device), subject_idx)

            # MSE loss
            mse_loss = MSE_loss(z, pred_z)

            # clip loss
            clip_loss = 0

            # loss
            loss = loss_weight * clip_loss + (1.0 - loss_weight) * mse_loss

            loss.backward()  # 誤差の逆伝播
            optimizer.step()  # パラメータの更新

            clip_losses_train.append(clip_loss.tolist())
            mse_losses_train.append(mse_loss.tolist())

        brain_module.eval()  # 評価モードにする

        for image_X, brain_X, y, subject_idx in valid_loader:
            z = image_module(image_X.to(device))
            pred_z = brain_module(brain_X.to(device), subject_idx)

            # MSE loss
            mse_loss = mse_loss(z, pred_z)

            # clip loss
            clip_loss = 0

            # loss
            loss = loss_weight * clip_loss + (1.0 - loss_weight) * mse_loss

            clip_losses_valid.append(clip_loss.tolist())
            mse_losses_valid.append(mse_loss.tolist())

        losses_train = [
            loss_weight * clip_loss + (1.0 - loss_weight) * mse_loss
            for clip_loss, mse_loss in zip(clip_losses_train, mse_losses_train)
        ]
        losses_valid = [
            loss_weight * clip_loss + (1.0 - loss_weight) * mse_loss
            for clip_loss, mse_loss in zip(clip_losses_valid, mse_losses_valid)
        ]

        print(
            "EPOCH: {}, Train [Loss: {:.3f}], Valid [Loss: {:.3f}]".format(
                epoch,
                np.mean(losses_train),
                np.mean(losses_valid),
            )
        )
        train_loss_list.append(np.mean(losses_train))
        valid_loss_list.append(np.mean(losses_valid))

        if scheduler is not None:
            scheduler.step()

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="valid loss")
    # 凡例を表示
    plt.legend()
    # plt.savefig("drive/MyDrive/Colab Notebooks/DLBasics2023_colab/Lecture05/loss.png")
    plt.show()


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
        embedding_model_id=args.image_module.repo + "-" + args.image_module.model,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDatasetWithImages(
        "val",
        args.data_dir,
        embedding_model_id=args.image_module.repo + "-" + args.image_module.model,
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
    )
    classifier = MLPHead(in_dim=latent_dim, out_dim=train_set.num_classes)
    brain_module = BrainModule(out_dim=latent_dim)

    # ------------------
    #     Optimizer
    # ------------------
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    brain_optimizer = torch.optim.Adam(brain_module.parameters(), lr=args.lr)

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

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(
        torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device)
    )

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")  # type: ignore


if __name__ == "__main__":
    run()
