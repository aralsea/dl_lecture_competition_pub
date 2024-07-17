import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.library.brain_module import BrainModule
from src.library.function import CLIP_loss, MSE_loss
from src.library.utils import get_model_id


def train_brain_module(
    args: DictConfig,
    logdir: str,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    image_module: nn.Module,
    brain_module: BrainModule,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    model_id = get_model_id(args)

    image_module.to(args.device)
    brain_module.to(args.device)
    train_loss_list = []
    valid_loss_list = []

    min_val_loss = np.inf
    for epoch in range(args.brain_module_epochs):
        losses_train = []  # 訓練誤差を格納しておくリスト
        losses_valid = []  # 検証データの誤差を格納しておくリスト

        brain_module.train()  # 訓練モードにする
        for image_X, brain_X, y, subject_idx in tqdm(train_loader):
            optimizer.zero_grad()  # 勾配の初期化

            z = image_module(image_X.to(args.device))
            pred_z = brain_module(brain_X.to(args.device), subject_idx)

            # MSE loss
            mse_loss = MSE_loss(z, pred_z)

            # clip loss
            clip_loss = CLIP_loss(z, pred_z, brain_module.temperature)

            # loss
            loss = args.loss_weight * clip_loss + (1.0 - args.loss_weight) * mse_loss

            loss.backward()  # 誤差の逆伝播
            optimizer.step()  # パラメータの更新

            losses_train.append(loss.tolist())

        brain_module.eval()  # 評価モードにする

        for image_X, brain_X, y, subject_idx in valid_loader:
            z = image_module(image_X.to(args.device))
            pred_z = brain_module(brain_X.to(args.device), subject_idx)

            # MSE loss
            mse_loss = MSE_loss(z, pred_z)

            # clip loss
            clip_loss = CLIP_loss(z, pred_z, brain_module.temperature)

            # loss
            loss = args.loss_weight * clip_loss + (1.0 - args.loss_weight) * mse_loss

            losses_valid.append(loss.tolist())

        print(
            "EPOCH: {}, Train [Loss: {:.3f}], Valid [Loss: {:.3f}]".format(
                epoch,
                np.mean(losses_train),
                np.mean(losses_valid),
            )
        )
        train_loss_list.append(np.mean(losses_train))
        valid_loss_list.append(np.mean(losses_valid))
        if args.use_wandb:
            wandb.log(
                {
                    "brain_module_train_loss": np.mean(losses_train),
                    "brain_module_val_loss": np.mean(losses_valid),
                }
            )
        if np.mean(np.mean(losses_valid)) < min_val_loss:
            cprint("New best.", "cyan")
            torch.save(
                brain_module.state_dict(),
                os.path.join(logdir, f"brain_module_best_{model_id}.pt"),
            )
            min_val_loss = np.mean(losses_valid)

        if scheduler is not None:
            scheduler.step()

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="valid loss")
    # 凡例を表示
    plt.legend()
    # plt.savefig("drive/MyDrive/Colab Notebooks/DLBasics2023_colab/Lecture05/loss.png")
    plt.show()
