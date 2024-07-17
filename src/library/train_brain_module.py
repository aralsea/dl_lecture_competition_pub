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
from torchmetrics import Accuracy
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
    classifier: nn.Module,
    brain_module: BrainModule,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    model_id = get_model_id(args)

    image_module.to(args.device)
    classifier.to(args.device)
    brain_module.to(args.device)

    classifier.eval()

    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []

    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_loader.dataset.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.brain_module_epochs):
        losses_train = []  # 訓練誤差を格納しておくリスト
        losses_valid = []  # 検証データの誤差を格納しておくリスト

        val_acc = []

        brain_module.train()  # 訓練モードにする
        for image_X, brain_X, y, subject_idx in tqdm(train_loader):
            image_X, brain_X, y, subject_idx = (
                image_X.to(args.device),
                brain_X.to(args.device),
                y.to(args.device),
                subject_idx.to(args.device),
            )
            optimizer.zero_grad()  # 勾配の初期化

            z = image_X if args.use_cache else image_module(image_X)
            clip_pred_z, mse_pred_z = brain_module(brain_X, subject_idx)

            # MSE loss
            mse_loss = MSE_loss(z, mse_pred_z)

            # clip loss
            clip_loss = CLIP_loss(z, clip_pred_z, brain_module.temperature)

            # loss
            loss = args.loss_weight * clip_loss + (1.0 - args.loss_weight) * mse_loss

            loss.backward()  # 誤差の逆伝播
            optimizer.step()  # パラメータの更新

            losses_train.append(loss.tolist())

        brain_module.eval()  # 評価モードにする

        for image_X, brain_X, y, subject_idx in tqdm(valid_loader, desc="Validation"):
            image_X, brain_X, y, subject_idx = (
                image_X.to(args.device),
                brain_X.to(args.device),
                y.to(args.device),
                subject_idx.to(args.device),
            )
            z = image_X if args.use_cache else image_module(image_X)
            with torch.no_grad():
                clip_pred_z, mse_pred_z = brain_module(brain_X, subject_idx)

                y_pred = classifier(clip_pred_z)

                # MSE loss
                mse_loss = MSE_loss(z, mse_pred_z)

                # clip loss
                clip_loss = CLIP_loss(z, clip_pred_z, brain_module.temperature)

                # loss
                loss = (
                    args.loss_weight * clip_loss + (1.0 - args.loss_weight) * mse_loss
                )

                losses_valid.append(loss.tolist())

                val_acc.append(accuracy(y_pred, y).item())  # 予測精度を計算

        print(
            f"EPOCH: {epoch}, Train [Loss: { np.mean(losses_train):.3f}], Valid [Loss: {np.mean(losses_valid):.3f}, acc:{np.mean(val_acc):.3f}]".format()
        )
        train_loss_list.append(np.mean(losses_train))
        valid_loss_list.append(np.mean(losses_valid))
        valid_acc_list.append(np.mean(val_acc))
        if args.use_wandb:
            wandb.log(
                {
                    "brain_module_train_loss": np.mean(losses_train),
                    "brain_module_val_loss": np.mean(losses_valid),
                    "val_acc": np.mean(val_acc),
                }
            )
        if np.mean(np.mean(val_acc)) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(
                brain_module.state_dict(),
                os.path.join(logdir, f"brain_module_best_{model_id}.pt"),
            )
            max_val_acc = np.mean(val_acc)

        if scheduler is not None:
            scheduler.step()

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="valid loss")
    plt.plot(valid_acc_list, label="valid acc")
    # 凡例を表示
    plt.legend()
    plt.savefig(os.path.join(logdir, "/brain_module_loss.png"))
    plt.show()
