import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.brain_module import BrainModule
from library.datasets import ThingsMEGDatasetWithImages
from library.function import MSE_loss
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm


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


def train_classifier(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    n_epochs: int,
    device: torch.device,
    image_module: nn.Module,
    classifier: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    image_module.to(device)
    classifier.to(device)
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []

    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    for epoch in range(n_epochs):
        losses_train = []
        losses_valid = []

        classifier.train()  # 訓練モードにする
        n_train = 0  # 訓練データ数
        acc_train = 0  # 訓練データに対する精度
        for image_X, _, y, _ in tqdm(train_loader):
            optimizer.zero_grad()  # 勾配の初期化

            z = image_module(image_X.to(device))
            outputs = classifier(z)

            _, predicted = torch.max(
                outputs.data, 1
            )  # 最大値を取るラベルを予測ラベルとする
            # loss
            loss = F.cross_entropy(pred, y)

            loss.backward()  # 誤差の逆伝播
            optimizer.step()  # パラメータの更新

            losses_train.append(loss.tolist())

        classifier.eval()  # 評価モードにする

        for image_X, _, y, _ in valid_loader:
            z = image_module(image_X.to(device))
            pred = classifier(z)

            # loss
            loss = nn.CrossEntropyLoss()(pred, y)
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

        if scheduler is not None:
            scheduler.step()

    plt.plot(train_loss_list, label="train loss")
    plt.plot(valid_loss_list, label="valid loss")
    # 凡例を表示
    plt.legend()
    # plt.savefig("drive/MyDrive/Colab Notebooks/DLBasics2023_colab/Lecture05/loss.png")
    plt.show()


if __name__ == "__main__":
    data_dir = "../../data"
    batch_size = 32
    num_workers = 1
    loader_args = {"batch_size": batch_size, "num_workers": num_workers}

    pretrained_model2latent_dim = {
        "dinov2_vits14": 384,
    }

    train_set = ThingsMEGDatasetWithImages("train", data_dir)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    valid_set = ThingsMEGDatasetWithImages("val", data_dir)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_args)
    n_epochs = 200
    device = torch.device("mps")

    loss_weight = 0  # clip lossの重み
    image_module = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    brain_module = BrainModule(out_dim=pretrained_model2latent_dim["dinov2_vits14"])
    optimizer = optim.Adam(brain_module.parameters(), lr=3e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 120, 160], 0.2)

    train_brain_module(
        train_loader=train_loader,
        valid_loader=valid_loader,
        n_epochs=n_epochs,
        loss_weight=loss_weight,
        device=device,
        image_module=image_module,
        brain_module=brain_module,
        optimizer=optimizer,
        scheduler=None,
    )
