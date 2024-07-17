import os
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
from tqdm import tqdm

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # すべての画像を同じサイズにリサイズ
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

DROPPED_SENSOR_IDX = 69  # 271個のチャンネルのうち、69番目のセンサーMLO42の情報がMNEのデータから欠けているため、削除する


def interpolate_image_file_path(image_file_path: str) -> str:
    if "/" in image_file_path:
        return image_file_path
    directory_name = image_file_path[: image_file_path.rindex("_")]
    return f"{directory_name}/{image_file_path}"


def load_image_as_tensor(
    image_path: str,
) -> torch.Tensor:
    # 画像を開く
    image_file = Image.open(image_path)

    # 画像をRGBモードに変換（画像がグレースケールの場合に必要）
    image = image_file.convert("RGB")

    transform = TRANSFORM

    tensor_image = transform(image)
    return tensor_image  # type: ignore


def drop_invalid_channels(brain_X: torch.Tensor) -> torch.Tensor:
    # brain_X: (271, 281)
    return torch.cat([brain_X[:DROPPED_SENSOR_IDX], brain_X[DROPPED_SENSOR_IDX + 1 :]])


class ThingsMEGDatasetWithImages(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str = "data",
        drop_invalid_channels: bool = True,
        embedding_model_id: str
        | None = None,  # repo_model_name, e.g. "facebookresearch_dinov2_dinov2_vits14"
    ) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.brain_data_dir = data_dir + "/data-omni"
        self.image_data_dir = data_dir + "/images"
        self.num_classes = 1854
        self.num_samples = len(
            glob(os.path.join(self.brain_data_dir, f"{split}_X", "*.npy"))
        )
        self.embedding_model_id = (
            embedding_model_id.replace("/", "-")
            if embedding_model_id is not None
            else None
        )
        self.use_embedded_images = embedding_model_id is not None

        self.drop_invalid_channels = drop_invalid_channels

        if split in ["train", "val"]:
            self.image_file_paths: list[str] = []
            with open(f"{self.brain_data_dir}/{split}_image_paths.txt", "r") as f:
                for line in f:
                    self.image_file_paths.append(
                        interpolate_image_file_path(line.strip())
                    )
            # self.categories = sorted(
            #     [category for category in os.listdir(self.image_data_dir)]
            # )
            # assert len(self.categories) == self.num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
        self, i: int
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor]
    ):
        brain_X_path = os.path.join(
            self.brain_data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy"
        )
        brain_X = torch.from_numpy(np.load(brain_X_path))
        if self.drop_invalid_channels:
            brain_X = drop_invalid_channels(brain_X)

        subject_idx_path = os.path.join(
            self.brain_data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy"
        )
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(
                self.brain_data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy"
            )
            y = torch.from_numpy(np.load(y_path))

            image_file_name: str = self.image_file_paths[i]
            if self.use_embedded_images:
                assert self.embedding_model_id is not None
                embedded_image_path = os.path.join(
                    self.image_data_dir,
                    image_file_name[: image_file_name.rindex(".")]
                    + "_"
                    + self.embedding_model_id
                    + ".npy",
                )
                image_X = torch.from_numpy(np.load(embedded_image_path))

            else:
                image_path = os.path.join(self.image_data_dir, image_file_name)
                image_X = load_image_as_tensor(image_path)

            # if self.categories[y] != self.image_file_paths[i].split("/")[0]:
            #     print(i)
            #     print(self.categories[y], self.image_file_paths[i].split("/")[0])
            #     self.error_count += 1
            # assert self.categories[y] == self.image_file_paths[i].split("/")[0]
            return image_X, brain_X, y, subject_idx
        else:
            return brain_X, subject_idx

    @property
    def num_channels(self) -> int:
        return int(
            np.load(
                os.path.join(self.brain_data_dir, f"{self.split}_X", "00000.npy")
            ).shape[0]
        )

    @property
    def seq_len(self) -> int:
        return int(
            np.load(
                os.path.join(self.brain_data_dir, f"{self.split}_X", "00000.npy")
            ).shape[1]
        )

    def save_embedded_images(self, model: nn.Module, model_id: str) -> None:
        model_id = model_id.replace("/", "-")
        for image_path in tqdm(self.image_file_paths):
            tensor_image = load_image_as_tensor(
                os.path.join(self.image_data_dir, image_path)
            ).unsqueeze(0)
            embedded_image = model(tensor_image).squeeze(0)
            save_path = os.path.join(
                self.image_data_dir,
                image_path[: image_path.rindex(".")] + "_" + model_id + ".npy",
            )
            np.save(save_path, embedded_image.detach().numpy())


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"

        self.split = split
        self.data_dir = data_dir + "/data-omni"
        self.num_classes = 1854
        self.num_samples = len(glob(os.path.join(data_dir, f"{split}_X", "*.npy")))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i):
        X_path = os.path.join(
            self.data_dir, f"{self.split}_X", str(i).zfill(5) + ".npy"
        )
        X = torch.from_numpy(np.load(X_path))

        subject_idx_path = os.path.join(
            self.data_dir, f"{self.split}_subject_idxs", str(i).zfill(5) + ".npy"
        )
        subject_idx = torch.from_numpy(np.load(subject_idx_path))

        if self.split in ["train", "val"]:
            y_path = os.path.join(
                self.data_dir, f"{self.split}_y", str(i).zfill(5) + ".npy"
            )
            y = torch.from_numpy(np.load(y_path))

            return X, y, subject_idx
        else:
            return X, subject_idx

    @property
    def num_channels(self) -> int:
        return np.load(
            os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")
        ).shape[0]

    @property
    def seq_len(self) -> int:
        return np.load(
            os.path.join(self.data_dir, f"{self.split}_X", "00000.npy")
        ).shape[1]
