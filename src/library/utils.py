import random

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_model_id(args: DictConfig) -> str:
    return args.image_module.repo.replace("/", "_") + "_" + args.image_module.model  # type: ignore
