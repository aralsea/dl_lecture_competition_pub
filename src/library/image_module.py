import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models

PRETRAINED_MODEL_TO_LATENT_DIMENSION = {
    # (repo, model): latent_dim
    ("facebookresearch/dinov2", "dinov2_vits14"): 384,
    ("pytorch/vision:v0.10.0", "resnet152"): 1000,
    ("pytorch/vision:v0.10.0", "resnet18"): 1000,
}


def get_image_module(args: DictConfig) -> nn.Module:
    match args.image_module.repo:
        case "facebookresearch/dinov2":
            model = torch.hub.load(args.image_module.repo, args.image_module.model)  # type:ignore
            return model  # type:ignore
        case "pytorch/vision:v0.10.0":
            match args.image_module.model:
                case "resnet152":
                    model = torch.hub.load(
                        args.image_module.repo,
                        args.image_module.model,
                        weights=models.ResNet152_Weights.DEFAULT,
                    )  # type:ignore
                    return model  # type:ignore
                case "resnet18":
                    model = torch.hub.load(
                        args.image_module.repo,
                        args.image_module.model,
                        weights=models.ResNet18_Weights.DEFAULT,
                    )  # type: ignore
                    return model  # type: ignore
                case _:
                    raise ValueError("Unknown model")
        case _:
            raise ValueError("Unknown repo")
