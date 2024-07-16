import torch
import torch.nn as nn


def get_image_module(model_name: str) -> nn.Module:
    match model_name:
        case "dinov2_vits14":
            model: nn.Module = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14"
            )  # type: ignore
            return model
        case _:
            raise ValueError(f"Unknown model: {model_name}")
