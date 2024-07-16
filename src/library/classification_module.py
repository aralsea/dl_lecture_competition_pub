import torch
import torch.nn as nn

NUM_CLASSES = 1854


# ViTのヘッド部分
class MLPHead(nn.Moduel):
    def __init__(self, in_dim: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim, num_classes),
            nn.Dropout(0.1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.head(X)  # type:ignore
