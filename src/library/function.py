import torch
import torch.nn.functional as F


def MSE_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dim() == 2
    return torch.mean((x - y) ** 2)


def test_mse_loss() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    loss = MSE_loss(x, y)
    assert torch.allclose(loss, torch.tensor(4.5))


def CLIP_loss(x: torch.Tensor, y: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dim() == 2

    # バッチサイズ
    B = x.shape[0]

    # 正規化
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    # コサイン類似度行列を計算
    sim_matrix = torch.matmul(x_norm, y_norm.t()) / tau

    # ラベル（対角要素が正例）
    labels = torch.arange(B, device=x.device)

    # 損失を計算（x->yとy->xの両方向）
    loss_x_y = F.cross_entropy(sim_matrix, labels)
    loss_y_x = F.cross_entropy(sim_matrix.t(), labels)

    # 両方向の損失の平均を取る
    total_loss = (loss_x_y + loss_y_x) / 2
    return total_loss


if __name__ == "__main__":
    test_mse_loss()
    print("All tests passed!")
