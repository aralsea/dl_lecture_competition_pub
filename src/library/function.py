import torch


def MSE_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    assert x.dim() == 2
    return torch.mean((x - y) ** 2)


def test_mse_loss() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    loss = MSE_loss(x, y)
    assert torch.allclose(loss, torch.tensor(4.5))


if __name__ == "__main__":
    test_mse_loss()
    print("All tests passed!")
