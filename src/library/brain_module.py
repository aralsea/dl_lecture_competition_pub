import mne
import torch
import torch.nn as nn
from mne.datasets import spm_face
from mne.io import (
    read_raw_ctf,
)

S = 4  # number of subjects
C = 270  # number of channels for 2D spatial attention
C3D = 271  # number of channels for 3D spatial attention
T = 281
F = 768  # latent dim for images
F1 = 2048
K = 32  # max frequency for spatial attention
NUM_BLOCKES = 5  # number of convolutional blocks
D2 = 320  # internal channels for convolutional blocks


# input: (N, C, T) = (batch_size, num_channels, num_time_steps)
# output: (N, F) = (batch_size, latent_dim)
class BrainModule(nn.Module):
    def __init__(self, out_dim: int = F) -> None:
        super().__init__()

        self.in_channels = C
        self.backbone = BackBone(in_channels=C, out_channels=F1)
        self.affine_aggrigation = nn.Linear(T, 1)
        self.clip_head = MLP(in_dim=F1, out_dim=out_dim)
        self.mse_head = MLP(in_dim=F1, out_dim=out_dim)

        self.temperature = nn.Parameter(torch.rand(1))  # used for temperature scaling

    def forward(
        self, X: torch.Tensor, subject_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # X: (N, C, T)
        # subject_idx: (N)
        assert X.shape[0] == subject_idx.shape[0]
        if X.shape[1] != self.in_channels:
            print(X.shape)
            print(self.in_channels)
            raise ValueError("X must have the same number of channels as the model")
        assert X.shape[1] == self.in_channels
        Y = self.backbone(X, subject_idx=subject_idx)  # (N, F1, T)
        Y_agg = self.affine_aggrigation(Y)  # (N, F1, 1)
        Y_agg = torch.squeeze(Y_agg, dim=2)  # (N, F1)

        Y_clip = self.clip_head(Y_agg)
        Y_mse = self.mse_head(Y_agg)
        return Y_clip, Y_mse


class BackBone(nn.Module):
    def __init__(self, in_channels: int = C, out_channels: int = F1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_attention = SpatialAttention2D(num_channels=self.in_channels)
        self.subject_layer = SubjectLayer(num_channels=self.in_channels)
        self.temporal_conv_block_stack = TemporalConvBlockStack(
            in_channels=self.in_channels, out_channels=self.out_channels
        )

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        # X: (N, C, T)
        assert X.shape[1] == self.in_channels
        assert X.shape[0] == subject_idx.shape[0]
        Y = self.spatial_attention(X)
        Y = self.subject_layer(Y, subject_idx)
        Y = self.temporal_conv_block_stack(Y)
        assert Y.shape[1] == self.out_channels
        assert Y.shape[2] == X.shape[2]
        return Y  # type:ignore


class MLP(nn.Module):
    def __init__(self, in_dim: int = F1, out_dim: int = F) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_norm1 = nn.LayerNorm(self.in_dim)
        self.gelu1 = nn.GELU()
        self.linear1 = nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.layer_norm2 = nn.LayerNorm(self.out_dim)
        self.gelu2 = nn.GELU()
        self.linear2 = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N, F1, T)
        assert X.shape[1] == self.in_dim
        Y = self.layer_norm1(X)
        Y = self.gelu1(Y)
        Y = self.linear1(Y)
        Y = self.layer_norm2(Y)
        Y = self.gelu2(Y)
        Y = self.linear2(Y)
        assert Y.shape[1] == self.out_dim
        return Y  # type:ignore


class SpatialAttention2D(nn.Module):
    def __init__(self, num_channels: int = C, max_freq: int = K) -> None:
        super().__init__()
        self.num_channels = num_channels  # C = num_channels
        self.max_freq = max_freq  # K = max_freq

        # Fourier coefficients
        self.real = nn.Parameter(
            torch.randn(self.num_channels, max_freq, max_freq, dtype=torch.float32)
        )  # Re z_j^(k, l), (C, K, K)
        self.imag = nn.Parameter(
            torch.randn(self.num_channels, max_freq, max_freq, dtype=torch.float32)
        )  # Im z_j^(k, l), (C, K, K)

        # sensor positions
        raw = read_raw_ctf(
            spm_face.data_path() / "MEG" / "spm" / "SPM_CTF_MEG_example_faces1_3D.ds"
        )
        sensor_names = [
            ch["ch_name"]
            for ch in raw.info["chs"]
            if ch["kind"] == mne.io.constants.FIFF.FIFFV_MEG_CH
        ]  # ["MLC11-2908", "MLC12-2908", ...], len=274

        excluded_sensor_names = ["MLF25", "MRF43", "MRO13", "MRO11"]
        not_found_sensor_names = [
            "MLO42"
        ]  # なぜか上記のsensor_namesに含まれないセンサー名

        layout = mne.channels.find_layout(raw.info, ch_type="meg")
        positions = layout.pos[:, :2][
            [
                (name[:5] not in excluded_sensor_names + not_found_sensor_names)
                for name in sorted(sensor_names + not_found_sensor_names)
            ]
        ]

        # Scale down positions to keep a margin of 0.1 on each side
        scaled_positions = positions * 0.8 + 0.1
        self.register_buffer(
            "positions", torch.tensor(scaled_positions, dtype=torch.float32)
        )  # (C, 2)

        # 1*1 conv
        self.conv = nn.Conv1d(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=1,
            stride=1,
        )

        self.k = torch.arange(
            1,
            self.max_freq + 1,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )  # (K)
        self.l = torch.arange(
            1,
            self.max_freq + 1,
            device=("cuda" if torch.cuda.is_available() else "cpu"),
        )  # (K)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N, C, T)
        # output: (N, C, T)

        # weights
        x, y = self.positions[:, 0], self.positions[:, 1]

        # Compute 2π(kx + ly) for all combinations of k, l, and positions
        kx = 2 * torch.pi * self.k.view(1, -1, 1) * x.view(-1, 1, 1)  # (C, K, 1)
        ly = 2 * torch.pi * self.l.view(1, 1, -1) * y.view(-1, 1, 1)  # (C, 1, K)
        phase = kx + ly  # (C, K, K)

        # Compute cos and sin terms
        cos_term = torch.cos(phase)  # (C, K, K)
        sin_term = torch.sin(phase)  # (C, K, K)

        # Compute a_j(x_n, y_n) for all output channels and positions
        weights = torch.einsum("jkl,nkl->jn", self.real, cos_term) + torch.einsum(
            "jkl,nkl->jn", self.imag, sin_term
        )  # (C, C)

        # Apply spatial dropout
        if self.training:
            drop_idx = torch.randint(0, self.num_channels, (1,)).item()
            xdrop, ydrop = self.positions[drop_idx]
            distances = torch.sqrt(
                (x - xdrop) ** 2 + (y - ydrop) ** 2
            )  # Calculate distances
            dropout_mask = (
                distances > 0.2
            )  # Create mask for sensors outside dropout area

            weights = weights * dropout_mask.float()  # Apply mask to weights
        softmax_weights = torch.softmax(weights, dim=1)  # (C, C)

        attended = torch.einsum(
            "nct,cd->ndt", X, softmax_weights
        )  # (N, C, T) @ (C, C) = (N, C, T)

        # 1*1 conv により、C方向の値を集約
        Y = self.conv(attended)  # (N, C, T)
        assert Y.shape == X.shape
        return Y  # type:ignore


class SubjectLayer(nn.Module):
    """
    input: (N, C, T)
    output: (N, C, T)
    """

    def __init__(self, num_subjects: int = S, num_channels: int = C) -> None:
        """
        SpatialAttention2D -> num_channels = C
        SpatialAttention3D -> num_channels = C3D
        """
        super().__init__()
        self.mat = [
            nn.Linear(num_channels, num_channels, bias=False)
            for _ in range(num_subjects)
        ]

        self.num_subjects = num_subjects

    def forward(self, X: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        # X: (N, C, T)
        # subject_idx: (N)
        if X.shape[0] != subject_idx.shape[0]:
            raise ValueError("X and subject_idx must have the same batch size")

        if (subject_idx < 0).any() or (subject_idx >= self.num_subjects).any():
            raise ValueError("subject_idx must be in [0, num_subjects)")

        XT = X.transpose(1, 2)  # (N, T, C)

        # バッチ全体に対して一度に処理を行う
        YT = torch.zeros_like(XT)

        for i in range(self.num_subjects):
            mask = subject_idx == i
            if mask.any():
                YT[mask] = self.mat[i](XT[mask])
        Y = YT.transpose(1, 2)  # (N, C, T)
        assert Y.shape == X.shape
        return Y  # type:ignore


class TemporalConvBlockStack(nn.Module):
    """
    input: (N, C, T)
    output: (N, F1, T)
    """

    def __init__(self, in_channels: int = C, out_channels: int = F1) -> None:
        super().__init__()
        self.blocks = [
            ConvBlock(
                idx=idx,
                in_channels=(in_channels if idx == 0 else D2),
            )
            for idx in range(NUM_BLOCKES)
        ]  # output: (N, D2, T)
        self.linear = nn.Linear(D2, out_channels, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N, C, T)
        assert X.shape[1] == C
        for i, block in enumerate(self.blocks):
            X = block(X)
        XT = X.transpose(1, 2)  # (N, T, D2)
        XT = self.linear(XT)  # (N, T, F1)
        assert XT.shape[2] == F1
        return XT.transpose(1, 2)  # type:ignore


class ConvBlock(nn.Module):
    """
    input: (N, in_channels, T)
    output: (N, out_channels, T)
    """

    def __init__(
        self,
        idx: int,
        in_channels: int = D2,
        out_channels: int = D2,
    ) -> None:
        super().__init__()
        if idx < 0:
            raise ValueError("idx must be non-negative")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_conv1 = ResConv1D(
            self.in_channels, self.out_channels, dilation=(2**idx) % 5
        )
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.act1 = nn.GELU()
        self.res_conv2 = ResConv1D(
            self.out_channels, self.out_channels, dilation=(2 ** (idx + 1)) % 5
        )
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.act2 = nn.GELU()
        self.res_conv3 = ResConv1D(
            self.out_channels, self.out_channels * 2, dilation=1, bias=True
        )
        self.act3 = nn.GLU(dim=1)  # (N, out_channels * 2, T) -> (N, out_channels, T)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (N, in_channels, T)
        Y = self.res_conv1(X)  # (N, out_channels, T)
        Y = self.bn1(Y)
        Y = self.act1(Y)  # (N, out_channels, T)
        Y = self.res_conv2(Y)  # (N, out_channels, T)
        Y = self.bn2(Y)
        Y = self.act2(Y)  # (N, out_channels, T)
        Y = self.res_conv3(Y)  # (N, out_channels * 2, T)
        Y = self.act3(Y)  # (N, out_channels, T) <- GLU halves the number of channels
        assert Y.shape[1] == self.out_channels
        return Y  # type:ignore


class ResConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )

        self.same_in_out = self.in_channels == self.out_channels
        self.shortcut = (
            nn.Identity()
            if self.same_in_out
            else nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # T方向に畳み込みを行う
        # X: (N, C, T)
        assert X.shape[1] == self.in_channels
        Y = self.conv(X)  # (N, C, T)
        Y = Y + self.shortcut(X)  # (N, C, T)
        assert Y.shape[1] == self.out_channels
        return Y  # type:ignore
