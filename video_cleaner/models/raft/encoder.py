import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, norm_type: str = "batch_norm"):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if norm_type == "group_norm":
            num_groups = out_channels // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

            if stride > 1:
                norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        elif norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)

            if stride > 1:
                norm3 = nn.BatchNorm2d(out_channels)
        elif norm_type == "instance_norm":
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)

            if stride > 1:
                norm3 = nn.InstanceNorm2d(out_channels)
        elif norm_type == "identity":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

            if stride > 1:
                norm3 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm3,
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = x

        x = F.relu_(self.norm1(self.conv1(x)))
        x = F.relu_(self.norm2(self.conv2(x)))

        return F.relu_(x + self.downsample(residual))


class BasicEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_channels: tuple[int, int, int, int] = (64, 64, 96, 128),
        block_strides: tuple[int, int, int] = (1, 2, 2),
        norm_type: str = "batch_norm",
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        if norm_type == "group_norm":
            num_groups = block_channels[0] // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=block_channels[0])
        elif norm_type == "batch_norm":
            self.norm1 = nn.BatchNorm2d(block_channels[0])
        elif norm_type == "instance_norm":
            self.norm1 = nn.InstanceNorm2d(block_channels[0])
        elif norm_type == "identity":
            self.norm1 = nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        assert len(block_channels) == 4, block_channels
        assert len(block_strides) == 3, block_strides

        self.layer1 = self._make_stage(
            in_channels=block_channels[0],
            out_channels=block_channels[1],
            stride=block_strides[0],
            norm_type=norm_type,
        )
        self.layer2 = self._make_stage(
            in_channels=block_channels[1],
            out_channels=block_channels[2],
            stride=block_strides[1],
            norm_type=norm_type,
        )
        self.layer3 = self._make_stage(
            in_channels=block_channels[2],
            out_channels=block_channels[3],
            stride=block_strides[2],
            norm_type=norm_type,
        )

        # output convolution
        self.conv2 = nn.Conv2d(block_channels[-1], out_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_type: str,
    ) -> nn.Module:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride, norm_type=norm_type),
            ResidualBlock(out_channels, out_channels, stride=1, norm_type=norm_type),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu_(self.norm1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)
        x = self.dropout(x)

        return x
