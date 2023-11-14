import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicMotionEncoder(nn.Module):
    def __init__(
        self,
        num_corr_levels: int,
        corr_radius: int,
    ):
        super().__init__()

        num_channels = num_corr_levels * (2 * corr_radius + 1) ** 2

        self.convc1 = nn.Conv2d(num_channels, 256, kernel_size=1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, kernel_size=3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, kernel_size=7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, kernel_size=3, padding=1)

    def forward(self, flow: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        cor = F.relu_(self.convc1(corr))
        cor = F.relu_(self.convc2(cor))
        flo = F.relu_(self.convf1(flow))
        flo = F.relu_(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu_(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim: int = 128, input_dim: int = 192 + 128):
        super().__init__()

        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size=(1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size=(1, 5), padding=(0, 2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, kernel_size=(5, 1), padding=(2, 0))

    def forward(self, net: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        # horizontal
        net_inp = torch.cat([net, inp], dim=1)
        z = torch.sigmoid(self.convz1(net_inp))
        r = torch.sigmoid(self.convr1(net_inp))
        q = torch.tanh(self.convq1(torch.cat([r * net, inp], dim=1)))
        net = (1 - z) * net + z * q

        # vertical
        net_inp = torch.cat([net, inp], dim=1)
        z = torch.sigmoid(self.convz2(net_inp))
        r = torch.sigmoid(self.convr2(net_inp))
        q = torch.tanh(self.convq2(torch.cat([r * net, inp], dim=1)))
        net = (1 - z) * net + z * q

        return net


class FlowHead(nn.Module):
    def __init__(self, in_channels: int = 128, num_channels: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu_(self.conv1(x)))


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_corr_levels: int,
        corr_radius: int,
    ):
        super().__init__()

        self.encoder = BasicMotionEncoder(
            num_corr_levels=num_corr_levels,
            corr_radius=corr_radius,
        )

        self.gru = SepConvGRU(hidden_dim, input_dim=hidden_dim + input_dim)

        self.flow_head = FlowHead(in_channels=hidden_dim, num_channels=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 9 * 8 * 8, kernel_size=1),
        )

    def forward(
        self, net: torch.Tensor, inp: torch.Tensor, corr: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        flow_delta = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)

        return net, mask, flow_delta
