import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .utils import to_pair


class P3DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        use_residual: bool = False,
        bias: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=bias,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=(3, 1, 1),
                stride=(1, 1, 1),
                padding=(2, 0, 0),
                dilation=(2, 1, 1),
                bias=bias,
            ),
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.use_residual:
            x += residual

        return x


class SecondOrderDeformableAlignment(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        max_residue_magnitude: int = 5,
        bias: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        self.stride = to_pair(stride)
        self.padding = to_pair(padding)
        self.dilation = to_pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.max_residue_magnitude = max_residue_magnitude

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(cond)
        offset, mask = torch.split(out, [18 * self.deform_groups, 9 * self.deform_groups], dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(offset)

        # mask
        mask = torch.sigmoid(mask)

        return deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )


class BidirectionalPropagation(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()

        for i, mod_name in enumerate(["backward_", "forward_"]):
            self.deform_align[mod_name] = SecondOrderDeformableAlignment(
                2 * num_channels, num_channels, kernel_size=3, padding=1, deform_groups=16
            )

            self.backbone[mod_name] = nn.Sequential(
                nn.Conv2d((2 + i) * num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            )

        self.fusion = nn.Conv2d(2 * num_channels, num_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, t, c, h, w = x.shape

        feats: dict[str, list[torch.Tensor]] = {}
        feats["spatial"] = [x[:, i, ...] for i in range(t)]

        for mod_name in ["backward_", "forward_"]:
            feats[mod_name] = []

            frame_indices = list(range(t))
            if "backward" in mod_name:
                frame_indices = frame_indices[::-1]

            mapping_indices = list(range(t))
            mapping_indices += mapping_indices[::-1]

            prop_feat = torch.zeros(bsz, c, h, w, dtype=x.dtype, device=x.device)
            for i, idx in enumerate(frame_indices):
                cur_feat = feats["spatial"][mapping_indices[idx]]
                if i > 0:
                    cond_n1 = prop_feat

                    # initialize second-order features
                    if i > 1:
                        feat_n2 = feats[mod_name][-2]
                        cond_n2 = feat_n2
                    else:
                        feat_n2 = torch.zeros_like(prop_feat)
                        cond_n2 = torch.zeros_like(cond_n1)

                    cond = torch.cat([cond_n1, cur_feat, cond_n2], dim=1)
                    prop_feat = torch.cat([prop_feat, feat_n2], dim=1)
                    prop_feat = self.deform_align[mod_name](prop_feat, cond)

                # fuse current features
                feat = (
                    [cur_feat] +
                    [feats[k][idx] for k in feats if k not in ("spatial", mod_name)] +
                    [prop_feat]
                )
                feat = torch.cat(feat, dim=1)
                # embed current features
                prop_feat = prop_feat + self.backbone[mod_name](feat)

                feats[mod_name].append(prop_feat)

            if "backward" in mod_name:
                feats[mod_name] = feats[mod_name][::-1]

        outputs = []
        for i in range(t):
            align_feats = [feats[k].pop(0) for k in feats if k != "spatial"]
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))

        return torch.stack(outputs, dim=1) + x


class Deconv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=True,
        )
        return self.conv(x)


class RecurrentFlowCompleteNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), padding_mode="replicate"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder1 = nn.Sequential(
            P3DBlock(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            P3DBlock(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ) # 4x

        self.encoder2 = nn.Sequential(
            P3DBlock(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            P3DBlock(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ) # 8x

        self.mid_dilation = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 3, 3), dilation=(1, 3, 3)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 2, 2), dilation=(1, 2, 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.feat_prop_module = BidirectionalPropagation(128)

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Deconv(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ) # 4x

        self.decoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Deconv(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ) # 2x

        self.upsample = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Deconv(32, 2, kernel_size=3, padding=1),
        )

    def forward(self, masked_flows: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        bsz, t, _, h, w = masked_flows.shape
        # b, t, 2, h, w -> b, 2, t, h, w
        masked_flows = masked_flows.permute(0, 2, 1, 3, 4)
        # b, t, 1, h, w -> b, 1, t, h, w
        masks = masks.permute(0, 2, 1, 3, 4)

        x = torch.cat((masked_flows, masks), dim=1)
        x = self.downsample(x)

        feat_e1: torch.Tensor = self.encoder1(x)
        feat_e2: torch.Tensor = self.encoder2(feat_e1)

        feat_mid: torch.Tensor = self.mid_dilation(feat_e2)
        # b, c, t, h, w -> b, t, c, h, w
        feat_mid = feat_mid.permute(0, 2, 1, 3, 4)

        feat_prop: torch.Tensor = self.feat_prop_module(feat_mid)
        feat_prop = feat_prop.flatten(0, 1)

        # b, c, t, h, w -> b * t, c, h, w
        feat_e1 = feat_e1.permute(0, 2, 1, 3, 4).flatten(0, 1)
        feat_d2: torch.Tensor = self.decoder2(feat_prop) + feat_e1

        feat_d1: torch.Tensor = self.decoder1(feat_d2)

        flow: torch.Tensor = self.upsample(feat_d1)

        return flow.view(bsz, t, 2, h, w)
