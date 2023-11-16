from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .recurrent_flow_completion import Deconv
from .sparse_transformer import TemporalSparseTransformer
from .utils import to_pair


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_groups = [1, 2, 4, 8, 1]

        self.layers = nn.ModuleList([
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=self.num_groups[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=self.num_groups[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=self.num_groups[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=self.num_groups[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=self.num_groups[4]),
            nn.LeakyReLU(0.2, inplace=True),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]

        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
                _, _, h, w = x0.shape

            if i > 8 and i % 2 == 0:
                g = self.num_groups[(i - 8) // 2]
                x = x0.view(bsz, g, -1, h, w)
                o = out.view(bsz, g, -1, h, w)
                out = torch.cat([x, o], dim=2).view(bsz, -1, h, w)

            out = layer(out)

        return out


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
        max_residue_magnitude: int = 3,
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
            nn.Conv2d(2 * out_channels + 2 + 1 + 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        out = self.conv_offset(cond)
        offset, mask = torch.split(out, [18 * self.deform_groups, 9 * self.deform_groups], dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(offset)
        offset += flow.flip(1).repeat(1, offset.shape[1] // 2, 1, 1)

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

@lru_cache
@torch.no_grad
def compute_grid_coords(height: int, width: int, dtype: torch.dtype, device: torch.device):
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    grid_coords = torch.stack((grid_x, grid_y), dim=2)
    return grid_coords.to(dtype=dtype, device=device)


def flow_warp(
    x: torch.Tensor,
    flows: torch.Tensor,
    interpolation: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True
) -> torch.Tensor:
    _, _, h, w = x.shape
    dtype = x.dtype
    device = x.device

    grid_coords = compute_grid_coords(h, w, dtype, device)

    grid_flows = flows + grid_coords
    grid_flows_x = 2.0 * grid_flows[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flows_y = 2.0 * grid_flows[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flows = torch.stack((grid_flows_x, grid_flows_y), dim=3)

    return F.grid_sample(
        x,
        grid=grid_flows,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def length_sq(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(x), dim=1, keepdim=True)


def forward_backward_consistency_check(
    flow_fw: torch.Tensor,
    flow_bw: torch.Tensor,
    alpha1: float = 0.01,
    alpha2: float = 0.5,
    interpolation: str = "bilinear",
) -> torch.Tensor:
    flow_bw_warped = flow_warp(flow_bw, flow_fw.permute(0, 2, 3, 1), interpolation=interpolation)
    flow_diff_fw = flow_fw + flow_bw_warped

    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    occ_thresh_fw = alpha1 * mag_sq_fw + alpha2

    return length_sq(flow_diff_fw) < occ_thresh_fw


def to_binary_mask(mask: torch.Tensor, threshold: float = 0.1):
    mask[mask > threshold] = 1.0
    mask[mask <= threshold] = 0.0
    return mask.to(torch.bool)


class BidirectionalPropagation(nn.Module):
    def __init__(
        self,
        num_channels: int,
        learnable: bool = True,
        interpolation: str = "bilinear",
    ):
        super().__init__()

        self.learnable = learnable
        self.interpolation = interpolation

        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()

        if learnable:
            for mod_name in ["backward_1", "forward_1"]:
                self.deform_align[mod_name] = SecondOrderDeformableAlignment(
                    num_channels, num_channels, kernel_size=3, padding=1, deform_groups=16
                )

                self.backbone[mod_name] = nn.Sequential(
                    nn.Conv2d(2 * num_channels + 2, num_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                )

            self.fusion = nn.Sequential(
                nn.Conv2d(2 * num_channels + 2, num_channels, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
            )

    def forward(
        self,
        x: torch.Tensor,
        flows_forward: torch.Tensor,
        flows_backward: torch.Tensor,
        input_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        bsz, t = x.shape[:2]

        feats: dict[str, list[torch.Tensor]] = {}
        feats["spatial"] = [x[:, i, ...] for i in range(t)]
        feats["input"] = [x[:, i, ...] for i in range(t)]

        masks: dict[str, list[torch.Tensor]] = {}
        masks["input"] = [input_masks[:, i, ...] for i in range(t)]

        prop_list = ["backward_1", "forward_1"]
        cache_list = ["input"] +  prop_list

        for p_i, mod_name in enumerate(prop_list):
            feats[mod_name] = []
            masks[mod_name] = []

            frame_indices = list(range(t))
            if "backward" in mod_name:
                frame_indices = frame_indices[::-1]
                flow_indices = frame_indices
                flows_for_prop = flows_forward
                flows_for_check = flows_backward
            else:
                flow_indices = list(range(-1, t - 1))
                flows_for_prop = flows_backward
                flows_for_check = flows_forward

            for i, idx in enumerate(frame_indices):
                feat_current = feats[cache_list[p_i]][idx]
                mask_current = masks[cache_list[p_i]][idx]

                if i == 0:
                    feat_prop = feat_current
                    mask_prop = mask_current
                else:
                    flow_prop = flows_for_prop[:, flow_indices[i], :, :, :]
                    flow_check = flows_for_check[:, flow_indices[i], :, :, :]
                    flow_vaild_mask = forward_backward_consistency_check(flow_prop, flow_check)
                    flow_prop_t = flow_prop.permute(0, 2, 3, 1)
                    feat_warped = flow_warp(
                        feat_prop,
                        flows=flow_prop_t,
                        interpolation=self.interpolation,
                    )

                    if self.learnable:
                        cond = torch.cat([feat_current, feat_warped, flow_prop, flow_vaild_mask, mask_current], dim=1)
                        feat_prop = self.deform_align[mod_name](feat_prop, cond, flow_prop)
                        mask_prop = mask_current
                    else:
                        mask_prop_valid = flow_warp(
                            mask_prop.to(flow_prop_t.dtype), flow_prop_t, interpolation=self.interpolation
                        )
                        mask_prop_valid = to_binary_mask(mask_prop_valid)

                        union_vaild_mask = mask_current * flow_vaild_mask * (~mask_prop_valid)

                        feat_prop = union_vaild_mask * feat_warped + (~union_vaild_mask) * feat_current
                        # update mask
                        mask_prop = mask_current * (~(flow_vaild_mask * (~mask_prop_valid)))

                if self.learnable:
                    feat = torch.cat([feat_current, feat_prop, mask_current], dim=1)
                    feat_prop = feat_prop + self.backbone[mod_name](feat)

                feats[mod_name].append(feat_prop)
                masks[mod_name].append(mask_prop)

            if "backward" in mod_name:
                feats[mod_name] = feats[mod_name][::-1]
                masks[mod_name] = masks[mod_name][::-1]

        outputs_b = torch.stack(feats["backward_1"], dim=1)
        outputs_f = torch.stack(feats["forward_1"], dim=1)

        if self.learnable:
            outputs = torch.cat([outputs_b, outputs_f, input_masks], dim=2)
            outputs: torch.Tensor = self.fusion(outputs.flatten(0, 1))
            outputs = outputs.view(bsz, t, *outputs.shape[1:]) + x
            updated_masks = None
        else:
            outputs = outputs_f
            updated_masks = torch.stack(masks["forward_1"], dim=1)

        return outputs_b, outputs_f, outputs, updated_masks


class SoftSplit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.kernel_size = to_pair(kernel_size)
        self.stride = to_pair(stride)
        self.padding = to_pair(padding)

        self.t2t = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        self.embedding = nn.Linear(
            self.kernel_size[0] * self.kernel_size[1] * in_channels, out_channels
        )

    def forward(self, x: torch.Tensor):
        bsz, t, _, h, w = x.shape

        x = x.flatten(0, 1)
        x = self.t2t(x)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)

        out_h = int(
            (h + 2 * self.padding[0] - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        out_w = int(
            (w + 2 * self.padding[1] - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )
        x = x.view(bsz, t, out_h, out_w, x.shape[-1])

        return x


class SoftComp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_pair(kernel_size)
        self.stride = to_pair(stride)
        self.padding = to_pair(padding)

        self.embedding = nn.Linear(
            out_channels, self.kernel_size[0] * self.kernel_size[1] * in_channels
        )

        self.bias_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]):
        bsz, t, _, _, c = x.shape
        x = x.view(bsz, -1, c)
        x = self.embedding(x)
        x = x.view(bsz * t, -1, x.shape[-1]).permute(0, 2, 1)
        x = F.fold(
            x,
            output_size=output_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        x = self.bias_conv(x)
        x = x.view(bsz, t, self.in_channels, *output_size)

        return x


class InpaintGenerator(nn.Module):
    def __init__(
        self,
        interpolation: str = "bilinear",
        temporal_dilation: int = 2,
    ):
        super().__init__()

        self.interpolation = interpolation
        self.temporal_dilation = temporal_dilation

        self.encoder = Encoder()

        self.decoder = nn.Sequential(
            Deconv(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            Deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

        self.img_prop_module = BidirectionalPropagation(3, learnable=False, interpolation=interpolation)
        self.feat_prop_module = BidirectionalPropagation(128, learnable=True, interpolation=interpolation)

        self.ss = SoftSplit(128, 512, kernel_size=7, stride=3, padding=3)

        self.transformers = TemporalSparseTransformer(
            d_model=512,
            nhead=4,
            window_size=(5, 9),
            pool_size=(4, 4),
            t2t_kernel_size=(7, 7),
            t2t_stride=(3, 3),
            t2t_padding=(3, 3),
            num_layers=8,
        )

        self.sc = SoftComp(128, 512, kernel_size=7, stride=3, padding=3)

    def forward(
        self,
        masked_frames: torch.Tensor,
        completed_flows: tuple[torch.Tensor, torch.Tensor],
        input_masks: torch.Tensor,
        updated_masks: torch.Tensor,
        num_local_frames: int,
    ) -> torch.Tensor:
        bsz, t, _, h0, w0 = masked_frames.shape

        feats = torch.cat([
            masked_frames.flatten(0, 1),
            input_masks.flatten(0, 1),
            updated_masks.flatten(0, 1),
        ], dim = 1)

        feats: torch.Tensor = self.encoder(feats)
        feats = feats.view(bsz, t, *feats.shape[1:])
        h, w = feats.shape[-2:]

        local_feats = feats[:, :num_local_frames, ...]
        ref_feats = feats[:, num_local_frames:, ...]

        ds_flows_f: torch.Tensor = F.interpolate(
            completed_flows[0].flatten(0, 1),
            scale_factor=1 / 4,
            mode="bilinear",
            align_corners=False,
        )
        ds_flows_f = ds_flows_f.view(bsz, num_local_frames - 1, 2, h, w) / 4.0

        ds_flows_b: torch.Tensor = F.interpolate(
            completed_flows[1].flatten(0, 1),
            scale_factor=1 / 4,
            mode="bilinear",
            align_corners=False,
        )
        ds_flows_b = ds_flows_b.view(bsz, num_local_frames - 1, 2, h, w) / 4.0

        ds_input_masks: torch.Tensor = F.interpolate(
            input_masks.flatten(0, 1).float(),
            scale_factor=1 / 4,
            mode="nearest",
        ).bool()
        ds_input_masks = ds_input_masks.view(bsz, t, 1, h, w)
        local_ds_input_masks = ds_input_masks[:, :num_local_frames]

        lcoal_ds_updated_masks: torch.Tensor = F.interpolate(
            updated_masks[:, :num_local_frames].flatten(0, 1).float(),
            scale_factor=1 / 4,
            mode="nearest",
        ).bool()
        lcoal_ds_updated_masks = lcoal_ds_updated_masks.view(bsz, num_local_frames, 1, h, w)

        _, _, local_feats, _ = self.feat_prop_module(
            local_feats,
            flows_forward=ds_flows_f,
            flows_backward=ds_flows_b,
            input_masks=torch.cat([local_ds_input_masks, lcoal_ds_updated_masks], dim=2),
        )
        feats = torch.cat([local_feats, ref_feats], dim=1)

        mask_pool_l = F.max_pool2d(local_ds_input_masks.flatten(0, 1).to(feats.dtype), kernel_size=7, stride=3, padding=3)
        mask_pool_l = mask_pool_l.view(bsz, num_local_frames, 1, *mask_pool_l.shape[-2:])
        mask_pool_l = mask_pool_l.permute(0, 1, 3, 4, 2)

        residual = feats

        output_size = feats.shape[-2:]

        # TODO: ss, trans, sc
        hidden_states = self.ss(feats)
        # b, t, h, w, c
        hidden_states = self.transformers(
            hidden_states,
            fold_size=output_size,
            masks=mask_pool_l,
            temporal_dilation=self.temporal_dilation,
        )
        feats = self.sc(hidden_states, output_size=output_size)

        feats += residual

        outputs = self.decoder(feats[:, :num_local_frames].flatten(0, 1))
        outputs = torch.tanh(outputs).view(bsz, num_local_frames, 3, h0, w0)

        return outputs

    def image_propagation(
        self,
        masked_frames: torch.Tensor,
        flows_fwd: torch.Tensor,
        flows_bwd: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, prop_frames, updated_masks = self.img_prop_module(
            masked_frames,
            flows_forward=flows_fwd,
            flows_backward=flows_bwd,
            input_masks=masks,
        )
        return prop_frames, updated_masks
