from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlock
from .encoder import BasicEncoder
from .update_block import BasicUpdateBlock


@lru_cache
@torch.no_grad
def compute_grid_coords(height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).to(device=device, dtype=dtype)
    return coords[None]


def upflow8(flow: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


class RAFT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        context_dim: int = 128,
        num_corr_levels: int = 4,
        corr_radius: int = 4,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_corr_levels = num_corr_levels
        self.corr_radius = corr_radius

        self.fnet = BasicEncoder(
            in_channels=3,
            out_channels=256,
            norm_type="instance_norm",
            dropout_prob=0.0,
        )

        self.cnet = BasicEncoder(
            in_channels=3,
            out_channels=hidden_dim + context_dim,
            norm_type="batch_norm",
            dropout_prob=0.0,
        )

        self.update_block = BasicUpdateBlock(
            hidden_dim=hidden_dim,
            input_dim=context_dim,
            num_corr_levels=num_corr_levels,
            corr_radius=corr_radius,
        )

    def init_flow(self, frame1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, h, w = frame1.shape

        coords0: torch.Tensor = compute_grid_coords(h // 8, w // 8, frame1.dtype, frame1.device)
        coords0 = coords0.expand(bsz, -1, -1, -1)
        coords1 = coords0.clone()

        return coords0, coords1

    def upsample_flow_by_mask(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        bsz, _, h, w = flow.shape

        mask = mask.view(bsz, 1, 9, 8, 8, h, w)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(bsz, 2, 9, 1, 1, h, w)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(bsz, 2, 8 * h, 8 * w)

    def forward(
        self, frame1: torch.Tensor, frame2: torch.Tensor, update_iters: int = 12
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fmap1 = self.fnet(frame1)
        fmap2 = self.fnet(frame2)

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        net_inp = self.cnet(frame1)
        net, inp = torch.split(net_inp, [self.hidden_dim, self.context_dim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.init_flow(frame1)

        for _ in range(update_iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)

            flow = coords1 - coords0
            net, up_mask, flow_delta = self.update_block(net, inp, corr, flow)

            coords1 = coords1 + flow_delta

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow_by_mask(coords1 - coords0, up_mask)

        return flow_up
