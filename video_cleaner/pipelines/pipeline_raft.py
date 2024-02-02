import os

import torch
from safetensors import safe_open

from video_cleaner.models.raft import RAFT
from video_cleaner.models.raft.corr import CorrBlock


class RAFTPipeline:
    def __init__(
        self,
        raft: RAFT,
    ):
        self.raft = raft

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | os.PathLike,
        device: torch.device = torch.device("cuda"),
    ) -> "RAFTPipeline":
        state_dict = {}
        with safe_open(model_name_or_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        raft = RAFT()
        raft.load_state_dict(state_dict)
        raft = raft.eval()
        raft = raft.to(device)

        return cls(raft=raft)

    @torch.no_grad()
    def __call__(
        self,
        frames: torch.Tensor,
        clip_length: int = 32,
        update_iters: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, video_length = frames.shape[:2]

        fmaps: torch.Tensor = self.raft.fnet(frames.flatten(0, 1))
        cmaps: torch.Tensor = self.raft.cnet(frames.flatten(0, 1))

        fmaps = fmaps.view(bsz, video_length, *fmaps.shape[1:])
        cmaps = cmaps.view(bsz, video_length, *cmaps.shape[1:])
        net, inp = torch.split(cmaps, [self.raft.hidden_dim, self.raft.context_dim], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        flows_fwd_list, flows_bwd_list = [], []
        for i in range(0, video_length, clip_length):
            start = max(i - 1, 0)
            end = min(i + clip_length, video_length)

            indices1 = list(range(start, end - 1))
            indices2 = list(range(start + 1, end))
            real_clip_length = len(indices1)

            fmap1 = fmaps[:, indices1].flatten(0, 1)
            fmap2 = fmaps[:, indices2].flatten(0, 1)

            corr_fn = CorrBlock(
                fmap1, fmap2,
                num_levels=self.raft.num_corr_levels,
                radius=self.raft.corr_radius,
                use_transpose=True,
            )

            net12 = net[:, indices1 + indices2].flatten(0, 1)
            inp12 = inp[:, indices1 + indices2].flatten(0, 1)

            coords0, coords1 = self.raft.init_flow(frames.flatten(0, 1), batch_size=real_clip_length * 2)

            for i in range(update_iters):
                coords1 = coords1.detach()
                coords1 = coords1.view(bsz, real_clip_length * 2, *coords1.shape[1:])
                coords11, coords21 = coords1.chunk(2, dim=1)
                coords1 = coords1.flatten(0, 1)
                corr1, corr2 = corr_fn(coords11.flatten(0, 1)), corr_fn(coords21.flatten(0, 1), transpose=True)
                corr1 = corr1.view(bsz, real_clip_length, *corr1.shape[1:])
                corr2 = corr2.view(bsz, real_clip_length, *corr2.shape[1:])
                corr12 = torch.cat([corr1, corr2], dim=1)
                corr12 = corr12.flatten(0, 1)

                flow = coords1 - coords0
                net12, flow_delta = self.raft.update_block(net12, inp12, corr12, flow)

                coords1: torch.Tensor = coords1 + flow_delta

            # scale mask to balence gradients
            up_mask: torch.Tensor = .25 * self.raft.update_block.mask(net12)
            flow_up = self.raft.upsample_flow_by_mask(coords1 - coords0, up_mask)

            flows_fwd, flows_bwd = flow_up.chunk(2, dim=0)
            flows_fwd = flows_fwd.view(bsz, real_clip_length, *flows_fwd.shape[1:])
            flows_bwd = flows_bwd.view(bsz, real_clip_length, *flows_bwd.shape[1:])

            flows_fwd_list.append(flows_fwd)
            flows_bwd_list.append(flows_bwd)

        flows_fwd = torch.cat(flows_fwd_list, dim=1)
        flows_bwd = torch.cat(flows_bwd_list, dim=1)

        return flows_fwd, flows_bwd
