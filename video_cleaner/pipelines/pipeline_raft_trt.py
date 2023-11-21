from __future__ import annotations

import os
from functools import lru_cache

import torch
import torch.nn.functional as F

from video_cleaner.utils.trt_engine import TRTEngine

ENGINES = {
    "encoder": "encoder_320x576_256.engine",
    "update_block": "update_block_320x576_32.engine",
    "flow_upsampler": "flow_upsampler_320x576_32.engine",
}


def compute_all_pairs_correlation(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
    bsz, c, h, w = fmap1.shape

    fmap1 = fmap1.flatten(2)
    fmap2 = fmap2.flatten(2)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(bsz, h, w, 1, h, w)
    return corr / c ** 0.5


@lru_cache
def compute_grid_coords(height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).to(device=device, dtype=dtype)
    return coords[None]


def init_flow(frames: torch.Tensor, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, _, h, w = frames.shape

    if batch_size is None:
        batch_size = bsz

    coords0: torch.Tensor = compute_grid_coords(h // 8, w // 8, frames.dtype, frames.device)
    coords0 = coords0.expand(batch_size, -1, -1, -1)
    coords1 = coords0.clone()

    return coords0, coords1


class RAFTTRTPipeline:
    def __init__(
        self,
        engines: dict[str, TRTEngine],
        stream: torch.cuda.Stream,
    ):
        self.engines = engines
        self.stream = stream

    @classmethod
    def from_engine(
        cls,
        engine_dir: str | os.PathLike,
        device: torch.device = torch.device("cuda"),
    ) -> "RAFTTRTPipeline":
        engines: dict[str, TRTEngine] = {}
        max_device_memory = 0

        for name, path in ENGINES.items():
            engine = TRTEngine(engine_path=os.path.join(engine_dir, path))
            engine.load()
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
            engines[name] = engine

        shared_device_memory = torch.empty(max_device_memory, dtype=torch.uint8, device=device)
        for engine in engines.values():
            engine.activate(shared_device_memory_ptr=shared_device_memory)
            engine.allocate_buffers(device=device)

        stream = torch.cuda.Stream(device)

        return cls(engines=engines, stream=stream)

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor, update_iters: int = 8):
        assert frames.is_cuda

        video_length = frames.shape[0]
        clip_length = 32

        outputs = self.engines["encoder"].infer(
            feed_dict={"frames": frames},
            stream=self.stream,
            use_cuda_graph=False,
        )

        fmaps, cmaps = outputs["fmaps"], outputs["cmaps"]
        net, inp = torch.chunk(cmaps, 2, dim=1)
        net = torch.tanh(net)
        inp = torch.relu_(inp)

        flows_fwd_list, flows_bwd_list = [], []
        for i in range(0, video_length, clip_length):
            start = max(i - 1, 0)
            end = min(i + clip_length, video_length)

            indices1 = list(range(start, end - 1))
            indices2 = list(range(start + 1, end))

            if len(indices1) < clip_length:
                padding = clip_length - len(indices1)
                indices1 += [0] * padding
                indices2 += [0] * padding
            else:
                padding = 0

            fmap1 = fmaps[indices1]
            fmap2 = fmaps[indices2]

            corr = compute_all_pairs_correlation(fmap1, fmap2)
            corr_t = corr.permute(0, 4, 5, 3, 1, 2)

            corr = corr.flatten(0, 2)
            corr_pyramid = [corr]
            for _ in range(4 - 1):
                corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
                corr_pyramid.append(corr)

            corr_t = corr_t.flatten(0, 2)
            corr_t_pyramid = [corr_t]
            for _ in range(4 - 1):
                corr_t = F.avg_pool2d(corr_t, kernel_size=2, stride=2)
                corr_t_pyramid.append(corr_t)

            net_i = net[indices1 + indices2]
            inp_i = inp[indices1 + indices2]

            coords0, coords1 = init_flow(frames, batch_size=net_i.shape[0])

            for i in range(update_iters):
                outputs = self.engines["update_block"].infer(
                    feed_dict=dict(
                        coords0=coords0,
                        coords1=coords1,
                        net=net_i,
                        inp=inp_i,
                        corr_pyramid_0=corr_pyramid[0],
                        corr_pyramid_1=corr_pyramid[1],
                        corr_pyramid_2=corr_pyramid[2],
                        corr_pyramid_3=corr_pyramid[3],
                        corr_t_pyramid_0=corr_t_pyramid[0],
                        corr_t_pyramid_1=corr_t_pyramid[1],
                        corr_t_pyramid_2=corr_t_pyramid[2],
                        corr_t_pyramid_3=corr_t_pyramid[3],
                    ),
                    stream=self.stream,
                    use_cuda_graph=False,
                )
                coords1 = outputs["updated_coords1"]
                net_i = outputs["updated_net"]

            flow = self.engines["flow_upsampler"].infer(
                feed_dict=dict(
                    coords0=coords0,
                    coords1=coords1,
                    net=net_i,
                ),
                stream=self.stream,
                use_cuda_graph=False,
            )["upsampled_flow"]

            flows_fwd, flows_bwd = flow.chunk(2, dim=0)
            if padding > 0:
                flows_fwd = flows_fwd[:-padding]
                flows_bwd = flows_bwd[:-padding]
            flows_fwd_list.append(flows_fwd)
            flows_bwd_list.append(flows_bwd)

        flows_fwd = torch.cat(flows_fwd_list, dim=0)
        flows_bwd = torch.cat(flows_bwd_list, dim=0)

        return flows_fwd, flows_bwd
