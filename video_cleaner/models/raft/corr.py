from functools import lru_cache

import torch
import torch.nn.functional as F


def compute_all_pairs_correlation(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
    bsz, c, h, w = fmap1.shape

    fmap1 = fmap1.flatten(2)
    fmap2 = fmap2.flatten(2)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(bsz, h, w, 1, h, w)
    return corr / c ** 0.5


@lru_cache
@torch.no_grad
def compute_delta(r: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    dx = torch.linspace(-r, r, 2 * r + 1)
    dy = torch.linspace(-r, r, 2 * r + 1)
    delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)
    return delta.to(dtype=dtype, device=device)


@torch.no_grad
def bilinear_sampler(x: torch.Tensor, coords: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """ Wrapper for grid_sample, uses pixel coordinates """
    h, w = x.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / max(w - 1, 1) - 1
    ygrid = 2 * ygrid / max(h - 1, 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    x = F.grid_sample(x, grid, mode=mode, align_corners=True)

    return x


class CorrBlock:
    def __init__(
        self,
        fmap1: torch.Tensor,
        fmap2: torch.Tensor,
        num_levels: int = 4,
        radius: int = 4,
    ):
        self.num_levels = num_levels
        self.radius = radius

        corr = compute_all_pairs_correlation(fmap1, fmap2)
        corr = corr.flatten(0, 2)

        self.corr_pyramid = [corr]
        for _ in range(num_levels - 1):
            corr = F.avg_pool2d(corr, kernel_size=2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        # bsz, 2, h, w -> bsz, h, w, 2
        coords = coords.permute(0, 2, 3, 1)
        bsz, h0, w0, _ = coords.shape
        coords = coords.view(bsz * h0 * w0, 1, 1, 2)

        delta = compute_delta(self.radius, coords.dtype, coords.device)
        delta = delta.view(1, 2 * self.radius + 1, 2 * self.radius + 1, 2)

        out_pyramid = []
        for i, corr in enumerate(self.corr_pyramid):
            centroid_lvl = coords / 2 ** i
            coords_lvl = centroid_lvl + delta

            corr = bilinear_sampler(corr, coords=coords_lvl)
            corr = corr.view(bsz, h0, w0, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2)
