import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import to_pair


class FusionFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        t2t_kernel_size: tuple[int, int],
        t2t_stride: tuple[int, int],
        t2t_padding: tuple[int, int],
    ):
        super().__init__()

        self.t2t_kernel_size = t2t_kernel_size
        self.t2t_stride = t2t_stride
        self.t2t_padding = t2t_padding
        self.kernel_shape = t2t_kernel_size[0] * t2t_kernel_size[1]

        self.fc1 = nn.Linear(d_model, 1960)
        self.fc2 = nn.Linear(1960, d_model)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        n_vecs = 1
        for size, kernel_size, stride, padding in zip(
            output_size, self.t2t_kernel_size, self.t2t_stride, self.t2t_padding
        ):
            n_vecs *= int(
                (size + 2 * padding - (kernel_size - 1) - 1) / stride + 1
            )

        x = self.fc1(x)

        bsz, n, c = x.shape
        normalizer = torch.ones(bsz * n // n_vecs, self.kernel_shape, n_vecs)
        normalizer = F.fold(
            normalizer,
            output_size=output_size,
            kernel_size=self.t2t_kernel_size,
            stride=self.t2t_stride,
            padding=self.t2t_padding,
        )

        x = F.fold(
            x.view(-1, n_vecs, c).permute(0, 2, 1),
            output_size=output_size,
            kernel_size=self.t2t_kernel_size,
            stride=self.t2t_stride,
            padding=self.t2t_padding,
        )

        x = F.unfold(
            x / normalizer,
            kernel_size=self.t2t_kernel_size,
            stride=self.t2t_stride,
            padding=self.t2t_padding,
        )
        x = x.permute(0, 2, 1).reshape(bsz, n, c)

        x = F.gelu(x)
        x = self.fc2(x)

        return x


def window_partition(x: torch.Tensor, window_size: tuple[int, int], nhead: int) -> torch.Tensor:
    bsz, t, h, w, c = x.shape
    x = x.view(bsz, t, h // window_size[0], window_size[0], w // window_size[1], window_size[1], nhead, c // nhead)
    # b, t, h_w, ws, w_w, ws, nh, c -> b, h_w, w_w, nh, t, ws, ws, c
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    # -> b, h_w * w_w, nh, t, ws, ws, c
    x = x.flatten(1, 2)
    # -> b, h_w * w_w, nh, t, ws * ws, c
    x = x.flatten(4, 5)
    return x


class SparseWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: tuple[int, int],
        pool_size: tuple[int, int],
    ):
        super().__init__()

        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.pool_size = pool_size

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.proj = nn.Linear(d_model, d_model)

        self.pool_layer = nn.Conv2d(
            d_model, d_model,
            kernel_size=pool_size[0],
            stride=pool_size[1],
            padding=0,
            groups=d_model,
        )

        self.expand_size = tuple((i + 1) // 2 for i in window_size)

        if any(i > 0 for i in self.expand_size):
            # get mask for rolled k and rolled v
            mask_tl = torch.ones(self.window_size[0], self.window_size[1])
            mask_tl[:-self.expand_size[0], :-self.expand_size[1]] = 0
            mask_tr = torch.ones(self.window_size[0], self.window_size[1])
            mask_tr[:-self.expand_size[0], self.expand_size[1]:] = 0
            mask_bl = torch.ones(self.window_size[0], self.window_size[1])
            mask_bl[self.expand_size[0]:, :-self.expand_size[1]] = 0
            mask_br = torch.ones(self.window_size[0], self.window_size[1])
            mask_br[self.expand_size[0]:, self.expand_size[1]:] = 0
            mask = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), dim=0).flatten(0)

            valid_ind_rolled = mask.nonzero(as_tuple=False).flatten(0)
            self.register_buffer("valid_ind_rolled", valid_ind_rolled)
            self.valid_ind_rolled: torch.Tensor

    def forward(self, x: torch.Tensor, masks: torch.Tensor, temporal_indices: torch.Tensor | None = None) -> torch.Tensor:
        bsz, t, h, w, c = x.shape
        w_h, w_w = self.window_size[0], self.window_size[1]
        head_dim = c // self.nhead
        n_wh = math.ceil(h / w_h)
        n_ww = math.ceil(w / w_w)
        padded_h = n_wh * w_h
        padded_w = n_ww * w_w
        pad_b = padded_h - h
        pad_r = padded_w - w

        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)
            masks = F.pad(masks, (0, 0, 0, pad_r, 0, pad_b, 0, 0), mode="constant", value=0)

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        win_q = window_partition(q, self.window_size, self.nhead)
        win_k = window_partition(k, self.window_size, self.nhead)
        win_v = window_partition(v, self.window_size, self.nhead)

        # roll_k and roll_v
        if any(i > 0 for i in self.expand_size):
            k_tl, v_tl = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (k, v))
            k_tr, v_tr = map(lambda a: torch.roll(a, shifts=(-self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (k, v))
            k_bl, v_bl = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], -self.expand_size[1]), dims=(2, 3)), (k, v))
            k_br, v_br = map(lambda a: torch.roll(a, shifts=(self.expand_size[0], self.expand_size[1]), dims=(2, 3)), (k, v))

            k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows = map(
                lambda a: window_partition(a, self.window_size, self.nhead),
                (k_tl, k_tr, k_bl, k_br)
            )
            v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows = map(
                lambda a: window_partition(a, self.window_size, self.nhead),
                (v_tl, v_tr, v_bl, v_br)
            )

            roll_k = torch.cat((k_tl_windows, k_tr_windows, k_bl_windows, k_br_windows), dim=4)
            roll_v = torch.cat((v_tl_windows, v_tr_windows, v_bl_windows, v_br_windows), dim=4)

            # mask out tokens in current window
            roll_k = roll_k[:, :, :, :, self.valid_ind_rolled]
            roll_v = roll_v[:, :, :, :, self.valid_ind_rolled]
            roll_n = roll_k.shape[4]
            roll_k = roll_k.view(bsz, n_wh * n_ww, self.nhead, t, roll_n, c // self.nhead)
            roll_v = roll_v.view(bsz, n_wh * n_ww, self.nhead, t, roll_n, c // self.nhead)
            win_k = torch.cat((win_k, roll_k), dim=4)
            win_v = torch.cat((win_v, roll_v), dim=4)

        # pool_k and pool_v
        pool_x: torch.Tensor = self.pool_layer(x.view(bsz * t, padded_h, padded_w, c).permute(0, 3, 1, 2))
        pool_h, pool_w = pool_x.shape[-2:]
        pool_x = pool_x.permute(0, 2, 3, 1).view(bsz, t, pool_h, pool_w, c)
        # pool_k
        pool_k: torch.Tensor = self.key(pool_x)
        pool_k = pool_k[:, None].repeat(1, n_wh * n_ww, 1, 1, 1, 1)
        pool_k = pool_k.view(bsz, n_wh * n_ww, t, pool_h, pool_w, self.nhead, head_dim).permute(0, 1, 5, 2, 3, 4, 6)
        pool_k = pool_k.reshape(bsz, n_wh * n_ww, self.nhead, t, pool_h * pool_w, head_dim)
        win_k = torch.cat((win_k, pool_k), dim=4)
        # pool_v
        pool_v: torch.Tensor = self.value(pool_x)
        pool_v = pool_v[:, None].repeat(1, n_wh * n_ww, 1, 1, 1, 1)
        pool_v = pool_v.view(bsz, n_wh * n_ww, t, pool_h, pool_w, self.nhead, head_dim).permute(0, 1, 5, 2, 3, 4, 6)
        pool_v = pool_v.reshape(bsz, n_wh * n_ww, self.nhead, t, pool_h * pool_w, head_dim)
        win_v = torch.cat((win_v, pool_v), dim=4)

        mask_t = masks.shape[1]
        masks = F.max_pool2d(
            masks.flatten(0, 1).squeeze(-1),
            kernel_size=self.window_size,
            stride=self.window_size,
            padding=0,
        )
        masks = masks.view(bsz, mask_t, n_wh * n_ww)
        masks = torch.sum(masks, dim=1)

        out = torch.zeros_like(win_q)

        for i in range(win_q.shape[0]):
            mask_ind_i = masks[i].nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            mask_n = len(mask_ind_i)
            if mask_n > 0:
                win_q_t = win_q[i, mask_ind_i].view(mask_n, self.nhead, t * w_h * w_w, head_dim)
                win_k_t = win_k[i, mask_ind_i]
                win_v_t = win_v[i, mask_ind_i]

                # mask out key and value
                if temporal_indices is not None:
                    # key [n_wh * n_ww, nhead, t, w_h * w_w, head_dim]
                    win_k_t = win_k_t[:, :, temporal_indices.flatten()].view(mask_n, self.nhead, -1, head_dim)
                    # value
                    win_v_t = win_v_t[:, :, temporal_indices.flatten()].view(mask_n, self.nhead, -1, head_dim)
                else:
                    win_k_t = win_k_t.view(n_wh * n_ww, self.nhead, t * w_h * w_w, head_dim)
                    win_v_t = win_v_t.view(n_wh * n_ww, self.nhead, t * w_h * w_w, head_dim)

                att_t = (win_q_t @ win_k_t.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_t.size(-1)))
                att_t = F.softmax(att_t, dim=-1)
                y_t = att_t @ win_v_t

                out[i, mask_ind_i] = y_t.view(-1, self.nhead, t, w_h * w_w, head_dim)

            ### For unmasked windows
            unmask_ind_i = (masks[i] == 0).nonzero(as_tuple=False).view(-1)
            # mask out quary in current window
            win_q_s = win_q[i, unmask_ind_i]
            win_k_s = win_k[i, unmask_ind_i, :, :, :w_h * w_w]
            win_v_s = win_v[i, unmask_ind_i, :, :, :w_h * w_w]

            att_s = (win_q_s @ win_k_s.transpose(-2, -1)) * (1.0 / math.sqrt(win_q_s.size(-1)))
            att_s = F.softmax(att_s, dim=-1)
            y_s = att_s @ win_v_s

            out[i, unmask_ind_i] = y_s

        # re-assemble all head outputs side by side
        out = out.view(bsz, n_wh, n_ww, self.nhead, t, w_h, w_w, head_dim)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(bsz, t, padded_h, padded_w, c)

        if pad_r > 0 or pad_b > 0:
            out = out[:, :, :h, :w, :]

        out = self.proj(out)

        return out


class TemporalSparseTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: tuple[int, int],
        pool_size: tuple[int, int],
        t2t_kernel_size: tuple[int, int],
        t2t_stride: tuple[int, int],
        t2t_padding: tuple[int, int],
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = SparseWindowAttention(
            d_model=d_model,
            nhead=nhead,
            window_size=window_size,
            pool_size=pool_size,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FusionFFN(
            d_model=d_model,
            t2t_kernel_size=t2t_kernel_size,
            t2t_stride=t2t_stride,
            t2t_padding=t2t_padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        fold_size:  tuple[int, int],
        masks: torch.Tensor,
        temporal_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, t, h, w, c = x.shape

        # Attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, masks=masks, temporal_indices=temporal_indices)
        x += residual

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x.view(bsz, t * h * w, c), output_size=fold_size)
        x = x.view(bsz, t, h, w, c)
        x += residual

        return x


class TemporalSparseTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int | tuple[int, int],
        pool_size: int | tuple[int, int],
        t2t_kernel_size: int | tuple[int, int],
        t2t_stride: int | tuple[int, int],
        t2t_padding: int | tuple[int, int],
        num_layers: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.window_size = to_pair(window_size)
        self.pool_size = to_pair(pool_size)
        self.t2t_kernel_size = to_pair(t2t_kernel_size)
        self.t2t_stride = to_pair(t2t_stride)
        self.t2t_padding = to_pair(t2t_padding)
        self.num_layers = num_layers

        self.transformer = nn.Sequential(*(
            TemporalSparseTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                window_size=self.window_size,
                pool_size=self.pool_size,
                t2t_kernel_size=self.t2t_kernel_size,
                t2t_stride=self.t2t_stride,
                t2t_padding=self.t2t_padding,
            )
            for _ in range(num_layers)
        ))

    def forward(
        self,
        x: torch.Tensor,
        fold_size: tuple[int, int],
        masks: torch.Tensor,
        temporal_dilation: int = 2,
    ) -> torch.Tensor:
        assert self.num_layers % temporal_dilation == 0, (self.num_layers, temporal_dilation)

        temporal_indices_list = compute_temporal_indices_list(
            x.shape[1], temporal_dilation=temporal_dilation, device=x.device
        )

        for i in range(self.num_layers):
            x = self.transformer[i](
                x,
                fold_size=fold_size,
                masks=masks,
                temporal_indices=temporal_indices_list[i % temporal_dilation],
            )

        return x


@lru_cache
@torch.no_grad
def compute_temporal_indices_list(
    temporal_length: int, temporal_dilation: int, device: torch.device
) -> list[torch.Tensor]:
    indices = [
        torch.arange(i, temporal_length, temporal_dilation, device=device)
        for i in range(temporal_dilation)
    ]
    return indices
