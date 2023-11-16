import os
from fractions import Fraction

import av
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation


def read_video(
    video_path: str | os.PathLike,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, Fraction | None]:
    frames = []

    if os.path.isdir(video_path):
        fps = None

        for frame_path in sorted(os.listdir(video_path)):
            if frame_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                frame_path = os.path.join(video_path, frame_path)
                frame = np.array(Image.open(frame_path).convert("RGB"))
                frame_tensor = torch.from_numpy(frame).to(device=device, non_blocking=True)
                frames.append(frame_tensor)
    else:
        with av.open(video_path) as container:
            assert len(container.streams.video) == 1, container.streams.video
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            assert stream.base_rate == stream.average_rate
            fps = stream.base_rate

            for frame in container.decode(stream):
                frame = np.array(frame.to_image())
                frame_tensor = torch.from_numpy(frame).to(device=device, non_blocking=True)
                frames.append(frame_tensor)

    frames = torch.stack(frames, dim=0).permute(0, 3, 1, 2)
    frames = frames.float()
    frames.div_(255.).mul_(2).sub_(1.0)

    return frames, fps


def save_video(
    frames: torch.Tensor,
    video_path: str | os.PathLike,
    fps: int | float | Fraction = 30,
) -> None:
    frames = (frames + 1.) * (0.5 * 255.)
    frames.clamp_(0., 255.)
    frames = frames.to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1)
    frames_np = frames.cpu().numpy()

    height, width = frames.shape[1:3]

    with av.open(video_path, mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"

        for frame_np in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def read_masks(
    mask_path: str | os.PathLike,
    dilation_iters: int = 8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    mask_paths = []
    if os.path.isdir(mask_path):
        mask_paths = sorted(filter(lambda p: p.endswith((".jpg", ".jpeg", ".png", ".bmp")), os.listdir(mask_path)))
        mask_paths = [os.path.join(mask_path, p) for p in mask_paths]
    else:
        mask_paths = [mask_path]

    masks = []
    for mask_path in mask_paths:
        mask_image = Image.open(mask_path).convert("L")
        mask_array = np.asarray(mask_image)
        mask_array = binary_dilation(mask_array, iterations=dilation_iters)
        mask_tensor = torch.from_numpy(mask_array).to(device, non_blocking=True)
        masks.append(mask_tensor)

    masks = torch.stack(masks, dim=0)
    masks = masks[:, None]

    return masks
