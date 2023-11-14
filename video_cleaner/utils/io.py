import os
from fractions import Fraction

import av
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def read_video(video_path: str | os.PathLike) -> tuple[np.ndarray, Fraction | None]:
    frames = []

    if os.path.isdir(video_path):
        fps = None

        for frame_path in sorted(os.listdir(video_path)):
            if frame_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                frame_path = os.path.join(video_path, frame_path)
                frame = np.asarray(Image.open(frame_path).convert("RGB"))
                frames.append(frame)
    else:
        with av.open(video_path) as container:
            assert len(container.streams.video) == 1, container.streams.video
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            assert stream.base_rate == stream.average_rate
            fps = stream.base_rate

            for frame in container.decode(stream):
                frame = np.asarray(frame.to_image())
                frames.append(frame)

    frames = np.stack(frames, axis=0)

    return frames, fps


def read_masks(mask_path: str | os.PathLike, dilation_iters: int = 8) -> np.ndarray:
    mask_paths = []
    if os.path.isdir(mask_path):
        mask_paths = sorted(filter(lambda p: p.endswith((".jpg", ".jpeg", ".png", ".bmp")), os.listdir(mask_path)))
        mask_paths = [os.path.join(mask_path, p) for p in mask_paths]
    else:
        mask_paths = [mask_path]

    masks = []
    for mask_path in mask_paths:
        mask_image = Image.open(mask_path).convert("L")
        mask_array = np.array(mask_image)
        mask_array = binary_dilation(mask_array, iterations=dilation_iters).astype(np.uint8)
        masks.append(mask_array)

    masks = np.stack(masks, axis=0)

    return masks
