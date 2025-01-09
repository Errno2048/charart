import math
import typing

import numpy as np
import moviepy
import torch

from . import color
from charart.utils.color import rgb2lab as np_rgb2lab
from charart.utils.filters import image_norm_filter


def video_norm_filter(video : moviepy.VideoClip, outlier_ratio=0.01, skip_frames=0) \
        -> typing.Callable[[torch.Tensor], torch.Tensor]:
    w, h = video.size
    sz = w * h
    outlier_index = math.floor(sz * outlier_ratio)
    left_index, right_index = outlier_index, sz - outlier_index
    frames = []
    fps = getattr(video, 'fps', None)
    if fps is None:
        index = 0
        for frame in video.iter_frames(logger='bar'):
            if skip_frames <= 0 or index % skip_frames == 0:
                frame_lab = np_rgb2lab(frame, illuminant='D65')
                frame_l = frame_lab[:, :, 0].reshape(-1)
                frames.append(frame_l)
            index += 1
    else:
        new_fps = fps / max(1, skip_frames)
        for frame in video.iter_frames(fps=new_fps, logger='bar'):
            frame_lab = np_rgb2lab(frame, illuminant='D65')
            frame_l = frame_lab[:, :, 0].reshape(-1)
            frames.append(frame_l)
    if not frames:
        return None

    frames = np.stack(frames, axis=0)
    frames.sort(axis=1)
    min_, max_ = frames[:, left_index].min(), frames[:, right_index].max()

    if min_ >= max_:
        filter = None
    else:
        def wrapper(min_, max_):
            def filter(image):
                nonlocal min_, max_
                image = color.rgb2lab(image, illuminant='D65')
                image[:, :, 0] = ((image[:, :, 0] - min_) / (max_ - min_)).clip(0., 1.) * 100.
                return color.lab2rgb(image, illuminant='D65')
            return filter
        filter = wrapper(min_, max_)
    return filter


__all__ = ['image_norm_filter', 'video_norm_filter']