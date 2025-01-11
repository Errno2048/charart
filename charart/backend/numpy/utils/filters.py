import math
import typing

import numpy as np
import moviepy
from PIL import Image
from .color import rgb2lab, lab2rgb


def image_norm_filter(image : Image.Image, outlier_ratio=0.01) -> Image.Image:
    w, h = image.size
    sz = w * h
    outlier_index = math.floor(sz * outlier_ratio)
    left_index, right_index = outlier_index, sz - outlier_index
    image_rgb = np.array(image).astype(float) / 255.
    image_lab = rgb2lab(image_rgb, illuminant='D65')
    image_l = image_lab[:, :, 0].reshape(-1).copy()
    image_l.sort()
    min_, max_ = image_l[left_index], image_l[right_index]
    image_lab[:, :, 0] = (image_lab[:, :, 0] - min_) / (max_ - min_) * 100.
    new_image_rgb = lab2rgb(image_lab, illuminant='D65')
    return Image.fromarray((new_image_rgb * 255).astype(np.uint8), 'RGB')


def video_norm_filter(video : moviepy.VideoClip, outlier_ratio=0.01, skip_frames=0) \
        -> typing.Callable[[np.ndarray], np.ndarray]:
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
                frame_lab = rgb2lab(frame, illuminant='D65')
                frame_l = frame_lab[:, :, 0].reshape(-1)
                frames.append(frame_l)
            index += 1
    else:
        new_fps = fps / max(1, skip_frames)
        for frame in video.iter_frames(fps=new_fps, logger='bar'):
            frame_lab = rgb2lab(frame, illuminant='D65')
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
                image = rgb2lab(image, illuminant='D65')
                image[:, :, 0] = ((image[:, :, 0] - min_) / (max_ - min_)).clip(0., 1.) * 100.
                return lab2rgb(image, illuminant='D65')
            return filter
        filter = wrapper(min_, max_)
    return filter


__all__ = ['image_norm_filter', 'video_norm_filter']
