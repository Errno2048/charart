import typing
from collections.abc import Iterable as _Iterable

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import color
import moviepy


class Box:
    def __init__(self, left=0, top=0, right=0, bottom=0):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def tuple(self):
        return (self.left, self.top, self.right, self.bottom)

    def copy(self):
        return Box(self.left, self.top, self.right, self.bottom)

    def union(self, box, inplace=False):
        if not inplace:
            return self.copy().union(box, inplace=True)
        self.left = min(self.left, box.left)
        self.top = min(self.top, box.top)
        self.right = max(self.right, box.right)
        self.bottom = max(self.bottom, box.bottom)
        return self

    def intersection(self, box, inplace=False):
        if not inplace:
            return self.copy().intersection(box, inplace=True)
        self.left = max(self.left, box.left)
        self.top = max(self.top, box.top)
        self.right = min(self.right, box.right)
        self.bottom = min(self.bottom, box.bottom)
        return self

    def move(self, x, y, inplace=False):
        if not inplace:
            return self.copy().move(x, y, inplace=True)
        self.left += x
        self.right += x
        self.top += y
        self.bottom += y
        return self

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top


class Charart:
    class char:
        def __init__(self, char, box : Box):
            self.char = char
            self.box = box

    def __init__(
        self,
        size : typing.Union[int, float],
        font : ImageFont,
        hparts : int = 1,
        vparts : int = 1,
        hspace : int = 0,
        vspace : int = 0,
        lab_illuminant = 'D65',
    ):
        self._font = font
        self._size = size
        self._hparts = hparts
        self._vparts = vparts
        self._hspace = hspace
        self._vspace = vspace
        self._lab_illuminant = lab_illuminant

        ref_color = color.rgb2lab(np.array([[[0., 0., 0.], [1., 1., 1.]]]), illuminant=lab_illuminant)
        self._lab_ref_color = (ref_color[0, 0], ref_color[0, 1])

        self.clear()

    def clear(self):
        self._charset = []
        self._charset_set = set()
        self._charset_box = None
        self._charset_features = None
        self._charset_prepared = False
        self._charset_buf = Image.new('RGB', (self._size * 3, self._size * 3), color='#ffffff')
        self._charset_lightness = None
        self._charset_images = None

    def add_character(self, *characters) -> None:
        self._charset_prepared = False
        chars = set()
        for c in characters:
            if isinstance(c, _Iterable):
                chars.update(c)
        chars -= self._charset_set
        chars = list(chars)
        for c in chars:
            draw = ImageDraw.Draw(self._charset_buf)
            draw.font = self._font
            box = Box(*draw.textbbox((0, 0), c, anchor='la'))
            if self._charset_box is None:
                self._charset_box = box
            else:
                self._charset_box.union(box, inplace=True)
            self._charset.append(self.char(c, box))
        self._charset_set.update(chars)

    def _check_and_prepare(self):
        if not self._charset_prepared:
            self.prepare()

    def _get_features(self, image : Image.Image, box=None):
        if isinstance(image, Image.Image):
            image_array = np.array(image)[:, :, :3].transpose((1, 0, 2)).astype(float) / 255
            image_lab = color.rgb2lab(image_array, illuminant=self._lab_illuminant)[:, :, 0]
        else:
            image_lab = image[:, :, 0]
        crop_width, crop_height = self._charset_box.width // self._hparts, self._charset_box.height // self._vparts
        if box:
            image_lab = image_lab[box.left : box.right, box.top : box.bottom]
        hstride, vstride = image_lab.strides
        hshape, vshape = image_lab.shape[0] // self._charset_box.width, image_lab.shape[1] // self._charset_box.height
        strided = np.lib.stride_tricks.as_strided(
            image_lab,
            (hshape, vshape, self._hparts, self._vparts, crop_width, crop_height),
            (
                hstride * self._charset_box.width, vstride * self._charset_box.height,
                hstride * crop_width, vstride * crop_height,
                hstride, vstride,
            ),
        )
        mean_pooled = strided.reshape(hshape, vshape, self._hparts, self._vparts, -1).mean(axis=-1)
        mean_pooled = mean_pooled.reshape(hshape, vshape, -1)
        return mean_pooled

    def _get_colors(self, image : Image, box=None):
        if isinstance(image, Image.Image):
            image_array = np.array(image)[:, :, :3].transpose((1, 0, 2)).astype(float) / 255
            image_lab = color.rgb2lab(image_array, illuminant=self._lab_illuminant)
        else:
            image_lab = image
        if box:
            image_lab = image_lab[box.left : box.right, box.top : box.bottom]

        hshape, vshape = image_lab.shape[0] // self._charset_box.width, image_lab.shape[1] // self._charset_box.height
        hstride, vstride, color_stride = image_lab.strides
        strided = np.lib.stride_tricks.as_strided(
            image_lab,
            (hshape, vshape, 3, self._charset_box.width, self._charset_box.height),
            (
                hstride * self._charset_box.width, vstride * self._charset_box.height,
                color_stride,
                hstride, vstride,
            ),
        )
        mean_pooled = strided.reshape(hshape, vshape, 3, -1).mean(axis=-1)
        return mean_pooled

    def prepare(self):
        if self._charset_prepared:
            return
        self._charset_prepared = True
        hspace_pad = -(self._charset_box.width + self._hspace) % self._hparts + self._hspace
        vspace_pad = -(self._charset_box.height + self._vspace) % self._vparts + self._vspace
        move_x, move_y = hspace_pad // 2, vspace_pad // 2
        self._charset_box.move(move_x, move_y, inplace=True)
        self._charset_box.right += hspace_pad
        self._charset_box.bottom += vspace_pad
        self._charset_buf = Image.new('RGB', (self._charset_box.width, self._charset_box.height), '#ffffff')
        ori_box = self._charset_box.move(-self._charset_box.left, -self._charset_box.top)
        self._charset_features = []
        self._charset_images = []
        for char in self._charset:
            self._charset_buf.paste('#ffffff', ori_box.tuple())
            draw = ImageDraw.Draw(self._charset_buf)
            draw.font = self._font
            draw.text((
                -self._charset_box.left + (self._charset_box.width - char.box.width) / 2,
                -self._charset_box.top + (self._charset_box.height - char.box.height) / 2,
            ), char.char, anchor='la', fill='#000000')
            image_array = np.array(self._charset_buf)[:, :, :3].transpose((1, 0, 2)).astype(float) / 255
            image_lab = color.rgb2lab(image_array, illuminant=self._lab_illuminant)
            char_features = self._get_features(image_lab).reshape(-1)
            self._charset_features.append(char_features)
            self._charset_images.append(image_lab)
        self._charset_features = np.stack(self._charset_features, axis=0)
        self._charset_images = np.stack(self._charset_images, axis=0)
        self._charset_lightness = (self._charset_features.min(), self._charset_features.max())

    def transform(self, image : Image.Image, dist_norm=2., set_color=True, char_normalize=False, return_array=False):
        self._check_and_prepare()

        boxes_h, boxes_v = (image.width + self._hspace) // self._charset_box.width, (image.height + self._vspace) // self._charset_box.height
        start_x, start_y = (image.width + self._hspace - boxes_h * self._charset_box.width) // 2, (image.height + self._vspace - boxes_v * self._charset_box.height) // 2
        feature_box = Box(start_x, start_y, start_x + boxes_h * self._charset_box.width, start_y + boxes_v * self._charset_box.height)

        image_array = np.array(image)[:, :, :3].transpose((1, 0, 2)).astype(float) / 255
        image_lab = color.rgb2lab(image_array, illuminant=self._lab_illuminant)
        image_features = self._get_features(image_lab, feature_box)
        image_features = image_features.reshape(1, *image_features.shape)
        charset_features = self._charset_features.reshape(self._charset_features.shape[0], 1, 1, -1)
        l_min, l_max = self._charset_lightness
        normalized_charset_features = (charset_features - l_min) / (l_max - l_min) if char_normalize else charset_features
        feature_dist = (np.abs(image_features - normalized_charset_features) ** dist_norm).mean(axis=-1)
        selected = feature_dist.argmin(axis=0)

        text_image = np.ones((feature_box.width, feature_box.height, 3), dtype=float)
        for v in range(boxes_v):
            for h in range(boxes_h):
                selected_image = self._charset_images[selected[h, v]]
                x, y = h * self._charset_box.width, v * self._charset_box.height
                text_image[x : x + self._charset_box.width, y : y + self._charset_box.height] = selected_image

        if set_color:
            hs, vs, cs = text_image.strides
            text_image_strided = np.lib.stride_tricks.as_strided(
                text_image,
                (boxes_h, boxes_v, 3, self._charset_box.width, self._charset_box.height),
                (
                    hs * self._charset_box.width, vs * self._charset_box.height,
                    cs,
                    hs, vs,
                ),
            )
            image_cropped = image_lab[feature_box.left : feature_box.right, feature_box.top : feature_box.bottom]
            ohs, ovs, ocs = image_cropped.strides
            image_strided = np.lib.stride_tricks.as_strided(
                image_cropped,
                (boxes_h, boxes_v, 3, self._charset_box.width, self._charset_box.height),
                (
                    ohs * self._charset_box.width, ovs * self._charset_box.height,
                    ocs,
                    ohs, ovs,
                ),
            )

            ref_black, ref_white = self._lab_ref_color
            ori_color = image_strided.reshape(boxes_h, boxes_v, 3, -1).mean(axis=-1)
            text_image_ratio = (text_image_strided[:, :, 0:1, ...] - ref_white[0].reshape(1, 1, 1, 1, 1)) \
                                / (ref_black[0] - ref_white[0])
            # must not copy text_image_strided and should always do inplace operations
            text_image_strided[:, :, 1:3, ...] = \
                text_image_ratio * ((ori_color - ref_white)[..., 1:3] / text_image_ratio.clip(1e-4).reshape(boxes_h, boxes_v, 1, -1).mean(axis=-1)).reshape(boxes_h, boxes_v, 2, 1, 1) + \
                (1 - text_image_ratio) * (text_image_strided - ref_white.reshape(1, 1, 3, 1, 1))[:, :, 1:3, ...] + \
                ref_white[1:3].reshape(1, 1, 2, 1, 1)
            text_image = np.lib.stride_tricks.as_strided(
                text_image_strided,
                (feature_box.width, feature_box.height, 3),
                (hs, vs, cs),
            )

        text_image = color.lab2rgb(text_image, illuminant=self._lab_illuminant)
        new_text_image = np.ones((image.width, image.height, 3), dtype=float)
        new_text_image[feature_box.left : feature_box.right, feature_box.top : feature_box.bottom] = text_image
        new_image_array = (new_text_image.transpose((1, 0, 2)) * 255).astype(np.uint8)

        if return_array:
            return new_image_array

        new_image = Image.fromarray(new_image_array, 'RGB')

        return new_image

    def video_transform(
        self,
        video : moviepy.VideoClip,
        skip_frames=0,
        filter : typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
        **kwargs,
    ):
        arr, frame_count = None, 0
        def transform(image : np.ndarray):
            nonlocal arr, frame_count
            if skip_frames <= 0 or frame_count % skip_frames == 0:
                image = Image.fromarray(image.astype(np.uint8), mode='RGB')
                arr = self.transform(image, return_array=True, **kwargs)
            if skip_frames > 0:
                frame_count = (frame_count + 1) % skip_frames
            return arr
        if filter:
            _transform = transform
            def transform(image : np.ndarray):
                return _transform(filter(image))
        return video.image_transform(transform)
