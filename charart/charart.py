import typing
from collections.abc import Iterable as _Iterable

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import moviepy

from .backend import NumpyBackend, PytorchBackend


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
        background_color = '#ffffff',
        foreground_color = '#000000',
        lab_illuminant = 'D65',
        lab_observer = '2',
        backend : str = 'numpy',
        **kwargs,
    ):
        self._backend_kwargs = {}
        if backend == 'numpy':
            self._backend = NumpyBackend
        elif backend == 'pytorch':
            self._backend = PytorchBackend
            self._backend_kwargs['device'] = kwargs.get('device', None)
        else:
            raise ValueError('invalid backend: {}'.format(backend))

        self._font = font
        self._size = size
        self._hparts = hparts
        self._vparts = vparts
        self._hspace = hspace
        self._vspace = vspace
        self._lab_illuminant = lab_illuminant
        self._lab_observer = lab_observer

        self._bg = ImageColor.getcolor(background_color, 'RGB')
        self._fg = ImageColor.getcolor(foreground_color, 'RGB')
        ref_color = self._backend.color.rgb2lab(
            self._backend.array(np.array([[list(self._fg), list(self._bg)]], dtype=float) / 255., **self._backend_kwargs),
            illuminant=lab_illuminant,
            observer=lab_observer,
        )
        self._lab_ref_color = (ref_color[0, 0], ref_color[0, 1])

        self.clear()

    @property
    def filters(self):
        return self._backend.filters

    def clear(self):
        self._charset = []
        self._charset_set = set()
        self._charset_box = None
        self._charset_features = None
        self._charset_prepared = False
        self._charset_buf = Image.new('RGB', (self._size * 3, self._size * 3), color=self._bg)
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
            image_array = self._backend.array(np.array(image), **self._backend_kwargs)
            image_array = self._backend.transpose(image_array, (1, 0, 2))
            image_array = self._backend.astype(image_array, self._backend.type('float')) / 255
            image_lab = self._backend.color.rgb2lab(image_array, illuminant=self._lab_illuminant, observer=self._lab_observer)[:, :, 0]
        else:
            image_lab = image[:, :, 0]
        crop_width, crop_height = self._charset_box.width // self._hparts, self._charset_box.height // self._vparts
        if box:
            image_lab = image_lab[box.left : box.right, box.top : box.bottom]
        hstride, vstride = self._backend.strides(image_lab)
        hshape, vshape = self._backend.shape(image_lab)[0] // self._charset_box.width, self._backend.shape(image_lab)[1] // self._charset_box.height
        strided = self._backend.as_strided(
            image_lab,
            (hshape, vshape, self._hparts, self._vparts, crop_width, crop_height),
            (
                hstride * self._charset_box.width, vstride * self._charset_box.height,
                hstride * crop_width, vstride * crop_height,
                hstride, vstride,
            ),
        )
        mean_pooled = self._backend.reshape(strided, (hshape, vshape, self._hparts, self._vparts, -1))
        mean_pooled = self._backend.mean(mean_pooled, axis=-1)
        mean_pooled = self._backend.reshape(mean_pooled, (hshape, vshape, -1))
        return mean_pooled

    def _get_colors(self, image : Image, box=None):
        if isinstance(image, Image.Image):
            image_array = self._backend.array(np.array(image), **self._backend_kwargs)
            image_array = self._backend.transpose(image_array, (1, 0, 2))
            image_array = self._backend.astype(image_array, self._backend.type('float')) / 255
            image_lab = self._backend.color.rgb2lab(image_array, illuminant=self._lab_illuminant, observer=self._lab_observer)
        else:
            image_lab = image
        if box:
            image_lab = image_lab[box.left : box.right, box.top : box.bottom]

        hshape, vshape = self._backend.shape(image_lab)[0] // self._charset_box.width, self._backend.shape(image_lab)[1] // self._charset_box.height
        hstride, vstride, color_stride = self._backend.strides(image_lab)
        strided = self._backend.as_strided(
            image_lab,
            (hshape, vshape, 3, self._charset_box.width, self._charset_box.height),
            (
                hstride * self._charset_box.width, vstride * self._charset_box.height,
                color_stride,
                hstride, vstride,
            ),
        )
        mean_pooled = self._backend.reshape(strided, (hshape, vshape, 3, -1))
        mean_pooled = self._backend.mean(mean_pooled, axis=-1)
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
        self._charset_buf = Image.new('RGB', (self._charset_box.width, self._charset_box.height), self._bg)
        ori_box = self._charset_box.move(-self._charset_box.left, -self._charset_box.top)
        self._charset_features = []
        self._charset_images = []
        for char in self._charset:
            self._charset_buf.paste(self._bg, ori_box.tuple())
            draw = ImageDraw.Draw(self._charset_buf)
            draw.font = self._font
            draw.text((
                -self._charset_box.left + (self._charset_box.width - char.box.width) / 2,
                -self._charset_box.top + (self._charset_box.height - char.box.height) / 2,
            ), char.char, anchor='la', fill=self._fg)
            image_array = self._backend.array(np.array(self._charset_buf), **self._backend_kwargs)
            image_array = self._backend.transpose(image_array, (1, 0, 2))
            image_array = self._backend.astype(image_array, self._backend.type('float')) / 255
            image_lab = self._backend.color.rgb2lab(image_array, illuminant=self._lab_illuminant, observer=self._lab_observer)
            char_features = self._backend.reshape(self._get_features(image_lab), (-1, ))
            self._charset_features.append(char_features)
            self._charset_images.append(image_lab)
        self._charset_features = self._backend.stack(self._charset_features, axis=0)
        self._charset_images = self._backend.stack(self._charset_images, axis=0)
        self._charset_lightness = (self._backend.min(self._charset_features), self._backend.max(self._charset_features))

    def transform(self, image : Image.Image, dist_norm=2., set_color=True, char_normalize=False, return_array=False):
        self._check_and_prepare()

        ref_black, ref_white = self._lab_ref_color
        if isinstance(image, Image.Image):
            image_array = self._backend.array(np.array(image), **self._backend_kwargs)
            image_array = self._backend.transpose(image_array, (1, 0, 2))
            image_array = self._backend.astype(image_array, self._backend.type('float')) / 255
            image_width, image_height = image.width, image.height
        else:
            image_array = image
            image_width, image_height = self._backend.shape(image_array)[:2]

        boxes_h, boxes_v = (image_width + self._hspace) // self._charset_box.width, (image_height + self._vspace) // self._charset_box.height
        start_x, start_y = (image_width + self._hspace - boxes_h * self._charset_box.width) // 2, (image_height + self._vspace - boxes_v * self._charset_box.height) // 2
        feature_box = Box(start_x, start_y, start_x + boxes_h * self._charset_box.width, start_y + boxes_v * self._charset_box.height)

        image_lab = self._backend.color.rgb2lab(image_array, illuminant=self._lab_illuminant, observer=self._lab_observer)
        image_features = self._get_features(image_lab, feature_box)
        image_features = self._backend.reshape(image_features, (1, *image_features.shape))
        charset_features = self._backend.reshape(self._charset_features, (self._charset_features.shape[0], 1, 1, -1))
        l_min, l_max = self._charset_lightness
        normalized_charset_features = (charset_features - l_min) / (l_max - l_min) if char_normalize else charset_features
        feature_dist = self._backend.mean(self._backend.pow(self._backend.abs(image_features - normalized_charset_features), dist_norm), axis=-1)
        selected = self._backend.argmin(feature_dist, axis=0)

        text_image = self._backend.ones((feature_box.width, feature_box.height, 3), dtype=self._backend.type('float'), **self._backend_kwargs)
        for v in range(boxes_v):
            for h in range(boxes_h):
                selected_image = self._charset_images[selected[h, v]]
                x, y = h * self._charset_box.width, v * self._charset_box.height
                text_image[x : x + self._charset_box.width, y : y + self._charset_box.height] = selected_image

        if set_color:
            hs, vs, cs = self._backend.strides(text_image)
            text_image_strided = self._backend.as_strided(
                text_image,
                (boxes_h, boxes_v, 3, self._charset_box.width, self._charset_box.height),
                (
                    hs * self._charset_box.width, vs * self._charset_box.height,
                    cs,
                    hs, vs,
                ),
            )
            image_cropped = image_lab[feature_box.left : feature_box.right, feature_box.top : feature_box.bottom]
            ohs, ovs, ocs = self._backend.strides(image_cropped)
            image_strided = self._backend.as_strided(
                image_cropped,
                (boxes_h, boxes_v, 3, self._charset_box.width, self._charset_box.height),
                (
                    ohs * self._charset_box.width, ovs * self._charset_box.height,
                    ocs,
                    ohs, ovs,
                ),
            )

            ori_color = self._backend.mean(self._backend.reshape(image_strided, (boxes_h, boxes_v, 3, -1)), axis=-1)
            text_image_ratio = (text_image_strided[:, :, 0:1, ...] - self._backend.reshape(ref_white[0], (1, 1, 1, 1, 1))) \
                                / (ref_black[0] - ref_white[0])
            # must not copy text_image_strided and should always do inplace operations
            text_image_ratio_reshape = self._backend.mean(self._backend.reshape(self._backend.clip(text_image_ratio, 1e-4), (boxes_h, boxes_v, 1, -1)), axis=-1)
            text_image_strided[:, :, 1:3, ...] = \
                text_image_ratio * self._backend.reshape(((ori_color - ref_white)[..., 1:3] / text_image_ratio_reshape), (boxes_h, boxes_v, 2, 1, 1)) + \
                (1 - text_image_ratio) * (text_image_strided - self._backend.reshape(ref_white, (1, 1, 3, 1, 1)))[:, :, 1:3, ...] + \
                self._backend.reshape(ref_white[1:3], (1, 1, 2, 1, 1))
            text_image = self._backend.as_strided(
                text_image_strided,
                (feature_box.width, feature_box.height, 3),
                (hs, vs, cs),
            )

        new_text_image = self._backend.zeros((image_width, image_height, 3), dtype=self._backend.type('float'), **self._backend_kwargs) + ref_white
        new_text_image[feature_box.left : feature_box.right, feature_box.top : feature_box.bottom] = text_image
        new_text_image = self._backend.color.lab2rgb(new_text_image, illuminant=self._lab_illuminant, observer=self._lab_observer)
        new_image_array = self._backend.astype(self._backend.transpose(new_text_image, (1, 0, 2)) * 255, self._backend.type('uint8'))

        if return_array:
            return new_image_array

        new_image_array = self._backend.numpy(new_image_array)
        new_image = Image.fromarray(new_image_array, 'RGB')

        return new_image

    def video_transform(
        self,
        video : moviepy.VideoClip,
        skip_frames=0,
        preprocess : typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None,
        postprocess : typing.Optional[typing.Callable[[typing.Any], typing.Any]] = None,
        **kwargs,
    ):
        arr, frame_count = None, 0
        def transform(image : np.ndarray):
            nonlocal arr, frame_count
            if skip_frames <= 0 or frame_count % skip_frames == 0:
                image = self._backend.array(image, **self._backend_kwargs)
                image = self._backend.astype(self._backend.transpose(image, (1, 0, 2)), self._backend.type('float')) / 255.
                if preprocess:
                    image = preprocess(image)
                arr = self._backend.numpy(self.transform(image, return_array=True, **kwargs))
                if postprocess:
                    arr = postprocess(arr)
            if skip_frames > 0:
                frame_count = (frame_count + 1) % skip_frames
            return arr
        return video.image_transform(transform)


__all__ = ['Charart']
