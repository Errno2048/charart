import typing

import numpy as np

from .._backend_base import _BackendBase
from .utils import color, filters


class NumpyBackend(_BackendBase):
    color, filters = color, filters

    @classmethod
    def type(cls, name: str) -> np.dtype:
        return {
            'float': float,
            'uint8': np.uint8,
        }.get(name, None)
    
    @classmethod
    def array(cls, value, dtype: np.dtype = None, *args, **kwargs) -> np.ndarray:
        if isinstance(value, np.ndarray):
            if dtype:
                value = value.astype(dtype)
            return value
        return np.array(value, dtype, *args, **kwargs)

    @classmethod
    def transpose(cls, value: np.ndarray, axes: typing.Optional[typing.Tuple[int, ...]] = None) -> np.ndarray:
        return np.transpose(value, axes)

    @classmethod
    def astype(cls, value: np.ndarray, dtype) -> np.ndarray:
        return value.astype(dtype)

    @classmethod
    def strides(cls, value: np.ndarray) -> typing.Tuple[int, ...]:
        return value.strides

    @classmethod
    def as_strided(cls, value: np.ndarray, shape: typing.Tuple[int, ...], strides: typing.Tuple[int, ...]) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(value, shape, strides)

    @classmethod
    def reshape(cls, value: np.ndarray, shape: typing.Tuple[int, ...]) -> np.ndarray:
        return np.reshape(value, shape)

    @classmethod
    def mean(cls, value: np.ndarray, axis: int = None, keepdims: bool = False):
        return np.mean(value, axis, keepdims=keepdims)

    @classmethod
    def min(cls, value: np.ndarray, axis: int = None, keepdims: bool = False):
        return np.min(value, axis, keepdims=keepdims)

    @classmethod
    def max(cls, value: np.ndarray, axis: int = None, keepdims: bool = False):
        return np.max(value, axis, keepdims=keepdims)

    @classmethod
    def argmin(cls, value: np.ndarray, axis: int = None, keepdims: bool = False):
        return np.argmin(value, axis, keepdims=keepdims)

    @classmethod
    def argmax(cls, value: np.ndarray, axis: int = None, keepdims: bool = False):
        return np.argmax(value, axis, keepdims=keepdims)

    @classmethod
    def stack(cls, values: typing.Sequence[np.ndarray], axis: int = 0):
        return np.stack(values, axis)

    @classmethod
    def shape(cls, value: np.ndarray) -> typing.Tuple[int, ...]:
        return value.shape

    @classmethod
    def abs(cls, value: np.ndarray) -> np.ndarray:
        return np.abs(value)

    @classmethod
    def pow(cls, value: np.ndarray, exponent) -> np.ndarray:
        return np.power(value, exponent)

    @classmethod
    def clip(cls, value: np.ndarray, min=None, max=None) -> np.ndarray:
        if min is None:
            min = -np.inf
        if max is None:
            max = np.inf
        return np.clip(value, min, max).astype(value.dtype)

    @classmethod
    def zeros(cls, shape, dtype: typing.Optional[np.dtype] = None, *args, **kwargs) -> np.ndarray:
        return np.zeros(shape, dtype, *args, **kwargs)

    @classmethod
    def numpy(cls, value: np.ndarray) -> np.ndarray:
        return value
