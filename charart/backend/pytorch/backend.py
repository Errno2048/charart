import typing

import numpy as np
import torch

from .._backend_base import _BackendBase
from .utils import color, filters


class PytorchBackend(_BackendBase):
    color, filters = color, filters

    @classmethod
    def type(cls, name: str) -> typing.Any:
        return {
            'float': torch.float,
            'uint8': torch.uint8,
        }.get(name, None)

    @classmethod
    def array(cls, value, dtype: torch.dtype = None, *args, **kwargs) -> torch.Tensor:
        return torch.tensor(value, dtype=dtype, device=kwargs.get('device', None), requires_grad=kwargs.get('requires_grad', False))

    @classmethod
    def transpose(cls, value: torch.Tensor, axes: typing.Optional[typing.Tuple[int, ...]] = None) -> torch.Tensor:
        if axes is None:
            return value.T
        return torch.permute(value, axes)

    @classmethod
    def astype(cls, value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return value.to(dtype=dtype)

    @classmethod
    def strides(cls, value: torch.Tensor) -> typing.Tuple[int, ...]:
        return value.stride()

    @classmethod
    def as_strided(cls, value: torch.Tensor, shape: typing.Tuple[int, ...], strides: typing.Tuple[int, ...]) -> torch.Tensor:
        return torch.as_strided(value, shape, strides)

    @classmethod
    def reshape(cls, value: torch.Tensor, shape: typing.Tuple[int, ...]) -> torch.Tensor:
        return torch.reshape(value, shape)

    @classmethod
    def mean(cls, value: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        return torch.mean(value, dim=axis, keepdim=keepdims)

    @classmethod
    def min(cls, value: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.min(value)
        return torch.min(value, dim=axis, keepdim=keepdims).values

    @classmethod
    def max(cls, value: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        if axis is None:
            return torch.max(value)
        return torch.max(value, dim=axis, keepdim=keepdims).values

    @classmethod
    def argmin(cls, value: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        return torch.argmin(value, dim=axis, keepdim=keepdims)

    @classmethod
    def argmax(cls, value: torch.Tensor, axis: int = None, keepdims: bool = False) -> torch.Tensor:
        return torch.argmax(value, dim=axis, keepdim=keepdims)

    @classmethod
    def stack(cls, values: typing.Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.stack(list(values), axis)

    @classmethod
    def shape(cls, value: torch.Tensor) -> torch.Size:
        return value.shape

    @classmethod
    def abs(cls, value: torch.Tensor) -> torch.Tensor:
        return torch.abs(value)

    @classmethod
    def pow(cls, value: torch.Tensor, exponent) -> torch.Tensor:
        return torch.pow(value, exponent)

    @classmethod
    def clip(cls, value: torch.Tensor, min=None, max=None) -> torch.Tensor:
        return torch.clip(value, min, max)

    @classmethod
    def zeros(cls, shape, dtype=None, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=kwargs.get('device', None), requires_grad=kwargs.get('requires_grad', False))

    @classmethod
    def numpy(cls, value: torch.Tensor) -> np.ndarray:
        return value.numpy(force=True)
