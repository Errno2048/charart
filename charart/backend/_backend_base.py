import typing
from abc import ABC, abstractmethod

import numpy as np


class _BackendBase(ABC):
    color, filters = None, None

    def __init__(self):
        raise NotImplementedError('cannot initialize objects of a backend class')

    @classmethod
    @abstractmethod
    def type(cls, name: str) -> typing.Any:
        """
        To return a dtype of the specified name.
        :param name: the name of the dtype
        :return: a dtype
        """
        return {
            'float': float,
            'uint8': int,
        }.get(name, None)

    @classmethod
    @abstractmethod
    def array(cls, value, dtype=None, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def transpose(cls, value, axes: typing.Optional[typing.Tuple[int, ...]] = None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def astype(cls, value, dtype):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def strides(cls, value):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def as_strided(cls, value, shape: typing.Tuple[int, ...], strides: typing.Tuple[int, ...]):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def reshape(cls, value, shape: typing.Tuple[int, ...]):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def mean(cls, value, axis: int = None, keepdims: bool = False):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def min(cls, value, axis: int = None, keepdims: bool = False):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def max(cls, value, axis: int = None, keepdims: bool = False):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def argmin(cls, value, axis: int = None, keepdims: bool = False):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def argmax(cls, value, axis: int = None, keepdims: bool = False):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def stack(cls, values: typing.Sequence[typing.Any], axis: int = 0):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def shape(cls, value):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def abs(cls, value):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def pow(cls, value, exponent):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def clip(cls, value, min=None, max=None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def zeros(cls, shape, dtype=None, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def ones(cls, shape, dtype=None, *args, **kwargs):
        return cls.zeros(shape, dtype, *args, **kwargs) + 1

    @classmethod
    @abstractmethod
    def numpy(cls, value) -> np.ndarray:
        raise NotImplementedError
