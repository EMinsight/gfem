"""Pytree registration for gfem's frozen physics values."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import jax
import numpy as np


def readonly(a: np.ndarray) -> np.ndarray:
    """A read-only C-contiguous copy of ``a`` (a frozen value's array)."""
    out = np.array(a, order="C")
    out.flags.writeable = False
    return out


def _freeze(v):
    """A hashable, equality-comparable key of a static value (arrays by content)."""
    if isinstance(v, np.ndarray):
        return ("array", v.shape, v.dtype.str, v.tobytes())
    if isinstance(v, Mapping):
        return ("mapping", tuple((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, tuple | list):
        return (type(v).__name__, tuple(_freeze(x) for x in v))
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return (
            "dataclass",
            type(v),
            tuple((f.name, _freeze(getattr(v, f.name))) for f in dataclasses.fields(v)),
        )
    return v


class _Static:
    """A static value with a content hash and equality, whatever its type.

    JAX compares and hashes the static part of a pytree (the cache key of
    ``jit``, the structure check of ``tree.map``); NumPy arrays and dicts
    would raise there, so they travel inside this wrapper and come back
    unchanged on unflatten.
    """

    __slots__ = ("key", "value")

    def __init__(self, value):
        self.value = value
        self.key = _freeze(value)

    def __hash__(self) -> int:
        try:
            return hash(self.key)
        except TypeError:
            return id(self.value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, _Static):
            return NotImplemented
        if self.value is other.value:
            return True
        try:
            return bool(self.key == other.key)
        except (TypeError, ValueError):
            return False


def _wrap(v):
    if isinstance(v, np.ndarray | Mapping):
        return _Static(v)
    if isinstance(v, tuple) and any(isinstance(x, np.ndarray | Mapping) for x in v):
        return _Static(v)
    return v


def _unwrap(v):
    return v.value if isinstance(v, _Static) else v


def register_fields[T](cls: type[T], children: tuple[str, ...]) -> type[T]:
    """Register a frozen dataclass as a pytree.

    The ``children`` fields are subtrees (array leaves, ``None`` for absent
    values, nested gfem pytrees), keyed by attribute name; every other field
    is static and must be hashable. Unflattening bypasses ``__init__``, so
    validation runs once, when the user builds the object, never on the
    placeholder leaves JAX passes around.
    """
    names = tuple(f.name for f in dataclasses.fields(cls))  # ty: ignore[invalid-argument-type]
    statics = tuple(n for n in names if n not in children)

    def flatten_with_keys(obj):
        kids = tuple((jax.tree_util.GetAttrKey(n), getattr(obj, n)) for n in children)
        return kids, tuple(_wrap(getattr(obj, n)) for n in statics)

    def flatten(obj):
        kids = tuple(getattr(obj, n) for n in children)
        return kids, tuple(_wrap(getattr(obj, n)) for n in statics)

    def unflatten(aux, kids):
        obj = object.__new__(cls)
        for n, v in zip(children, kids, strict=True):
            object.__setattr__(obj, n, v)
        for n, v in zip(statics, aux, strict=True):
            object.__setattr__(obj, n, _unwrap(v))
        return obj

    jax.tree_util.register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)
    return cls
