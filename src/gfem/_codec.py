"""Generic JSON + ``.npz`` encoding of gfem's value classes.

A value is written as a JSON document plus a table of arrays. Every array
goes into the table (the ``.npz`` sidecar) under its pytree leaf path, the
`jax.tree_util.keystr` of the leaf, e.g. ``.model.materials['oxide'].eps_r``;
the JSON holds ``{"$array": "<path>"}`` in its place. Everything else is plain
JSON:

- ``None``, booleans, integers, strings and finite floats as themselves;
  non-finite floats as ``{"$float": "nan" | "inf" | "-inf"}``, complex
  numbers as ``{"$complex": [re, im]}``;
- tuples and lists as lists (read back as tuples: gfem values are immutable);
- mappings with string keys as objects (``{"$map": {...}}`` when a key starts
  with ``$``);
- a registered class as ``{"$type": "<tag>", <argument>: <value>, ...}``.

A class is serialized through its constructor arguments. `register` adds a
class: for a dataclass whose constructor takes its fields, the default
(every init field, read back by calling the class) is all it needs; other
classes pass ``to_args`` / ``from_args``, and a ``from_args`` comes with
``names``, the arguments it takes. Reading refuses any other key of a
``$type`` object by name: an entry of a newer gfem is never dropped.
Argument names should be the pytree attribute names, so that the sidecar
keys are the leaf paths.

Reading never imports or calls anything that is not registered, and the
sidecar is read with ``allow_pickle=False``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer

#: The format version of job.json and result.json. Bump it, and add a
#: migration, only when this gfem cannot read an older file as it is: an entry
#: renamed, removed or stored differently, or a constructor default that
#: changes. A class that gains an entry with a default needs neither: older
#: files read through the default, and an older gfem refuses the entry it
#: does not know by name (`Decoder`).
#: Version 2 adds the mesh stored as arrays (a refined mesh, which has no file).
SCHEMA_VERSION = 2

#: ``MIGRATIONS[v]`` upgrades a document of schema version ``v`` to ``v + 1``.
MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    1: lambda doc: doc,  # version 2 only adds a mesh form; 1 reads as it is
}


@dataclasses.dataclass(frozen=True)
class Context:
    """What encoders and decoders may need besides the value.

    Attributes:
        base: The directory of the JSON document; file references (the mesh)
            are relative to it when they lie inside it.
        comm: The communicator a reader was given, as it was given.
        key: Encode for the solve key: a mesh by its in-memory topology,
            never by a file path or the file's sha256.
        model_mesh: The model's mesh, which an encoder may write as a
            reference instead of inline (the solve cache's entries).
    """

    base: Path
    comm: Any = None
    key: bool = False
    model_mesh: Any = None


class UnencodableError(TypeError):
    """A value of a type that is not registered, a subclass included."""


ToArgs = Callable[[Any, Context], Mapping[str, Any]]
FromArgs = Callable[[dict[str, Any], Context], Any]


@dataclasses.dataclass(frozen=True)
class _Codec:
    cls: type
    tag: str
    to_args: ToArgs
    from_args: FromArgs
    names: frozenset[str]


_BY_TYPE: dict[type, _Codec] = {}
_BY_TAG: dict[str, _Codec] = {}


def register(
    cls: type,
    *,
    tag: str | None = None,
    to_args: ToArgs | None = None,
    from_args: FromArgs | None = None,
    names: Iterable[str] | None = None,
) -> None:
    """Make a class serializable.

    Args:
        cls: The class.
        tag: Its name in JSON (``"$type"``); defaults to the class name. It is
            part of the file format: never rename it.
        to_args: ``(value, context) -> {argument: value}``; defaults to the
            dataclass's init fields.
        from_args: ``(arguments, context) -> value``; defaults to
            ``cls(**arguments)``. Arguments absent from an older file are
            left out, so the constructor's defaults apply.
        names: The arguments ``from_args`` takes; a file with any other is
            refused. Defaults to the dataclass's init fields, for the default
            ``from_args`` only.

    Raises:
        TypeError: Not a dataclass and no ``to_args``, or a ``from_args``
            without ``names``.
        ValueError: The tag is taken.
    """
    tag = tag or cls.__name__
    if tag in _BY_TAG and _BY_TAG[tag].cls is not cls:
        raise ValueError(f"Serialization tag {tag!r} is already taken.")
    if to_args is None:
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} is not a dataclass: give to_args.")
        written = _init_fields(cls)

        def to_args(obj: Any, _: Context) -> dict[str, Any]:
            return {n: getattr(obj, n) for n in written}

    if names is None:
        if from_args is not None or not dataclasses.is_dataclass(cls):
            raise TypeError(
                f"{cls.__name__} is read by its own from_args: give names, the "
                "arguments it takes."
            )
        names = _init_fields(cls)
    if from_args is None:

        def from_args(args: dict[str, Any], _: Context) -> Any:
            return cls(**args)

    codec = _Codec(cls, tag, to_args, from_args, frozenset(names))
    _BY_TYPE[cls] = codec
    _BY_TAG[tag] = codec


def _init_fields(cls: Any) -> tuple[str, ...]:
    return tuple(f.name for f in dataclasses.fields(cls) if f.init)


# --- encoding -------------------------------------------------------------------


class Encoder:
    """Encodes values to JSON, collecting their arrays by leaf path."""

    def __init__(self, context: Context) -> None:
        """An empty encoder writing file references relative to ``context``."""
        self.context = context
        self.arrays: dict[str, np.ndarray] = {}

    def encode(self, value: Any, path: str = "") -> Any:  # noqa: PLR0911
        """The JSON form of ``value``, whose pytree path is ``path``."""
        if isinstance(value, Tracer):
            raise TypeError(
                f"Cannot write {path or 'the value'}: it is computed inside "
                "jax.grad, jax.jit or jax.vmap. Write jobs and results from "
                "concrete values."
            )
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return _float(float(value))
        if isinstance(value, (complex, np.complexfloating)):
            return {"$complex": [_float(value.real), _float(value.imag)]}
        if isinstance(value, (jax.Array, np.ndarray)):
            self.arrays[path] = np.asarray(value)
            out: dict[str, Any] = {"$array": path}
            if isinstance(value, np.ndarray):
                out["numpy"] = True
            return out
        codec = _BY_TYPE.get(type(value))
        if codec is not None:
            args = codec.to_args(value, self.context)
            return {"$type": codec.tag} | self._mapping(args, path, attrs=True)
        if isinstance(value, (tuple, list)):
            return [self.encode(v, f"{path}[{i}]") for i, v in enumerate(value)]
        if isinstance(value, Mapping):
            obj = self._mapping(value, path, attrs=False)
            return {"$map": obj} if any(k.startswith("$") for k in obj) else obj
        raise UnencodableError(
            f"Cannot write {path or 'the value'}: {type(value).__name__} is not "
            "a serializable gfem value."
        )

    def _mapping(self, value: Mapping, path: str, *, attrs: bool) -> dict[str, Any]:
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Cannot write {path}: key {key!r} is not a string.")
            sub = f"{path}.{key}" if attrs else f"{path}[{key!r}]"
            out[key] = self.encode(item, sub)
        return out


def _float(x: float) -> float | dict[str, str]:
    return x if math.isfinite(x) else {"$float": repr(x)}


# --- decoding -------------------------------------------------------------------


class Decoder:
    """Decodes JSON written by `Encoder`, reading arrays from a table."""

    def __init__(
        self,
        context: Context,
        arrays: Mapping[str, np.ndarray],
        *,
        path: Path | None = None,
        written_by: str | None = None,
    ) -> None:
        """A decoder resolving file references against ``context``.

        Args:
            context: Where the document is, and the reader's communicator.
            arrays: The sidecar's arrays by leaf path.
            path: The document's file, for error messages.
            written_by: The document's ``gfem_version``, for error messages.
        """
        self.context = context
        self.arrays = arrays
        self.path = path
        self.written_by = written_by

    def decode(self, value: Any) -> Any:  # noqa: PLR0911
        """The value of a JSON form."""
        if isinstance(value, list):
            return tuple(self.decode(v) for v in value)
        if not isinstance(value, dict):
            return value
        if "$type" in value:
            tag = value["$type"]
            codec = _BY_TAG.get(tag)
            if codec is None:
                raise ValueError(
                    f"The file uses the type {tag!r}, which this gfem "
                    f"({version('gfem')}) does not know: upgrade gfem."
                )
            unknown = sorted(value.keys() - codec.names - {"$type"})
            if unknown:
                raise ValueError(self._unknown(tag, unknown))
            args = {k: self.decode(v) for k, v in value.items() if k != "$type"}
            return codec.from_args(args, self.context)
        if "$array" in value:
            key = value["$array"]
            if key not in self.arrays:
                raise ValueError(f"The array {key!r} is missing from the .npz sidecar.")
            arr = self.arrays[key]
            return np.array(arr) if value.get("numpy") else jnp.asarray(arr)
        if "$float" in value:
            return float(value["$float"])
        if "$complex" in value:
            re, im = (self.decode(v) for v in value["$complex"])
            return complex(re, im)
        if "$map" in value:
            value = value["$map"]
        return {k: self.decode(v) for k, v in value.items()}

    def _unknown(self, tag: str, keys: list[str]) -> str:
        """The refusal of entries the class of ``tag`` does not take."""
        this = version("gfem")
        where = "" if self.path is None else f"{self.path}: "
        listed = ", ".join(repr(k) for k in keys)
        head = f"{where}{tag} has no {listed} in this gfem ({this})"
        if self.written_by in (None, this):
            return f"{head}: remove the entry or check its spelling."
        return (
            f"{head}. The file was written by gfem {self.written_by}: upgrade "
            "gfem (pip install -U gfem)."
        )


# --- documents ------------------------------------------------------------------


def header(fmt: str) -> dict[str, Any]:
    """The common head of a gfem document of format ``fmt``."""
    return {
        "format": fmt,
        "schema_version": SCHEMA_VERSION,
        "gfem_version": version("gfem"),
    }


def check_header(doc: Any, fmt: str, what: str, path: Path) -> dict:
    """Check a document's head and migrate it to the current schema.

    Raises:
        ValueError: Not a ``fmt`` document, or one written by a newer gfem.
    """
    v = doc.get("schema_version") if isinstance(doc, dict) else None
    if (
        not isinstance(doc, dict)
        or doc.get("format") != fmt
        or not isinstance(v, int)
        or isinstance(v, bool)
        or v < 1
    ):
        raise ValueError(f"{path} is not a {what} (format {fmt!r}).")
    if v > SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {v} (written by gfem "
            f"{doc.get('gfem_version', '?')}), but this gfem ({version('gfem')}) "
            f"reads schema_version {SCHEMA_VERSION} and older: upgrade gfem "
            "(pip install -U gfem)."
        )
    while v < SCHEMA_VERSION:
        doc = MIGRATIONS[v](doc)
        v += 1
        doc["schema_version"] = v
    return doc
