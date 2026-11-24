r"""GDSII export of layer patterns, with no layout library.

`write_gds` writes the minimal record sequence of the Calma GDSII Stream
Format (HEADER ... ENDLIB), one flat cell holding every pattern, reals in the
format's excess-64 base-16 encoding. GDS has no hole record, so each hole joins
its outer ring by a zero-width cut line (a keyhole), as other layout writers do;
a layout tool's merge recovers the holes. Strict readers refuse a BOUNDARY above
4094 points (its XY record must stay within 32 767 bytes), so a larger ring is
bisected along database-grid lines until every piece fits.

Coordinates are integers in database units, so the geometry here is exact
integer arithmetic: snapping is the only rounding, plus the keyhole and cut
points, each within half a database unit of the exact point.
"""

from __future__ import annotations

import numbers
import re
import struct
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from gfem._pattern import LayerPattern, Polygon

__all__ = ["write_gds"]

#: Points per BOUNDARY, the closing repeat not counted: the XY record then holds
#: at most 4095 pairs, 32 764 bytes, within the 32 767 bytes a record's signed
#: length allows (strict readers refuse more, others warn).
_MAX_VERTICES = 4094
#: The user unit, fixed: one readers ignore (the database unit in metres is
#: what places the vertices) or use only to display numbers.
_USER_UNIT = 1e-6
#: GDS stamps are calendar fields with no time zone, hence a naive datetime.
_EPOCH = datetime(1970, 1, 1)  # noqa: DTZ001
_CELL = re.compile(r"[A-Za-z0-9_?$]{1,32}")
_INT32 = 2**31 - 1

_HEADER, _BGNLIB, _LIBNAME, _UNITS, _ENDLIB = 0x0002, 0x0102, 0x0206, 0x0305, 0x0400
_BGNSTR, _STRNAME, _ENDSTR = 0x0502, 0x0606, 0x0700
_BOUNDARY, _LAYER, _DATATYPE, _XY, _ENDEL = 0x0800, 0x0D02, 0x0E02, 0x1003, 0x1100


def write_gds(
    path: str | Path,
    layers: Mapping[tuple[int, int], LayerPattern | Sequence[LayerPattern]],
    *,
    cell: str | None = None,
    database_unit: float = 1e-9,
    timestamp: datetime = _EPOCH,
) -> Path:
    r"""Write layer patterns to a GDSII file, one flat cell, for a layout tool or fab.

    Not collective: across ranks a script calls it on rank 0.

    Each pattern goes on its ``(layer, datatype)`` key; layer numbers come from
    the foundry's layer map, so there is no default. One key may hold several
    patterns (two design regions on one layer, written side by side, not
    merged), and several keys several layers or variants (e.g. nominal,
    eroded and dilated on datatypes 0, 1, 2 for review; the fab gets the
    nominal). The polygons stay in the mesh's xy frame: the cell placed at
    the origin of the layout that built the mesh abuts the parts drawn
    there (feed guides), which gfem does not write.

    Every vertex is snapped to the database grid; repeated and collinear
    vertices are then dropped, and a ring left with fewer than 3 vertices
    (narrower than one database unit) is dropped. GDS has no holes: each
    hole joins its outer ring through a zero-width cut line (a keyhole),
    which a layout tool's merge removes. A ring above 4094 vertices is cut
    along database-grid lines into pieces that fit, which abut exactly.

    The user unit is 1 µm, the library name ``gfem``. The same arguments
    give the same bytes: the file is stamped with ``timestamp``, not the
    time of writing.

    Args:
        path: The file; its directory must exist. An existing file is
            overwritten.
        layers: ``{(layer, datatype): pattern}``, each value a
            `LayerPattern` or a sequence of them; layer and datatype are
            integers in :math:`[0, 32767]`. A pattern with no polygon writes
            nothing.
        cell: The cell name, 1 to 32 characters from ``A-Z a-z 0-9 _ ? $``.
            Default: the file name's stem, other characters replaced by
            ``_``, cut to 32 (``bend-v2.gds`` gives ``bend_v2``).
        database_unit: The database grid step, m, in :math:`(0, 10^{-6}]`:
            finer for e-beam, or a foundry's manufacturing grid (``5e-9``
            puts every vertex on 5 nm).
        timestamp: The modification and access time written in the file;
            ``datetime.now()`` stamps the time of writing.

    Returns:
        The path written.

    Raises:
        TypeError: ``layers`` is not a mapping, a value is not a
            `LayerPattern` or a sequence of them (a density: pass
            ``grid.pattern(filtered, eta=...)``), ``cell`` is not a string,
            ``database_unit`` not a number, or ``timestamp`` not a
            `datetime.datetime`.
        ValueError: ``layers`` is empty, a key is not a ``(layer,
            datatype)`` pair of integers in :math:`[0, 32767]`, ``cell`` is
            not a GDS cell name, ``database_unit`` is outside
            :math:`(0, 10^{-6}]` m, a vertex is not finite or lies beyond
            the :math:`\pm 2^{31}` database units a GDS file holds, or a
            hole does not lie inside its outer ring. Raised before the file
            is opened: a failed call leaves no file.
        FileNotFoundError: The file's directory does not exist.
    """
    path = Path(path)
    if not isinstance(layers, Mapping):
        raise TypeError(
            "write_gds: layers maps (layer, datatype) to layer patterns, e.g. "
            f"{{(1, 0): pattern}}; got {type(layers).__name__}."
        )
    if not layers:
        raise ValueError(
            "write_gds: no layer pattern to write; layers is empty. Pass "
            "{(layer, datatype): pattern}."
        )
    groups = [(_key(key), _patterns(key, value)) for key, value in layers.items()]
    name = _cell(path, cell)
    unit = _database_unit(database_unit)
    if not isinstance(timestamp, datetime):
        raise TypeError(
            "write_gds: timestamp is a datetime.datetime, e.g. datetime(1970, 1, 1) "
            f"or datetime.now(); got {type(timestamp).__name__}."
        )
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"write_gds: the directory {str(path.parent)!r} does not exist; create "
            "it first."
        )
    elements = []
    for (layer, datatype), patterns in groups:
        tags = _record(_LAYER, struct.pack(">h", layer)) + _record(
            _DATATYPE, struct.pack(">h", datatype)
        )
        for p, pattern in enumerate(patterns):
            of = f" of pattern {p}" if len(patterns) > 1 else ""
            for k, polygon in enumerate(pattern.polygons):
                where = f"polygon {k}{of} on layer ({layer}, {datatype})"
                for ring in _boundaries(polygon, unit, where):
                    xy = np.vstack([ring, ring[:1]]).astype(">i4").tobytes()
                    elements.append(
                        _record(_BOUNDARY) + tags + _record(_XY, xy) + _record(_ENDEL)
                    )
    stamp = _stamp(timestamp)
    data = b"".join(
        [
            _record(_HEADER, struct.pack(">h", 600)),
            _record(_BGNLIB, stamp),
            _record(_LIBNAME, _ascii("gfem")),
            _record(_UNITS, _real8(unit / _USER_UNIT) + _real8(unit)),
            _record(_BGNSTR, stamp),
            _record(_STRNAME, _ascii(name)),
            *elements,
            _record(_ENDSTR),
            _record(_ENDLIB),
        ]
    )
    path.write_bytes(data)
    return path


# --- arguments ----------------------------------------------------------------


def _key(key: object) -> tuple[int, int]:
    if (
        isinstance(key, tuple)
        and len(key) == 2
        and all(
            isinstance(v, int | np.integer)
            and not isinstance(v, bool)
            and 0 <= v <= 32767
            for v in key
        )
    ):
        return int(key[0]), int(key[1])
    raise ValueError(
        f"write_gds: the key {key!r} is not a (layer, datatype) pair of integers in "
        "[0, 32767]; layer numbers come from the foundry's layer map, e.g. "
        "{(1, 0): pattern}."
    )


def _patterns(key: object, value: object) -> tuple[LayerPattern, ...]:
    if isinstance(value, LayerPattern):
        return (value,)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | Polygon)
        and all(isinstance(v, LayerPattern) for v in value)
    ):
        return tuple(value)
    raise TypeError(
        f"write_gds: the value for {key!r} is a {type(value).__name__}, not a "
        "LayerPattern or a sequence of them. Draw a density as polygons first, "
        "grid.pattern(filtered, eta=...), and your own polygons with "
        "LayerPattern.from_polygons."
    )


def _cell(path: Path, cell: object) -> str:
    if cell is None:
        cell = re.sub(r"[^A-Za-z0-9_?$]", "_", path.stem)[:32]
    elif not isinstance(cell, str):
        raise TypeError(f"write_gds: cell is a str; got {type(cell).__name__}.")
    if not _CELL.fullmatch(cell):
        raise ValueError(
            f"write_gds: the cell name {cell!r} is not a GDS cell name: 1 to 32 "
            "characters from A-Z a-z 0-9 _ ? $."
        )
    return cell


def _database_unit(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            "write_gds: database_unit is a length in metres, e.g. 1e-9; got "
            f"{type(value).__name__}."
        )
    unit = float(value)
    if not 0 < unit <= _USER_UNIT:  # also refuses NaN; inf is above the bound
        raise ValueError(
            f"write_gds: database_unit must be in (0, 1e-6] m, got {value}; "
            "e.g. 1e-9 for a 1 nm grid."
        )
    return unit


# --- records ------------------------------------------------------------------


def _record(kind: int, payload: bytes = b"") -> bytes:
    return struct.pack(">HH", 4 + len(payload), kind) + payload


def _ascii(text: str) -> bytes:
    data = text.encode("ascii")
    return data + b"\0" if len(data) % 2 else data


def _real8(x: float) -> bytes:
    r"""The GDSII 8-byte real: sign bit, excess-64 base-16 exponent, 56-bit mantissa.

    :math:`x = \pm m \, 16^{e - 64}` with :math:`1/16 \le m < 1`, the
    mantissa :math:`m \, 2^{56}` rounded to the nearest integer.
    """
    if x == 0:
        return bytes(8)
    sign, m, e = (0x80 if x < 0 else 0), abs(x), 0
    while m >= 1:  # scaling by 16 is exact in binary floating point
        m, e = m / 16, e + 1
    while m < 1 / 16:
        m, e = m * 16, e - 1
    mantissa = round(m * 2**56)
    if mantissa == 2**56:  # rounded up to 1: renormalize
        mantissa, e = mantissa // 16, e + 1
    return struct.pack(">Q", ((sign | (e + 64)) << 56) | mantissa)


def _stamp(t: datetime) -> bytes:
    """BGNLIB / BGNSTR: modification then access time, full four-digit year."""
    return struct.pack(">6h", t.year, t.month, t.day, t.hour, t.minute, t.second) * 2


# --- geometry, in integer database units --------------------------------------


def _boundaries(polygon: Polygon, unit: float, where: str) -> list[np.ndarray]:
    """The polygon as open integer rings, at most `_MAX_VERTICES` points each."""
    outer = _snap(polygon.exterior, unit, where)
    if len(outer) < 3:  # narrower than a database unit
        return []
    if _area2(outer) < 0:
        outer = outer[::-1]
    holes = []
    for i, hole in enumerate(polygon.holes):
        ring = _snap(hole, unit, where)
        if len(ring) >= 3:
            holes.append((i, ring[::-1] if _area2(ring) > 0 else ring))
    return _split(_keyhole(outer, holes, where), _MAX_VERTICES)


def _snap(ring: np.ndarray, unit: float, where: str) -> np.ndarray:
    q = np.rint(np.asarray(ring, dtype=float) / unit)
    if not np.isfinite(q).all():
        raise ValueError(f"write_gds: {where} has a non-finite vertex.")
    if np.abs(q).max(initial=0) > _INT32:
        raise ValueError(
            f"write_gds: {where} has a vertex beyond the 2^31 database units a GDS "
            f"file holds ({_INT32 * unit:.3g} m from the origin at database_unit="
            f"{unit:g} m). Lengths are in metres; or choose a coarser database_unit."
        )
    return _clean(q.astype(np.int64))


def _clean(ring: np.ndarray) -> np.ndarray:
    """An open ring without repeated, then collinear vertices, until stable.

    The collinear test is an exact integer cross product, so it also drops
    zero-area spikes. Differences of int32 coordinates reach :math:`2^{32}`, so
    their products are taken as Python integers past :math:`2^{30}`.
    """
    if len(ring) and np.abs(ring).max() >= 2**30:
        ring = ring.astype(object)
    while len(ring) >= 3:
        ring = _drop_repeats(ring)
        if len(ring) < 3:
            break
        a = np.roll(ring, 1, axis=0) - ring
        b = np.roll(ring, -1, axis=0) - ring
        keep = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0] != 0
        if keep.all():
            return ring.astype(np.int64)
        ring = ring[keep]
    return np.zeros((0, 2), dtype=np.int64)


def _drop_repeats(ring: np.ndarray) -> np.ndarray:
    """An open ring without consecutive repeats (the last against the first too).

    Of consecutive repeats the cyclically last copy stays, so a closed ring
    loses its closing repeat and keeps its first vertex.
    """
    return ring[np.any(ring != np.roll(ring, -1, axis=0), axis=1)]


def _area2(ring: np.ndarray) -> int:
    """Twice the signed area of an open integer ring, exactly (Python integers)."""
    x, y = ring[:, 0].astype(object), ring[:, 1].astype(object)
    return int(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _keyhole(
    ring: np.ndarray, holes: list[tuple[int, np.ndarray]], where: str
) -> np.ndarray:
    """Join each hole to the ring by a horizontal cut from its rightmost vertex.

    Holes go in decreasing rightmost x, so a cut ends on the outer ring or on
    a hole already joined: the cut's open segment lies in the solid.
    """
    for i, hole in sorted(holes, key=lambda h: -int(h[1][:, 0].max())):
        start = int(np.argmax(hole[:, 0]))
        px, py = hole[start]
        a, b = ring, np.roll(ring, -1, axis=0)
        dy = b[:, 1] - a[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (py - a[:, 1]) / dy
            x = a[:, 0] + t * (b[:, 0] - a[:, 0])
        hit = (dy != 0) & (t >= 0) & (t <= 1) & (x >= px)
        if not hit.any():
            raise ValueError(
                f"write_gds: hole {i} of {where} does not lie inside its outer ring."
            )
        j = int(np.flatnonzero(hit)[np.argmin(x[hit])])
        cut = np.array([[np.rint(x[j]), py]], dtype=np.int64)
        loop = np.roll(hole, -start, axis=0)
        ring = np.vstack([ring[: j + 1], cut, loop, loop[:1], cut, ring[j + 1 :]])
        ring = _drop_repeats(ring)  # a cut ending on a vertex
    return ring


def _split(ring: np.ndarray, cap: int, depth: int = 0) -> list[np.ndarray]:
    """Bisect a ring above ``cap`` points along database-grid lines until each fits.

    The cut is across the bounding box's longer side, at the median vertex
    coordinate: each half then has about half the vertices.
    """
    if len(ring) <= cap:
        return [ring]
    if depth > 64:  # halving: a valid ring needs far fewer levels
        raise RuntimeError("write_gds: could not split a polygon under the vertex cap.")
    low, high = ring.min(axis=0), ring.max(axis=0)
    axis = int(np.argmax(high - low))
    cut = int(np.median(ring[:, axis]))
    if not low[axis] < cut < high[axis]:  # most vertices on one side of the box
        cut = int(low[axis] + high[axis]) // 2
    pieces = []
    for keep_low in (True, False):
        piece = _clip(ring, axis, cut, keep_low)
        if len(piece) >= 3 and _area2(piece) != 0:
            pieces += _split(piece, cap, depth + 1)
    return pieces


def _clip(ring: np.ndarray, axis: int, cut: int, keep_low: bool) -> np.ndarray:
    """The ring's part on one side of the line ``ring[:, axis] == cut``.

    Sutherland-Hodgman against one half-plane. A crossing point is rounded
    from the edge's lower-coordinate end, so both halves get the same point
    and abut exactly. Where the half-plane separates two parts, they stay
    joined by a zero-width bridge along the cut line, like a keyhole.
    """
    s = ring[:, axis] - cut
    inside = s <= 0 if keep_low else s >= 0
    q = np.roll(ring, -1, axis=0)
    sq = np.roll(s, -1)
    cross = ((s < 0) & (sq > 0)) | ((sq < 0) & (s > 0))
    swap = (ring[:, axis] > q[:, axis])[:, None]
    lo, hi = np.where(swap, q, ring), np.where(swap, ring, q)
    span = np.where(cross, hi[:, axis] - lo[:, axis], 1)
    t = np.where(cross, (cut - lo[:, axis]) / span, 0.0)
    point = np.empty_like(ring)
    point[:, axis] = cut
    other = 1 - axis
    point[:, other] = np.rint(lo[:, other] + t * (hi[:, other] - lo[:, other]))
    both = np.stack([ring, point], axis=1).reshape(-1, 2)
    return _clean(both[np.stack([inside, cross], axis=1).ravel()])
