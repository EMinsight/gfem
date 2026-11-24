"""Angle sweeps: one fixed-angle frequency sweep per incidence on a Floquet surface."""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from gfem import _floquet, _progress
from gfem._checks import check_fixed
from gfem._pytree import register_fields
from gfem._text import hertz_band
from gfem.errors import UnrequestedOrderWarning

if TYPE_CHECKING:
    import os

    from numpy.typing import ArrayLike

    from gfem.model import Model
    from gfem.options import SolverOptions
    from gfem.results import DrivenResult

__all__ = ["AngleSweepResult", "angle_sweep"]

_TRACED = (
    "angle_sweep: theta and phi must be fixed numbers: each incidence is a "
    "constant of its solve, not a design variable"
)
#: At most this many incidences are listed in the unrequested-order warning.
_LISTED = 4


def _deg(x: float) -> float:
    return math.degrees(x)


def _label(theta: float, phi: float) -> str:
    return f"theta = {_deg(theta):.6g}°, phi = {_deg(phi):.6g}°"


@dataclasses.dataclass(frozen=True, eq=False)
class AngleSweepResult(collections.abc.Sequence):
    r"""The driven results of a list of incidences on one Floquet surface.

    A `collections.abc.Sequence` of `gfem.DrivenResult`, one per incidence in
    the order asked, so every cell output of `gfem.diffraction`
    (`efficiencies`, `power_balance`, `jones`) takes it as it is and stacks
    its members on a leading incidence axis. Incidences with the same wave
    vector (:math:`\theta = 0` at any :math:`\varphi`, a repeated pair) share
    one solve: the same member stands at each of their positions.

    Attributes:
        results: The member results, in the order asked.
        theta: Each incidence's angle from the inward normal of ``surface``,
            rad (static).
        phi: Each incidence's azimuth from :math:`\hat e_1`, rad, as asked
            (static).
        surface: The Floquet surface the light enters through.
        frequency: The frequency grid every member was solved on, Hz.
    """

    results: tuple[DrivenResult, ...]
    theta: tuple[float, ...]
    phi: tuple[float, ...]
    surface: str
    frequency: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.results)

    @overload
    def __getitem__(self, index: int) -> DrivenResult: ...
    @overload
    def __getitem__(self, index: slice) -> AngleSweepResult: ...
    def __getitem__(self, index: Any) -> Any:
        """One member result, or a slice of the sweep with its labels."""
        if isinstance(index, slice):
            return AngleSweepResult(
                self.results[index],
                self.theta[index],
                self.phi[index],
                self.surface,
                self.frequency,
            )
        return self.results[index]

    def __repr__(self) -> str:
        n = len(self.theta)
        f = self.frequency
        band = hertz_band(min(f), max(f), 4) if f else "no frequency"
        angles = (
            f"theta {_deg(min(self.theta)):.4g}° to {_deg(max(self.theta)):.4g}°, "
            f"phi {_deg(min(self.phi)):.4g}° to {_deg(max(self.phi)):.4g}°"
            if n
            else "no incidence"
        )
        return (
            f"AngleSweepResult({n} incidences on {self.surface!r}, {angles}, "
            f"{len(f)} frequencies {band})"
        )


register_fields(AngleSweepResult, ("results",))


def _flat_angles(theta: Any, phi: Any) -> tuple[np.ndarray, np.ndarray]:
    """``theta`` and ``phi`` broadcast to one flat list (SW1)."""
    who = "angle_sweep"
    if np.size(theta) == 0:
        raise ValueError(f"{who}: theta is empty: give at least one incidence.")
    try:
        t, p = np.broadcast_arrays(np.asarray(theta, float), np.asarray(phi, float))
        ok = t.ndim <= 1
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(
            f"{who}: theta and phi must broadcast to one flat list of incidences, "
            f"got shapes {np.shape(theta)} and {np.shape(phi)}; pass a flat "
            "list: ravel a grid (np.meshgrid(...)[k].ravel())."
        )
    return np.atleast_1d(t).astype(float), np.atleast_1d(p).astype(float)


def _excited(model: Model, surface: str, excite: Any) -> Any:
    """The names to excite: the specular Floquet ports of ``surface`` for ``None``."""
    from gfem._boundaries import FloquetPort

    if excite is not None:
        return excite
    names = tuple(
        p.name
        for p in model.ports
        if isinstance(p, FloquetPort) and p.surface == surface and p.order == (0, 0)
    )
    if not names:
        raise ValueError(
            f"angle_sweep: excite=None excites the specular Floquet ports on "
            f"{surface!r}, but {surface!r} has none. Declare them, e.g. with "
            f"gfem.diffraction.floquet_ports(model, {surface!r}, frequency, ...), "
            "or pass excite=."
        )
    return names


def _unrequested(
    tables: Sequence[dict[str, Any]], labels: Sequence[str]
) -> list[tuple[str, tuple[int, int], tuple[str, ...], list[str]]]:
    """The (surface, order, polarizations) that propagate without a port, W2.

    ``labels[i]`` names the incidence of ``tables[i]``; each entry lists the
    incidences where the order propagates at some frequency.
    """
    found: dict[tuple[str, tuple[int, int]], tuple[tuple[str, ...], list[str]]] = {}
    for table, label in zip(tables, labels, strict=True):
        for surface, so in table.items():
            declared = {(p.order, p.polarization) for p in so.ports}
            propagating = so.rows.propagating
            for k, order in enumerate(so.orders):
                if not propagating[:, k].any():
                    continue
                pols = tuple(q for q in ("TE", "TM") if (order, q) not in declared)
                if pols:
                    found.setdefault((surface, order), (pols, []))[1].append(label)
    return [(s, o, pols, where) for (s, o), (pols, where) in found.items()]


def _warn_unrequested(
    listing: list[tuple[str, tuple[int, int], tuple[str, ...], list[str]]],
) -> None:
    if not listing:
        return
    parts = []
    for surface, order, pols, where in sorted(listing, key=lambda r: (r[0], r[1])):
        shown = ", ".join(where[:_LISTED]) + (", ..." if len(where) > _LISTED else "")
        parts.append(f"{order} {' and '.join(pols)} on {surface!r} at {shown}")
    warnings.warn(
        UnrequestedOrderWarning(
            f"angle_sweep: propagating order(s) {'; '.join(parts)} carry power "
            "that S does not report (they are absorbed, not reflected); declare "
            "them for every incidence with gfem.diffraction.floquet_ports(model, "
            "surface, frequency, wave_vector=<the sweep's wave vectors>, "
            "wave_vector_frequency=<its first frequency>), or filter "
            "UnrequestedOrderWarning."
        ),
        stacklevel=3,
    )


def angle_sweep(
    model: Model,
    frequency: ArrayLike,
    *,
    surface: str,
    theta: ArrayLike,
    phi: ArrayLike = 0.0,
    excite: str | Sequence[str] | None = None,
    keep_fields: bool | float | Sequence[float] = False,
    options: SolverOptions | None = None,
    cache: str | os.PathLike | None = None,
) -> AngleSweepResult:
    r"""Solve a list of incidences on one Floquet surface, each over a frequency grid.

    Each incidence :math:`(\theta_i, \varphi_i)` is one fixed-angle frequency
    sweep through `gfem.solve`: the wave vector
    :math:`\mathbf k_i = k_0 n \sin\theta_i (\cos\varphi_i\,\hat e_1 +
    \sin\varphi_i\,\hat e_2)` of `gfem.Model.bloch_wave_vector` at the first
    frequency, scaled with the frequency so that the angle stays put. The
    members stack on a leading axis in `gfem.diffraction.efficiencies`,
    `gfem.diffraction.power_balance` and `gfem.diffraction.jones`.

    :math:`\theta` and :math:`\varphi` broadcast together to one flat list
    (no outer product: build a grid with ``np.meshgrid(...)[k].ravel()``).
    :math:`\theta = 0` solves with plain periodicity, and incidences with the
    same wave vector are solved once. With ``cache`` each incidence is its own
    cache entry, so extending the list solves only the new incidences and an
    interrupted sweep resumes.

    Everything is checked before the first solve: the angles (fixed numbers
    in range), the ``excite`` names, and that every excited Floquet port
    propagates at every incidence and frequency. Propagating orders without a
    Floquet port warn once for the whole grid with
    `gfem.UnrequestedOrderWarning`; declare them for every incidence with
    `gfem.diffraction.floquet_ports` and the sweep's wave vectors.

    Conical incidence:

    - :math:`\varphi` is measured from :math:`\hat e_1`, the translation of
      the first periodic pair, toward :math:`\hat e_2`. For a grating whose
      lines run along :math:`\hat e_2`, :math:`\varphi = 0` is the planar
      mount and :math:`\varphi \ne 0, \pi` is conical.
    - TE and TM are relative to each order's own plane of incidence. In a
      conical mount the diffracted orders leave on a cone outside the plane
      of incidence, so TE and TM couple (the cross-polarized rows are not
      zero); in a planar mount they read zero.
    - At :math:`\theta = 0` the references are :math:`(\hat e_1, \hat e_2)`
      whatever :math:`\varphi`, so a :math:`\theta` scan at
      :math:`\varphi \ne 0` sees the linear TE and TM entries jump at
      :math:`\theta = 0` by the rotation :math:`\varphi`. Efficiencies summed
      over the output polarization, unpolarized reflectance, transmittance
      and absorptance are continuous.

    Cost: one factorization per distinct incidence and frequency, none shared
    across incidences. Outside `jax.grad`, ``SolverOptions(retain_memory=0)``
    stops earlier incidences' factorizations from filling the memory budget;
    separate processes over chunks of the incidence list that share ``cache``
    run the incidences in parallel (entries are published atomically), and
    one call over the whole list then reads them back. A gradient with
    respect to a design (materials, interior nodes) flows through every
    member; the angles are constants.

    Args:
        model: A periodic cell with Floquet ports.
        frequency: One frequency or several, Hz.
        surface: The Floquet surface the light enters through; the angles
            are in the medium behind it.
        theta: Angles from the inward normal of ``surface``, rad,
            :math:`|\theta| \le \pi/2`; one or a flat list.
        phi: Azimuths from :math:`\hat e_1` toward :math:`\hat e_2`, rad.
        excite: The ports to excite, as `gfem.Driven`; ``None`` is every
            Floquet port of ``surface`` with order (0, 0).
        keep_fields: As `gfem.Driven`.
        options: The numerics; ``None`` is ``SolverOptions()``.
        cache: A solve-cache directory, as `gfem.solve`.

    Returns:
        An `AngleSweepResult`.

    Raises:
        NonDifferentiableError: ``theta``, ``phi`` or ``frequency`` is traced
            by JAX.
        ValueError: ``theta`` and ``phi`` do not make one flat non-empty
            list, an angle is out of range, ``excite=None`` and ``surface``
            has no specular Floquet port, an ``excite`` name is not a port,
            or an excited Floquet port does not propagate at some incidence
            and frequency (the message names it).
    """
    import gfem
    from gfem._engine.driven import wave_vectors
    from gfem.options import SolverOptions
    from gfem.problems import Driven

    who = "angle_sweep"
    check_fixed((theta, phi), _TRACED)
    check_fixed(frequency, f"{who}: frequency must be fixed numbers")
    t, p = _flat_angles(theta, phi)
    options = SolverOptions() if options is None else options
    grid = Driven(frequency).frequency
    f_ref = grid[0]
    # the range checks of `bloch_wave_vector`, for every incidence
    vectors = [
        model.bloch_wave_vector(surface, f_ref, theta=float(a), phi=float(b))
        for a, b in zip(t, p, strict=True)
    ]
    names = _excited(model, surface, excite)

    # one solve per distinct wave vector; normal incidence is plain periodicity
    keys = [None if not any(v) else tuple(float(x) + 0.0 for x in v) for v in vectors]
    first: dict[Any, int] = {}
    for i, key in enumerate(keys):
        first.setdefault(key, i)
    problems = {
        key: Driven(
            frequency,
            excite=names,
            keep_fields=keep_fields,
            wave_vector=key,
            wave_vector_frequency=None if key is None else f_ref,
        )
        for key in first
    }
    port_names = model.port_names
    for name in next(iter(problems.values())).excite or ():
        if name not in port_names:
            raise ValueError(
                f"Driven.excite names port {name!r}, but the model's ports are "
                f"{', '.join(map(repr, port_names))}."
            )

    # every excited port propagates everywhere (SW3), before any solve
    tables, labels = [], []
    for key, i in first.items():
        problem = problems[key]
        table = _floquet.order_table(model, grid, wave_vectors(problem, grid), options)
        label = _label(float(t[i]), float(p[i]))
        if table:
            try:
                _floquet_check(table, problem.excite or (), grid)
            except ValueError as error:
                raise ValueError(
                    f"{who}: at incidence {i} ({label}): {error}"
                ) from None
        tables.append(table)
        labels.append(label)
    _warn_unrequested(_unrequested(tables, labels))

    solved: dict[Any, Any] = {}
    with _progress.task(_progress.ANGLE_SWEEP, total=len(first)) as row:
        for key, i in first.items():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UnrequestedOrderWarning)
                solved[key] = gfem.solve(
                    model, problems[key], options=options, cache=cache
                )
            row.advance(theta=_deg(float(t[i])), phi=_deg(float(p[i])))
    return AngleSweepResult(
        results=tuple(solved[key] for key in keys),
        theta=tuple(float(a) for a in t),
        phi=tuple(float(b) for b in p),
        surface=surface,
        frequency=tuple(float(f) for f in grid),
    )


def _floquet_check(table: dict[str, Any], excite: Sequence[str], grid: Any) -> None:
    from gfem._engine.floquet_ports import check_excitable

    check_excitable(table, excite, grid)
