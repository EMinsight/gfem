r"""Incident fields as host values: a source in its background stack.

An incident-field column (`gfem.PlaneWave`, `gfem.GaussianBeam`,
`gfem.IncidentField`) enters a
driven solve through its electric field at points (`values`), evaluated on
the host as NumPy values; the magnetic field a wall check reads comes from
the same values by central differences (`magnetic`). A kept field of such a
column carries an `IncidentPart`, from which its incident and scattered
parts are rebuilt on demand rather than stored.

Under :math:`e^{+i\omega t}` a plane wave in a homogeneous, lossless,
isotropic background of relative permittivity :math:`\varepsilon_r` and
permeability :math:`\mu_r` is

.. math::

    E(r) = E_0\,\hat e\,e^{-i k \hat k\cdot r},
    \qquad k = \frac{\omega}{c}\sqrt{\varepsilon_r\mu_r},

with :math:`\hat e` the unit polarization vector of
`gfem._sources.polarization_vector`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import scipy.constants as const

from gfem import _beam
from gfem._background import Background
from gfem._sources import (
    GaussianBeam,
    IncidentField,
    PlaneWave,
    polarization_vector,
    propagation_basis,
)

__all__ = [
    "IncidentPart",
    "background_material",
    "check_solvable",
    "impedance",
    "magnetic",
    "values",
    "wavenumber",
]

#: The central-difference step of `magnetic`, times the wavelength over
#: :math:`2\pi`: its truncation error is about :math:`(k\delta)^2 / 6`, far
#: below the wall check's :math:`10^{-3}`, and its round-off stays near
#: :math:`10^{-12}`.
_STEP = 1e-4

Incident = PlaneWave | GaussianBeam | IncidentField


@dataclasses.dataclass(frozen=True)
class IncidentPart:
    """What a kept field of an incident column needs to rebuild its incident part.

    The incident part is the moment interpolant of the source's field on
    the field's own space, set to zero in the perfectly matched layer: the
    field there is the scattered field (`gfem.conventions`, Incident
    fields). It is evaluated again each time it is read, never stored.

    Attributes:
        source: The incident source.
        background: The model's background stack.
        layers: The perfectly matched layer's regions, where the incident
            part is zero.
        is_incident: The field is the incident part itself
            (``e_field(..., part="incident")``), not the total field.
    """

    source: Incident
    background: Background
    layers: tuple[str, ...] = ()
    is_incident: bool = False


def background_material(background: Background) -> Any:
    """The one layer of a homogeneous background.

    Raises:
        NotImplementedError: The background has several layers or a ground
            plane.
    """
    if len(background.layers) > 1 or background.ground is not None:
        raise NotImplementedError(
            "An incident field in a layered background (several layers or a "
            "ground plane) is not solved in this gfem version; give a "
            "homogeneous background, Model(background=gfem.Material(...))."
        )
    return background.layers[0]


def _lossless(material: Any) -> bool:
    return not np.any(material.tan_delta) and not np.any(material.conductivity)


def check_solvable(source: Incident, background: Background) -> None:
    """Raise if this version cannot solve ``source`` in ``background``.

    Raises:
        NotImplementedError: A layered background, or a `gfem.PlaneWave` or
            `gfem.GaussianBeam` in a lossy background.
    """
    if isinstance(source, GaussianBeam):
        _beam.check_background(source, background)
        return
    material = background_material(background)
    if isinstance(source, PlaneWave) and not _lossless(material):
        raise NotImplementedError(
            f"The PlaneWave {source.name!r} comes from a lossy background "
            f"({material!r}); a plane wave is solved in a lossless one. Give "
            "the background without tan_delta or conductivity."
        )


def wavenumber(background: Background, frequency: float) -> float:
    """The background's real wavenumber at ``frequency``, rad/m."""
    material = background_material(background)
    n = math.sqrt(float(material.eps_r) * float(material.mu_r))
    return 2 * math.pi * frequency / const.c * n


def impedance(background: Background) -> float:
    r"""The wave impedance of the background, ohm.

    :math:`\eta_0\sqrt{\mu_r/\varepsilon_r}`.
    """
    material = background_material(background)
    eta0 = const.mu_0 * const.c
    return eta0 * math.sqrt(float(material.mu_r) / float(material.eps_r))


def values(
    source: Incident,
    background: Background,
    points: np.ndarray,
    frequency: float,
) -> np.ndarray:
    """The incident electric field of ``source`` at ``points``, V/m.

    Args:
        source: The incident source.
        background: The model's background stack.
        points: ``(n, 3)``, m, model coordinates.
        frequency: Hz.

    Returns:
        ``(n, 3)`` complex peak phasors.

    Raises:
        ValueError: An `gfem.IncidentField` function returned a wrong shape,
            a non-numeric array or a non-finite value; a `gfem.GaussianBeam`
            whose waist is too small at ``frequency`` (`gfem._beam`).
        gfem.NonDifferentiableError: An `gfem.IncidentField` function
            computed its value from a JAX tracer.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    if isinstance(source, PlaneWave):
        k_hat, _, _ = propagation_basis(source.theta, source.phi)
        e = source.amplitude * polarization_vector(
            source.theta, source.phi, source.polarization
        )
        k = wavenumber(background, frequency)
        return np.exp(-1j * k * (points @ k_hat))[:, None] * e[None, :]
    if isinstance(source, IncidentField):
        return _function_values(source, points, frequency)
    if isinstance(source, GaussianBeam):
        return _beam.values(source, background, points, frequency)
    check_solvable(source, background)  # raises for what is not solved here
    raise AssertionError(source)  # pragma: no cover


def _function_values(
    source: IncidentField, points: np.ndarray, frequency: float
) -> np.ndarray:
    import jax.errors

    from gfem._engine.moments import incident_values
    from gfem.errors import NonDifferentiableError

    try:
        return incident_values(source.function, points, frequency, source.name)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        raise NonDifferentiableError(
            f"The IncidentField {source.name!r} computes its field from a JAX "
            "tracer: its function is evaluated on the host with NumPy values, "
            "and gfem does not differentiate through it. Close over concrete "
            "values (jax.lax.stop_gradient) instead."
        ) from None


def magnetic(
    source: Incident,
    background: Background,
    points: np.ndarray,
    frequency: float,
) -> np.ndarray:
    r"""The incident magnetic field at ``points``, A/m, from the electric one.

    :math:`H = \nabla\times E / (-i\omega\mu)` under :math:`e^{+i\omega t}`,
    the curl taken by central differences of `values` with step
    :math:`10^{-4}/k`.

    Returns:
        ``(n, 3)`` complex peak phasors.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    k = wavenumber(background, frequency)
    step = _STEP / k
    n = len(points)
    shifted = [points + s * step * np.eye(3)[j] for j in range(3) for s in (1, -1)]
    e = values(source, background, np.concatenate(shifted), frequency)
    e = e.reshape(3, 2, n, 3)
    # row j holds the derivatives of E along x_j
    d = (e[:, 0] - e[:, 1]) / (2 * step)
    curl = np.stack(
        [d[1][:, 2] - d[2][:, 1], d[2][:, 0] - d[0][:, 2], d[0][:, 1] - d[1][:, 0]],
        axis=1,
    )
    material = background_material(background)
    omega = 2 * math.pi * frequency
    return curl / (-1j * omega * const.mu_0 * float(material.mu_r))
