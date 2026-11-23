r"""Incident fields as host values: a source in its background stack.

An incident-field column (`gfem.PlaneWave`, `gfem.IncidentField`) enters a
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
`gfem._sources.polarization_vector`. In a stack of several layers, or over
a ground plane, it is the transfer matrix's field
(`gfem._transfer_matrix.StackWave`).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import scipy.constants as const

from gfem._background import Background
from gfem._materials import Material
from gfem._sources import (
    GaussianBeam,
    IncidentField,
    PlaneWave,
    polarization_vector,
    propagation_basis,
)
from gfem._transfer_matrix import StackWave, incidence_layer, layer_index, number

__all__ = [
    "IncidentPart",
    "background_material",
    "check_solvable",
    "impedance",
    "impedance_at",
    "impedance_of",
    "incidence_material",
    "layered",
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


def layered(background: Background) -> bool:
    """Whether the stack has several layers or a ground plane."""
    return len(background.layers) > 1 or background.ground is not None


def background_material(background: Background) -> Any:
    """The one layer of a homogeneous background.

    Raises:
        NotImplementedError: The background has several layers or a ground
            plane.
    """
    if layered(background):
        raise NotImplementedError(
            "This measurement of an incident field in a layered background "
            "(several layers or a ground plane) is not made in this gfem "
            "version; give a homogeneous background, "
            "Model(background=gfem.Material(...))."
        )
    return background.layers[0]


def _lossless(material: Any) -> bool:
    return not np.any(material.tan_delta) and not np.any(material.conductivity)


def incidence_material(source: PlaneWave, background: Background) -> Material:
    """The layer ``source`` comes from (the incidence layer)."""
    return background.layers[incidence_layer(background, source.theta)]


def _check_layered(source: Incident, background: Background) -> None:
    if isinstance(source, GaussianBeam):
        raise NotImplementedError(
            f"The GaussianBeam {source.name!r} meets a background of several "
            "layers or a ground plane, where gfem does not solve a beam in this "
            "version. Feed the beam from its homogeneous side and read it with "
            "gfem.beam_coupling (reciprocity), or give its field in the stack "
            "as a gfem.IncidentField."
        )
    if not isinstance(source, PlaneWave):
        return
    if abs(math.cos(source.theta)) < 1e-12:
        raise ValueError(
            f"The PlaneWave {source.name!r} is at grazing incidence (theta = "
            "pi/2) on a layered background, where it has no incidence layer; "
            "tilt it off the stack's planes."
        )
    if background.ground is not None and math.cos(source.theta) > 0:
        raise ValueError(
            f"The PlaneWave {source.name!r} travels up (theta < pi/2), from "
            f"below the ground plane at z = {background.ground} m, where there "
            "is no field; give it from above (theta > pi/2)."
        )
    j = incidence_layer(background, source.theta)
    material = background.layers[j]
    if not _lossless(material):
        raise NotImplementedError(
            f"The PlaneWave {source.name!r} comes from layer {j} of the "
            f"background stack ({material!r}), which is lossy: its wave would be "
            "inhomogeneous and its intensity undefined. Make the incidence "
            "layer lossless; the other layers may be lossy."
        )


def check_solvable(source: Incident, background: Background) -> None:
    """Raise if this version cannot solve ``source`` in ``background``.

    Raises:
        NotImplementedError: A `gfem.GaussianBeam`, or a `gfem.PlaneWave`
            from a lossy layer.
        ValueError: A `gfem.PlaneWave` at grazing incidence on a layered
            background, or from below its ground plane.
    """
    if layered(background):
        _check_layered(source, background)
        return
    if isinstance(source, GaussianBeam):
        raise NotImplementedError(
            f"The GaussianBeam {source.name!r} is not solved in this gfem "
            "version; give its field as a gfem.IncidentField."
        )
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
    return impedance_of(material)


def impedance_of(material: Material) -> float:
    """The lossless wave impedance of ``material``, ohm."""
    eta0 = const.mu_0 * const.c
    return eta0 * math.sqrt(number(material.mu_r) / number(material.eps_r))


def impedance_at(background: Background, points: np.ndarray) -> np.ndarray:
    r"""The lossless wave impedance of the stack layer at each point, ohm.

    :math:`\eta_0\sqrt{\mu_r/\varepsilon_r}` of the layer's real
    :math:`\varepsilon_r` and :math:`\mu_r`: the scale of a magnetic
    field against an electric one.
    """
    per_layer = np.array([impedance_of(m) for m in background.layers])
    return per_layer[layer_index(background, np.asarray(points)[:, 2])]


def _stack_wave(source: PlaneWave, background: Background, frequency: float) -> Any:
    return StackWave(background, source, frequency)


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
            a non-numeric array or a non-finite value.
        gfem.NonDifferentiableError: An `gfem.IncidentField` function
            computed its value from a JAX tracer.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    if isinstance(source, PlaneWave) and layered(background):
        return _stack_wave(source, background, frequency).electric(points)
    if isinstance(source, PlaneWave):
        k_hat, _, _ = propagation_basis(source.theta, source.phi)
        e = source.amplitude * polarization_vector(
            source.theta, source.phi, source.polarization
        )
        k = wavenumber(background, frequency)
        return np.exp(-1j * k * (points @ k_hat))[:, None] * e[None, :]
    if isinstance(source, IncidentField):
        return _function_values(source, points, frequency)
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
    if isinstance(source, PlaneWave) and layered(background):
        return _stack_wave(source, background, frequency).magnetic(points)
    n_max = max(math.sqrt(number(m.eps_r) * number(m.mu_r)) for m in background.layers)
    k = 2 * math.pi * frequency / const.c * n_max
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
    mu_r = np.array([number(m.mu_r) for m in background.layers])
    mu_r = mu_r[layer_index(background, points[:, 2])]
    omega = 2 * math.pi * frequency
    return curl / (-1j * omega * const.mu_0 * mu_r[:, None])
