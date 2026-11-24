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
`gfem._sources.polarization_vector`. In a stack of several layers, or over
a ground plane, it is the transfer matrix's field
(`gfem._transfer_matrix.StackWave`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.constants as const

from gfem import _beam
from gfem._background import Background, lossless
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
    "INCIDENT_KINDS",
    "IncidentKind",
    "IncidentPart",
    "amplitude",
    "background_material",
    "check_frequency",
    "check_solvable",
    "field_and_curl",
    "impedance",
    "impedance_at",
    "impedance_of",
    "incidence_material",
    "intensity",
    "kind_of",
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


def incidence_material(source: PlaneWave, background: Background) -> Material:
    """The layer ``source`` comes from (the incidence layer)."""
    return background.layers[incidence_layer(background, source.theta)]


def check_solvable(source: Incident, background: Background) -> None:
    """Raise if this version cannot solve ``source`` in ``background``.

    Raises:
        NotImplementedError: A `gfem.PlaneWave` from a lossy layer, or a
            `gfem.GaussianBeam` in a layered or lossy background.
        ValueError: A `gfem.PlaneWave` at grazing incidence on a layered
            background, or from below its ground plane.
    """
    kind_of(source).check(source, background)


def check_frequency(source: Incident, background: Background, frequency: float) -> None:
    """Raise if ``source`` cannot be normalized at ``frequency``.

    Raises:
        ValueError: A `gfem.GaussianBeam` whose waist is too small there
            (`gfem._beam.check_waist`).
    """
    kind_of(source).check_frequency(source, background, frequency)


def intensity(
    source: Incident, background: Background, frequencies: Sequence[float]
) -> np.ndarray | None:
    r"""The intensity of ``source`` per frequency, W/m², or ``None``.

    A plane wave's :math:`|E_0|^2 / 2\eta` in its incidence layer, a
    beam's at its focus; ``None`` for a field gfem only evaluates (a
    `gfem.IncidentField`), whose intensity the caller gives.
    """
    return kind_of(source).intensity(source, background, frequencies)


def amplitude(source: Incident) -> float | None:
    """The peak field :math:`|E_0|` of a uniform incident wave, V/m, or ``None``.

    ``None`` for a field with no one amplitude (a beam, a function).
    """
    return kind_of(source).amplitude(source)


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
    return kind_of(source).electric(source, background, points, frequency)


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
    r"""The incident magnetic field at ``points``, A/m.

    In closed form where the kind has one (a plane wave in a stack), else
    from the electric field: :math:`H = \nabla\times E / (-i\omega\mu)`
    under :math:`e^{+i\omega t}`, the curl taken by central differences of
    `values` with step :math:`10^{-4}/k`.

    Returns:
        ``(n, 3)`` complex peak phasors.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    return kind_of(source).magnetic(source, background, points, frequency)


def field_and_curl(
    source: Incident, material: Material, points: np.ndarray, frequency: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""``source``'s :math:`E` and :math:`\nabla\times E` in ``material``, NumPy.

    A `gfem.GaussianBeam` in closed form; another field's curl from its
    magnetic field (`magnetic`).
    """
    points = np.asarray(points, float).reshape(-1, 3)
    return kind_of(source).field_and_curl(source, material, points, frequency)


def _differenced(
    source: Incident, background: Background, points: np.ndarray, frequency: float
) -> np.ndarray:
    """`magnetic` by central differences of `values`."""
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


# --- the kinds of incident field ----------------------------------------------------


class IncidentKind:
    """What gfem reads of one kind of incident field (`INCIDENT_KINDS`).

    The defaults are a field gfem only evaluates: solvable in any
    background, with no closed-form magnetic field, intensity or amplitude.
    """

    def check(self, source: Any, background: Background) -> None:
        """Raise if this version cannot solve ``source`` in ``background``."""

    def check_frequency(
        self, source: Any, background: Background, frequency: float
    ) -> None:
        """Raise if ``source`` cannot be normalized at ``frequency``."""

    def electric(
        self, source: Any, background: Background, points: np.ndarray, frequency: float
    ) -> np.ndarray:
        """The electric field at ``(n, 3)`` points, V/m."""
        raise NotImplementedError

    def magnetic(
        self, source: Any, background: Background, points: np.ndarray, frequency: float
    ) -> np.ndarray:
        """The magnetic field at ``(n, 3)`` points, A/m."""
        return _differenced(source, background, points, frequency)

    def field_and_curl(
        self, source: Any, material: Material, points: np.ndarray, frequency: float
    ) -> tuple[np.ndarray, np.ndarray]:
        r""":math:`E` and :math:`\nabla\times E` in ``material``."""
        background = Background(material)
        e = self.electric(source, background, points, frequency)
        h = self.magnetic(source, background, points, frequency)
        mu = const.mu_0 * float(np.real(complex(np.asarray(material.mu_r))))
        return e, -1j * 2 * math.pi * frequency * mu * h

    def intensity(
        self, source: Any, background: Background, frequencies: Sequence[float]
    ) -> np.ndarray | None:
        """The intensity per frequency, W/m², or ``None`` if gfem cannot tell."""

    def amplitude(self, source: Any) -> float | None:
        """The peak field of a uniform wave, V/m, or ``None``."""


class _PlaneWaveKind(IncidentKind):
    def check(self, source: PlaneWave, background: Background) -> None:
        if not layered(background):
            material = background.layers[0]
            if not lossless(material):
                raise NotImplementedError(
                    f"The PlaneWave {source.name!r} comes from a lossy background "
                    f"({material!r}); a plane wave is solved in a lossless one. "
                    "Give the background without tan_delta or conductivity."
                )
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
        if not lossless(material):
            raise NotImplementedError(
                f"The PlaneWave {source.name!r} comes from layer {j} of the "
                f"background stack ({material!r}), which is lossy: its wave would "
                "be inhomogeneous and its intensity undefined. Make the incidence "
                "layer lossless; the other layers may be lossy."
            )

    def electric(
        self,
        source: PlaneWave,
        background: Background,
        points: np.ndarray,
        frequency: float,
    ) -> np.ndarray:
        if layered(background):
            return StackWave(background, source, frequency).electric(points)
        k_hat, _, _ = propagation_basis(source.theta, source.phi)
        e = source.amplitude * polarization_vector(
            source.theta, source.phi, source.polarization
        )
        k = wavenumber(background, frequency)
        return np.exp(-1j * k * (points @ k_hat))[:, None] * e[None, :]

    def magnetic(
        self,
        source: PlaneWave,
        background: Background,
        points: np.ndarray,
        frequency: float,
    ) -> np.ndarray:
        if layered(background):
            return StackWave(background, source, frequency).magnetic(points)
        return super().magnetic(source, background, points, frequency)

    def intensity(
        self,
        source: PlaneWave,
        background: Background,
        frequencies: Sequence[float],
    ) -> np.ndarray:
        eta = impedance_of(incidence_material(source, background))
        return np.full(len(frequencies), abs(source.amplitude) ** 2 / (2 * eta))

    def amplitude(self, source: PlaneWave) -> float:
        return abs(source.amplitude)


class _BeamKind(IncidentKind):
    def check(self, source: GaussianBeam, background: Background) -> None:
        _beam.check_background(source, background)

    def check_frequency(
        self, source: GaussianBeam, background: Background, frequency: float
    ) -> None:
        _beam.check_waist(source, background, frequency)

    def electric(
        self,
        source: GaussianBeam,
        background: Background,
        points: np.ndarray,
        frequency: float,
    ) -> np.ndarray:
        return _beam.values(source, background, points, frequency)

    def field_and_curl(
        self,
        source: GaussianBeam,
        material: Material,
        points: np.ndarray,
        frequency: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _beam.field_and_curl(source, Background(material), points, frequency)

    def intensity(
        self,
        source: GaussianBeam,
        background: Background,
        frequencies: Sequence[float],
    ) -> np.ndarray:
        eta = _beam.impedance(source, background)
        return np.array(
            [
                _beam.focal_field(source, background, f) ** 2 / (2 * eta)
                for f in frequencies
            ]
        )


class _FunctionKind(IncidentKind):
    def electric(
        self,
        source: IncidentField,
        background: Background,  # noqa: ARG002 - the function holds its medium
        points: np.ndarray,
        frequency: float,
    ) -> np.ndarray:
        return _function_values(source, points, frequency)


#: Per kind of incident field (its class), what gfem reads of it.
INCIDENT_KINDS: tuple[tuple[type, IncidentKind], ...] = (
    (PlaneWave, _PlaneWaveKind()),
    (GaussianBeam, _BeamKind()),
    (IncidentField, _FunctionKind()),
)


def kind_of(source: Incident) -> IncidentKind:
    """The reads of ``source``'s kind.

    Raises:
        TypeError: ``source`` is no incident field.
    """
    for cls, kind in INCIDENT_KINDS:
        if isinstance(source, cls):
            return kind
    raise TypeError(f"{type(source).__name__} is not an incident field.")
