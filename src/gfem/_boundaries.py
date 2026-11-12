"""Boundary conditions and ports: frozen values naming mesh surfaces."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar, Literal

import jax
import numpy as np

from gfem._checks import check_fixed, is_traced, optional_positive, static_float
from gfem._pytree import register_fields
from gfem._sheet import SheetField

_FIXED = (
    "Of the boundary and port values, gfem differentiates a lumped port's "
    "inductance and capacitance and an Impedance's resistance, inductance and "
    "capacitance only; wrap this value in jax.lax.stop_gradient or keep it a "
    "plain number."
)
#: The lumped values of a lumped port or an `Impedance`, with their units.
_RLC = (("resistance", "ohm"), ("inductance", "H"), ("capacitance", "F"))


def _surface_name(value: Any, owner: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"{owner}: surface must be one surface name (a string), got {value!r}. "
            "Give one boundary condition per surface."
        )
    return value


def _check_fill(fill: Any, owner: str) -> None:
    if fill is not None and not isinstance(fill, SheetField):
        raise TypeError(
            f"{owner}: fill takes a gfem.SheetField (one fill value per face of "
            f"the surface), got {type(fill).__name__}; build one with "
            "gfem.SheetField(fill=values) or gfem.SheetField.interpolate(density), "
            "or leave fill=None for the solid sheet."
        )


@dataclasses.dataclass(frozen=True)
class PEC:
    """Perfect electric conductor on a surface: n x E = 0.

    Attributes:
        surface: The surface region's name.
    """

    surface: str
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "PEC")


@dataclasses.dataclass(frozen=True)
class PMC:
    """Perfect magnetic conductor (symmetry plane) on a surface: n x H = 0.

    Exterior surfaces that no boundary condition or port names get this
    condition by default in electromagnetic problems.

    Attributes:
        surface: The surface region's name.
    """

    surface: str
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "PMC")


@dataclasses.dataclass(frozen=True)
class Absorbing:
    """First- or second-order absorbing boundary on an exterior surface.

    Under e^{+iωt}, with k = k0 sqrt(eps_r mu_r) in the material behind the
    surface and n its outward normal, the condition is

        n x (mu_r^-1 curl E) = i (k / mu_r) E_t + β curl[n (n · mu_r^-1 curl E)]

    with β = 1 / (2ik) at order 2 and β = 0 at order 1: the plane-wave
    expansion k_n ≈ k - k_t² / (2k) of an outgoing wave (Jin, *The Finite
    Element Method in Electromagnetics*, §9.3, with the radius of curvature
    taken infinite). Its weak form adds i (k / mu_r) ∮ E_t · v_t and, at
    order 2, (1 / (2ik)) ∮ mu_r^-1 (n · curl E)(n · curl v). Only the curl part of the
    second-order term is kept; the β ∇_t(∇ · E_t) part is dropped, as in
    COMSOL, because the tangential divergence of a Nédélec trace is not
    square-integrable. gfem uses the textbook sign of the curl term only;
    the opposite sign reflects more than order 1, so on a tight air box a
    solver using it differs from gfem at order 2 (they agree at order 1 and
    on adequately sized boxes).

    Both orders are exact at normal incidence only. A plane wave arriving at
    the angle θ from the normal is reflected with amplitude (1 - cos θ) /
    (1 + cos θ) at order 1 (0.017 at 15°, 0.072 at 30°, 0.17 at 45°), for
    both polarizations. Order 2 lowers this about tenfold up to 30° for the
    polarization with E parallel to the surface and normal to k_t; for the
    other one (E in the plane of incidence, no normal magnetic field) the
    curl term vanishes and the reflection is that of order 1. In a periodic
    cell under a Bloch wave vector with a component k_t along the surface,
    ``sin θ = |k_t| / k``.

    Next to an anisotropic material, order 1 uses the normal-incidence
    admittance tensor of that material for the face normal (i k0 ∮ (Y E_t)
    · v_t, exact for a plane wave at normal incidence; real parts of ε and
    μ, as for isotropic faces). Order 2 raises `NotImplementedError` there.

    Attributes:
        surface: The surface region's name.
        order: 1 or 2.
    """

    surface: str
    order: Literal[1, 2] = 1
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Absorbing")
        if (
            is_traced(self.order)
            or self.order not in (1, 2)
            or isinstance(self.order, bool)
        ):
            raise ValueError(f"Absorbing: order must be 1 or 2, got {self.order!r}.")
        object.__setattr__(self, "order", int(self.order))


@dataclasses.dataclass(frozen=True)
class Conductor:
    """A finite-conductivity metal surface, modelled by its surface impedance.

    Each face adds iω / Z_s(ω) ∫ E_t · v_t, with Z_s the surface impedance of
    one face of a metal film (HFSS finite-thickness form). On an
    interior surface it is one term for the whole sheet, so a film much
    thinner than its skin depth is a sheet admittance of about sigma t / 2; on
    an exterior surface the metal is one-sided and its thickness counts
    twice (sigma t); see `gfem.conventions`.

    Attributes:
        surface: The surface region's name.
        conductivity: Electrical conductivity of the metal, S/m, > 0.
        thickness: Metal thickness in m, or ``None`` for a thick conductor
            (skin effect only).
        fill: A `SheetField` scaling the sheet's admittance face by face
            (a sheet design, interior surfaces only), or ``None`` for the
            solid sheet.
    """

    surface: str
    conductivity: float
    thickness: float | None = None
    fill: SheetField | None = None
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Conductor")
        sigma = static_float(self.conductivity, "Conductor.conductivity", _FIXED)
        if sigma <= 0:
            raise ValueError(
                f"Conductor.conductivity must be > 0 S/m, got {sigma}. For a "
                "perfect conductor use gfem.PEC."
            )
        object.__setattr__(self, "conductivity", sigma)
        object.__setattr__(
            self,
            "thickness",
            optional_positive(self.thickness, "Conductor.thickness", "m", _FIXED),
        )
        _check_fill(self.fill, "Conductor")


@dataclasses.dataclass(frozen=True)
class Impedance:
    """A surface impedance: per-square R (ohm), L (H) and C (F) in parallel.

    The three values are design inputs (pytree leaves; ADR 0100): driven
    S-parameters, kept fields and eigenfrequencies differentiate with respect
    to them. A value built under `jax.grad`, `jax.jit` or `jax.vmap` is
    checked when the solve runs, where a non-finite or non-positive one
    raises `ValueError`. On a surface that shares an edge with a wave port's
    face the port mode reads them, so differentiating them there raises
    `gfem.PortGeometryError`.

    Attributes:
        surface: The surface region's name.
        resistance: Sheet resistance, ohm per square, or ``None``.
        inductance: Sheet inductance, H per square, or ``None``.
        capacitance: Sheet capacitance, F per square, or ``None``.
        fill: A `SheetField` scaling the sheet's admittance face by face
            (a sheet design, interior surfaces only), or ``None`` for the
            solid sheet.
    """

    surface: str
    resistance: float | None = None
    inductance: float | None = None
    capacitance: float | None = None
    fill: SheetField | None = None
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        _surface_name(self.surface, "Impedance")
        _set_rlc(self, "Impedance", traced=("resistance", "inductance", "capacitance"))
        if all(v is None for v in (self.resistance, self.inductance, self.capacitance)):
            raise ValueError(
                "Impedance needs at least one of resistance, inductance or "
                "capacitance (per square, in parallel)."
            )
        _check_fill(self.fill, "Impedance")


@dataclasses.dataclass(frozen=True)
class Periodic:
    """A periodic pair: two exterior surfaces related by a translation.

    The fields on the two surfaces are identified node for node. The
    translation a runs from ``surfaces[0]`` to ``surfaces[1]`` and is
    measured from the mesh (`Model.periodic_translations`); the two surface
    meshes must be identical up to it (gmsh: ``setPeriodic``). A model takes
    one to three pairs, along independent directions. With no Bloch wave
    vector on the problem the fields are equal on both surfaces (plain
    periodicity); with ``wave_vector=k`` (rad/m) on `gfem.Driven` or
    `gfem.Eigenmode`, E(r + a) = exp(-i k·a) E(r) (`gfem.conventions`,
    Bloch periodicity). The order of the two surfaces is not physical:
    swapping them flips a and gives the same field.

    `gfem.Eigenmode` and `gfem.Driven` solve periodic models: a driven
    problem with a wave vector on the uniform sweep only, without one on
    both sweeps. `gfem.Electrostatic` raises on a periodic model. A paired
    surface takes no other boundary or port, and the default PMC skips it.
    The two meshes are checked when the `gfem.Model` is built. See the
    "Solve a periodic cell" recipe.

    Attributes:
        surfaces: The two surface region names (first, second).

    Raises:
        TypeError: ``surfaces`` is not two surface names.
        ValueError: The same name is given twice.

    Example:
        ``gfem.Periodic(("x_min", "x_max"))``
    """

    surfaces: tuple[str, str]
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        value = self.surfaces
        if (
            isinstance(value, str)
            or not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(isinstance(s, str) for s in value)
        ):
            raise TypeError(
                "Periodic: surfaces must be two surface names (first, second), "
                f"got {value!r}."
            )
        if value[0] == value[1]:
            raise ValueError(
                f"Periodic: the two surfaces must differ, got {value[0]!r} twice; "
                "a periodic pair joins two opposite surfaces of the cell."
            )
        object.__setattr__(self, "surfaces", (value[0], value[1]))


def _set_rlc(obj: Any, owner: str, traced: tuple[str, ...]) -> None:
    """Check the lumped values of ``obj``; those named in ``traced`` may be traced.

    A traced value is kept as it is: its value is checked on the host, on every
    forward (`check_lumped_values`). A concrete one is checked here.
    """
    for name, unit in _RLC:
        value = getattr(obj, name)
        if name in traced and is_traced(value):
            continue
        value = optional_positive(value, f"{owner}.{name}", unit, _FIXED)
        object.__setattr__(obj, name, value)


def lumped_label(item: LumpedPort | Impedance) -> str:
    """``LumpedPort 'p1'`` or ``Impedance on surface 's'``, for messages."""
    if isinstance(item, LumpedPort):
        return f"LumpedPort {item.name!r}"
    return f"Impedance on surface {item.surface!r}"


def check_lumped_values(item: LumpedPort | Impedance) -> None:
    """Raise unless every lumped value of ``item`` is a finite number > 0.

    The constructor checks concrete values; a value built under `jax.grad`,
    `jax.jit` or `jax.vmap` reaches the host unchecked, so every forward checks
    the values it reads here.

    Raises:
        ValueError: A value is not finite, or not > 0.
    """
    for name, unit in _RLC:
        value = getattr(item, name)
        if value is None:
            continue
        out = np.asarray(value)
        if out.shape or np.iscomplexobj(out) or not np.isfinite(out) or out <= 0:
            raise ValueError(
                f"{lumped_label(item)}: {name} must be a finite number > 0 {unit}, "
                f"got {value!r}."
            )


for _cls in (PEC, PMC, Absorbing, Periodic):
    jax.tree_util.register_static(_cls)
register_fields(Conductor, ("fill",))
register_fields(Impedance, ("resistance", "inductance", "capacitance", "fill"))

#: A boundary condition of a model: any one of the classes this union names.
Boundary = PEC | PMC | Absorbing | Conductor | Impedance | Periodic


# --- ports ----------------------------------------------------------------------

_AXES = {"x": 0, "y": 1, "z": 2}

Direction = (
    Literal[
        "x",
        "y",
        "z",
        "X",
        "Y",
        "Z",
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
        "+X",
        "-X",
        "+Y",
        "-Y",
        "+Z",
        "-Z",
    ]
    | tuple[float, float, float]
)


def _port_name(name: Any, owner: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"{owner}: name must be a string, got {name!r}.")
    if not name:
        raise ValueError(f"{owner}: name must not be empty.")
    return name


def _direction(value: Any, owner: str) -> tuple[float, float, float]:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _AXES:  # no sign means the positive axis
            v = "+" + v
        if len(v) == 2 and v[0] in "+-" and v[1] in _AXES:
            out = [0.0, 0.0, 0.0]
            out[_AXES[v[1]]] = 1.0 if v[0] == "+" else -1.0
            return (out[0], out[1], out[2])
        raise ValueError(
            f"{owner}: direction {value!r} is not an axis; use 'x', 'y', 'z' with "
            "an optional sign ('+x', '-z'; no sign means '+') or a 3-vector."
        )
    check_fixed(value, f"{owner}: direction must be fixed numbers")
    try:
        d = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{owner}: direction {value!r} is not a 3-vector.") from None
    norm = float(np.linalg.norm(d)) if d.shape == (3,) else 0.0
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{owner}: direction {value!r} is not a non-zero 3-vector.")
    d = d / norm
    return (float(d[0]), float(d[1]), float(d[2]))


def _is_one_direction(value: Any) -> bool:
    """A single direction (axis name or 3 numbers), not a sequence of them."""
    if isinstance(value, str):
        return True
    return not any(isinstance(v, str) for v in value) and np.ndim(value) == 1


@dataclasses.dataclass(frozen=True)
class LumpedPort:
    """A lumped port on one surface, or several surfaces driven together.

    Each element (surface) has its own excitation direction; a coplanar
    waveguide port is two elements with opposite directions. The port is
    terminated by R, L and C in parallel, or by a resistive ``impedance``.

    Example:
        A coplanar waveguide port: one gap on each side of the signal line,
        each excited from the signal line towards ground::

            port = gfem.LumpedPort(
                "p1", surface=("gap_left", "gap_right"), direction=("-x", "+x")
            )

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: One surface name, or a sequence of them (multi-element).
        direction: The excitation direction of each element, along its
            surface from one conductor to the other: the reference direction
            of E across the gap, so it sets the polarity of the port voltage.
            An axis ('+x', '-y', ...; no sign means +) or a 3-vector. With
            several surfaces, a sequence of the same length.
        resistance: Termination resistance, ohm. It is the port's reference
            impedance, a fixed value: a traced one raises
            `gfem.NonDifferentiableError`.
        inductance: Termination inductance, H. A design input (a pytree leaf;
            ADR 0100): driven S-parameters, kept fields and eigenfrequencies
            differentiate with respect to it.
        capacitance: Termination capacitance, F. A design input, like
            ``inductance``.
        impedance: A resistive termination in ohm, the same as
            ``resistance``. If no termination is given, 50 ohm.

    A traced inductance or capacitance is checked when the solve runs, where a
    non-finite or non-positive one raises `ValueError`.
    """

    name: str
    surface: str | tuple[str, ...]
    direction: Direction | tuple[Direction, ...]
    resistance: float | None = None
    inductance: float | None = None
    capacitance: float | None = None
    impedance: float | None = None
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        owner = f"LumpedPort {self.name!r}"
        _port_name(self.name, "LumpedPort")
        single = isinstance(self.surface, str)
        if single:
            surfaces: tuple[str, ...] = (self.surface,)
            directions = [self.direction]
        else:
            surfaces = tuple(_surface_name(s, owner) for s in self.surface)
            if not surfaces:
                raise ValueError(f"{owner}: needs at least one surface.")
            if len(set(surfaces)) != len(surfaces):
                raise ValueError(f"{owner}: a surface is listed twice.")
            if _is_one_direction(self.direction):
                raise ValueError(
                    f"{owner}: a multi-element port needs one direction per element "
                    f"(surface); got one direction for {len(surfaces)} surfaces."
                )
            directions = list(self.direction)
            if len(directions) != len(surfaces):
                raise ValueError(
                    f"{owner}: needs one direction per element (surface); got "
                    f"{len(directions)} directions for {len(surfaces)} surfaces."
                )
        dirs = tuple(_direction(d, owner) for d in directions)
        object.__setattr__(self, "surface", surfaces[0] if single else surfaces)
        object.__setattr__(self, "direction", dirs[0] if single else dirs)
        _set_rlc(self, "LumpedPort", traced=("inductance", "capacitance"))
        object.__setattr__(
            self,
            "impedance",
            optional_positive(self.impedance, "LumpedPort.impedance", "ohm", _FIXED),
        )
        none = all(
            v is None for v in (self.resistance, self.inductance, self.capacitance)
        )
        if self.impedance is not None and not none:
            raise ValueError(
                f"{owner}: give either impedance (a resistive termination) or "
                "resistance / inductance / capacitance, not both."
            )
        if self.impedance is None and none:
            object.__setattr__(self, "impedance", 50.0)

    @property
    def surfaces(self) -> tuple[str, ...]:
        """The port's surfaces, one per element."""
        return (self.surface,) if isinstance(self.surface, str) else self.surface

    @property
    def elements(self) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        """``(surface, unit direction)`` per element."""
        dirs = (self.direction,) if isinstance(self.surface, str) else self.direction
        return tuple(zip(self.surfaces, dirs, strict=True))  # ty: ignore[invalid-return-type]

    @property
    def rlc(self) -> tuple[float | None, float | None, float | None]:
        """The termination as ``(R, L, C)`` in parallel, ``impedance`` read as R."""
        if self.impedance is not None:
            return (self.impedance, None, None)
        return (self.resistance, self.inductance, self.capacitance)


@dataclasses.dataclass(frozen=True)
class WavePort:
    """A wave port on an exterior surface, fed by its port mode.

    Several wave ports on one surface each request a different mode (a
    shared port surface). A propagating mode no port on the surface requests
    is partly reflected, not converted.

    The cross-section may contain anisotropic materials when every tensor
    has a principal axis along the port normal (any orientation in the port
    plane); off-axis tensors would make forward and backward port modes
    differ. Elsewhere in the model the principal axes may point in any
    direction.

    Attributes:
        name: The port's name; ports index S-parameters by name, in the
            order of the model's ``ports``.
        surface: The port's surface region.
        mode: Which port mode excites the port: its rank at the lowest
            frequency among the modes the model admits (a symmetry plane
            halves them), 1 = the most confined.
        offset: De-embedding distance in m, >= 0.
        voltage_path: Points (m) of a path across the port; it fixes the
            mode's polarity and defines a voltage, hence a reference
            impedance. ``None``: no voltage.
        pec_edges: Surfaces whose edges on the port are perfect conductors
            in the port mode problem only (where the 3D problem models them
            otherwise, e.g. as a `Conductor`).
    """

    name: str
    surface: str
    mode: int = 1
    offset: float = 0.0
    voltage_path: tuple[tuple[float, float, float], ...] | None = None
    pec_edges: tuple[str, ...] = ()
    _physics: ClassVar[str] = "em"

    def __post_init__(self) -> None:
        owner = f"WavePort {self.name!r}"
        _port_name(self.name, "WavePort")
        _surface_name(self.surface, owner)
        if (
            isinstance(self.mode, bool)
            or not isinstance(self.mode, (int, np.integer))
            or self.mode < 1
        ):
            raise ValueError(
                f"{owner}: mode must be an integer >= 1, got {self.mode!r}."
            )
        object.__setattr__(self, "mode", int(self.mode))
        offset = static_float(self.offset, "WavePort.offset", _FIXED)
        if offset < 0:
            raise ValueError(f"{owner}: offset must be >= 0 m, got {offset}.")
        object.__setattr__(self, "offset", offset)
        if self.voltage_path is not None:
            object.__setattr__(self, "voltage_path", _path(self.voltage_path, owner))
        edges = (self.pec_edges,) if isinstance(self.pec_edges, str) else self.pec_edges
        object.__setattr__(
            self, "pec_edges", tuple(_surface_name(s, owner) for s in edges)
        )


def _path(value: Any, owner: str) -> tuple[tuple[float, float, float], ...]:
    check_fixed(value, f"{owner}: voltage_path must be fixed numbers")
    try:
        p = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        p = np.empty(0)
    if p.ndim != 2 or p.shape[-1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError(
            f"{owner}: voltage_path must be two or more points (m), each a "
            f"3-vector, got {value!r}."
        )
    return tuple((float(r[0]), float(r[1]), float(r[2])) for r in p.reshape(-1, 3))


jax.tree_util.register_static(WavePort)
register_fields(LumpedPort, ("inductance", "capacitance"))

#: A port of a model: a lumped port or a wave port.
Port = LumpedPort | WavePort


def sheet_of(item: Boundary | Port) -> SheetField | None:
    """The sheet field (``fill=``) a boundary carries, or ``None``."""
    return item.fill if isinstance(item, (Conductor, Impedance)) else None


def surfaces_of(item: Boundary | Port) -> tuple[str, ...]:
    """The surfaces a boundary condition or port occupies."""
    if isinstance(item, (LumpedPort, Periodic)):
        return item.surfaces
    return (item.surface,)


def wave_port_surfaces(ports: Iterable[Port]) -> dict[str, tuple[WavePort, ...]]:
    """The wave ports among ``ports`` by surface, in model order.

    Surfaces come in the order of their first wave port; a surface with
    several wave ports is a shared port surface.
    """
    surfaces: dict[str, list[WavePort]] = {}
    for port in ports:
        if isinstance(port, WavePort):
            surfaces.setdefault(port.surface, []).append(port)
    return {surface: tuple(group) for surface, group in surfaces.items()}


def lead_port(group: Sequence[WavePort]) -> WavePort:
    """The wave port with the smallest ``mode`` on a surface.

    Its propagation constant is the surface's Robin ``gamma_0``.
    """
    return min(group, key=lambda p: p.mode)


def wave_port_label(
    group: Sequence[WavePort], *, lone_surface: bool = False, modes: bool = False
) -> str:
    """``WavePorts 'a', 'b' on surface 's'`` for a surface's wave ports.

    Args:
        group: The wave ports of one surface.
        lone_surface: Name the surface of a lone wave port too
            (``WavePort 'a' on surface 's'``); else ``WavePort 'a'``.
        modes: Follow each name with its mode (``'a' (mode 1)``).
    """
    names = ", ".join(
        f"{p.name!r} (mode {p.mode})" if modes else repr(p.name) for p in group
    )
    surface = f" on surface {group[0].surface!r}"
    if len(group) == 1:
        return f"WavePort {names}{surface if lone_surface else ''}"
    return f"WavePorts {names}{surface}"
