"""Problem types: what to solve on a model, as frozen values.

A problem names the physics analysis and its physical inputs (terminals,
frequencies, ...). The model holds everything about the device; the solver
options hold the numerics. ``gfem.solve(model, problem)`` dispatches on the
problem's type.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal, get_args

import jax
import numpy as np

from gfem._boundaries import _temperature
from gfem._checks import check_fixed, static_float
from gfem._pytree import register_fields
from gfem.thermal import HeatSource, Trace, _check_finite

__all__ = [
    "DCConduction",
    "Driven",
    "Eigenmode",
    "Electrostatic",
    "PortMode",
    "SteadyThermal",
    "TransientThermal",
]


def _surfaces(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)) and all(isinstance(s, str) for s in value):
        return tuple(value)
    raise TypeError(
        f"{what} must be a surface name or a list of surface names, got {value!r}."
    )


def _terminals(
    terminals: Any, ground: Any, owner: str
) -> tuple[list[tuple[str, tuple[str, ...]]], tuple[str, ...]]:
    """``terminals`` and ``ground`` normalized: ``(name, surfaces)`` pairs, surfaces.

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.
    """
    items: list[tuple[str, tuple[str, ...]]]
    if isinstance(terminals, Mapping):
        items = []
        for name, surfaces in terminals.items():
            if not isinstance(name, str):
                raise TypeError(f"Terminal names must be strings, got {name!r}.")
            items.append((name, _surfaces(surfaces, f"terminals[{name!r}]")))
    else:
        items = [(s, (s,)) for s in _surfaces(terminals, "terminals")]
    if not items:
        raise ValueError(
            f"{owner} needs at least one terminal, e.g. "
            "terminals={'trace': 'trace_surface'}."
        )
    ground = _surfaces(ground, "ground") if ground else ()
    owners: dict[str, str] = {}
    for name, surfaces in items:
        if len(set(surfaces)) != len(surfaces):
            raise ValueError(f"Terminal {name!r} lists a surface twice.")
        for s in surfaces:
            if s in owners:
                raise ValueError(
                    f"Surface {s!r} is in terminals {owners[s]!r} and {name!r}; "
                    "a surface belongs to one terminal."
                )
            owners[s] = name
    for s in ground:
        if s in owners:
            raise ValueError(
                f"Surface {s!r} is both in terminal {owners[s]!r} and in the "
                "ground; remove it from one of them."
            )
    return items, tuple(dict.fromkeys(ground))


def _wave_vector(value: Any, owner: str) -> tuple[float, float, float] | None:
    """``None``, or a Bloch wave vector as three finite floats (rad/m)."""
    if value is None:
        return None
    check_fixed(value, f"{owner}.wave_vector must be fixed numbers")
    try:
        k = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        k = np.empty(0)
    if k.shape != (3,) or not np.isfinite(k).all():
        raise ValueError(
            f"{owner}.wave_vector must be a 3-vector in rad/m (kx, ky, kz), or "
            f"None for plain periodicity; got {value!r}."
        )
    return (float(k[0]), float(k[1]), float(k[2]))


def _reference_frequency(value: Any, wave_vector: Any) -> float | None:
    """``None``, or the frequency (Hz) at which ``wave_vector`` holds (ADR 0151)."""
    if value is None:
        return None
    check_fixed(value, "Driven.wave_vector_frequency must be a fixed number")
    if wave_vector is None:
        raise ValueError(
            "Driven.wave_vector_frequency scales Driven.wave_vector with "
            "frequency, but wave_vector is None. Give the Bloch wave vector at "
            "that frequency, e.g. Model.bloch_wave_vector(surface, f, theta=...)."
        )
    ok = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    if not ok or not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Driven.wave_vector_frequency must be a positive finite frequency in "
            f"Hz, or None for the same wave vector at every frequency; got {value!r}."
        )
    return float(value)


@dataclasses.dataclass(frozen=True, init=False)
class Electrostatic:
    """The Maxwell capacitance matrix between named terminals.

    For each terminal in turn, the potential is 1 V on that terminal and
    0 V on every other terminal and on ground, and the charge-free Laplace
    equation div(eps grad V) = 0 is solved in the electromagnetic solve
    domain (the model's excluded volumes are left out). Surfaces that are
    neither terminal nor ground carry no surface charge (zero normal
    displacement), except that every surface of a `PEC` or `Conductor`
    boundary of the model that is not a terminal is grounded, as conductors
    are. Ports and other boundaries are charge-free.

    Only the real part of the permittivity is read: losses
    (``tan_delta``, ``conductivity``, a negative imaginary ``eps_r``) and
    ``mu_r`` do not enter the electrostatic problem.

    Example:
        The capacitance of a trace made of two surfaces, against ground::

            problem = gfem.Electrostatic(
                terminals={"trace": ["m1_xy", "m1_z"]}, ground="ground"
            )

    Args:
        terminals: Terminal name -> its surface name or names (a conductor's
            shell), in capacitance-matrix order. A list of surface names
            makes one terminal per surface, named after it.
        ground: The surface name or names held at 0 V besides the model's
            conductors. Empty: only the model's `PEC` / `Conductor`
            surfaces, or no ground at all (then each row of the matrix sums
            to zero).

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.

    `gfem.solve` raises NotImplementedError on a model with a
    `gfem.Periodic` pair.
    """

    terminals: tuple[tuple[str, tuple[str, ...]], ...]
    ground: tuple[str, ...]
    _rule: ClassVar[str] = "gfem._engine.electrostatic:ElectrostaticRule"

    def __init__(
        self,
        terminals: Mapping[str, str | list[str] | tuple[str, ...]] | list[str],
        ground: str | list[str] | tuple[str, ...] = (),
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        items, ground = _terminals(terminals, ground, "Electrostatic")
        object.__setattr__(self, "terminals", tuple(items))
        object.__setattr__(self, "ground", tuple(dict.fromkeys(ground)))

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Terminal names, in capacitance-matrix order."""
        return tuple(name for name, _ in self.terminals)

    def __repr__(self) -> str:
        terms = ", ".join(f"{n!r}: {list(s)}" for n, s in self.terminals)
        ground = f", ground={list(self.ground)}" if self.ground else ""
        return f"Electrostatic(terminals={{{terms}}}{ground})"


jax.tree_util.register_static(Electrostatic)


@dataclasses.dataclass(frozen=True, init=False)
class DCConduction:
    """The DC conductance matrix between named terminals of conductors.

    For each terminal in turn, the potential is 1 V on that terminal and
    0 V on every other terminal and on the ground, and the steady current
    equation div(sigma grad V) = 0 is solved in the conductors. Every
    other surface of a conductor is insulated (no current crosses it).

    The conductors are not named: they are every volume whose material
    has ``conductivity`` > 0, minus the volumes excluded from DC conduction
    (``Model(excluded={"dc_conduction": (...)})``). A conducting piece
    that touches no terminal and no ground carries no current: it is left
    out, and its heat is exactly zero. A tensor ``conductivity`` is read as
    a tensor.

    The result (`gfem.DCConductionResult`) holds the conductance and the
    dissipation per element at unit voltages; `gfem.joule_heating` applies
    a voltage, current or power drive to it and gives the heat source of a
    thermal problem, with no new solve.

    On a cross-section mesh (`gfem.Mesh.from_gmsh_cross_section`), put the
    terminals on its end faces ``"front"`` and ``"back"``: the current runs
    along the waveguide. A conducting volume that touches both end faces
    (a doped substrate, say) then joins the circuit; exclude it from DC
    conduction if no current flows in it.

    Example:
        A heater between two pads::

            dc = gfem.solve(
                model,
                gfem.DCConduction(terminals={"pad_a": "pad_a_top"}, ground="pad_b_top"),
            )
            dc.resistance  # ohm

    Args:
        terminals: Terminal name -> its surface name or names, in
            conductance-matrix order. A list of surface names makes one
            terminal per surface, named after it.
        ground: The surface name or names held at 0 V. Empty: no ground
            (then each row of the conductance matrix sums to zero).

    Raises:
        TypeError: A name is not a string.
        ValueError: No terminal, or a surface in two terminals or in a
            terminal and the ground.

    `gfem.solve` raises ValueError when no volume conducts, when a
    terminal or the ground touches no conducting volume, or when an
    element of a conducting volume has zero conductivity (a
    `gfem.MaterialField` with exact zeros: give its void a small positive
    value).
    """

    terminals: tuple[tuple[str, tuple[str, ...]], ...]
    ground: tuple[str, ...]
    _rule: ClassVar[str] = "gfem._engine.dc_conduction:DCConductionRule"
    _physics: ClassVar[str] = "dc_conduction"

    def __init__(
        self,
        terminals: Mapping[str, str | list[str] | tuple[str, ...]] | list[str],
        ground: str | list[str] | tuple[str, ...] = (),
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        items, ground = _terminals(terminals, ground, "DCConduction")
        object.__setattr__(self, "terminals", tuple(items))
        object.__setattr__(self, "ground", ground)

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """Terminal names, in conductance-matrix order."""
        return tuple(name for name, _ in self.terminals)

    def __repr__(self) -> str:
        terms = ", ".join(f"{n!r}: {list(s)}" for n, s in self.terminals)
        ground = f", ground={list(self.ground)}" if self.ground else ""
        return f"DCConduction(terminals={{{terms}}}{ground})"


jax.tree_util.register_static(DCConduction)


def _heat_sources(heat: Any, owner: str) -> tuple[HeatSource, ...]:
    """``heat`` as a tuple of `gfem.HeatSource` (one, or a sequence of them)."""
    if isinstance(heat, HeatSource):
        return (heat,)
    if isinstance(heat, Sequence) and not isinstance(heat, str):
        sources = tuple(heat)
        if all(isinstance(s, HeatSource) for s in sources):
            return sources
    raise TypeError(
        f"{owner}.heat must be a gfem.HeatSource or a sequence of them (from "
        "gfem.HeatSource.uniform or gfem.joule_heating), got "
        f"{type(heat).__name__}."
    )


@dataclasses.dataclass(frozen=True, init=False)
class SteadyThermal:
    """The steady temperature of a model under heat sources.

    Solves -div(k grad T) = q on the thermal domain: every volume of the
    mesh minus those excluded from thermal problems
    (``Model(excluded={"thermal": (...)})``), with k the materials'
    ``thermal_conductivity``. The boundary conditions are the model's
    `gfem.FixedTemperature`, `gfem.Convection` and `gfem.HeatFlux`; every
    other exterior surface of the thermal domain is insulated
    (`gfem.Model.default_insulated`), and unnamed interior interfaces are in
    perfect thermal contact. Temperatures are absolute, in kelvin.

    The result (`gfem.SteadyThermalResult`) holds the temperature and its
    rise: the part linear in the heat sources given here, which scales
    with their power. The boundary data (fixed temperatures, ambients and
    heat fluxes) set the source-free temperature the rise is added to.

    Example:
        A heater dissipating 10 mW over a substrate held at 293.15 K::

            model = gfem.Model(
                mesh,
                materials=materials,
                boundaries=[gfem.FixedTemperature("bottom", temperature=293.15)],
            )
            heat = gfem.HeatSource.uniform("heater", power=10e-3)
            result = gfem.solve(model, gfem.SteadyThermal(heat))
            result.rise.at(point)  # K

    Args:
        heat: A `gfem.HeatSource` or a sequence of them; they superpose.
            Empty: the temperature set by the boundaries alone.

    Raises:
        TypeError: ``heat`` is not a heat source or a sequence of them.

    `gfem.solve` raises ValueError when a connected piece of the thermal
    domain has no heat sink (no `gfem.FixedTemperature` or
    `gfem.Convection` on it: its steady temperature is undetermined), and
    `gfem.errors.MissingMaterialPropertyError` when a region of the
    thermal domain has no ``thermal_conductivity``.
    """

    heat: tuple[HeatSource, ...]
    _rule: ClassVar[str] = "gfem._engine.thermal:SteadyThermalRule"
    _physics: ClassVar[str] = "thermal"

    def __init__(self, heat: HeatSource | Sequence[HeatSource] = ()) -> None:
        """Validate and normalize (see the class docstring)."""
        object.__setattr__(self, "heat", _heat_sources(heat, "SteadyThermal"))

    def __repr__(self) -> str:
        return f"SteadyThermal(heat={list(self.heat)!r})"


register_fields(SteadyThermal, ("heat",))


#: Two grid times closer than this fraction of the step are the same time.
_GRID_RTOL = 1e-9
_FIXED_TIME = (
    "Build the problem from concrete values, outside the transformed function; "
    "the waveform scales are the differentiable part of a waveform."
)


def _grid_index(time: float, dt: float, steps: int, what: str) -> int:
    """The grid index ``n`` of ``time = n dt``, or a ValueError naming the nearest.

    Raises:
        ValueError: ``time`` is outside [0, duration] or between grid times.
    """
    k = time / dt
    n = round(k)
    if not -_GRID_RTOL <= k <= steps * (1 + _GRID_RTOL):
        raise ValueError(
            f"TransientThermal: {what} {time:g} s is outside the run, from 0 to "
            f"duration={steps * dt:g} s."
        )
    if abs(k - n) > _GRID_RTOL * max(1.0, abs(k)):
        low = math.floor(k)
        raise ValueError(
            f"TransientThermal: {what} {time:g} s is not on the time grid (dt = "
            f"duration / steps = {dt:g} s); the nearest grid times are "
            f"{low * dt:g} s and {(low + 1) * dt:g} s. Move it to a grid time, or "
            "choose steps so that it is one."
        )
    return n


def _waveform(value: Any, dt: float, steps: int) -> tuple[tuple[float, ...], Any]:
    """The waveform's switch times (static) and scales (a JAX array)."""
    try:
        pairs = [tuple(p) for p in value]
    except TypeError:
        pairs = []
    if not pairs or any(len(p) != 2 for p in pairs):
        raise TypeError(
            "TransientThermal.waveform must be a non-empty sequence of (start time "
            f"in s, scale) pairs, e.g. [(0.0, 1.0), (30e-6, 0.0)]; got {value!r}."
        )
    times = tuple(
        static_float(t, "A TransientThermal.waveform start time", _FIXED_TIME)
        for t, _ in pairs
    )
    if any(b <= a for a, b in itertools.pairwise(times)):
        raise ValueError(
            "TransientThermal.waveform start times must increase strictly, got "
            f"{list(times)}."
        )
    for t in times:
        _grid_index(t, dt, steps, "the waveform switch time")
    scales = jax.numpy.stack(
        [_check_finite(v, "A TransientThermal.waveform scale") for _, v in pairs]
    )
    return times, scales


def _kept_times(value: Any, dt: float, steps: int) -> bool | tuple[float, ...]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    check_fixed(value, "TransientThermal.keep_fields must be fixed times")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            "TransientThermal.keep_fields takes True, False or grid times in s, got "
            f"{value!r}."
        ) from None
    kept = {_grid_index(float(t), dt, steps, "the kept time") for t in wanted}
    return tuple(n * dt for n in sorted(kept))


def _traces(value: Any) -> dict[str, Trace]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(
        isinstance(k, str) and isinstance(v, Trace) for k, v in value.items()
    ):
        raise TypeError(
            "TransientThermal.traces must be a mapping from name to gfem.Trace, "
            f"e.g. {{'core': gfem.Trace.mean('core')}}; got {value!r}."
        )
    return dict(value)


@dataclasses.dataclass(frozen=True, init=False)
class TransientThermal:
    """The temperature of a model over time, under switched heat sources.

    Solves rho c_p dT/dt - div(k grad T) = s(t) q on the thermal domain
    (as `gfem.SteadyThermal`, with the materials' ``mass_density`` and
    ``specific_heat`` too) on the uniform time grid t_n = n dt, dt =
    ``duration / steps``. The heat sources q follow the waveform s(t):
    piecewise constant, off before its first start time, the value on
    [t_n, t_n+1) applied over that step. Every switch time must be a grid
    time. `gfem.SolverOptions.time_scheme` picks the scheme (TR-BDF2 by
    default; one factorization serves the whole run).

    The result (`gfem.TransientThermalResult`) records each trace at all
    ``steps + 1`` grid times and keeps the temperature field at the final
    time and at the ``keep_fields`` times. Temperatures are absolute, in
    kelvin; the rise is the temperature minus the source-free temperature
    (the steady solve with the heat sources off).

    Example:
        A heater switched on for 30 us, then off for 30 us::

            problem = gfem.TransientThermal(
                gfem.HeatSource.uniform("heater", power=10e-3),
                duration=60e-6,
                steps=240,
                waveform=[(0.0, 1.0), (30e-6, 0.0)],
                traces={"core": gfem.Trace.mean("core")},
            )
            result = gfem.solve(model, problem)
            result.traces["core"]  # K, at result.time

    Args:
        heat: A `gfem.HeatSource` or a sequence of them; they superpose and
            share the waveform.
        duration: The run's length in s (> 0).
        steps: The number of time steps (>= 1).
        waveform: ``(start time in s, scale)`` pairs with increasing start
            times on the grid; the scale multiplies every source from its
            start time to the next one. The default is a step on at t = 0.
        initial_temperature: ``None`` starts from the source-free
            temperature (the steady temperature with the sources off), so
            the rise starts at zero; a number starts from that uniform
            temperature in K (> 0). A model with a part of the thermal
            domain that touches no heat sink needs a number.
        traces: A mapping from name to `gfem.Trace`, recorded every step.
        keep_fields: Which temperature fields the result keeps besides the
            final one: ``False`` none, ``True`` every grid time, or a
            sequence of grid times in s.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: ``duration`` or ``steps`` out of range; a switch time
            or a kept time off the grid (the message names the nearest
            grid times) or outside the run; start times not increasing; a
            temperature <= 0 K.
        NonDifferentiableError: ``duration``, ``steps``, a waveform start
            time, ``initial_temperature`` or a kept time is traced by JAX
            (the waveform scales and the heat sources are differentiable).
    """

    heat: tuple[HeatSource, ...]
    duration: float
    steps: int
    initial_temperature: float | None
    traces: dict[str, Trace]
    keep_fields: bool | tuple[float, ...]
    #: The waveform's start times in s.
    _switches: tuple[float, ...]
    #: The waveform's scales, a JAX array (one per start time).
    _scales: Any
    _rule: ClassVar[str] = "gfem._engine.thermal:TransientThermalRule"
    _physics: ClassVar[str] = "thermal"

    def __init__(
        self,
        heat: HeatSource | Sequence[HeatSource] = (),
        *,
        duration: float,
        steps: int,
        waveform: Sequence[tuple[float, Any]] = ((0.0, 1.0),),
        initial_temperature: float | None = None,
        traces: Mapping[str, Trace] | None = None,
        keep_fields: bool | Sequence[float] = False,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        set_ = object.__setattr__
        set_(self, "heat", _heat_sources(heat, "TransientThermal"))
        duration = static_float(duration, "TransientThermal.duration", _FIXED_TIME)
        if duration <= 0:
            raise ValueError(
                f"TransientThermal.duration must be > 0 s, got {duration}."
            )
        check_fixed(steps, "TransientThermal.steps must be a fixed integer")
        if (
            not isinstance(steps, (int, np.integer))
            or isinstance(steps, bool)
            or steps < 1
        ):
            raise ValueError(
                f"TransientThermal.steps must be an integer >= 1, got {steps!r}."
            )
        steps = int(steps)
        dt = duration / steps
        set_(self, "duration", duration)
        set_(self, "steps", steps)
        times, scales = _waveform(waveform, dt, steps)
        set_(self, "_switches", times)
        set_(self, "_scales", scales)
        if initial_temperature is not None:
            initial_temperature = _temperature(
                initial_temperature, "TransientThermal.initial_temperature"
            )
        set_(self, "initial_temperature", initial_temperature)
        set_(self, "traces", _traces(traces))
        set_(self, "keep_fields", _kept_times(keep_fields, dt, steps))

    @property
    def dt(self) -> float:
        """The time step in s: ``duration / steps``."""
        return self.duration / self.steps

    @property
    def waveform(self) -> tuple[tuple[float, Any], ...]:
        """The ``(start time in s, scale)`` pairs."""
        return tuple(zip(self._switches, self._scales, strict=True))

    def __repr__(self) -> str:
        try:
            scales = [float(v) for v in np.asarray(self._scales)]
            wave = [(t, v) for t, v in zip(self._switches, scales, strict=True)]
        except (TypeError, jax.errors.TracerArrayConversionError):
            wave = list(self._switches)
        return (
            f"TransientThermal(heat={list(self.heat)!r}, duration={self.duration:g}, "
            f"steps={self.steps}, waveform={wave}, traces={list(self.traces)})"
        )


register_fields(TransientThermal, ("heat", "traces", "_scales"))


def _frequencies(value: Any, owner: str = "PortMode") -> tuple[float, ...]:
    check_fixed(value, f"{owner}.frequency must be fixed numbers")
    try:
        f = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        f = np.empty((0, 0))
    if f.ndim > 1 or f.size == 0:
        raise ValueError(
            f"{owner}.frequency must be one frequency or a 1D sequence of them, in "
            f"Hz; got {value!r}."
        )
    f = f.reshape(-1)
    if not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError(f"{owner}.frequency must be finite and > 0 Hz, got {value!r}.")
    out = tuple(float(x) for x in f)
    seen: set[float] = set()
    for x in out:
        if x in seen:
            raise ValueError(
                f"{owner}.frequency names {x:g} Hz twice: each frequency is solved "
                "once. Drop the repeats, for example with numpy.unique(frequency)."
            )
        seen.add(x)
    return out


@dataclasses.dataclass(frozen=True, init=False)
class PortMode:
    """The 2D port mode problem of a wave port over a frequency grid.

    Solves the port's cross-section for its propagating mode at each
    frequency: propagation constant, effective index, mode field and the
    power-voltage impedance. The cross-section's materials are those of the
    volume elements behind the port; its edges take the boundary conditions
    of the surfaces that meet the port (`PEC` edges are perfect conductors,
    `Absorbing` edges first-order absorbing, all others natural, i.e. PMC).

    A grid runs the port mode sweep that driven problems use for their wave
    ports: a few exact port mode solves (typically 5-12 per band) and a
    reduced port mode model checked at every frequency against
    ``SolverOptions.port_mode_tolerance``. With up to three frequencies each
    is solved exactly. The mode is picked at the lowest frequency and
    followed through the band by its overlap with the neighbouring
    frequency's mode, so it stays the same physical mode through crossings.
    On a shared port surface it runs the surface's sweep, with the other
    wave ports of its surface, so it gives the same mode a driven problem
    uses; an explicit ``mode`` sweeps that rank alone.

    Example:
        The mode of wave port ``"p1"`` from 1 to 100 GHz::

            problem = gfem.PortMode("p1", np.linspace(1e9, 100e9, 100))
            modes = gfem.solve(model, problem)

    Args:
        port: The name of a `WavePort` of the model.
        frequency: One frequency or a 1D sequence of distinct frequencies, Hz,
            in any order; results keep the order given.
        mode: Which propagating mode at the lowest frequency, in rank
            order: by decreasing effective index for lossless modes (1 = the
            most confined), with strongly attenuated modes ranked lower.
            ``None``: the wave port's own ``mode``.

    Raises:
        TypeError: ``port`` is not a name.
        ValueError: A frequency is not finite and positive, or ``mode`` is
            not an integer >= 1.
        NonDifferentiableError: ``frequency`` is computed inside `jax.grad`,
            `jax.jit` or `jax.vmap` from their arguments; pass a fixed value.

    `gfem.solve` raises `PortModeError` when the mode is below cutoff
    somewhere in the band (naming the estimated cutoff) or degenerate, or
    follows the same mode as another wave port of its surface, and
    warns with `PortModeTrackingWarning` when it crosses another mode. The
    2D problem always runs SuperLU and SciPy on the CPU, so it ignores
    `SolverOptions.device`, ``linear_solver`` and ``eigen_solver``, as a
    driven solve's port mode sweep does: one `SolverOptions` serves a
    `PortMode` check and the driven solve after it.

    Port modes are constants: under `jax.grad` an input the problem reads
    (an electromagnetic material property, the node coordinates) raises
    `gfem.NonDifferentiableError` and a thermal property gets an exact
    zero. Under `jax.jit` and `jax.vmap` the solve runs, but the result's
    ``e_field`` is not available.
    """

    port: str
    frequency: tuple[float, ...]
    mode: int | None
    _rule: ClassVar[str] = "gfem._sweep.port_mode:PortModeRule"

    def __init__(self, port: str, frequency: Any, *, mode: int | None = None) -> None:
        """Validate and normalize (see the class docstring)."""
        if not isinstance(port, str):
            raise TypeError(
                f"PortMode.port must be the name of a WavePort, got {port!r}."
            )
        if mode is not None and (
            isinstance(mode, bool)
            or not isinstance(mode, (int, np.integer))
            or mode < 1
        ):
            raise ValueError(
                f"PortMode.mode must be an integer >= 1 or None, got {mode!r}."
            )
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "frequency", _frequencies(frequency))
        object.__setattr__(self, "mode", None if mode is None else int(mode))


jax.tree_util.register_static(PortMode)


Sweep = Literal["uniform", "adaptive"]
_SWEEPS = get_args(Sweep)
#: Frequency grids up to this size are listed in full in error messages.
_LISTED = 8


def _port_names(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = (value,) if isinstance(value, str) else tuple(value)
    if not all(isinstance(n, str) for n in names):
        raise TypeError(f"Driven.excite takes port names (strings), got {value!r}.")
    if not names:
        raise ValueError(
            "Driven.excite needs at least one port name; None excites every port."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"Driven.excite lists a port twice: {value!r}.")
    return names


@dataclasses.dataclass(frozen=True, init=False)
class Driven:
    """The frequency-domain response of a model to its ports.

    At each frequency, each excited port in turn is fed 1 W time-averaged
    incident power while every other port is terminated in its own
    impedance, and the electric field solves the time-harmonic Maxwell
    equations (e^{+iωt}) in the electromagnetic solve domain. The result is
    the scattering matrix (power waves, each lumped port's reference
    impedance its resistance; see `gfem.NetworkParameters`) and, on request,
    the electric fields (see `gfem.conventions` for their amplitude).

    A `WavePort` is fed by its own port mode, solved before the 3D solve to
    ``SolverOptions.port_mode_tolerance``; S is the field's projection on
    that mode, and ``WavePort(offset=...)`` de-embeds it. The port absorbs
    its own mode without reflection, hybrid modes included, and the modes
    are on ``DrivenResult.port_modes``.

    Example:
        Solve 100 frequencies in full, or read a 300-point sweep from a few
        full solves::

            uniform = gfem.Driven(frequency=np.linspace(1e9, 100e9, 100))
            adaptive = gfem.Driven(
                frequency=np.linspace(1e9, 100e9, 300), sweep="adaptive"
            )
            result = gfem.solve(model, adaptive)

    Args:
        frequency: One frequency or a 1D sequence of distinct frequencies, Hz,
            in any order; results keep the order given.
        excite: The port name or names fed in turn, one column of S each, in
            this order. ``None``: every port of the model, in model order.
        sweep: ``"uniform"`` solves every frequency in full.
            ``"adaptive"`` solves a few of them in full (every excited port
            at once, and one more right-hand side per port left out of
            ``excite``, which keeps gradients exact there) and reads S and
            the kept fields at every frequency from a reduced model built
            on those solutions: the greedy picks each next frequency from
            ``frequency`` where the reduced model is least accurate, until
            ``SolverOptions.adaptive_memory`` samples in a row agree with it to
            ``SolverOptions.adaptive_tolerance`` (relative error of the
            field). S is then accurate to roughly the square of that error.
            Worth it above about 20 frequencies; the result's ``sweep``
            tells which frequencies were solved.
        samples: With ``sweep="adaptive"``, the frequencies to solve in full,
            in order, instead of the greedy's choice: each must be one of
            ``frequency``. Freezing the samples of a first sweep
            (``samples=result.sweep.sampled_frequencies``) keeps the reduced
            model's sample set fixed while the model changes, e.g. across
            optimizer iterations, so the objective is smooth in the design.
            Under `jax.grad`, `jax.jit` or `jax.vmap` an adaptive sweep
            needs frozen samples. Its gradients hold the reduced basis
            fixed: equal to the full solve's at the sample frequencies,
            accurate to the sweep's field error between them.
        keep_fields: Which electric fields the result keeps, one per kept
            frequency and excited port: ``False`` (the default) none,
            ``True`` all, or a frequency or list of frequencies of
            ``frequency``. A field is a complex vector per DOF (16 bytes per
            DOF; 160 MB at 10M DOFs), so keep only what you use.
        wave_vector: The Bloch wave vector k = (kx, ky, kz), rad/m
            (Cartesian), of a model with `gfem.Periodic` pairs: across each
            pair, E(r + a) = exp(-i k·a) E(r), with a the pair's translation
            (`gfem.conventions`, Bloch periodicity). The same k at every
            frequency. ``None``: plain periodicity, the fields equal on both
            surfaces of every pair. Any vector, (0, 0, 0) included, declares
            a Bloch solve: a uniform sweep on the CPU, with
            ``linear_solver`` ``"direct"`` or ``"bddc_gmres"``, whose S is
            not symmetric (S(k)ᵀ = S(-k)) and is referenced to the field E,
            not to its periodic envelope (`gfem.conventions`).
        wave_vector_frequency: The frequency, Hz, at which ``wave_vector``
            holds, for a fixed-angle sweep: the solve at frequency f uses the
            wave vector ``wave_vector * f / wave_vector_frequency``. A Floquet
            port's half-space has a real constant index, so this keeps the
            incidence angle fixed in every port medium at once. Build
            ``wave_vector`` with `gfem.Model.bloch_wave_vector` at this
            frequency. ``None``: the same wave vector at every frequency.

    Raises:
        TypeError: ``excite`` is not port names, or ``keep_fields`` is not a
            bool or frequencies.
        ValueError: A frequency is not finite and positive, ``excite`` is
            empty or repeats a port, ``sweep`` is unknown, ``samples`` is
            given without ``sweep="adaptive"``, is empty or repeats a
            frequency, a ``samples`` or ``keep_fields`` frequency is not one
            of ``frequency``, ``wave_vector`` is not a finite 3-vector, or
            ``wave_vector_frequency`` is not a positive finite frequency or
            is given without ``wave_vector``. With ``wave_vector``,
            `gfem.solve` also raises it on a model without a `gfem.Periodic`
            pair, for a k with a component outside the span of the pairs'
            translations, and with
            ``linear_solver="petsc"``; use ``"direct"`` or ``"bddc_gmres"``.
        NotImplementedError: `gfem.solve` raises it for ``wave_vector`` with
            ``sweep="adaptive"`` or a GPU ``device``; solve a uniform sweep
            on the CPU.
        NonDifferentiableError: ``frequency``, ``samples``, ``keep_fields``,
            ``wave_vector`` or ``wave_vector_frequency`` is computed inside
            `jax.grad`, `jax.jit` or `jax.vmap` from their arguments; pass a
            fixed value.
        LinearSolverConvergenceError: `gfem.solve` raises it when an
            iterative linear solve (``linear_solver="bddc_gmres"``, or
            ``"petsc"`` with a Krylov ``ksp_type``) does not converge at some
            frequency. A uniform sweep stops there and keeps what it solved:
            the error's ``.result`` holds the frequencies solved before it,
            its ``.unsolved`` the rest; solve ``err.unsolved`` again with
            ``linear_solver="direct"``. An adaptive sweep keeps neither,
            since its reduced model needs every sample; rerun it with
            ``linear_solver="direct"``.

    Notes:
        Below the mesh's low-frequency floor, k0·h_min = 1e-6 (f·h_min ≈
        48 Hz·m, with h_min the shortest edge, at order 2), the solve warns
        with `gfem.LowFrequencyWarning`. The floor is 48 MHz with 1 µm
        elements and 5 GHz with 10 nm elements. Below it, the E-field
        formulation's rounding error grows like 1e-16/(k0·h_min)², so the
        fields and gradients lose digits. With an exact factorization
        (``linear_solver="direct"``, or ``"petsc"`` with its default MUMPS
        factorization) the capacitive part of S loses digits too. The
        inductance read from S and reciprocity are unaffected down to at
        least k0·h_min = 1e-8. MKL PARDISO becomes unreliable below
        k0·h_min ≈ 3e-8, and ``"direct"`` uses it whenever the ``mkl`` extra
        is installed. Below that, use ``linear_solver="petsc"`` (MUMPS), or
        ``"direct"`` in an environment without the ``mkl`` extra. With
        ``bddc_gmres``, S stays at ``SolverOptions.tolerance``, but a
        capacitance read from S is good only to about tolerance/(ωCZ0). For
        capacitance at low frequency, solve `gfem.Electrostatic`.

        A driven solve ignores ``SolverOptions.eigen_solver``.
    """

    frequency: tuple[float, ...]
    excite: tuple[str, ...] | None
    sweep: Sweep
    keep_fields: bool | tuple[float, ...]
    samples: tuple[float, ...] | None = None
    wave_vector: tuple[float, float, float] | None = None
    wave_vector_frequency: float | None = None
    _rule: ClassVar[str] = "gfem._engine.driven:DrivenRule"

    def __init__(
        self,
        frequency: Any,
        *,
        excite: str | Sequence[str] | None = None,
        sweep: Sweep = "uniform",
        samples: Sequence[float] | None = None,
        keep_fields: bool | float | Sequence[float] = False,
        wave_vector: Sequence[float] | None = None,
        wave_vector_frequency: float | None = None,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        if sweep not in _SWEEPS:
            raise ValueError(
                f"Driven.sweep must be 'uniform' or 'adaptive', got {sweep!r}."
            )
        frequencies = _frequencies(frequency, "Driven")
        object.__setattr__(self, "frequency", frequencies)
        object.__setattr__(self, "excite", _port_names(excite))
        object.__setattr__(self, "sweep", sweep)
        object.__setattr__(self, "keep_fields", _kept(keep_fields, frequencies))
        object.__setattr__(self, "samples", _samples(samples, sweep, frequencies))
        object.__setattr__(self, "wave_vector", _wave_vector(wave_vector, "Driven"))
        object.__setattr__(
            self,
            "wave_vector_frequency",
            _reference_frequency(wave_vector_frequency, self.wave_vector),
        )

    @property
    def kept_frequencies(self) -> tuple[float, ...]:
        """The frequencies whose fields are kept, in ``frequency`` order."""
        if self.keep_fields is True:
            return self.frequency
        if self.keep_fields is False:
            return ()
        return self.keep_fields


def _kept(value: Any, frequencies: tuple[float, ...]) -> bool | tuple[float, ...]:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    check_fixed(value, "Driven.keep_fields must be fixed frequencies")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            f"Driven.keep_fields takes True, False or frequencies in Hz, got {value!r}."
        ) from None
    kept = set(_on_grid(wanted, frequencies, "keep_fields"))
    return tuple(frequencies[i] for i in sorted(kept))


def _samples(
    value: Any, sweep: str, frequencies: tuple[float, ...]
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if sweep != "adaptive":
        raise ValueError(
            "Driven.samples freezes the samples of an adaptive sweep; pass "
            "sweep='adaptive' with it."
        )
    check_fixed(value, "Driven.samples must be fixed frequencies")
    try:
        wanted = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raise TypeError(
            f"Driven.samples takes frequencies in Hz, got {value!r}."
        ) from None
    if not wanted.size:
        raise ValueError("Driven.samples needs at least one frequency.")
    indices = _on_grid(wanted, frequencies, "samples")
    if len(set(indices)) != len(indices):
        raise ValueError(f"Driven.samples names a frequency twice: {list(wanted)} Hz.")
    return tuple(frequencies[i] for i in indices)


def _on_grid(
    wanted: np.ndarray, frequencies: tuple[float, ...], what: str
) -> list[int]:
    """Index in ``frequencies`` of each of ``wanted`` (1e-9 relative)."""
    grid = np.asarray(frequencies)
    out = []
    for f in wanted:
        match = np.flatnonzero(np.isclose(grid, f, rtol=1e-9, atol=0))
        if not match.size:
            listed = (
                ", ".join(f"{g:g}" for g in grid)
                if len(grid) <= _LISTED
                else f"{len(grid)} frequencies, {grid.min():g} to {grid.max():g}"
            )
            raise ValueError(
                f"Driven.{what} frequency {f:g} Hz is not one of the "
                f"problem's frequencies ({listed} Hz)."
            )
        out.append(int(match[0]))
    return out


jax.tree_util.register_static(Driven)


@dataclasses.dataclass(frozen=True, init=False)
class Eigenmode:
    """The resonant modes of a model nearest a target frequency.

    Finds the source-free solutions of the time-harmonic Maxwell equations
    (e^{+iωt}) in the electromagnetic solve domain: complex angular
    frequencies ``omega`` (``Im omega > 0`` for a decaying mode) and their
    electric fields. Losses (``tan_delta``, ``conductivity``), resistive
    lumped ports and `Impedance` boundaries, and first-order `Absorbing`
    boundaries make the modes decay; each mode's Q is
    ``|omega| / (2 Im omega)``. The `n_modes` modes whose frequency
    ``Re omega / 2 pi`` is nearest ``target`` are returned, nearest first.

    A `Conductor` (its surface impedance grows as the square root of the
    frequency) or a second-order `Absorbing` boundary (a term in 1 /
    frequency) makes the eigenproblem nonlinear. The eigen solver's first pass
    then freezes those factors at ``target``. Each mode is refined on the
    exact frequency dependence, reusing that pass's factorization, so
    frequency and Q do not depend on ``target``. Conductor and radiation
    losses must be perturbations (Q above about 5); for a strongly radiating
    structure, find the resonance from a `gfem.Driven` sweep.

    Lumped ports are terminations here: nothing is excited. A port with a
    resistance gets a coupling Q, a port with an inductance a participation
    ratio (see `gfem.EigenmodeResult`).

    Example:
        The two modes nearest 5 GHz::

            result = gfem.solve(model, gfem.Eigenmode(target=5e9, n_modes=2))

    Args:
        target: The frequency to search around, Hz.
        n_modes: How many modes.
        wave_vector: The Bloch wave vector k = (kx, ky, kz), rad/m
            (Cartesian), of a model with `gfem.Periodic` pairs: across each
            pair, E(r + a) = exp(-i k·a) E(r), with a the pair's translation
            (`gfem.conventions`, Bloch periodicity). A band diagram is one
            problem per k on the same model (one k per problem). ``None``:
            plain periodicity, the fields equal on both surfaces of every
            pair. Any vector, (0, 0, 0) included, declares a Bloch solve:
            a lossless, undamped cell gives real ``omega``, and each mode's
            phase makes its largest coefficient real and positive.

    Raises:
        ValueError: ``target`` is not a finite positive number,
            ``n_modes`` is not an integer >= 1, or ``wave_vector`` is not a
            finite 3-vector.
        NonDifferentiableError: ``target`` or ``wave_vector`` is computed
            inside `jax.grad`, `jax.jit` or `jax.vmap` from their arguments;
            pass a fixed value.

    `gfem.solve` raises `gfem.EigenmodeConvergenceError` when fewer than
    ``n_modes`` modes converge (never a partial result), including when the
    eigen solver's first pass stops at its bound of 100 restarts. That happens
    where the modes near ``target`` are closer together than their linewidth:
    above a modal overlap of about 10 in a lossy region, "nearest" is
    ill-defined, and the message names the region. Close to a lossless resonance
    of a `gfem.PermittivityLaw` the modes crowd too, and a mode one step farther
    can come back in place of a nearer one: ask for more modes and compare two
    targets. A target or a mode in a law region's unresolved band, where its
    wavelength is shorter than two of its longest edges, raises ValueError. It
    raises NotImplementedError for a `WavePort` or a shared port surface (its
    propagation constants would have to be solved at the complex
    eigenfrequency): replace a lone wave port with a `LumpedPort` of the line
    impedance, or use a `gfem.Driven` sweep. With ``wave_vector``, it raises
    ValueError on a model without a `gfem.Periodic` pair, for a k with a
    component outside the span of the pairs' translations, and with any
    ``linear_solver`` but ``"direct"``; and NotImplementedError on a GPU
    ``device``.

    Notes:
        When ``target`` or the lowest mode found is below the mesh's
        low-frequency floor, k0·h_min = 1e-6 (f·h_min ≈ 48 Hz·m, with h_min
        the shortest edge; see `gfem.Driven`, Notes), the solve warns with
        `gfem.LowFrequencyWarning`. Below the floor the eigenfrequencies,
        the fields and the participations lose digits like
        1e-16/(k0·h_min)², and `gfem.EigenmodeResult.error` does not show
        it: a mode at k0·h_min = 2e-7 had a frequency off by 4e-4 with a
        reported error of 1e-16.
    """

    target: float
    n_modes: int
    wave_vector: tuple[float, float, float] | None = None
    _rule: ClassVar[str] = "gfem._eigen.rule:EigenmodeRule"

    def __init__(
        self,
        target: float,
        *,
        n_modes: int = 1,
        wave_vector: Sequence[float] | None = None,
    ) -> None:
        """Validate and normalize (see the class docstring)."""
        check_fixed(target, "Eigenmode.target must be a fixed number")
        try:
            value = float(target)
        except (TypeError, ValueError):
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Eigenmode.target must be a finite frequency > 0 Hz, got {target!r}."
            )
        if (
            isinstance(n_modes, bool)
            or not isinstance(n_modes, (int, np.integer))
            or n_modes < 1
        ):
            raise ValueError(
                f"Eigenmode.n_modes must be an integer >= 1, got {n_modes!r}."
            )
        object.__setattr__(self, "target", value)
        object.__setattr__(self, "n_modes", int(n_modes))
        object.__setattr__(self, "wave_vector", _wave_vector(wave_vector, "Eigenmode"))


jax.tree_util.register_static(Eigenmode)
