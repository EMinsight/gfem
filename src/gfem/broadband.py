r"""Broadband models and time responses of network parameters.

Two views of S beyond the frequencies it was solved at:

- a rational model, :math:`S(f) = D + \sum_n R_n / (2\pi i f - p_n)`, fitted
  to S by vector fitting, made passive, and written as a SPICE subcircuit or
  a Verilog-A module;
- time responses: the impulse and step responses of S, from which the
  reflected step (TDR), the transmitted step (TDT) and the impedance profile
  are read.

Both start from S on the frequencies of a result (`gfem.NetworkParameters`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from gfem._broadband._poles import evaluate
from gfem._checks import is_traced
from gfem._pytree import readonly, register_fields
from gfem._text import hertz, hertz_band
from gfem.errors import NonDifferentiableError
from gfem.results import NetworkParameters, _variable, _xarray

if TYPE_CHECKING:
    import xarray

__all__ = ["RationalModel", "TimeResponse"]


# --- Time responses -------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class TimeResponse:
    r"""Impulse or step responses of S over time, labelled by port name.

    Built by `gfem.NetworkParameters.impulse_response` and
    `gfem.NetworkParameters.step_response`. ``values[m, i, j]`` is the
    response at port ``ports[i]`` at time ``time[m]`` to a unit impulse or
    step of the incident wave at port ``excited[j]``, every other port
    terminated in its reference impedance ``z_ref``. Values are
    dimensionless wave ratios: a step settles at :math:`S(0)`, and an
    impulse is scaled so that a frequency-independent reflection
    :math:`\Gamma` peaks at :math:`\Gamma`. Positive time is later under
    gfem's :math:`e^{+i\omega t}` (`gfem.conventions`, "Time responses").

    The reflected step at a port (TDR) is ``response[p, p]`` and its
    impedance profile `impedance`; the transmitted step (TDT) from port
    ``q`` to port ``p`` is ``response[p, q]``. `values` and `z_ref` are JAX
    arrays that differentiate with respect to S and the reference.

    Attributes:
        time: The sample times in s, a read-only NumPy array of shape
            ``(n_time,)`` (a pytree child, like `values`; under
            `jax.jit` it is a traced array): increasing for the windowed
            responses, whose negative times hold the window's
            pre-response; as given for a rational model's closed form.
        values: The responses, real, ``(n_time, n_ports, n_excited)``.
        z_ref: The real reference impedance of each port in ohm,
            ``(n_ports,)``; NaN for every port of a network without
            reference impedances (wave ports without ``voltage_path``).
        ports: Port names, the row order of `values`.
        excited: Names of the excited ports, the column order of `values`.
        kind: ``"impulse"`` or ``"step"``.
        mode: How the response was computed: ``"low_pass"`` (windowed
            transform of S on a harmonic grid), ``"band_pass"`` (magnitude
            of the windowed transform over a band) or ``"closed_form"``
            (exact response of a rational model).
    """

    time: np.ndarray
    values: jax.Array
    z_ref: jax.Array
    ports: tuple[str, ...]
    excited: tuple[str, ...]
    kind: Literal["impulse", "step"]
    mode: Literal["low_pass", "band_pass", "closed_form"]

    def __post_init__(self) -> None:
        """Store concrete `time` as a flat read-only float array."""
        if not is_traced(self.time):
            t = np.asarray(self.time, dtype=float).reshape(-1)
            object.__setattr__(self, "time", readonly(t))

    def __getitem__(self, key: tuple[str, str]) -> jax.Array:
        """The response at port ``to`` to port ``from``, ``(n_time,)``.

        E.g. ``step["p2", "p1"]``, the transmitted step from ``p1`` to ``p2``.

        Raises:
            KeyError: ``to`` is not a port, or ``from`` is not an excited port.
        """
        to, source = key
        i = self._port(to)
        if source not in self.excited:
            raise KeyError(
                f"Port {source!r} was not excited; the excited ports are "
                f"{', '.join(map(repr, self.excited))} (see Driven.excite)."
            )
        return self.values[:, i, self.excited.index(source)]

    def _port(self, port: str) -> int:
        """The row of ``port`` in `values`.

        Raises:
            KeyError: ``port`` is not a port.
        """
        if port not in self.ports:
            raise KeyError(
                f"No port {port!r}; the ports are {', '.join(map(repr, self.ports))}."
            )
        return self.ports.index(port)

    def impedance(self, port: str) -> jax.Array:
        r"""The impedance profile seen from ``port`` over `time`, ohm, ``(n_time,)``.

        :math:`Z(t) = Z_0 \frac{1 + \rho(t)}{1 - \rho(t)}`, with
        :math:`\rho` the reflected step ``self[port, port]`` and :math:`Z_0`
        the port's real reference impedance. Like an instrument's TDR
        profile it is exact only up to the first round trip past the first
        discontinuity: later, multiple reflections mix in (a 30 Ω load
        behind a 75 Ω line on a 50 Ω port reads 32.55 Ω on the segment after
        the line), and gfem does not correct for them. On a lossy line the
        profile drifts upward along the line, as measured profiles do.

        Raises:
            KeyError: ``port`` is not a port.
            ValueError: The response is not a step, ``port`` was not
                excited, or it has no reference impedance (NaN `z_ref`).
        """
        i = self._port(port)
        if self.kind != "step":
            raise ValueError(
                f"The impedance profile is read from a step response, not an "
                f"{self.kind} response; use NetworkParameters.step_response()."
            )
        if port not in self.excited:
            raise ValueError(
                f"The impedance profile of port {port!r} needs its reflected "
                f"step, but it was not excited (excited: "
                f"{', '.join(map(repr, self.excited))}). Solve with "
                "Driven(excite=...) including it."
            )
        if not is_traced(self.z_ref) and np.isnan(np.asarray(self.z_ref)[i]):
            raise ValueError(
                f"Port {port!r} has no reference impedance (z_ref is NaN: a "
                "network of wave ports without voltage_path), so its step is "
                "not a voltage ratio and has no impedance profile. Give each "
                "WavePort a voltage_path and solve again: S then has a "
                "reference impedance to read the profile on."
            )
        rho = self[port, port]
        return self.z_ref[i] * (1 + rho) / (1 - rho)

    def to_xarray(self) -> xarray.Dataset:
        """The responses and reference impedances as an `xarray.Dataset`.

        ``response`` has dimensions ``("time", "port", "excited")`` (as
        `values`) and ``z_ref`` (ohm) ``("port",)``; the coordinates are the
        times (s) and the port names; the attributes ``kind`` and ``mode``
        say what the responses are. Needs the ``xarray`` extra:
        ``pip install "gfem[xarray]"``.

        Raises:
            ImportError: xarray is not installed.
            TypeError: The response holds JAX tracers.
        """
        xr = _xarray(self)
        return xr.Dataset(
            {
                "response": _variable(("time", "port", "excited"), self.values),
                "z_ref": _variable(("port",), self.z_ref, "ohm"),
            },
            coords={
                "time": _variable(("time",), self.time, "s"),
                "port": list(self.ports),
                "excited": list(self.excited),
            },
            attrs={"kind": self.kind, "mode": self.mode},
        )

    def __repr__(self) -> str:
        t = np.asarray(self.time)
        span = f", {t[0]:.6g} s to {t[-1]:.6g} s" if t.size else ""
        return (
            f"TimeResponse({self.kind}, {self.mode}, ports={list(self.ports)}, "
            f"excited={list(self.excited)}, {t.size} times{span})"
        )


register_fields(TimeResponse, ("time", "values", "z_ref"))


# --- Rational models -------------------------------------------------------

#: Relative slack above the fitted range that `RationalModel.network` accepts,
#: so that a grid rebuilt as ``df * np.arange(...)`` over the same band, which
#: can end one rounding above the data's last frequency, is not refused.
_RANGE_SLACK = 1e-9
#: Relative mismatch accepted between the two poles (or residues) of a pair.
_PAIR_RTOL = 1e-12


@dataclasses.dataclass(frozen=True, eq=False)
class RationalModel:
    r"""A rational model of S: common poles, residue matrices and a constant.

    .. math::

        S(f) = D + \sum_n \frac{R_n}{2\pi i f - p_n}

    under :math:`e^{+i\omega t}` (so :math:`s = i\omega` is the Laplace
    variable), with poles :math:`p_n` in rad/s, :math:`\mathrm{Re}\, p_n < 0`,
    shared by every element of S. Poles are real or come in conjugate pairs
    whose residues are conjugate, so the model is real in the time domain.
    S is referred to one constant real reference impedance per port,
    `z_ref`, fixed when the model was fitted; to use another reference,
    fit again with ``z0=``. There is no term proportional to :math:`s`: S
    tends to :math:`D` at high frequency.

    `gfem.NetworkParameters.fit_rational` builds a model from S. A model can
    also be built from known poles and residues; the diagnostic fields then
    keep their defaults. Arrays are converted to read-only NumPy arrays: the
    model is a NumPy object, not differentiated by JAX.

    Attributes:
        poles: ``(n_poles,)`` complex, rad/s, :math:`\mathrm{Re}\, p < 0`;
            the two poles of a pair are adjacent, positive imaginary part
            first.
        residues: ``(n_poles, n_ports, n_ports)`` complex, rad/s; ``residues
            [n, i, j]`` belongs to S at port ``ports[i]`` excited from
            ``ports[j]``. Conjugate for the two poles of a pair, real for a
            real pole.
        d: ``(n_ports, n_ports)`` real, S as :math:`f \to \infty`.
        z_ref: ``(n_ports,)`` real ohm, the reference impedance of each port.
        ports: Port names, the row and column order of S.
        frequency_range: The lowest and highest data frequency of the fit,
            Hz; `network` refuses frequencies above it. None: no range.
        error: Largest per-element RMS of :math:`|S_\mathrm{model} -
            S_\mathrm{data}|` over the data frequencies.
        fit_error: The same before passivity enforcement.
        passivity_violations: Frequency intervals (Hz) where the largest
            singular value of the fitted S exceeded 1 before enforcement;
            the last one ends at ``inf`` when :math:`D` itself exceeded 1.
        nominal_reference: Whether `z_ref` is only a label: the data had no
            reference impedance (wave ports without ``voltage_path``, S on
            unit modal power), so S was not renormalized.
    """

    poles: np.ndarray
    residues: np.ndarray
    d: np.ndarray
    z_ref: np.ndarray
    ports: tuple[str, ...]
    frequency_range: tuple[float, float] | None = None
    error: float = math.nan
    fit_error: float = math.nan
    passivity_violations: tuple[tuple[float, float], ...] = ()
    nominal_reference: bool = False

    def __post_init__(self) -> None:
        """Check the shapes, the poles' stability and pairing, and `z_ref`."""
        _concrete(
            (self.poles, self.residues, self.d, self.z_ref),
            "RationalModel needs concrete poles, residues, d and z_ref",
        )
        ports = tuple(self.ports)
        if not all(isinstance(p, str) for p in ports) or len(set(ports)) != len(ports):
            raise ValueError(
                f"RationalModel.ports must be distinct names, got {self.ports!r}."
            )
        n = len(ports)
        # copies: the model's arrays are made read-only, the caller's are not
        poles = np.array(self.poles, complex).reshape(-1)
        residues = np.array(self.residues, complex)
        d = np.array(self.d)
        z_ref = np.array(self.z_ref)
        if residues.shape != (poles.size, n, n):
            raise ValueError(
                f"RationalModel.residues must have shape (n_poles, n_ports, "
                f"n_ports) = {(poles.size, n, n)}, got {residues.shape}."
            )
        if d.shape != (n, n) or z_ref.shape != (n,):
            raise ValueError(
                f"RationalModel.d must have shape {(n, n)} and z_ref {(n,)} for "
                f"{n} ports, got {d.shape} and {z_ref.shape}."
            )
        if np.iscomplexobj(d) and np.any(d.imag != 0):
            raise ValueError("RationalModel.d must be real (S at infinite frequency).")
        if np.iscomplexobj(z_ref) and np.any(z_ref.imag != 0):
            raise ValueError("RationalModel.z_ref must be real reference impedances.")
        d = np.real(d).astype(float)
        z_ref = np.real(z_ref).astype(float)
        if not (np.isfinite(d).all() and np.isfinite(residues).all()):
            raise ValueError("RationalModel residues and d must be finite.")
        if not (np.isfinite(z_ref).all() and (z_ref > 0).all()):
            raise ValueError(
                f"RationalModel.z_ref must be finite and > 0 ohm, got {z_ref.tolist()}."
            )
        if not (np.isfinite(poles).all() and (poles.real < 0).all()):
            raise ValueError(
                "RationalModel poles must be finite with a negative real part "
                f"(stable), got {poles.tolist()} rad/s."
            )
        _check_pairs(poles, residues)
        object.__setattr__(self, "ports", ports)
        for name, value in (
            ("poles", poles),
            ("residues", residues),
            ("d", d),
            ("z_ref", z_ref),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.frequency_range is not None:
            lo, hi = (float(x) for x in self.frequency_range)
            if not (math.isfinite(hi) and 0 <= lo <= hi):
                raise ValueError(
                    "RationalModel.frequency_range must be (lowest, highest) "
                    f"with 0 <= lowest <= highest Hz, got {self.frequency_range!r}."
                )
            object.__setattr__(self, "frequency_range", (lo, hi))
        object.__setattr__(
            self,
            "passivity_violations",
            tuple((float(a), float(b)) for a, b in self.passivity_violations),
        )
        object.__setattr__(self, "error", float(self.error))
        object.__setattr__(self, "fit_error", float(self.fit_error))
        object.__setattr__(self, "nominal_reference", bool(self.nominal_reference))

    @property
    def n_poles(self) -> int:
        """The number of poles, counting both poles of a pair."""
        return int(self.poles.size)

    @functools.cached_property
    def passive(self) -> bool:
        """Whether S has no singular value above 1 at any frequency.

        Computed from the model as it is (Hamiltonian test), on first use;
        the model is immutable, so the answer is kept.
        """
        from gfem._broadband._passivity import violations

        return not violations(self.poles, self.residues, self.d)

    def network(self, frequency: ArrayLike) -> NetworkParameters:
        r"""The model's S on the given frequencies, as `gfem.NetworkParameters`.

        ``network([0.0])`` is the DC limit :math:`D - \sum_n R_n / p_n`. A
        model whose `z_ref` is a nominal label gives a network without
        reference impedance (``z_ref`` NaN), like the data it was fitted to.

        Args:
            frequency: Frequencies in Hz, :math:`f \ge 0`, up to the highest
                of `frequency_range`; any order.

        Returns:
            S with every port excited, in `ports` order.

        Raises:
            NonDifferentiableError: ``frequency`` is traced by JAX.
            ValueError: A frequency is negative, not finite or above the
                fitted range (the model never extrapolates above its data).
        """
        _concrete(frequency, "RationalModel.network needs concrete frequencies")
        f = np.asarray(frequency, dtype=float).reshape(-1)
        if not (np.isfinite(f).all() and (f >= 0).all()):
            raise ValueError(
                f"RationalModel.network takes frequencies >= 0 Hz, got {f.tolist()}."
            )
        if self.frequency_range is not None and f.size:
            hi = self.frequency_range[1]
            if f.max() > hi * (1 + _RANGE_SLACK):
                raise ValueError(
                    f"RationalModel.network: {hertz(float(f.max()))} is above the "
                    f"fitted range {hertz_band(*self.frequency_range)}; a rational "
                    "model is not extrapolated above its data. Fit over a wider "
                    "band."
                )
        values = evaluate(2j * math.pi * f, self.poles, self.residues, self.d)
        n = len(self.ports)
        return NetworkParameters(
            values=jnp.asarray(values),
            z_ref=jnp.asarray(np.broadcast_to(self._response_z_ref(), (f.size, n))),
            frequency=tuple(float(x) for x in f),
            ports=self.ports,
            excited=self.ports,
        )

    def step_response(self, time: ArrayLike) -> TimeResponse:
        r"""The model's exact step response, a `gfem.TimeResponse`.

        The inverse Laplace transform of :math:`S(s)/s`,

        .. math::

            D + \sum_n \frac{R_n}{p_n} \left(e^{p_n t} - 1\right)

        for :math:`t \ge 0`, and 0 before: it jumps to :math:`D` at
        :math:`t = 0` and settles at the DC value ``network([0.0])``. This is
        the response a transient simulation of the model's netlist must
        reproduce; unlike `gfem.NetworkParameters.step_response` it is not
        band-limited, so it includes the model's behaviour outside the fitted
        band.

        Args:
            time: Times in s, any order; flattened to ``(n_time,)``.

        Returns:
            The step with every port excited, ``mode="closed_form"``; `z_ref`
            is NaN when the model's reference is a nominal label.

        Raises:
            NonDifferentiableError: ``time`` is traced by JAX.
            ValueError: A time is not finite.
        """
        _concrete(time, "RationalModel.step_response needs concrete times")
        t = np.asarray(time, dtype=float).reshape(-1)
        if not np.isfinite(t).all():
            raise ValueError(
                f"RationalModel.step_response takes finite times, got {t.tolist()}."
            )
        after = t >= 0
        # e^{p t} - 1 only where t >= 0: before the edge it would grow
        grow = np.expm1(np.where(after, t, 0.0)[:, None] * self.poles[None, :])
        scaled = self.residues / self.poles[:, None, None]
        step = self.d[None] + np.einsum("tn,nij->tij", grow, scaled).real
        values = np.where(after[:, None, None], step, 0.0)
        return TimeResponse(
            time=t,
            values=jnp.asarray(values),
            z_ref=jnp.asarray(self._response_z_ref()),
            ports=self.ports,
            excited=self.ports,
            kind="step",
            mode="closed_form",
        )

    def _response_z_ref(self) -> np.ndarray:
        """The reference of the model's responses: NaN when `z_ref` is a label."""
        if self.nominal_reference:
            return np.full(len(self.ports), np.nan)
        return self.z_ref

    def to_spice(self, path: str | Path, *, name: str | None = None) -> Path:
        r"""Write the model as a SPICE subcircuit that reproduces its S.

        The subcircuit has two pins per port, in `ports` order:
        ``<port>`` and ``<port>_ref``, with names lower-cased, every
        character outside ``a-z``, ``0-9`` and ``_`` replaced by ``_``, and
        ``p_`` put first when the name does not start with a letter. Between
        the two pins, port :math:`i` is a Norton circuit on its reference
        impedance :math:`z_i`: driven by power waves on :math:`z_i`, the
        subcircuit gives the model's S. Tie each ``*_ref`` pin to the
        port's return (``0`` for ground).

        Only resistors, capacitors, 0 V current-sense sources and linear
        controlled sources (``R``, ``C``, ``V``, ``G``, ``F`` cards) are
        used, numbers in exponent form without unit suffixes, and ``*``
        comments, so one file runs in common SPICE simulators. The header
        comments give the original port names, the reference impedances,
        the fitted range, `error`, the gfem version and the time
        convention.

        Not collective: across ranks a script calls it on rank 0.

        Args:
            path: The file to write; any suffix (``.sp``, ``.cir``, ``.lib``).
            name: The subcircuit name, sanitized as the pins are; default
                the file name without its suffix.

        Returns:
            The path written.

        Raises:
            ValueError: The model is not `passive` (a netlist of an active
                model can make a transient simulation diverge; fit with
                ``enforce_passivity=True``), two pins get one name or a pin
                is named ``gnd`` (rename the port), or ``name`` is empty.
        """
        from gfem._broadband import _netlist

        path = Path(path)
        name = self._netlist_name("to_spice", path, name)
        path.write_text(_netlist.spice(self, name))
        return path

    def to_verilog_a(self, path: str | Path, *, name: str | None = None) -> Path:
        r"""Write the model as a Verilog-A module that reproduces its S.

        The module has the pins of `to_spice` (``<port>``, ``<port>_ref`` in
        `ports` order, sanitized the same way) and the same realization:
        each pole's states are internal nodes integrated by ``ddt``, never a
        Laplace filter, whose coefficients overflow at high order. The code
        is fully unrolled (no arrays, ``genvar`` or ``parameter``), uses
        flow contributions only with each ``ddt`` alone in its statement,
        and numbers in exponent form, so it compiles with the strictest
        Verilog-A front ends. The header comments are those of `to_spice`.

        Not collective: across ranks a script calls it on rank 0.

        Args:
            path: The file to write, usually ``*.va``.
            name: The module name, sanitized as the pins are; default the
                file name without its suffix.

        Returns:
            The path written.

        Raises:
            ValueError: As `to_spice`, and when a pin or the module name is
                a Verilog-A reserved word (``input``, ``output``, ...;
                rename the port or pass another ``name``).
        """
        from gfem._broadband import _netlist

        path = Path(path)
        name = self._netlist_name("to_verilog_a", path, name)
        path.write_text(_netlist.verilog_a(self, name))
        return path

    def _netlist_name(self, method: str, path: Path, name: str | None) -> str:
        """The sanitized netlist name, after the passivity and pin checks."""
        from gfem._broadband import _netlist

        if not self.passive:
            raise ValueError(
                f"RationalModel.{method}: the model is not passive (a singular "
                "value of S exceeds 1), and a netlist of it can make a "
                "transient simulation diverge. Fit with enforce_passivity=True."
            )
        _netlist.pins(self.ports)
        name = path.stem if name is None else name
        if not name:
            raise ValueError(f"RationalModel.{method}: name must not be empty.")
        return _netlist.sanitize(name)

    def __repr__(self) -> str:
        band = (
            "no frequency range"
            if self.frequency_range is None
            else hertz_band(*self.frequency_range)
        )
        return (
            f"RationalModel(ports={list(self.ports)}, n_poles={self.n_poles}, "
            f"{band}, error={self.error:.3g}, passive={self.passive})"
        )


def _concrete(value: Any, what: str) -> None:
    """Refuse a JAX tracer where the NumPy model needs a value.

    Raises:
        NonDifferentiableError: ``value`` is traced by JAX.
    """
    if is_traced(value):
        raise NonDifferentiableError(
            f"{what}: the model is a NumPy object, not differentiated by JAX. "
            "Use it outside jax.grad, jax.jit or jax.vmap."
        )


def _check_pairs(poles: np.ndarray, residues: np.ndarray) -> None:
    """Real poles carry real residues; a complex pole is followed by its pair.

    The second pole and residue of each pair are set to the exact conjugates.
    """
    k = 0
    while k < poles.size:
        p = poles[k]
        if p.imag == 0:
            if np.any(residues[k].imag != 0):
                raise ValueError(
                    f"RationalModel: the real pole {p.real!r} rad/s has a complex "
                    "residue; a real pole's residue must be real."
                )
            k += 1
            continue
        pair_ok = (
            p.imag > 0
            and k + 1 < poles.size
            and abs(poles[k + 1] - np.conj(p)) <= _PAIR_RTOL * abs(p)
            and _close(residues[k + 1], np.conj(residues[k]))
        )
        if not pair_ok:
            raise ValueError(
                f"RationalModel: the complex pole {p!r} rad/s must be followed by "
                "its conjugate, positive imaginary part first, and the two "
                "residues must be conjugate (a model real in time)."
            )
        # exact conjugates, so that the model is real in time to the last bit
        poles[k + 1] = np.conj(p)
        residues[k + 1] = np.conj(residues[k])
        k += 2


def _close(a: Any, b: Any) -> bool:
    """Equal to the pair tolerance, relative to the larger magnitude."""
    scale = max(float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)))
    return bool(np.abs(a - b).max(initial=0.0) <= _PAIR_RTOL * scale)
