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
from typing import TYPE_CHECKING, Literal

import jax
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import register_fields
from gfem.results import _variable, _xarray

if TYPE_CHECKING:
    import xarray

__all__ = ["TimeResponse"]


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
        time: The sample times in s, ``(n_time,)``, increasing; negative
            times hold the window's pre-response.
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

    def __getitem__(self, key: tuple[str, str]) -> jax.Array:
        """The response at port ``to`` to port ``from``, ``(n_time,)``.

        E.g. ``step["p2", "p1"]``, the transmitted step from ``p1`` to ``p2``.

        Raises:
            KeyError: ``to`` is not a port, or ``from`` is not an excited port.
        """
        to, source = key
        if to not in self.ports:
            raise KeyError(
                f"No port {to!r}; the ports are {', '.join(map(repr, self.ports))}."
            )
        if source not in self.excited:
            raise KeyError(
                f"Port {source!r} was not excited; the excited ports are "
                f"{', '.join(map(repr, self.excited))} (see Driven.excite)."
            )
        return self.values[:, self.ports.index(to), self.excited.index(source)]

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
        if port not in self.ports:
            raise KeyError(
                f"No port {port!r}; the ports are {', '.join(map(repr, self.ports))}."
            )
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
        i = self.ports.index(port)
        if not is_traced(self.z_ref) and np.isnan(np.asarray(self.z_ref)[i]):
            raise ValueError(
                f"Port {port!r} has no reference impedance (z_ref is NaN: a "
                "network of wave ports without voltage_path), so its step is "
                "not a voltage ratio and has no impedance profile. Pass z0= to "
                "step_response with a voltage_path on each WavePort, or give "
                "the ports one."
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


register_fields(TimeResponse, ("values", "z_ref"))
