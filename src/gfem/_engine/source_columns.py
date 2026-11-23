"""The source columns of a driven solve, beside the excited ports' columns.

A driven solve's block holds the excited ports' right-hand sides, then one
column per source (``Driven(sources=...)``), in the problem's order, on the
one factorization. `SourceColumns` is the seam between the port network
(`gfem._engine.port_network`), which knows the ports' columns, and each
source kind's load: it builds the whole block, splits the outgoing waves
into S and the sources' port waves, and reads each source's own outputs (a
dipole's emitted power).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const

from gfem._sweep.adaptive import AffineTerm

if TYPE_CHECKING:
    from gfem._engine.dipoles import PointLoads
    from gfem._engine.port_network import PortNetwork

__all__ = ["SourceColumns", "real_up_to_phase"]

MU0 = const.mu_0


def real_up_to_phase(moment: np.ndarray) -> bool:
    r"""Whether a moment is a real vector times one phase: :math:`\bar m \parallel m`.

    Then a dipole's emitted power has the dipole's own solution as its
    adjoint.
    """
    m = np.asarray(moment, complex)
    size = float(np.vdot(m, m).real)
    return abs(np.dot(m, m)) >= (1 - 1e-12) * size


@dataclasses.dataclass(frozen=True, eq=False)
class SourceColumns:
    """The block layout of one driven solve: ports first, then sources.

    Attributes:
        n_excited: The excited ports' columns, first in the block.
        names: The sources' names, in column order after them.
        dipoles: The point dipoles' loads, or ``None`` without dipoles.
        dipole_columns: The sources' positions (0-based after the ports) of
            the dipoles, in `PointLoads` column order.
    """

    n_excited: int
    names: tuple[str, ...] = ()
    dipoles: PointLoads | None = None
    dipole_columns: tuple[int, ...] = ()

    @property
    def width(self) -> int:
        """The block's columns."""
        return self.n_excited + len(self.names)

    def rhs(self, ports: PortNetwork, omega: float, size: int) -> np.ndarray:
        """``(size, width)``: the excited ports' loads, then the sources'."""
        out = np.zeros((size, self.width), complex)
        if ports.excitation:
            out[:, : self.n_excited] = ports.rhs(omega)
        if self.dipoles is not None:
            columns = [self.n_excited + c for c in self.dipole_columns]
            out[:, columns] = self.dipoles.rhs(omega)
        return out

    def outgoing(self, ports: PortNetwork, omega: float, x: np.ndarray) -> np.ndarray:
        """``(n_ports, width)``: each port's outgoing wave in each column of ``x``."""
        if not ports.outputs:
            return np.zeros((0, x.shape[1]), complex)
        return np.asarray(ports.outgoing(omega, x))

    def emitted_power(self, omega: float, x: np.ndarray) -> np.ndarray:
        """``(n_dipoles,)`` W, from the block's solutions ``x`` ``(N, width)``."""
        if self.dipoles is None:
            return np.zeros(0)
        return self.dipoles.emitted_power(omega, x[:, self._dipole_block()])

    def _dipole_block(self) -> list[int]:
        return [self.n_excited + c for c in self.dipole_columns]

    def power_loads(self, omega: float, cotangent: np.ndarray) -> np.ndarray:
        """``(N, width)``: the cotangent on ``x`` of a cotangent on the emitted power.

        Zero in the ports' and incident columns.
        """
        loads = self.dipoles
        if loads is None:  # pragma: no cover - callers check
            raise ValueError("no dipole")
        out = np.zeros((loads.vectors.shape[0], self.width), complex)
        out[:, self._dipole_block()] = loads.power_loads(omega, cotangent)
        return out

    def adjoint_loads(self, omega: float, size: int) -> np.ndarray:
        """``(size, q)``: the emitted powers' adjoint loads its solutions miss.

        A dipole whose moment is real up to one phase is its own adjoint
        (its power's load is parallel to its own); a circular one's is the
        conjugate load, which an adaptive sweep solves at each sample.
        """
        loads = self.dipoles
        if loads is None:
            return np.zeros((size, 0), complex)
        complex_ = [k for k, m in enumerate(loads.moments) if not real_up_to_phase(m)]
        return np.conj(loads.rhs(omega)[:, complex_])

    def reduced_power(
        self, omega: float, projected: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        """``(n_dipoles,)`` W from reduced coordinates ``(n, width)``.

        ``projected`` is ``Vᵀ`` `PointLoads.vectors`, ``(n, n_dipoles)``,
        on the real basis V.
        """
        loads = self.dipoles
        if loads is None:
            return np.zeros(0)
        y = coordinates[:, self._dipole_block()]
        pairs = np.conj(loads.coefficients(omega)) * np.sum(np.conj(projected) * y, 0)
        return -np.imag(pairs) / (2 * omega * MU0 * loads.scale)

    def reduced_power_loads(
        self, omega: float, projected: np.ndarray, cotangent: np.ndarray
    ) -> np.ndarray:
        """``(n, width)``: `power_loads` on a real basis, given its ``projected``."""
        loads = self.dipoles
        if loads is None:  # pragma: no cover - callers check
            raise ValueError("no dipole")
        out = np.zeros((projected.shape[0], self.width), complex)
        scale = np.asarray(cotangent, float) / (2 * omega * MU0 * loads.scale)
        c = np.conj(loads.coefficients(omega)) * scale
        out[:, self._dipole_block()] = 1j * np.conj(projected) * c[None, :]
        return out

    def excitation(self, port_terms: Sequence[AffineTerm]) -> list[AffineTerm]:
        """The adaptive sweep's excitation terms of the whole block.

        The ports' terms with their coefficients widened by zero columns,
        then one term per dipole kind (its load is a fixed vector times a
        scalar of ω, `PointLoads.coefficients`).
        """
        width, n = self.width, self.n_excited

        def widen(term: AffineTerm) -> AffineTerm:
            def coefficients(w: float, term: AffineTerm = term) -> np.ndarray:
                c = np.asarray(term.coefficients(w))
                out = np.zeros((c.shape[0], width), complex)
                out[:, :n] = c
                return out

            return AffineTerm(term.vectors, coefficients, term.conjugate)

        out = [widen(t) for t in port_terms]
        loads = self.dipoles
        if loads is not None:
            block = self._dipole_block()

            def dipole_coefficients(w: float) -> np.ndarray:
                out = np.zeros((len(block), width), complex)
                out[range(len(block)), block] = loads.coefficients(w)
                return out

            out.append(AffineTerm(loads.vectors, dipole_coefficients))
        return out
