"""What the rows of a driven port pass share: its records and sample factor.

The planar port pass (`gfem._engine.port_pass`) and the line port pass
(`gfem._engine.line_port_pass`) read the same three records, so they live
below both.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from gfem._engine.line_mode import LineModeSpace
from gfem._engine.port_mode import PortModeSpace
from gfem._engine.port_mode_adjoint import shift_invert_factor
from gfem._sweep.port_mode import _wavenumbers

__all__ = ["Mode", "ModeValues", "Row", "RowInputs", "sample_factor"]


@dataclasses.dataclass(eq=False)
class ModeValues:
    r"""The modes of one surface at one frequency, ``nk`` of them.

    Attributes:
        field: ``(n_free, nk)`` the normalized field, :math:`(E_t, E_n)` on the
            free DOFs, at 1 W.
        gamma: ``(nk,)`` the propagation constants per file unit.
        voltage: ``(nk,)`` the voltages along the paths, NaN without one.
        phase: ``(nk,)`` the power phases :math:`P/|P|`.
    """

    field: np.ndarray
    gamma: np.ndarray
    voltage: np.ndarray
    phase: np.ndarray


@dataclasses.dataclass(eq=False)
class RowInputs:
    r"""What one frequency's port terms read from the 3D solve and the outputs.

    Attributes:
        omega: The angular frequency, rad/s.
        adjoint: ``(n_t, width)`` the traces :math:`a_j` of the adjoint columns.
        solution: ``(n_t, width)`` the traces :math:`u_j` of the solutions.
        weights: ``(nk, width)`` :math:`C_{ij} d_i s_j`, the cotangent of S
            (and of the port waves) with the de-embedding factors, per mode
            and column.
        gamma: ``(nk,)`` the cotangent of :math:`\gamma` per file unit that
            comes from outside the terms (the outputs, the de-embedding).
        impedance: ``(nk,)`` the cotangent of :math:`Z_{PV}`, ``0`` where a
            mode has no voltage path.
    """

    omega: float
    adjoint: np.ndarray
    solution: np.ndarray
    weights: np.ndarray
    gamma: np.ndarray
    impedance: np.ndarray


@dataclasses.dataclass(eq=False)
class Row:
    """One frequency's stored terms."""

    omega: float
    traces: list[tuple[np.ndarray, np.ndarray]]
    weights: np.ndarray
    gamma: np.ndarray
    impedance: np.ndarray
    #: The traces of the line ports' adjoint columns and solutions, per line.
    line_traces: list[tuple[np.ndarray, np.ndarray]] = dataclasses.field(
        default_factory=list
    )


@dataclasses.dataclass(eq=False)
class Mode:
    """One mode's polished pair and its normalization at one row."""

    x: np.ndarray
    lam: complex
    gamma: complex
    norm: Any
    reference: np.ndarray


def sample_factor(space: PortModeSpace | LineModeSpace, frequency: float) -> Any:
    """The shift-invert factor of the pencil at an exact sample's frequency."""
    k0, _, shift = _wavenumbers(space.section, frequency)
    a, b = space.pencil.at(k0)
    return shift_invert_factor(a, b, shift)
