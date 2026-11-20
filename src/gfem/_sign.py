"""The reference coefficient of a mode's sign (or phase) convention."""

from __future__ import annotations

from typing import Any

import numpy as np

# Coefficients this close to the largest magnitude (relative) tie with it.
TIE_RTOL = 1e-12


def largest_entry(x: np.ndarray) -> int:
    """Index of the largest-magnitude entry of ``x``, ties to the lowest index.

    Symmetric structures give modes whose largest coefficients agree in
    magnitude up to rounding, with opposite signs; a plain ``argmax`` then
    picks one by rounding, so the sign of the mode would depend on the CPU.
    Every entry within `TIE_RTOL` of the maximum ties, and the lowest index
    among them wins.
    """
    magnitude = np.abs(x)
    return int(np.argmax(magnitude >= (1 - TIE_RTOL) * magnitude.max()))


def largest_value(x: np.ndarray, layout: Any = None, numbers: Any = None) -> complex:
    """The entry `largest_entry` picks, as one process would pick it.

    On one process (``layout`` ``None``, or one without a communicator) it is
    ``x[largest_entry(x)]``. Under ranks ``x`` is a state of ``layout``
    (`gfem._linalg.layout.RankLayout`) and ``numbers`` each local DOF's
    one-process number: the largest owned magnitude by a max-reduction, the
    lowest one-process number among the owned entries that tie with it by a
    min-reduction, and that entry from its owner by a sum. Every rank gets the
    same value, one process's entry whatever the partition.
    """
    comm = None if layout is None else layout.comm
    if comm is None:
        return x[largest_entry(x)]
    from gfem import _mpi

    owned = layout.owners
    values, mine = x[owned], np.asarray(numbers)[owned]
    magnitude = np.abs(values)
    top = _mpi.max_bcast(comm, np.array([magnitude.max(initial=0.0)]))[0]
    ties = magnitude >= (1 - TIE_RTOL) * top
    lowest = _mpi.agree_min(comm, int(mine[ties].min()) if ties.any() else _NONE)
    chosen = np.zeros(1, dtype=complex)
    chosen[0] = values[ties & (mine == lowest)].sum()
    return complex(_mpi.reduce_bcast(comm, chosen)[0])


#: Above every one-process DOF number: a rank with no tie offers this.
_NONE = 2**62
