r"""The band group velocity of Bloch modes, by Hellmann-Feynman.

A Bloch operator is :math:`T(k_0;\mathbf{k}) = P(\mathbf{k})^H A(k_0)
P(\mathbf{k})`, with :math:`A` the plain operator on the unpaired space, which
does not depend on :math:`\mathbf{k}`, and :math:`P` the expansion of the
periodic space onto it. Each stored entry of :math:`P` is a real coefficient
times :math:`e^{-i\,\mathbf{m}\cdot\mathbf{k}}` (:math:`\mathbf{m}` a lattice
vector), so :math:`\partial P/\partial k_d = -i\,m_d \circ P` entry by entry,
and the eigenvalue of a mode :math:`(x, y)` moves as

.. math::

    \frac{\partial k_0}{\partial k_d} =
    -\frac{(\overline{\partial_d P}\,y)^T A\,P x + (\overline{P}\,y)^T A\,
    \partial_d P\,x}{y^T T'(k_0)\,x}.

This is the exact derivative of the discrete band, complex on a lossy cell, and
it carries the dispersion of a permittivity law through :math:`T'`. A group of
modes within `GROUP_RTOL` of each other gets the matrices :math:`-G^{-1}
N_d` (:math:`G = Y^T T' X`, :math:`N_d` the numerator above in the group's
basis): their eigenvalues are the branch slopes along axis :math:`d`. The group
shares the mean slope when they agree, and has none (NaN) when they differ:
there the frequency is not differentiable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.constants as const
import scipy.sparse as sp

if TYPE_CHECKING:
    from gfem._eigen.operator import EigenOperator
    from gfem._eigen.reach import GapSet
    from gfem._engine.driven import DrivenOperator

#: Modes whose wavenumbers lie within this relative distance of the next one
#: below (in ascending real part) form one group. Looser than
#: ``SolverOptions.cluster_rtol``: a mesh without the cell's symmetry splits a
#: symmetry doublet by 7e-6 to 1.6e-4, and a split pair has no meaningful
#: slope of its own.
GROUP_RTOL = 1e-3
#: The slopes of a group agree when no branch's differs from their mean by more
#: than this fraction of the speed of light.
SLOPE_SPREAD = 1e-3
#: A group's ``G = Y^T T' X`` with a condition number above this is defective.
DEFECTIVE = 1e8


def groups(ks: np.ndarray, rtol: float = GROUP_RTOL) -> list[list[int]]:
    """Mode indices of each group, groups and members in ascending real part."""
    order = np.argsort(ks.real, kind="stable")
    out: list[list[int]] = []
    for j in order:
        if out and abs(ks[j] - ks[out[-1][-1]]) <= rtol * abs(ks[j]):
            out[-1].append(int(j))
        else:
            out.append([int(j)])
    return out


def unreturned(ks: np.ndarray, gaps: GapSet) -> np.ndarray:
    """The ``k`` of the pairs the search saw beyond the modes returned.

    The pool (converged pairs) and the Ritz values (the seed solve's, twice the
    modes asked for) each hold the modes among them: each mode takes its nearest
    entry of either out.
    """
    out = []
    for values in (gaps.pool, gaps.ritz):
        rest = list(np.asarray(values, dtype=complex))
        for k in ks:
            if rest:
                del rest[int(np.argmin(np.abs(np.asarray(rest) - k)))]
        out.extend(rest)
    return np.asarray(out, dtype=complex)


def _assembled(plain: DrivenOperator, omega: complex) -> sp.csr_matrix:
    """The plain operator at ``omega`` as a full CSR matrix."""
    form = plain.form(omega)
    form.Assemble()
    values, cols, rowptr = form.mat.CSR()
    n = plain.space.ndof
    matrix = sp.csr_matrix(
        (np.array(values, dtype=complex), np.array(cols), np.array(rowptr)),
        shape=(n, n),
    )
    if plain.symmetric and sp.triu(matrix, 1).nnz == 0:
        # a symmetric form stores its lower triangle only
        matrix = (matrix + matrix.T - sp.diags(matrix.diagonal())).tocsr()
    return matrix


def group_velocity(
    op: DrivenOperator,
    plain: DrivenOperator,
    eop: EigenOperator,
    *,
    ks: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    gaps: GapSet,
) -> np.ndarray:
    r"""``(n, 3)`` complex :math:`\partial\omega/\partial\mathbf{k}`, m/s, of the modes.

    Args:
        op: The Bloch driven operator, whose `Expansion` gives :math:`P` and
            the lattice vectors of its entries.
        plain: The same operator on the unpaired space (``paired=False``).
        eop: The eigen operator the modes were found on.
        ks: The modes' wavenumbers per file unit.
        x: The modes on the solve space, ``(ndof, n)``.
        y: Their left eigenvectors.
        gaps: The gap set of the search: its pool and Ritz values name the
            pairs the modes returned leave out.

    Returns:
        The group velocity of each mode, NaN where it is not defined: no
        phase template for ``P``, a group the returned modes cut, a defective
        group, or slopes that differ inside a group. Not snapped real: the
        caller does it with ``omega``.
    """
    out = np.full((len(ks), 3), np.nan + 0j)
    expansion = op.expansion
    lattice = None if expansion is None else expansion.lattice_vectors
    if expansion is None or lattice is None:
        return out
    p = expansion.matrix
    # an axis outside the span of the translations has no slope
    axes = [d for d in range(3) if lattice[:, d].any()]
    missing = unreturned(ks, gaps)
    # d k0 / dk per (file unit, rad/m) to d omega / dk, m/s
    to_ms = const.c / op.built.scale
    for members in groups(ks):
        if len(missing) and any(
            np.any(np.abs(missing - ks[j]) <= GROUP_RTOL * abs(ks[j])) for j in members
        ):
            continue
        mean = ks[members].mean()
        xs, ys = x[:, members], y[:, members]
        gram = ys.T @ np.stack([eop.derivative(mean, v) for v in xs.T], axis=1)
        if np.linalg.cond(gram) > DEFECTIVE:
            continue
        a = _assembled(plain, mean * to_ms)
        px, py = p @ xs, p.conj() @ ys
        ax = a @ px
        slopes = []
        for d in axes:
            dp = sp.csr_matrix(
                (-1j * lattice[:, d] * p.data, p.indices, p.indptr), shape=p.shape
            )
            num = (dp.conj() @ ys).T @ ax + py.T @ (a @ (dp @ xs))
            slopes.append(np.linalg.eigvals(-np.linalg.solve(gram, num)) * to_ms)
        mean_slope = np.array([s.mean() for s in slopes])
        spread = max(
            np.abs(s - m).max() for s, m in zip(slopes, mean_slope, strict=True)
        )
        if spread <= SLOPE_SPREAD * const.c:
            for j in members:
                out[j] = 0
                out[j, axes] = mean_slope
    return out
