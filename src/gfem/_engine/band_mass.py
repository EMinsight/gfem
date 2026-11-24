r"""The mass of the band-tracking overlaps (`gfem._band_tracking`'s host)."""

from __future__ import annotations

import ngsolve as ngs
import numpy as np
import scipy.sparse as sp

from gfem._engine.coefficients import coefficient
from gfem._engine.driven import driven_operator
from gfem._engine.eigen import _physical_eps
from gfem.model import Model


def mass_matrix(model: Model, order: int) -> sp.csr_matrix:
    r"""The mass :math:`\int \mathrm{Re}\,\varepsilon_r\, u\cdot v`, unpaired space.

    ``model`` carries concrete values (its leaves under
    `jax.lax.stop_gradient`); ``order`` is the solve's H(curl) order. A
    perfectly matched layer holds no weight, as the mode energy.
    """
    n_tets = len(model.mesh.tetrahedra)
    n_tris = len(model.mesh.triangles)
    op = driven_operator(
        model,
        order,
        elements=np.zeros(n_tets, bool),
        faces=np.zeros(n_tris, bool),
        paired=False,
    )
    eps = coefficient(op.built, np.real(_physical_eps(model)))
    u, v = op.space.TnT()
    form = ngs.BilinearForm(op.space, check_unused=False)
    form += (eps * u) * v * ngs.dx(bonus_intorder=2)
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    n = op.space.ndof
    return sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
    )
