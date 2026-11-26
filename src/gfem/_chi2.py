r"""The second-order nonlinear tensor :math:`d` of a material, contracted with fields.

A material's ``second_order_nonlinear_tensor`` is the contracted
:math:`d_{il} = d_{ijk}` (3x6, m/V, columns xx, yy, zz, yz, xz, xy, no factor 2
on the shear columns) in the frame of its ``axes``. The three-wave overlaps
of `gfem.three_wave` contract it with two fields, and a polarization source
would do the same: both go through `contract`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from gfem._materials import _VOIGT_ROW, piezoelectric_in_frame

__all__ = ["contract", "expand", "in_global_frame"]


def expand(d: Any) -> jax.Array:
    r""":math:`d_{ijk}`, ``(..., 3, 3, 3)``, from the contracted ``(..., 3, 6)``."""
    return jnp.asarray(d)[..., _VOIGT_ROW]


def in_global_frame(d: Any, axes: Any) -> Any:
    r"""The contracted tensor of the frame of ``axes`` (``None``: global), rotated.

    :math:`d^t_{pqs} = \sum_{lmn} Q_{pl} Q_{qm} Q_{sn} d^s_{lmn}`, :math:`Q =
    \mathrm{axes}^T`, as the piezoelectric tensor (a 3x6 tensor symmetric in
    its last pair rotates as one). NumPy in, NumPy out; JAX in, JAX out.
    """
    if axes is None:
        return d
    return piezoelectric_in_frame(d, axes, None)


def contract(d: Any, f: Any, g: Any) -> Any:
    r""":math:`\sum_{jk} d_{ijk} F_j G_k` of a contracted global-frame ``d``.

    ``d`` is ``(..., 3, 6)``, ``f`` and ``g`` ``(..., 3)`` (complex), broadcast
    against each other.
    """
    return jnp.einsum("...ijk,...j,...k->...i", expand(d), f, g)
