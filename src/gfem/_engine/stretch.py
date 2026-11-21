r"""The PML tensor of a perfectly matched layer as NGSolve coefficient functions.

Under gfem's :math:`e^{+i\omega t}` the stretch factor along a stretch axis
is :math:`s = 1 - i\sigma(\xi)`, with :math:`\sigma = \sigma_\max(\xi/d)^2`
at the depth :math:`\xi` beyond the interior box's face (`gfem._pml`); the
PML tensor is
:math:`\Lambda = \operatorname{diag}(s_y s_z / s_x,\ s_x s_z / s_y,\ s_x s_y / s_z)`.
"""

from __future__ import annotations

from typing import Any

from ngsolve import CF, IfPos, x, y, z

from gfem._pml import Stretch

__all__ = ["pml_tensor", "stretch_factors"]


def stretch_factors(
    stretch: Stretch,
    scale: float = 1.0,
    coords: tuple[Any, Any, Any] | None = None,
) -> tuple[Any, Any, Any]:
    r"""The stretch factors :math:`(s_x, s_y, s_z)` as coefficient functions.

    :math:`s_i = 1 - i\sigma_i`, :math:`\sigma_i = \sigma_\max(\xi/d)^2` at
    the depth :math:`\xi` beyond a covered face along axis :math:`i`, zero
    elsewhere, with each side's stretch maximum :math:`\sigma_\max`
    (`gfem._pml.Stretch.stretch_maximum`).

    Args:
        stretch: The layer's stretch for the solve.
        scale: Metres per unit of the coefficient functions' coordinates
            (the mesh file unit for gfem's discretizations).
        coords: The global coordinates as coefficient functions, in units
            of ``scale``; ``(x, y, z)`` by default (a 3D mesh). A port
            section passes its plane's map to global coordinates.
    """
    coord = (x, y, z) if coords is None else coords
    sigma = [CF(0.0), CF(0.0), CF(0.0)]
    for side in stretch.layers.sides:
        smax = stretch.stretch_maximum(side)
        d = side.thickness / scale
        xi = side.sign * (coord[side.axis] - side.face / scale)
        sigma[side.axis] = sigma[side.axis] + IfPos(xi, smax * (xi / d) ** 2, 0.0)
    sx, sy, sz = (1 - 1j * s for s in sigma)
    return sx, sy, sz


def pml_tensor(stretch: Stretch, scale: float = 1.0) -> tuple[Any, Any]:
    r"""The PML tensor :math:`\Lambda` and its inverse, 3x3 coefficient functions.

    :math:`\Lambda = \operatorname{diag}(s_y s_z/s_x,\ s_x s_z/s_y,\ s_x s_y/s_z)`
    multiplies the background permittivity and conductivity,
    :math:`\Lambda^{-1}` the inverse permeability. Arguments as
    `stretch_factors`.
    """
    sx, sy, sz = stretch_factors(stretch, scale)
    lam = (sy * sz / sx, sx * sz / sy, sx * sy / sz)
    inv = (sx / (sy * sz), sy / (sx * sz), sz / (sx * sy))

    def diag(v):
        return CF((v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]), dims=(3, 3))

    return diag(lam), diag(inv)
