"""The elements a measurement's side selects: named volume regions or a predicate.

`gfem.power_flow` and `gfem.heat_flow` (and a heat-out trace) take a side
as volume region names or a predicate on the element centroids; this
module turns either into a per-tetrahedron mask.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gfem.mesh import Mesh

__all__ = ["SideSpec", "selection"]

#: A side: names, or a centroid predicate.
SideSpec = tuple[str, ...] | Callable[[np.ndarray], np.ndarray]


def selection(
    mesh: Mesh,
    side: SideSpec,
    *,
    subject: str = "The side",
    refuse_surfaces: str | None = None,
) -> np.ndarray:
    """The selected tetrahedra: named volume regions or a centroid predicate.

    Args:
        mesh: The mesh.
        side: Volume region names, or ``inside(centroids) -> bool`` on the
            ``(n_tetrahedra, 3)`` element centroids.
        subject: How messages name the side (``"The side"``,
            ``"heat_flow's side"``).
        refuse_surfaces: The measurement that refuses a surface name, named
            in its message; ``None`` when the caller has kept volume
            regions only.

    Raises:
        RegionNotFoundError: A named region does not exist.
        ValueError: A name is a surface (with ``refuse_surfaces``), or the
            predicate does not return one boolean per tetrahedron.
    """
    n = len(mesh.tetrahedra)
    if callable(side):
        centroids = np.asarray(mesh.nodes)[mesh.tetrahedra].mean(axis=1)
        out = np.asarray(side(centroids))
        if out.dtype != bool or out.shape != (n,):
            raise ValueError(
                f"{subject} predicate must return one boolean per tetrahedron "
                f"({n},) from the (n_tetrahedra, 3) element centroids, got an "
                f"array of {out.dtype} of shape {out.shape}."
            )
        return out
    out = np.zeros(n, bool)
    for name in side:
        if refuse_surfaces is not None and mesh.region(name).dim != mesh.dim:
            raise ValueError(
                f"{name!r} is a surface region; {refuse_surfaces}'s side is a set "
                "of elements: volume regions, or a predicate on element centroids."
            )
        out[mesh.elements(name)] = True
    return out
