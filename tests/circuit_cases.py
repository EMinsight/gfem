r"""A 2D slab guide with a design patch, shared by the SAX component tests.

A straight dielectric slab 2 um long along x: core :math:`|y| < 0.4` um,
n = 3.5, in n = 1.44, PEC sides at :math:`y = \pm 1.5` um, wave-port lines
``port1`` (x = 0) and ``port2`` (x = L). The triangles whose centroid lies
in :math:`0.6 < x < 1.4`, :math:`|y| < 0.6` um form the region ``design``,
filled by a per-element permittivity (core or cladding by centroid by
default). Both ports carry a polarity reference at the guide's centre, so
the even mode's S21 is close to :math:`e^{-\gamma L}`. Ez, order 2, direct.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import structured_meshes as sm

import gfem

C0 = 299792458.0
W, L = 0.8, 2.0  # um
N_CORE, N_CLAD = 3.5, 1.44
H = 0.1  # um, cell size
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)
#: Five wavelengths over 1.50-1.60 um, in Hz (decreasing).
BAND = C0 / (np.linspace(1.50, 1.60, 5) * 1e-6)


def count_solves(monkeypatch) -> list:
    """Count the solves of every `gfem.SaxComponent`, which still run.

    One entry per call of the component's solve entry point.
    """
    count = []
    real = gfem.solve

    def counted(*args, **kwargs):
        count.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr("gfem.circuit.solve", counted)
    return count


def slab_mesh(path: Path) -> gfem.Mesh:
    """Write and read the slab mesh (um)."""
    xs = np.linspace(0.0, L, round(L / H) + 1)
    ys = np.unique(
        np.r_[
            np.linspace(-1.5, -W / 2, 12),
            np.linspace(-W / 2, W / 2, 9),
            np.linspace(W / 2, 1.5, 12),
        ]
    )
    nx, ny = len(xs), len(ys)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.c_[gx.ravel(), gy.ravel(), np.zeros(nx * ny)]

    def idx(i, j):
        return i * ny + j

    tris, names = [], []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            for tri in ((a, b, c), (a, c, d)):
                x, y = nodes[list(tri), :2].mean(axis=0)
                if 0.6 < x < 1.4 and abs(y) < 0.6:
                    names.append("design")
                else:
                    names.append("core" if abs(y) < W / 2 else "clad")
                tris.append(tri)
    segments = [(idx(i, 0), idx(i + 1, 0)) for i in range(nx - 1)]
    segments += [(idx(i, ny - 1), idx(i + 1, ny - 1)) for i in range(nx - 1)]
    segments += [(idx(0, j), idx(0, j + 1)) for j in range(ny - 1)]
    segments += [(idx(nx - 1, j), idx(nx - 1, j + 1)) for j in range(ny - 1)]
    seg_names = ["side"] * (2 * (nx - 1)) + ["port1"] * (ny - 1) + ["port2"] * (ny - 1)
    mesh = (nodes, np.array(tris), names, np.array(segments), seg_names)
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, mesh), length_unit="um")


def design_eps(mesh: gfem.Mesh) -> np.ndarray:
    """The design patch's permittivity per element: the plain slab."""
    elements = np.asarray(mesh.volume_elements)[mesh.elements("design")]
    y = np.asarray(mesh.nodes)[elements].mean(axis=1)[:, 1]
    return np.where(np.abs(y) < W / 2 * 1e-6, N_CORE**2, N_CLAD**2)


def slab(
    mesh: gfem.Mesh, eps, *, offset: float = 0.0, polarity: bool = True
) -> gfem.Model:
    """The slab with ``eps`` on the design patch; ``offset`` (m) on port 2."""

    def arrow(x):
        return ((x, 0.0), "+z") if polarity else None

    return gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=N_CORE**2),
            "clad": gfem.Material(eps_r=N_CLAD**2),
            "design": gfem.MaterialField(eps_r=eps),
        },
        boundaries=[gfem.PEC("side")],
        ports=(
            gfem.WavePort("p1", "port1", polarity=arrow(0.0)),
            gfem.WavePort("p2", "port2", offset=offset, polarity=arrow(L * 1e-6)),
        ),
    )


def slab_function(mesh: gfem.Mesh, **kwargs) -> Callable[..., gfem.Model]:
    """``slab_at(*, eps=...)``: the slab as a function of its design patch."""
    base = design_eps(mesh)

    def slab_at(*, eps=base):
        return slab(mesh, eps, **kwargs)

    return slab_at


def scaled_function(mesh: gfem.Mesh, **kwargs) -> Callable[..., gfem.Model]:
    """``slab_at(*, scale=1.0)``: the design patch's permittivity times ``scale``."""
    base = design_eps(mesh)

    def slab_at(*, scale=1.0):
        return slab(mesh, scale * base, **kwargs)

    return slab_at
