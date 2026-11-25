"""A 2D layered guide with line ports, for the line port gradient checks.

6 x 4 mm, structured triangles. A core layer (1 <= y <= 2 mm) is split at
mid-length into ``core1`` (touching ``port1``, x = 0) and ``core2`` (touching
``port2``, x = 6 mm) in a ``clad`` cladding; ``side`` holds both y = 0 and
y = 4 mm.
"""

from pathlib import Path

import jax.numpy as jnp

import gfem

LX, H, Y0, Y1 = 6.0, 4.0, 1.0, 2.0
NX, NY = 12, 8
FREQUENCY = 40e9


def write_guide(path: Path, nx: int = NX, ny: int = NY) -> Path:
    """Write the guide as a gmsh 2.2 file (mm) and return its path."""
    dx, dy = LX / nx, H / ny

    def nid(i, j):
        return 1 + i + j * (nx + 1)

    nodes = [(i * dx, j * dy) for j in range(ny + 1) for i in range(nx + 1)]
    tags = {"clad": 1, "core1": 2, "core2": 3, "side": 4, "port1": 5, "port2": 6}
    els = []
    for j in range(ny):
        for i in range(nx):
            yc, xc = (j + 0.5) * dy, (i + 0.5) * dx
            reg = (
                "clad" if not (Y0 < yc < Y1) else ("core1" if xc < LX / 2 else "core2")
            )
            t = tags[reg]
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            if (i + j) % 2:
                els += [f"2 2 {t} {t} {a} {b} {c}", f"2 2 {t} {t} {a} {c} {d}"]
            else:
                els += [f"2 2 {t} {t} {a} {b} {d}", f"2 2 {t} {t} {b} {c} {d}"]
    for i in range(nx):
        els.append(f"1 2 4 4 {nid(i, 0)} {nid(i + 1, 0)}")
        els.append(f"1 2 4 4 {nid(i + 1, ny)} {nid(i, ny)}")
    for j in range(ny):
        els.append(f"1 2 5 5 {nid(0, j + 1)} {nid(0, j)}")
        els.append(f"1 2 6 6 {nid(nx, j)} {nid(nx, j + 1)}")
    dims = {"side": 1, "port1": 1, "port2": 1}
    text = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$PhysicalNames",
        str(len(tags)),
    ]
    text += [f'{dims.get(n, 2)} {t} "{n}"' for n, t in tags.items()]
    text += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
    text += [f"{k} {x} {y} 0" for k, (x, y) in enumerate(nodes, start=1)]
    text += ["$EndNodes", "$Elements", str(len(els))]
    text += [f"{k} {e}" for k, e in enumerate(els, start=1)]
    text += ["$EndElements", ""]
    path.write_text("\n".join(text))
    return path


def guide(mesh, core1, core2, *, tan_delta=0.0, boundaries=None, clad=None):
    """The guide with ``core1`` and ``core2`` as materials (arrays of
    per-element permittivities become fields)."""

    def material(name, eps):
        n = len(mesh.elements(name))
        eps = jnp.broadcast_to(jnp.asarray(eps), (n,))
        return gfem.MaterialField(eps_r=eps, tan_delta=jnp.full(n, tan_delta))

    return gfem.Model(
        mesh,
        materials={
            "clad": gfem.Material() if clad is None else material("clad", clad),
            "core1": material("core1", core1),
            "core2": material("core2", core2),
        },
        boundaries=[gfem.PEC("side")] if boundaries is None else boundaries,
        ports=[gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")],
    )
