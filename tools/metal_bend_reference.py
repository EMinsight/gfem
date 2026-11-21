"""The metal-bend tutorial's case, and its fine-mesh reference value.

Usage: python tools/metal_bend_reference.py [OUTDIR]

Writes ``OUTDIR/metal-bend/fine.json`` (OUTDIR defaults to the current
directory): S21 and S11 of the 45 degree gold mirror, driven from p1 at 1.50,
1.55 and 1.60 um, on the tutorial's 2D section meshed with 0.0125 um cells
(eight times finer than ``docs/tutorials/data/bend2d.msh``, the z = 0 face of
`make_tutorial_meshes.bend` at that cell size) at element order 2. Copy it
into the reference data (``GFEM_REFERENCE_DIR``); the reference tier's
metal-bend test reads it from there. The solve takes about a minute and
4 GB on eight threads.

The case, shared with that test: the tutorial's bend (core 2.25 and cladding
1.444 in effective index, wave ports p1 and p2, a second-order absorbing
boundary on "outer"), its design square an index blend of the cladding and a
gold-like Drude metal (plasma frequency 2.183e15 Hz, damping 1.28e13 Hz) on a
0.05 um pixel grid. The mirror holds gold where x + y > 0.6 um, each pixel its
area fraction of gold. Posed with E along z on a triangle mesh, or as a slab
one cell thick between PEC plates on ``bend.msh``.
"""

import json
import os
import sys
import time
from pathlib import Path

import make_tutorial_meshes as mtm
import numpy as np
import structured_meshes as sm
from scipy.constants import c

import gfem
from gfem import design

DATA = Path(__file__).parents[1] / "docs" / "tutorials" / "data"
TUTORIAL_MESH = DATA / "bend2d.msh"
SLAB_MESH = DATA / "bend.msh"
FINE_FILE = "metal-bend/fine.json"
#: Cell size (um) and element order of the fine value.
FINE_H, FINE_ORDER = 0.0125, 2

um = 1e-6
FREQUENCIES = [c / (w * um) for w in (1.50, 1.55, 1.60)]
CLADDING = gfem.Material(eps_r=1.444**2)
CORE = gfem.Material(eps_r=2.25**2)
GOLD = gfem.Material(
    eps_r=gfem.PermittivityLaw(
        terms=[gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)]
    )
)


def case(mesh_path: Path, *, slab: bool = False):
    """The bend model on ``mesh_path`` (um) and its design's layer grid."""
    mesh = gfem.Mesh.from_gmsh(mesh_path, length_unit="um")
    boundaries = [gfem.Absorbing("outer", order=2)]
    if slab:
        boundaries.insert(0, gfem.PEC("plates"))
    model = gfem.Model(
        mesh,
        materials={"cladding": CLADDING, "core": CORE, "design": CLADDING},
        boundaries=boundaries,
        ports=[
            gfem.WavePort("p1", "port1", pec_edges=("outer",)),
            gfem.WavePort("p2", "port2", pec_edges=("outer",)),
        ],
    )
    return model, design.LayerGrid(mesh, "design", pitch=0.05 * um)


def problem(*, slab: bool = False) -> gfem.Driven:
    """The band solve, from p1; E along z on the 2D model."""
    if slab:
        return gfem.Driven(frequency=FREQUENCIES, excite="p1")
    return gfem.Driven(frequency=FREQUENCIES, excite="p1", polarization="Ez")


def mirror(grid) -> np.ndarray:
    """The 45 degree mirror: each pixel's area fraction of x + y > 0.6 um."""
    offsets = ((np.arange(8) + 0.5) / 8 - 0.5) * grid.pitch
    xs = grid.x[None, :, None, None] + offsets[None, None, None, :]
    ys = grid.y[:, None, None, None] + offsets[None, None, :, None]
    return ((xs + ys) > 0.6 * um).mean(axis=(2, 3))


def noisy_mirror(grid) -> np.ndarray:
    """The mirror with a little pixel noise, clipped to [0, 1].

    No filtered density then sits on the projection's clamp, where the
    gradient is zero.
    """
    noise = 0.05 * np.random.default_rng(0).standard_normal(grid.shape)
    return np.clip(mirror(grid) + noise, 0.0, 1.0)


def solve(model, grid, density, driven, order=None):
    """Solve ``driven`` with the design square holding ``density``."""
    field = gfem.MaterialField.interpolate(
        grid.to_elements(density), CLADDING, GOLD, scheme="index"
    )
    orders = {} if order is None else {"order": order}
    options = gfem.SolverOptions(linear_solver="direct", **orders)
    return gfem.solve(model.with_material("design", field), driven, options=options)


def band_s(model, grid, density, order=None, *, slab=False):
    """S21 and S11 over the band, as complex arrays."""
    s = solve(model, grid, density, problem(slab=slab), order).s
    return np.asarray(s["p2", "p1"]), np.asarray(s["p1", "p1"])


def write_fine_mesh(out: Path, h: float = FINE_H) -> Path:
    """The tutorial's 2D section with ``h`` um cells, written into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    return sm.write_msh22(out / f"bend2d_h{h:g}.msh", sm.z0_face(mtm.bend(h=h)))


def main(out: Path) -> Path:
    """Solve the fine value and write it under ``out``; return its path."""
    start = time.perf_counter()
    model, grid = case(write_fine_mesh(out))
    meshed = time.perf_counter() - start
    s21, s11 = band_s(model, grid, mirror(grid), FINE_ORDER)
    path = out / FINE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "case": "metal-bend gold mirror, 2D model, E along z",
        "h_um": FINE_H,
        "order": FINE_ORDER,
        "frequencies": FREQUENCIES,
        "s21": [[z.real, z.imag] for z in s21.tolist()],
        "s11": [[z.real, z.imag] for z in s11.tolist()],
        "gfem": gfem.__version__,
        "commit": os.environ.get("GFEM_SHA"),
        "mesh_seconds": meshed,
        "seconds": time.perf_counter() - start,
    }
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return path


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
