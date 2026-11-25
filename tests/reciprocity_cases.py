"""The cases of the reciprocal port waves: the terminated line and a guide.

`eigen_cases.terminated_line` (two 25 ohm lumped ports, 20 hexahedra of the
10 x 2 x 1 mm strip line), solved at order 2 (1214 DOFs): a sweep of tens of
frequencies with a handful of dipoles takes a second. The dipoles are
literals (mm), so a failure reproduces without a random generator.
"""

from pathlib import Path

import eigen_cases as ec
import numpy as np
import structured_meshes as sm
import wave_port_cases as w

import gfem
from gfem.mesh import LENGTH_UNITS

#: The band the line is swept over, Hz: its first resonances.
BAND = (4e9, 11e9)
ORDER2 = gfem.SolverOptions(order=2, linear_solver="direct")
#: Five positions inside the line, off the ends, mm.
POSITIONS_MM = (
    (6.287, 1.66, 0.69),
    (2.836, 1.171, 0.202),
    (0.685, 1.367, 0.714),
    (0.455, 1.07, 0.22),
    (7.945, 1.696, 0.638),
)
MOMENT = (0.0, 0.0, 1e-15)


def line_model(directory, *, held=()) -> gfem.Model:
    """The terminated line (ports "p1" and "p2", 25 ohm each).

    ``held``: dipoles whose elements form the volume "holder" (`hold`), of the
    line's material, so "dielectric" can take a `gfem.MaterialField`.
    """
    mesh = ec.line_mesh(directory, (20, 1, 1))
    if not held:
        return ec.terminated_line(mesh, 25.0)
    mesh = hold(mesh, Path(directory) / "held.msh", [d.position for d in held])
    material = gfem.Material(eps_r=ec.EPS)
    return gfem.Model(
        mesh,
        materials={"dielectric": material, HOLDER: material},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=25.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=25.0),
        ],
    )


def lumped_line(mesh: gfem.Mesh, capacitance) -> gfem.Model:
    """The line on ``mesh``, its ports 25 ohm in parallel with ``capacitance`` (F)."""
    return ec.line(
        mesh,
        ports=[
            gfem.LumpedPort(name, face, "+z", resistance=25.0, capacitance=capacitance)
            for name, face in (("p1", "port1"), ("p2", "port2"))
        ],
    )


#: The volume of the elements that hold dipoles (`hold`).
HOLDER = "holder"


def hold(mesh: gfem.Mesh, path: Path, positions) -> gfem.Mesh:
    """``mesh`` again, its tetrahedra holding ``positions`` (m) in volume "holder".

    A dipole lies in a constant `gfem.Material`: the rest of its region can
    then take a `gfem.MaterialField` design that reaches up to its element.
    A position on a face or an edge takes every element that holds it.
    """
    nodes = np.asarray(mesh.nodes, float)
    tets = np.asarray(mesh.tetrahedra)
    names = mesh._region_names()
    volumes = [names[int(t)] for t in mesh.tetrahedron_tags]
    corners = nodes[tets]
    edges = np.transpose(corners[:, :3] - corners[:, 3:], (0, 2, 1))
    for point in np.asarray(positions, float):
        local = np.linalg.solve(edges, (point - corners[:, 3])[..., None])[..., 0]
        bary = np.column_stack([local, 1 - local.sum(axis=1)])
        for k in np.flatnonzero(bary.min(axis=1) >= -1e-9):
            volumes[k] = HOLDER
    surfaces = mesh._region_names(surfaces=True)
    scale = LENGTH_UNITS[mesh.length_unit]
    written = (
        np.asarray(mesh.imported_nodes, float) / scale,
        tets,
        volumes,
        np.asarray(mesh.triangles),
        [surfaces[int(t)] for t in mesh.triangle_tags],
    )
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, written), mesh.length_unit)


def dipoles(n: int = 5, *, magnetic: bool = True) -> list:
    """``n`` dipoles at the literal positions: electric ones, the last one magnetic.

    ``n`` above 5 repeats the positions, shifted along x by 0.011 mm each time
    so that every position is distinct.
    """
    out = []
    for k in range(n):
        x, y, z = POSITIONS_MM[k % 5]
        position = (x * 1e-3 + 1e-5 * (k // 5) * 1.1, y * 1e-3, z * 1e-3)
        if magnetic and k == n - 1 and n > 1:
            out.append(
                gfem.MagneticDipole(f"d{k}", position=position, moment=(0, 1e-12, 0))
            )
        else:
            out.append(gfem.ElectricDipole(f"d{k}", position=position, moment=MOMENT))
    return out


def grid(n: int = 41) -> tuple[float, ...]:
    return tuple(np.linspace(*BAND, n))


#: W1: the guide B3 of `wave_port_cases` (WR-90, three eps_r = 10 windows, a
#: wave port on each end, Q 52 at 9.97 GHz), 11972 DOFs at order 2.
GUIDE_BAND = (9e9, 11e9)
#: Ten positions in its air sections "air_1" and "air_2", mm.
GUIDE_POSITIONS_MM = (
    (14.943, 5.558, 21.501),
    (9.312, 3.785, 6.052),
    (4.54, 5.72, 22.291),
    (15.564, 3.899, 6.638),
    (11.877, 4.28, 33.731),
    (7.833, 7.821, 31.761),
    (11.134, 3.113, 22.4),
    (18.722, 5.925, 36.837),
    (19.56, 2.097, 11.695),
    (8.727, 7.412, 34.029),
)


def guide_dipoles() -> list:
    """Electric dipoles along y (the TE10 field) at the guide's positions."""
    return [
        gfem.ElectricDipole(
            f"d{k}", position=tuple(1e-3 * np.asarray(p)), moment=(0.0, 1e-15, 0.0)
        )
        for k, p in enumerate(GUIDE_POSITIONS_MM)
    ]


def guide_model(directory, *, held=()) -> gfem.Model:
    """W1, ports "left" and "right"; ``held`` dipoles' elements apart (`hold`)."""
    mesh = w.ported_guide(Path(directory) / "b3.msh", w.CASE_B3, left_port=True)
    if not held:
        return w.ported_model(mesh, w.CASE_B3, w.TWO_PORTS)
    mesh = hold(mesh, Path(directory) / "held.msh", [d.position for d in held])
    layers = (*w.CASE_B3, (HOLDER, 1.0, 0.0))
    return w.ported_model(mesh, layers, w.TWO_PORTS)


def guide_grid(n: int = 41) -> tuple[float, ...]:
    return tuple(np.linspace(*GUIDE_BAND, n))
