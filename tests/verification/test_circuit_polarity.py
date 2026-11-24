r"""Relative port signs survive rigid motions of the model.

A WR-style E-plane bend (mm): a 10 x 20 mm guide along x from x = -30 to its
corner, then along y up to y = 40, PEC walls, TE10 at 10 GHz with E along the
10 mm side, so the bend turns E with the guide. The default sign of a 3D port
mode is tied to the global axes, so a quarter turn of the model can flip S21;
a polarity reference, or a voltage path, that moves with the model fixes it.
Port 1's arrow points along +y at the guide's centre, port 2's along -x: the
field carried round the bend. The rigid motions are exact in floating point
(quarter turns and a mirror), so S21 agrees to rounding.
"""

import numpy as np
import pytest
import structured_meshes as sm

import gfem

H = 5.0  # mm, cell size: the motions are exact, so the mesh need not be fine
F = 10e9
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)
# (quarter turns about z, mirror x first)
MOTIONS = ((0, False), (1, False), (2, False), (3, False), (0, True), (1, True))
# per port: surface, polarity (point mm, direction), voltage path ends (mm)
PORTS = {
    "p1": ("port1", ((-30, 5, 10), (0, 1, 0)), ((-30, 0, 10), (-30, 10, 10))),
    "p2": ("port2", ((5, 40, 10), (-1, 0, 0)), ((10, 40, 10), (0, 40, 10))),
}


def _move(points, quarter, mirror):
    p = np.array(points, float)
    if mirror:
        p[..., 0] = -p[..., 0]
    for _ in range(quarter):
        p = np.stack([-p[..., 1], p[..., 0], p[..., 2]], axis=-1)
    return p


def _bend(path, quarter, mirror):
    edges = (
        np.arange(-30, 10 + H / 2, H),
        np.arange(0, 40 + H / 2, H),
        np.arange(0, 20 + H / 2, H),
    )

    def volume(c):
        return "air" if c[1] < 10 or c[0] > 0 else None

    def surface(c):
        if abs(c[0] + 30) < 1e-6:
            return "port1"
        if abs(c[1] - 40) < 1e-6:
            return "port2"
        return "wall"

    mesh = sm.carve(edges, volume, surface, lambda p: _move(p, quarter, mirror))
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, mesh), length_unit="mm")


def _port(name, quarter, mirror, reference):
    surface, (point, direction), path = PORTS[name]
    if reference == "polarity":
        arrow = (
            tuple(_move(point, quarter, mirror) * 1e-3),
            tuple(_move(direction, quarter, mirror)),
        )
        return gfem.WavePort(name, surface, polarity=arrow)
    ends = _move(path, quarter, mirror) * 1e-3
    return gfem.WavePort(name, surface, voltage_path=tuple(map(tuple, ends)))


@pytest.fixture(scope="module")
def s21(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bends")
    out = {}
    for quarter, mirror in MOTIONS:
        mesh = _bend(folder / f"bend_{quarter}_{int(mirror)}.msh", quarter, mirror)
        for reference in ("polarity", "voltage_path"):
            model = gfem.Model(
                mesh,
                materials={"air": gfem.Material()},
                boundaries=[gfem.PEC("wall")],
                ports=[_port(name, quarter, mirror, reference) for name in PORTS],
            )
            result = gfem.solve(model, gfem.Driven(F, excite=None), options=OPTIONS)
            out[quarter, mirror, reference] = complex(
                np.asarray(result.s.values)[0, 1, 0]
            )
    return out


def test_an_e_plane_bend_keeps_its_s21_under_rotation_and_mirroring(s21):
    first = s21[0, False, "polarity"]
    assert abs(first) > 0.9
    for key, value in s21.items():
        assert abs(value - first) < 1e-8, key
