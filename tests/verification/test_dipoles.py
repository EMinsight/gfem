r"""Point dipoles against closed forms.

Each case solves its dipoles as columns of one factorization at order 3,
:math:`h \le \lambda/5` in the dipole's medium (the decided verification
mesh: the emitted power converges with the discretization of the whole
domain, not of the point load), and asserts the emitted power to 3 %:

- T8: in vacuum, electric moments along x and z and a magnetic moment
  along z at three positions, against the bulk power :math:`P_0`;
- T9: in a medium of :math:`\varepsilon_r = 4`, against :math:`n P_0` with
  :math:`n = 2` (the Purcell factor, which divides by the local bulk power,
  near 1);
- T10: above a ground plane at five heights, normal and tangential, against
  image theory, as a perfect conductor and then as a perfect magnetic wall,
  whose images flip sign (a ground named on part of the plane only gives
  the other wall's answer: both orientations of both walls are checked);
- T13: in a PEC guide where TE10 and TE20 propagate, the beta factors
  :math:`\beta_m = \sum_\text{ends}|b_m|^2 / P` against the guide's modal
  closed form to 1 %, and the guided powers summing to the emitted power.

Measured (order 3, one thread), relative to the closed form: T8 -0.22 %
to +0.23 % (magnetic included); T9 -0.3 % to +0.04 % (the Purcell factor);
T10 -0.18 % to +0.68 % over the conductor, -0.60 % to +2.06 % over the
magnetic wall (the largest where its normal dipole's power is 4 % of
:math:`P_0`); T13 beta -0.016 % and +0.006 %, the guided powers summing to
the emitted power to 3e-6.
"""

import math

import free_space_cases as fsc
import numpy as np
import pml_cases as pc
import pytest
import structured_meshes as sm

import gfem

pytestmark = pytest.mark.verification

OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")
#: The emitted power against its closed form.
REL = 3e-2
#: Moment sizes: any, the closed forms scale with them.
P = 1e-30
M = 1e-22


def _box(path, *, scale=1.0, sides=pc.SIDES, surface=None):
    """`fsc.dipole_box`, h = 0.2 um and a 0.4 um layer, every length times ``scale``."""
    return pc.box(
        path,
        sides=sides,
        thickness=0.4,
        cells=5,
        surface=surface,
        mapping=lambda x: x * scale,
    )


def _model(mesh, material=None, **kwargs):
    material = material or gfem.Material()
    return gfem.Model(
        mesh,
        materials={"air": material, "pml": material},
        pml=gfem.PML("pml"),
        **kwargs,
    )


def _ratios(model, dipoles, frequency=fsc.F0, eps_r=1.0):
    result = gfem.solve(model, gfem.Driven(frequency, sources=dipoles), options=OPTIONS)
    omega = 2 * math.pi * frequency
    return {
        d.name: float(result.emitted_power[d.name][0])
        / fsc.bulk_power(
            omega,
            np.asarray(d.moment),
            magnetic=isinstance(d, gfem.MagneticDipole),
            eps_r=eps_r,
        )
        for d in dipoles
    }


#: Generic points of the unit cube, off the grid's faces (um).
POSITIONS = ((0.47, 0.53, 0.56), (0.33, 0.61, 0.45), (0.58, 0.42, 0.37))


def _dipoles(positions, scale=1.0):
    out = []
    for k, point in enumerate(positions):
        r = np.asarray(point) * scale * fsc.UM
        out += [
            gfem.ElectricDipole(f"px{k}", position=r, moment=(P, 0, 0)),
            gfem.ElectricDipole(f"pz{k}", position=r, moment=(0, 0, P)),
            gfem.MagneticDipole(f"mz{k}", position=r, moment=(0, 0, M)),
        ]
    return out


def test_t8_a_dipole_in_vacuum_emits_its_bulk_power(tmp_path):
    mesh = gfem.Mesh.from_gmsh(_box(tmp_path / "box.msh"))
    ratios = _ratios(_model(mesh), _dipoles(POSITIONS))
    for name, ratio in ratios.items():
        assert ratio == pytest.approx(1.0, rel=REL), (name, ratio)


def test_t9_a_dipole_in_a_dielectric_emits_n_times_more(tmp_path):
    # the box at half size: h = lambda_n / 5 in eps_r = 4
    mesh = gfem.Mesh.from_gmsh(_box(tmp_path / "box.msh", scale=0.5))
    model = _model(mesh, gfem.Material(eps_r=4.0))
    dipoles = _dipoles(POSITIONS[:1], scale=0.5)
    vacuum = _ratios(model, dipoles)
    purcell = _ratios(model, dipoles, eps_r=4.0)
    for name, ratio in vacuum.items():
        assert ratio == pytest.approx(2.0, rel=REL), (name, ratio)
        assert purcell[name] == pytest.approx(1.0, rel=REL), (name, purcell[name])


#: Heights above the ground (um, in wavelengths), off the grid's z planes.
HEIGHTS = (0.05, 0.11, 0.21, 0.31, 0.41)


def _ground(c):
    return "ground" if abs(c[2]) < 1e-9 else ""


@pytest.fixture(scope="module")
def grounded(tmp_path_factory):
    # the layer on every side but -z: the plane z = 0 is the ground
    path = _box(
        tmp_path_factory.mktemp("ground") / "ground.msh",
        sides=("-x", "+x", "-y", "+y", "+z"),
        surface=_ground,
    )
    return gfem.Mesh.from_gmsh(path)


@pytest.mark.parametrize(("wall", "sign"), [(gfem.PEC, 1), (gfem.PMC, -1)])
def test_t10_a_dipole_above_a_ground_plane_meets_its_image(grounded, wall, sign):
    model = _model(grounded, boundaries=[wall("ground")])
    dipoles = []
    for k, h in enumerate(HEIGHTS):
        r = np.array([0.47, 0.53, h]) * fsc.UM
        dipoles += [
            gfem.ElectricDipole(f"n{k}", position=r, moment=(0, 0, P)),
            gfem.ElectricDipole(f"t{k}", position=r, moment=(P, 0, 0)),
        ]
    ratios = _ratios(model, dipoles)
    for k, h in enumerate(HEIGHTS):
        for normal, name in ((True, f"n{k}"), (False, f"t{k}")):
            image = fsc.image_power(h, normal=normal, sign=sign)
            assert ratios[name] == pytest.approx(image, rel=REL), (name, h, image)


# --- T13: a dipole in a two-mode guide ----------------------------------------------

#: The guide's section (um) and length, and its wavenumber: TE10 and TE20
#: propagate, TE01 does not (cutoff k = pi / b).
GUIDE_A, GUIDE_B, GUIDE_LENGTH = 1.0, 0.4, 2.0
GUIDE_K = 2.2 * math.pi
#: The dipole's x (both modes excited), at the middle of the guide.
GUIDE_X0 = 0.32


def _guide_mesh(path, h=0.1):
    edges = tuple(
        np.linspace(0.0, s, round(s / h) + 1) for s in (GUIDE_A, GUIDE_B, GUIDE_LENGTH)
    )
    eps = 1e-9

    def surface(c):
        if abs(c[2]) < eps:
            return "end1"
        if abs(c[2] - GUIDE_LENGTH) < eps:
            return "end2"
        return "wall"

    return sm.write_msh22(path, sm.structured(edges, lambda _: "air", surface))


def _guide_powers() -> np.ndarray:
    r"""Each mode's share :math:`P_m \propto \sin^2(m\pi x_0/a)/\beta_m`."""
    shares = []
    for m in (1, 2):
        beta = math.sqrt(GUIDE_K**2 - (m * math.pi / GUIDE_A) ** 2)
        shares.append(math.sin(m * math.pi * GUIDE_X0 / GUIDE_A) ** 2 / beta)
    shares = np.array(shares)
    return shares / shares.sum()


def test_t13_beta_factors_of_a_dipole_in_a_two_mode_guide(tmp_path):
    mesh = gfem.Mesh.from_gmsh(_guide_mesh(tmp_path / "guide.msh"))
    ports = [
        gfem.WavePort(f"{end}_m{m}", end, mode=m)
        for end in ("end1", "end2")
        for m in (1, 2)
    ]
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("wall")],
        ports=ports,
    )
    frequency = GUIDE_K / (2 * math.pi) * fsc.C0 / fsc.UM
    r0 = np.array([GUIDE_X0, 0.17, 0.53 * GUIDE_LENGTH]) * fsc.UM
    dipole = gfem.ElectricDipole("d", position=r0, moment=(0, P, 0))
    problem = gfem.Driven(frequency, excite=(), sources=dipole)
    result = gfem.solve(model, problem, options=OPTIONS)
    power = float(result.emitted_power["d"][0])
    guided = {
        p.name: float(np.abs(result.port_wave(p.name, "d")[0]) ** 2) for p in ports
    }
    beta = (
        np.array(
            [sum(guided[f"{end}_m{m}"] for end in ("end1", "end2")) for m in (1, 2)]
        )
        / power
    )
    np.testing.assert_allclose(beta, _guide_powers(), rtol=1e-2)
    assert sum(guided.values()) == pytest.approx(power, rel=5e-3)
