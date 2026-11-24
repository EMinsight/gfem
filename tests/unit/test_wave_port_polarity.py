r"""A wave port's polarity reference fixes its mode's sign from one point.

The 2D cases are a straight slab guide 2 um long along x (core
:math:`|y| < 0.4` um, n = 3.5 in n = 1.44, PEC sides at :math:`y = \pm 1.5`
um) at 1.55 um, meshed so that the two port lines carry mirrored node
layouts: the default sign of its odd mode then flips S21 with the meshing.
"""

import json
import warnings

import jax
import numpy as np
import pml_cases
import pytest
import structured_meshes as sm

import gfem

C0 = 299792458.0
F = C0 / 1.55e-6
W, L = 0.8, 2.0  # um
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct", threads=1)

# y grid lines of a port: dense in the lower core half (A) or its mirror (B)
_CLAD_LO = np.linspace(-1.5, -W / 2, 10)
_CLAD_HI = np.linspace(W / 2, 1.5, 10)
_CORE_A = np.r_[np.linspace(-W / 2, 0.15, 8), np.linspace(0.15, W / 2, 4)[1:]]
_A = np.r_[_CLAD_LO, _CORE_A[1:], _CLAD_HI[1:]]
_B = -_A[::-1]
# each port's layout: two with mirrored port meshes, two with equal ones
LAYOUTS = {
    "A to B": (_A, _B),
    "B to A": (_B, _A),
    "A to A": (_A, _A),
    "B to B": (_B, _B),
}


def _slab_mesh(path, y1, y2, nx=21):
    """The slab, its y grid morphing from ``y1`` (x = 0) to ``y2`` (x = L)."""
    xs = np.linspace(0, L, nx)
    ny = len(y1)
    t = (xs / L)[:, None]
    y = (1 - t) * y1[None, :] + t * y2[None, :]
    nodes = np.c_[np.repeat(xs, ny), y.ravel(), np.zeros(nx * ny)]

    def idx(i, j):
        return i * ny + j

    tris, names = [], []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            yc = 0.25 * (y[i, j] + y[i + 1, j] + y[i, j + 1] + y[i + 1, j + 1])
            for tri in ((a, b, c), (a, c, d)):
                tris.append(tri)
                names.append("core" if abs(yc) < W / 2 else "clad")
    segments = [(idx(i, 0), idx(i + 1, 0)) for i in range(nx - 1)]
    segments += [(idx(i, ny - 1), idx(i + 1, ny - 1)) for i in range(nx - 1)]
    segments += [(idx(0, j), idx(0, j + 1)) for j in range(ny - 1)]
    segments += [(idx(nx - 1, j), idx(nx - 1, j + 1)) for j in range(ny - 1)]
    seg_names = ["side"] * (2 * (nx - 1)) + ["port1"] * (ny - 1) + ["port2"] * (ny - 1)
    mesh = (nodes, np.array(tris), names, np.array(segments), seg_names)
    return gfem.Mesh.from_gmsh(sm.write_msh22(path, mesh), length_unit="um")


@pytest.fixture(scope="module")
def slabs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("slabs")
    return {
        name: _slab_mesh(folder / f"{i}.msh", *ys)
        for i, (name, ys) in enumerate(LAYOUTS.items())
    }


def _slab(mesh, *, mode=2, polarity=((0.0, 0.2e-6), (2e-6, 0.2e-6)), direction="+z"):
    ports = tuple(
        gfem.WavePort(
            name,
            surface,
            mode=mode,
            polarity=None if point is None else (point, direction),
        )
        for name, surface, point in zip(
            ("p1", "p2"), ("port1", "port2"), polarity, strict=True
        )
    )
    return gfem.Model(
        mesh,
        materials={
            "core": gfem.Material(eps_r=3.5**2),
            "clad": gfem.Material(eps_r=1.44**2),
        },
        boundaries=[gfem.PEC("side")],
        ports=ports,
    )


def _solve(model, problem):
    with warnings.catch_warnings():
        # mode 2 alone on a surface: mode 1's reflection is not under test
        warnings.simplefilter("ignore", gfem.SkippedPortModeWarning)
        return gfem.solve(model, problem, options=OPTIONS)


# --- the argument ----------------------------------------------------------------


def test_a_polarity_is_stored_as_a_point_and_a_unit_direction():
    port = gfem.WavePort("p", "s", polarity=([0, 1e-6, 0], (0, 2, 0)))
    assert port.polarity == ((0.0, 1e-6, 0.0), (0.0, 1.0, 0.0))
    assert gfem.WavePort("p", "s", polarity=((0, 1e-6), "-z")).polarity == (
        (0.0, 1e-6),
        (0.0, 0.0, -1.0),
    )
    assert gfem.WavePort("p", "s").polarity is None


@pytest.mark.parametrize(
    ("polarity", "message"),
    [
        (((0, 0, 0, 0), "+y"), r"'p'.*polarity.*point.*2 or 3"),
        (((0,), "+y"), r"'p'.*polarity.*point.*2 or 3"),
        (((0, np.nan, 0), "+y"), r"'p'.*polarity.*point.*finite"),
        (((0, 0, 0), "+w"), r"'p'.*polarity.*direction '\+w' is not an axis"),
        (((0, 0, 0), (0, 0, 0)), r"'p'.*polarity.*non-zero 3-vector"),
        ((0, 0, 0), r"'p'.*polarity.*\(point, direction\)"),
    ],
)
def test_a_malformed_polarity_raises(polarity, message):
    with pytest.raises(ValueError, match=message):
        gfem.WavePort("p", "s", polarity=polarity)


def test_a_polarity_and_a_voltage_path_raise_together():
    with pytest.raises(ValueError, match=r"'p'.*polarity.*voltage_path.*not both"):
        gfem.WavePort(
            "p",
            "s",
            voltage_path=((0, 0, 0), (0, 1e-6, 0)),
            polarity=((0, 0, 0), "+y"),
        )


def test_a_traced_polarity_raises():
    def build(y):
        gfem.WavePort("p", "s", polarity=((0.0, y, 0.0), "+y"))
        return y

    with pytest.raises(gfem.NonDifferentiableError, match=r"polarity.*fixed numbers"):
        jax.grad(build)(1e-6)


def test_the_model_checks_the_point_against_the_mesh_dimension(slabs, tmp_path):
    mesh2 = slabs["A to A"]
    with pytest.raises(ValueError, match=r"'p1'.*polarity.*2D.*\(x, y\)"):
        _slab(mesh2, polarity=((0.0, 0.2e-6, 0.0), None))
    with pytest.raises(ValueError, match=r"'p1'.*polarity.*line port.*'\+z' or '-z'"):
        _slab(mesh2, polarity=((0.0, 0.2e-6), None), direction="+y")
    with pytest.raises(ValueError, match=r"'w'.*polarity.*3D.*\(x, y, z\)"):
        _plate_ports(tmp_path, [gfem.WavePort("w", "top", polarity=((0, 0), "+x"))])


# --- the job file ----------------------------------------------------------------


def _plate_ports(tmp_path, ports):
    path = sm.write_msh22(
        tmp_path / "plate.msh", sm.plate(4.0, 3.0, (("oxide", 1.0),), (1, 1, 1))
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"oxide": gfem.Material()}, ports=ports
    )


def test_a_polarity_round_trips_through_a_job_and_is_left_out_when_unset(tmp_path):
    polar = gfem.WavePort("w1", "top", polarity=((1e-6, 1e-6, 1e-6), (0, 1, 1)))
    model = _plate_ports(tmp_path, [polar, gfem.WavePort("w2", "bottom")])
    path = gfem.Job(model, gfem.Driven(1e12)).to_json(tmp_path / "job.json")
    written = json.loads(path.read_text())["job"]["model"]["ports"]
    assert "polarity" in written[0]
    assert "polarity" not in written[1]
    assert gfem.Job.from_json(path).model.ports == model.ports


# --- the sign it fixes -----------------------------------------------------------


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_odd_mode_transmits_with_the_same_sign_on_any_port_meshing(
    slabs, layout, polarization
):
    # the same arrow at both ends of a straight guide: S21 is +e^{-gamma L}
    problem = gfem.Driven(F, excite=None, polarization=polarization)
    result = _solve(_slab(slabs[layout]), problem)
    gamma = complex(np.asarray(result.port_modes["p1"].gamma)[0])
    s21 = complex(np.asarray(result.s.values)[0, 1, 0])
    assert (s21 / np.exp(-gamma * L * 1e-6)).real > 0.99


def test_arrows_on_opposite_sides_of_the_odd_mode_flip_its_transmission(slabs):
    model = _slab(slabs["A to B"], polarity=((0.0, 0.2e-6), (2e-6, -0.2e-6)))
    result = _solve(model, gfem.Driven(F, excite=None, polarization="Ez"))
    gamma = complex(np.asarray(result.port_modes["p1"].gamma)[0])
    s21 = complex(np.asarray(result.s.values)[0, 1, 0])
    assert (s21 / np.exp(-gamma * L * 1e-6)).real < -0.99


@pytest.mark.parametrize("polarization", ["Ez", "Hz"])
def test_a_point_on_a_node_of_the_field_raises(slabs, polarization):
    model = _slab(slabs["A to B"], polarity=((0.0, 0.0), (2e-6, 0.2e-6)))
    problem = gfem.Driven(F, excite=None, polarization=polarization)
    with pytest.raises(
        gfem.PortModeError,
        match=r"WavePort 'p1'.*polarity.*of the port's peak at the point \(0, 0\) m "
        r"along \(0, 0, 1\) at .*THz.*Move the point towards the field's maximum",
    ):
        _solve(model, problem)


@pytest.mark.parametrize(
    ("point", "message"),
    [
        ((0.1e-6, 0.2e-6), r"'p1'.*polarity point.*off the port line"),
        ((0.0, 2e-6), r"'p1'.*polarity point.*off the port's cross-section"),
    ],
)
def test_a_point_off_the_port_raises(slabs, point, message):
    model = _slab(slabs["A to A"], polarity=(point, (2e-6, 0.2e-6)))
    with pytest.raises(ValueError, match=message):
        _solve(model, gfem.Driven(F, excite=None, polarization="Ez"))


def test_a_point_in_the_perfectly_matched_layer_raises(tmp_path):
    # the quarter strip of the layer tests: the +x layer starts at x = 0.9 um
    path = pml_cases.strip_guide(tmp_path / "strip.msh", 0.4, length=0.15)
    model = pml_cases.strip_model(path)
    arrow = gfem.WavePort("p1", "p1", polarity=((1.2e-6, 0.1e-6, 0.0), "+x"))
    model = model.replace(ports=[arrow, gfem.WavePort("p2", "p2")])
    problem = gfem.PortMode("p1", (pml_cases.STRIP_F,))
    with pytest.raises(
        ValueError, match=r"'p1'.*polarity point.*perfectly matched layer \('pml'\)"
    ):
        gfem.solve(model, problem, options=OPTIONS)


# --- the port mode problem -------------------------------------------------------


def test_the_port_mode_problem_reads_the_mode_the_driven_solve_uses(slabs):
    point = (0.0, 0.2e-6)
    model = _slab(slabs["B to A"])
    frequency = (F, 1.1 * F)
    problem = gfem.PortMode(port="p1", frequency=frequency, polarization="Ez")
    mode = _solve(model, problem)
    values = np.asarray(mode.e_field.at(point))
    assert values.shape == (2,)
    np.testing.assert_allclose(values.imag, 0.0, atol=1e-9 * abs(values).max())
    assert (values.real > 0).all()
    driven = _solve(model, gfem.Driven(F, excite=None, polarization="Ez"))
    np.testing.assert_allclose(
        np.asarray(driven.port_modes["p1"].e_field.at(point))[0], values[0], rtol=1e-6
    )


def test_the_port_mode_problem_drops_the_polarity_for_another_mode(slabs):
    # on mode 2's node: the arrow is mode 1's, so mode 2 takes the default sign
    model = _slab(slabs["B to A"], mode=1, polarity=((0.0, 0.0), (2e-6, 0.0)))
    problem = gfem.PortMode(port="p1", frequency=F, mode=2, polarization="Ez")
    other = _solve(model, problem)
    assert other.mode == 2
