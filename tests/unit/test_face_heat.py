"""Face heat: a heat source's per-face density (W/m^2) loads the thermal problem.

A source built from a solve may carry heat on faces beside its per-tetrahedron
density (sheet loss). The thermal load adds the face density times the test
function over the faces, and the face's thermal boundary decides where that
heat goes: a fixed temperature takes all of it, a convection boundary
splits it, an insulated face sends all of it into the model. These tests use
the private constructor, as a solve-built source would.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import thermal_cases as tc

import gfem

UM = 1e-6
L = 4.0  # um, slab height
K = 1.4
Q_S = 2e6  # W/m^2 on the top face
DIRECT = gfem.SolverOptions(order=2, linear_solver="direct")
EDGES = [
    (lambda p: abs(p[1]) < 1e-9, "bottom"),
    (lambda p: abs(p[1] - L) < 1e-9, "top"),
]


@pytest.fixture(scope="module")
def slab(tmp_path_factory):
    path = tmp_path_factory.mktemp("face") / "slab.msh"
    return tc.section(path, [(0.0, 0.0, 2.0, L, "slab", None)], 1.0, EDGES)


def sheet(mesh, surface, q):
    """A source with ``q`` W/m^2 on ``surface`` and no volume heat."""
    face = np.zeros(len(mesh.triangles))
    face[mesh.elements(surface)] = 1.0
    return gfem.HeatSource(
        regions=(),
        _kind="electromagnetic",
        _density=jnp.zeros(len(mesh.tetrahedra)),
        _face_density=jnp.asarray(face) * q,
    )


def _model(mesh, top=None):
    boundaries = [gfem.FixedTemperature("bottom", temperature=300.0)]
    if top is not None:
        boundaries.append(top)
    return gfem.Model(
        mesh,
        materials={"slab": gfem.Material(thermal_conductivity=K)},
        boundaries=boundaries,
    )


def _top(mesh):
    mid = mesh.section_thickness / 2
    x = np.array([0.3, 1.0, 1.7]) * UM
    return np.stack([x, np.full(3, L * UM), np.full(3, mid)], axis=1)


H = 2e5  # W/(m^2 K)


@pytest.mark.parametrize(
    ("top", "rise"),
    [
        # insulated: all the heat goes in, through the slab to the sink
        (None, Q_S * L * UM / K),
        # convection: the top keeps the share h / (h + k / L) of it
        (
            gfem.Convection("top", coefficient=H, ambient=300.0),
            Q_S / (H + K / (L * UM)),
        ),
        # fixed: the sink on the face takes it all
        (gfem.FixedTemperature("top", temperature=300.0), 0.0),
    ],
    ids=["insulated", "convection", "fixed"],
)
def test_face_heat_on_a_slab_top_gives_the_closed_form_rise(slab, top, rise):
    model = _model(slab, top)
    result = gfem.solve(
        model, gfem.SteadyThermal(sheet(slab, "top", Q_S)), options=DIRECT
    )
    # the rise is linear in depth: exact at order 2, round-off only
    np.testing.assert_allclose(result.rise.at(_top(slab)), rise, rtol=1e-9, atol=1e-12)
    # the power counts the face heat: q_s times the 2 um top, per length
    np.testing.assert_allclose(result.power_per_length, Q_S * 2 * UM, rtol=1e-12)


def _face_source(mesh, face):
    return gfem.HeatSource(
        regions=(),
        _kind="electromagnetic",
        _density=jnp.zeros(len(mesh.tetrahedra)),
        _face_density=face,
    )


def _profile(mesh):
    """A face density varying over the top's triangles, zero elsewhere."""
    rng = np.random.default_rng(0)
    face = np.zeros(len(mesh.triangles))
    top = mesh.elements("top")
    face[top] = Q_S * (1.0 + rng.random(len(top)))
    return face


def _check_fd(objective, face):
    grad = np.asarray(jax.grad(objective)(jnp.asarray(face)))
    rng = np.random.default_rng(1)
    direction = rng.standard_normal(len(face))
    h = 1e-3 * Q_S
    fd = (
        objective(jnp.asarray(face + h * direction))
        - objective(jnp.asarray(face - h * direction))
    ) / (2 * h)
    np.testing.assert_allclose(grad @ direction, float(fd), rtol=1e-6)
    return grad


def test_the_steady_face_density_gradient_matches_finite_differences(slab):
    model = _model(slab, gfem.Convection("top", coefficient=H, ambient=300.0))
    weights = np.linspace(1.0, 2.0, 40)

    def objective(face):
        problem = gfem.SteadyThermal(_face_source(slab, face))
        result = gfem.solve(model, problem, options=DIRECT)
        return jnp.sum(result.rise.coefficients[:40] * weights)

    grad = _check_fd(objective, _profile(slab))
    # heat on the bottom sink raises nothing: its derivative is zero
    np.testing.assert_allclose(grad[slab.elements("bottom")], 0.0, atol=1e-30)


STEPS = 6
DURATION = 2e-6


def test_the_transient_face_density_gradient_matches_finite_differences(slab):
    materials = {
        "slab": gfem.Material(
            thermal_conductivity=K, mass_density=2200.0, specific_heat=700.0
        )
    }
    model = gfem.Model(
        slab,
        materials=materials,
        boundaries=[
            gfem.FixedTemperature("bottom", temperature=300.0),
            gfem.Convection("top", coefficient=H, ambient=300.0),
        ],
    )
    weights = np.linspace(1.0, 2.0, STEPS + 1)

    def objective(face):
        problem = gfem.TransientThermal(
            _face_source(slab, face),
            duration=DURATION,
            steps=STEPS,
            waveform=[(0.0, 1.0), (DURATION / 2, 0.3)],
            traces={"slab": gfem.Trace.mean("slab")},
        )
        result = gfem.solve(model, problem, options=DIRECT)
        return jnp.sum(result.traces["slab"] * weights)

    _check_fd(objective, _profile(slab))


@pytest.fixture(scope="module")
def boxed(tmp_path_factory):
    """The slab under a 1 um box, whose lid is its outer face."""
    path = tmp_path_factory.mktemp("face") / "boxed.msh"
    rects = [(0.0, 0.0, 2.0, L, "slab", None), (0.0, L, 2.0, L + 1, "box", None)]
    edges = [*EDGES, (lambda p: abs(p[1] - L - 1) < 1e-9, "lid")]
    return tc.section(path, rects, 1.0, edges)


def _boxed_model(mesh):
    k = gfem.Material(thermal_conductivity=K)
    return gfem.Model(
        mesh,
        materials={"slab": k, "box": k},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        excluded={"thermal": ("box",)},
    )


def test_heat_on_an_excluded_tet_raises_with_its_region_and_watts(boxed):
    density = np.zeros(len(boxed.tetrahedra))
    density[boxed.elements("box")] = 1e6
    source = gfem.HeatSource(
        regions=(), _kind="electromagnetic", _density=jnp.asarray(density)
    )
    # 1e6 W/m^3 on the 2 um x 1 um box: 2e-6 W/m
    with pytest.raises(ValueError, match=r"2e-06 W/m on region 'box', which is excl"):
        gfem.solve(_boxed_model(boxed), gfem.SteadyThermal(source), options=DIRECT)


def test_face_heat_with_no_thermal_side_raises_with_its_surface_and_watts(boxed):
    source = sheet(boxed, "lid", 1e3)
    # 1e3 W/m^2 on the 2 um lid: 0.002 W/m
    with pytest.raises(ValueError, match=r"0.002 W/m on surface 'lid', which touch"):
        gfem.solve(_boxed_model(boxed), gfem.SteadyThermal(source), options=DIRECT)


def test_face_heat_with_one_thermal_side_enters_there(boxed):
    # the slab's top meets the excluded box: an insulated face of the slab
    result = gfem.solve(
        _boxed_model(boxed),
        gfem.SteadyThermal(sheet(boxed, "top", Q_S)),
        options=DIRECT,
    )
    np.testing.assert_allclose(result.rise.at(_top(boxed)), Q_S * L * UM / K, rtol=1e-9)


def test_the_repr_splits_volume_and_sheet_watts_when_sheet_heat_is_nonzero():
    def source(total, sheet, thickness=None):
        return gfem.HeatSource(
            regions=("slab",),
            _kind="electromagnetic",
            _total=total,
            _sheet_total=sheet,
            _thickness=thickness,
        )

    assert repr(source(3e-3, 1e-3)) == (
        "HeatSource(electromagnetic, regions=['slab'], total 0.003 W: "
        "volume 0.002 W, sheet 0.001 W)"
    )
    assert repr(source(3e-3, 0.0)) == (
        "HeatSource(electromagnetic, regions=['slab'], total 0.003 W)"
    )
    assert repr(source(3e-6, 1e-6, thickness=1e-6)) == (
        "HeatSource(electromagnetic, regions=['slab'], total 3 W/m: "
        "volume 2 W/m, sheet 1 W/m)"
    )
