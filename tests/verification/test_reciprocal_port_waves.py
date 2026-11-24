r"""Gradients of port waves read by reciprocity (`Driven(source_fields=False)`).

A dipole's port wave is :math:`W_{ic} = d_i\, b_c^\mathsf{T} z_i`, with
:math:`z_i` port :math:`i`'s own solution, :math:`b_c` the dipole's point
load and :math:`d_i` the de-embedding. Its backward solves one adjoint column
per port, :math:`y_i = A^{-1}\sum_c \bar W_{ic} b_c`, whatever the number of
dipoles, and pairs it with :math:`z_i`. The cases are the terminated line of
`reciprocity_cases` (L1, two lumped ports) and the guide B3 of
`wave_port_cases` (W1, two wave ports):

- on a uniform sweep the gradient equals the dipoles' own columns' (the
  column path) to round-off, in every input kind a dipole column takes;
- on an adaptive sweep the adjoint columns are solved at full size and
  paired with the ports' reduced solutions: exact at the samples, and
  accurate to the sweep's own field error between them, for a design next to
  a dipole (where a gradient on the frozen port basis is off by 10 %).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases
import pytest
import reciprocity_cases as rc

import gfem
from gfem._engine.system_operator import SystemOperator
from gfem.errors import NonDifferentiableError
from gfem.shape import boundary_morph

#: Relative to the max-norm of the column path's gradient: both are exact, so
#: round-off of differently ordered sums (6e-15 measured on the line).
UNIFORM_GATE = 1e-10


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    """The line, its dipoles' elements apart: "dielectric" takes a design."""
    return rc.line_model(tmp_path_factory.mktemp("line"), held=rc.dipoles())


def _eps_design(model, region):
    n = len(model.mesh.elements(region))
    eps = model.materials[region].eps_r
    return jnp.full(n, eps) * jnp.asarray(np.linspace(0.95, 1.05, n))


def _relative(got, want) -> float:
    got, want = np.asarray(got), np.asarray(want)
    return float(np.abs(got - want).max() / np.abs(want).max())


#: A fixed complex weight per (frequency, port, dipole): a linear objective.
def _weights(shape):
    rng = np.random.default_rng(7)
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _objectives(model, region, problem, options):
    """``sum |W|^2`` and ``Re sum a W`` in ``region``'s per-element eps_r."""

    def waves(eps):
        moved = model.with_material(region, gfem.MaterialField(eps_r=eps))
        return gfem.solve(moved, problem, options=options).port_waves

    def power(eps):
        w = waves(eps)
        return jnp.sum(jnp.abs(w) ** 2) / jnp.max(jnp.abs(jax.lax.stop_gradient(w)))

    def linear(eps):
        w = waves(eps)
        return jnp.real(jnp.sum(_weights(w.shape) * w))

    return power, linear


@pytest.fixture(scope="module")
def guide(tmp_path_factory):
    """W1, its dipoles' elements apart: "air_1" takes a design."""
    return rc.guide_model(tmp_path_factory.mktemp("guide"), held=rc.guide_dipoles())


def _column_and_reciprocal(model, region, objective, frequency, sources, **kwargs):
    """``objective``'s gradient with ``source_fields`` True, then False."""
    p = _eps_design(model, region)
    grads = []
    for source_fields in (True, False):
        problem = gfem.Driven(
            frequency, sources=sources, source_fields=source_fields, **kwargs
        )
        power, linear = _objectives(model, region, problem, rc.ORDER2)
        f = {"power": power, "linear": linear}[objective]
        grads.append(jax.grad(f)(p))
    return grads


@pytest.mark.parametrize("objective", ["power", "linear"])
def test_uniform_reciprocal_gradients_are_the_column_path_s(line, guide, objective):
    for model, region, frequency, sources, kwargs in (
        (line, "dielectric", rc.grid(3), rc.dipoles(), {}),
        (guide, "air_1", rc.guide_grid(3), rc.guide_dipoles(), {"excite": ()}),
    ):
        column, reciprocal = _column_and_reciprocal(
            model, region, objective, frequency, sources, **kwargs
        )
        assert np.abs(np.asarray(column)).max() > 0
        assert _relative(reciprocal, column) <= UNIFORM_GATE


# --- W1 values (lossless wave ports) ---------------------------------------------


@pytest.fixture(scope="module")
def plain_guide(tmp_path_factory):
    return rc.guide_model(tmp_path_factory.mktemp("plain_guide"))


#: Relative to max |W|: both read the same solves (1-2e-15 measured).
VALUE_GATE = 1e-12
#: Relative to max |W|, adaptive at adaptive_tolerance=1e-4 against the
#: uniform columns (8e-10 measured on this guide).
ADAPTIVE_VALUE_GATE = 1e-6


@pytest.mark.parametrize("excite", [None, ("left",), ()])
def test_wave_port_reciprocal_port_waves_are_the_dipoles_own_columns(
    plain_guide, excite
):
    # lossless lone wave ports are parallel: excite=None reads the excited
    # columns scaled, () the wave ports' adjoint loads
    results = [
        gfem.solve(
            plain_guide,
            gfem.Driven(
                rc.guide_grid(),
                excite=excite,
                sources=rc.guide_dipoles(),
                source_fields=source_fields,
            ),
            options=rc.ORDER2,
        )
        for source_fields in (True, False)
    ]
    columns, reciprocal = results
    assert reciprocal.port_waves.shape == columns.port_waves.shape
    assert _relative(reciprocal.port_waves, columns.port_waves) <= VALUE_GATE
    assert reciprocal.emitted_power == {}


def test_wave_port_adaptive_reciprocal_port_waves_match_the_columns(plain_guide):
    sources = rc.guide_dipoles()
    uniform = gfem.solve(
        plain_guide,
        gfem.Driven(rc.guide_grid(), excite=(), sources=sources),
        options=rc.ORDER2,
    )
    options = gfem.SolverOptions(
        order=2, linear_solver="direct", adaptive_tolerance=1e-4
    )
    adaptive = gfem.solve(
        plain_guide,
        gfem.Driven(
            rc.guide_grid(),
            excite=(),
            sources=sources,
            sweep="adaptive",
            source_fields=False,
        ),
        options=options,
    )
    assert adaptive.sweep.converged
    assert _relative(adaptive.port_waves, uniform.port_waves) <= ADAPTIVE_VALUE_GATE


# --- the uniform backward's cost --------------------------------------------------


def test_the_uniform_backward_solves_one_column_per_port(line, monkeypatch):
    widths = []
    solve_adjoint = SystemOperator.solve_adjoint

    def recorded(self, n, omega, loads):
        widths.append(loads.shape[1])
        return solve_adjoint(self, n, omega, loads)

    monkeypatch.setattr(SystemOperator, "solve_adjoint", recorded)
    sources = rc.dipoles(20, magnetic=False)
    problem = gfem.Driven(rc.grid(3), excite=(), sources=sources, source_fields=False)
    power, _ = _objectives(line, "dielectric", problem, rc.ORDER2)
    jax.grad(power)(_eps_design(line, "dielectric"))
    # one column per port (two), whatever the number of dipoles
    assert widths
    assert max(widths) <= 2


# --- the input kinds --------------------------------------------------------------

#: The gradient matrix's dipole line (`pml_cases.slab_line`): the layer's TEM
#: line with a wave port, a design slab next to the dipoles and a conducting
#: sheet on the slab's front face (um).
NEAR = np.array([0.07, 0.13, 0.44]) * 1e-6
FAR = np.array([0.13, 0.07, 0.24]) * 1e-6
ELECTRIC = (
    gfem.ElectricDipole("d", position=NEAR, moment=(0, 1e-28, 3e-29)),
    gfem.ElectricDipole("e", position=FAR, moment=(0, 1e-28, 0)),
)
WITH_MAGNETIC = (
    ELECTRIC[0],
    gfem.MagneticDipole("m", position=FAR, moment=(1e-20, 0, 0)),
)
#: A Lorentz law whose resonance lies above the line's frequency.
SLAB_LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Lorentz(
            delta_eps=0.5,
            resonance_frequency=2 * pml_cases.F0,
            damping=0.2 * pml_cases.F0,
        )
    ],
)
#: Each kind's starting values on the slab (per element) or its sheet.
SLAB_VALUES = {
    "eps_r": (2.0, 3.0),
    "mu_r": (1.0, 2.0),
    "tan_delta": (0.01, 0.2),
    "conductivity": (1e3, 3e3),
}


@pytest.fixture(scope="module")
def dipole_line(tmp_path_factory):
    path = pml_cases.slab_line(tmp_path_factory.mktemp("dipole") / "line.msh")
    mesh = gfem.Mesh.from_gmsh(path)
    air = gfem.Material()
    n = len(mesh.elements("front"))
    sheet = gfem.Conductor(
        "front", 100.0, 1e-6, fill=gfem.SheetField(fill=np.linspace(0.2, 0.9, n))
    )
    return gfem.Model(
        mesh,
        materials={"air": air, "pml": air, "slab": air},
        boundaries=[gfem.PEC("pec"), sheet],
        ports=[gfem.WavePort("p1", "port")],
        pml=gfem.PML("pml"),
    )


def _slab_kind(model, kind):
    """``(model of p, p0)``: the dipole line in input ``kind`` of its slab."""
    if kind in SLAB_VALUES:
        n = len(model.mesh.elements("slab"))
        p0 = jnp.asarray(np.linspace(*SLAB_VALUES[kind], n))
        return (
            lambda p: model.with_material("slab", gfem.MaterialField(**{kind: p})),
            p0,
        )
    if kind == "fill":
        p0 = jnp.asarray(model.boundaries[-1].fill.fill)

        def filled(p):
            sheet = gfem.Conductor("front", 100.0, 1e-6, fill=gfem.SheetField(fill=p))
            return model.replace(boundaries=[*model.boundaries[:-1], sheet])

        return filled, p0
    if kind == "law":
        leaves, tree = jax.tree.flatten(SLAB_LAW)
        t0 = jnp.zeros(len(leaves))

        def scaled(t):
            law = jax.tree.unflatten(
                tree, [v * (1 + t[i]) for i, v in enumerate(leaves)]
            )
            return model.with_material("slab", gfem.Material(eps_r=law))

        return scaled, t0
    if kind == "nodes":
        morph = boundary_morph(model, "front", grows="slab")
        return (lambda t: model.replace(mesh=morph(t))), jnp.zeros(morph.n_params)
    raise AssertionError(kind)


def _wave_gradient(build, p, sources, frequency, *, source_fields, options):
    def objective(p):
        problem = gfem.Driven(
            frequency, excite=(), sources=sources, source_fields=source_fields
        )
        w = gfem.solve(build(p), problem, options=options).port_waves
        return jnp.sum(jnp.abs(w) ** 2)

    return jax.grad(objective)(p)


DIPOLE_OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")


@pytest.mark.parametrize(
    "kind", ["eps_r", "mu_r", "tan_delta", "conductivity", "law", "fill", "nodes"]
)
def test_reciprocal_gradients_take_the_column_path_s_inputs(dipole_line, kind):
    build, p = _slab_kind(dipole_line, kind)
    sources = ELECTRIC if kind == "nodes" else WITH_MAGNETIC
    column, reciprocal = (
        _wave_gradient(
            build,
            p,
            sources,
            pml_cases.F0,
            source_fields=source_fields,
            options=DIPOLE_OPTIONS,
        )
        for source_fields in (True, False)
    )
    assert np.abs(np.asarray(column)).max() > 0
    assert _relative(reciprocal, column) <= UNIFORM_GATE


def test_a_lumped_capacitance_gradient_is_the_column_path_s(tmp_path):
    # the line's ports terminated by R and C in parallel; p1's C is the input
    mesh = rc.line_model(tmp_path).mesh

    def build(t):
        c = 2e-13 * (1 + t[0])
        return rc.lumped_line(mesh, c)

    column, reciprocal = (
        _wave_gradient(
            build,
            jnp.zeros(1),
            rc.dipoles(),
            rc.grid(3),
            source_fields=source_fields,
            options=rc.ORDER2,
        )
        for source_fields in (True, False)
    )
    assert np.abs(np.asarray(column)).max() > 0
    assert _relative(reciprocal, column) <= UNIFORM_GATE


def test_a_magnetic_dipole_refuses_the_node_coordinates_alike(dipole_line):
    build, t = _slab_kind(dipole_line, "nodes")
    messages = []
    for source_fields in (True, False):
        with pytest.raises(NonDifferentiableError, match="MagneticDipole") as raised:
            _wave_gradient(
                build,
                t,
                WITH_MAGNETIC,
                pml_cases.F0,
                source_fields=source_fields,
                options=DIPOLE_OPTIONS,
            )
        messages.append(str(raised.value))
    assert messages[0] == messages[1]
