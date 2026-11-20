"""Gradients by a permittivity law's parameters on the driven sweep.

The law pass weighs each law piece's form with d theta / dp at each pair's
frequency. The finite-difference cells are in the gradient matrix
(tests/verification/test_gradient_matrix.py, input ``"law"``); here: the
hand-written d eps / dp against the law's own JAX derivative, the host
re-check of a law built under a transformation, the adaptive sweep at its
samples against the full solve, and the combinations that still raise by
name.

The line is a 10 x 2 x 1 mm parallel-plate line with lumped ports at both
ends; its 2 mm "core" (x in [4, 6]) holds the law.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import dispersion
from gfem._refine import refine_model

GHZ = 1e9
OPTIONS = gfem.SolverOptions(linear_solver="direct")

#: One law with every term kind (Re eps > 0 at 7 GHz).
LAW = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[
        gfem.Drude(plasma_frequency=3 * GHZ, damping=0.5 * GHZ),
        gfem.Lorentz(delta_eps=1.0, resonance_frequency=20 * GHZ, damping=1 * GHZ),
        gfem.Debye(delta_eps=2.0, relaxation_frequency=5 * GHZ),
        gfem.DjordjevicSarkar(delta_eps=1.1, low_frequency=1e3, high_frequency=1e12),
    ],
)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, law, *, boundaries=()):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=4.0),
            "core": gfem.Material(eps_r=law),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *boundaries],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _s21(mesh, problem, **kwargs):
    def objective(law):
        result = gfem.solve(_line(mesh, law, **kwargs), problem, options=OPTIONS)
        return jnp.sum(jnp.abs(result.s.values[:, 1, 0]) ** 2)

    return objective


# --- d eps / dp ---------------------------------------------------------------------


@pytest.mark.parametrize("f", [0.7 * GHZ, 7 * GHZ, 19.5 * GHZ, (7 - 0.4j) * GHZ])
def test_d_eps_by_each_parameter_is_the_law_s_jax_derivative(f):
    leaves, tree = jax.tree.flatten(LAW)

    def eps(values):
        return jax.tree.unflatten(tree, list(values))(f)

    values = jnp.asarray([float(v) for v in leaves])
    want = jax.jacfwd(lambda v: jnp.stack([jnp.real(eps(v)), jnp.imag(eps(v))]))(values)
    want = np.asarray(want[0] + 1j * want[1])
    got = dispersion.d_eps(LAW, 2 * np.pi * f)
    paths = [
        ("eps_inf",) if len(p) == 1 else (p[1].idx, p[2].name)
        for p, _ in jax.tree_util.tree_flatten_with_path(LAW)[0]
    ]
    assert list(got) == paths
    np.testing.assert_allclose([got[p] for p in paths], want, rtol=1e-12)


# --- the host re-check --------------------------------------------------------------


def test_a_traced_invalid_law_is_rechecked_on_the_host_with_the_region_named(mesh):
    def objective(damping):
        terms = list(LAW.terms)
        terms[0] = dataclasses.replace(terms[0], damping=damping)
        law = dataclasses.replace(LAW, terms=terms)
        return _s21(mesh, gfem.Driven(frequency=[7 * GHZ]))(law)

    with pytest.raises(
        (ValueError, jax.errors.JaxRuntimeError), match=r"not passive.*region 'core'"
    ):
        jax.grad(objective)(-0.5 * GHZ)


def test_a_traced_law_is_refused_by_refinement(mesh):
    model = _line(mesh, LAW)
    marked = np.zeros(len(mesh.tetrahedra), bool)
    marked[0] = True

    def refined(eps_inf):
        refine_model(_line(mesh, dataclasses.replace(LAW, eps_inf=eps_inf)), marked)
        return eps_inf

    with pytest.raises(gfem.NonDifferentiableError, match=r"materials\['core'\]"):
        jax.grad(refined)(2.0)

    def refined_mesh(eps_inf):
        law = dataclasses.replace(LAW, eps_inf=eps_inf)
        gfem.refine_mesh(
            model.with_material("core", gfem.Material(eps_r=law)),
            gfem.Driven(frequency=[7 * GHZ]),
            estimator=gfem.FieldEstimator(),
            max_cycles=0,
        )
        return eps_inf

    with pytest.raises(gfem.NonDifferentiableError, match="refine_mesh"):
        jax.grad(refined_mesh)(2.0)


# --- the adaptive sweep ---------------------------------------------------------------


def test_law_parameters_on_the_adaptive_sweep_equal_the_full_solve_at_the_samples(
    mesh,
):
    # the frozen-basis backward builds one pair set per law key; at a sample
    # the full solution lies in the basis, so the gradient is the full one
    samples = [5 * GHZ, 9 * GHZ, 7 * GHZ]
    adaptive = gfem.Driven(
        frequency=np.linspace(5 * GHZ, 9 * GHZ, 5).tolist(),
        sweep="adaptive",
        samples=samples,
    )

    def objective(problem, rows):
        def at_rows(law):
            result = gfem.solve(_line(mesh, law), problem, options=OPTIONS)
            return jnp.sum(jnp.abs(result.s.values[jnp.asarray(rows), 1, 0]) ** 2)

        return at_rows

    got = jax.grad(objective(adaptive, [0, 4, 2]))(LAW)
    want = jax.grad(objective(gfem.Driven(frequency=samples), [0, 1, 2]))(LAW)
    got, want = (np.asarray(jax.tree.leaves(g), dtype=float) for g in (got, want))
    assert np.all(want != 0)
    np.testing.assert_allclose(got, want, rtol=1e-7)


# --- still raising by name ----------------------------------------------------------


def test_law_parameters_differentiate_the_eigenfrequency(mesh):
    # the eigenmode pairs carry the law pass's weights too:
    # tests/unit/test_eigenmode_dispersive.py checks them by differences
    def objective(law):
        result = gfem.solve(
            _line(mesh, law), gfem.Eigenmode(target=10 * GHZ), options=OPTIONS
        )
        return jnp.real(result.omega[0])

    grad = jax.grad(objective)(LAW)
    values = np.asarray(jax.tree.leaves(grad), dtype=float)
    assert np.all(np.isfinite(values))
    assert np.any(values != 0)


def test_law_parameters_differentiate_beside_an_unread_eps(mesh):
    # the gradient reaches every leaf of the law, in the law's own shape
    grad = jax.grad(_s21(mesh, gfem.Driven(frequency=[7 * GHZ])))(LAW)
    assert isinstance(grad, gfem.PermittivityLaw)
    values = np.asarray(jax.tree.leaves(grad), dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values != 0)


# --- no law work without a law input -------------------------------------------------


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_gradient_without_a_law_input_carries_no_law_weights(
    mesh, monkeypatch, sweep
):
    # a design beside a law: the law's dθ/dp is neither computed nor factored
    from gfem._engine import sensitivity

    seen = []
    original = sensitivity.sensitivity

    def spy(model, pairs, wants, **kwargs):
        seen.extend(key for pair in pairs for key in pair.weights)
        return original(model, pairs, wants, **kwargs)

    monkeypatch.setattr(sensitivity, "sensitivity", spy)
    n = len(mesh.elements("dielectric"))
    problem = (
        gfem.Driven(
            frequency=np.linspace(5 * GHZ, 9 * GHZ, 5).tolist(),
            sweep="adaptive",
            samples=[5 * GHZ, 9 * GHZ, 7 * GHZ],
        )
        if sweep == "adaptive"
        else gfem.Driven(frequency=[7 * GHZ])
    )

    def objective(eps):
        model = _line(mesh, LAW).with_material(
            "dielectric", gfem.MaterialField(eps_r=eps)
        )
        s = gfem.solve(model, problem, options=OPTIONS).s.values
        return jnp.sum(jnp.abs(s[:, 1, 0]) ** 2)

    grad = jax.grad(objective)(jnp.full(n, 4.0))
    assert np.isfinite(np.asarray(grad)).all()
    assert seen
    assert not [key for key in seen if " d/d " in key]
