"""On one process the backward refactorizes the forward's base (ADR 0065, amended).

A differentiated serial driven forward on the CPU (``direct`` or ``bddc_gmres``)
moves its base handle to its entry. A backward that needs adjoint solves
refactorizes that base at each adjoint frequency, starting from the base's
frequency, instead of preparing a handle per frequency. Every live entry keeps
its own base; ``retain_memory`` frees bases before entries.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import cache
from gfem._engine.driven import DrivenRule
from gfem._linalg import ResolvedLinearSolver
from gfem._linalg.base import PlannedHandle
from gfem._linalg.bddc import BddcHandle

F = (21e9, 24e9)
#: Excites p1 only: the S21 cotangent takes an adjoint solve at each frequency
PROBLEM = gfem.Driven(frequency=F, excite="p1")
SOLVERS = ("direct", "bddc_gmres")


@pytest.fixture(scope="module")
def path(tmp_path_factory):
    out = tmp_path_factory.mktemp("line") / "line.msh"
    return sm.write_msh22(out, sm.strip_line(3.0, 2.0, 1.0, (6, 2, 1), core=(1.0, 2.0)))


def _model(mesh, eps):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=2.0),
            "core": gfem.MaterialField(eps_r=eps),
        },
        boundaries=[
            gfem.PEC("bottom"),
            gfem.PEC("top"),
            gfem.Absorbing("side", order=2),
        ],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


class Counts:
    """Handles prepared and refactorized, in the forwards and in the backwards."""

    def __init__(self):
        self.backward = False
        self.prepared = {"forward": 0, "backward": 0}
        self.refactorized = {"forward": 0, "backward": 0}

    @property
    def phase(self):
        return "backward" if self.backward else "forward"


@pytest.fixture
def counts(monkeypatch):
    seen = Counts()
    prepare = ResolvedLinearSolver.prepare
    backward = DrivenRule.backward

    def counted_prepare(self, *args, **kwargs):
        seen.prepared[seen.phase] += 1
        return prepare(self, *args, **kwargs)

    def counted_backward(self, *args, **kwargs):
        seen.backward = True
        try:
            return backward(self, *args, **kwargs)
        finally:
            seen.backward = False

    monkeypatch.setattr(ResolvedLinearSolver, "prepare", counted_prepare)
    monkeypatch.setattr(DrivenRule, "backward", counted_backward)
    for cls in (PlannedHandle, BddcHandle):
        refactorize = cls.refactorize

        def counted_refactorize(self, *args, _refactorize=refactorize, **kwargs):
            seen.refactorized[seen.phase] += 1
            return _refactorize(self, *args, **kwargs)

        monkeypatch.setattr(cls, "refactorize", counted_refactorize)
    cache.ENTRIES.clear()
    # earlier tests' discretizations keep fixed parts and port state, which
    # the cache counts and a walk drops before any base
    cache._DISCRETIZATIONS.clear()
    yield seen
    cache.ENTRIES.clear()


def _s21(mesh, solver, retain_memory=None):
    options = gfem.SolverOptions(linear_solver=solver, retain_memory=retain_memory)

    def s21(eps):
        s = gfem.solve(_model(mesh, eps), PROBLEM, options=options).s
        return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

    return s21


def _eps(mesh):
    return jnp.full(len(mesh.elements("core")), 3.0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_the_backward_refactorizes_the_forwards_base(path, counts, solver):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    jax.grad(_s21(mesh, solver))(_eps(mesh))
    assert counts.prepared["backward"] == 0
    # starting from the base's frequency: one refactorization fewer
    assert counts.refactorized["backward"] == len(F) - 1


@pytest.mark.parametrize("solver", SOLVERS)
def test_an_entry_too_big_with_its_base_is_kept_without_it(path, counts, solver):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    eps = _eps(mesh)
    s21 = _s21(mesh, solver)
    # the second sight of the design keeps the fixed parts, as the run below
    for _ in range(2):
        _, vjp = jax.vjp(s21, eps)
        (want,) = vjp(1.0)
    (entry,) = cache.ENTRIES._entries.values()
    base = entry.handle.system._adjoint_sweep._base
    with_base = entry.nbytes
    # the cache also counts what the entry shares (its pieces' pattern, kept
    # fixed parts, ADR 0091), not the other frequency's pooled base (ADR 0121)
    pooled = sum(int(p.handle.nbytes) for p in cache.ENTRIES._plans.values())
    budget = cache.ENTRIES.nbytes - pooled - base.nbytes // 2
    cache.ENTRIES.clear()

    counts.prepared = {"forward": 0, "backward": 0}
    _, vjp = jax.vjp(_s21(mesh, solver, retain_memory=budget), eps)
    (entry,) = cache.ENTRIES._entries.values()
    assert entry.nbytes < with_base  # kept, its base freed
    (got,) = vjp(1.0)
    # the backward prepares one handle, as the forward did, and runs no forward
    assert counts.prepared == {"forward": 1, "backward": 1}
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0)


@pytest.mark.parametrize("solver", SOLVERS)
def test_vmapped_variants_each_back_onto_their_own_base(path, counts, solver):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    scales = jnp.array([[0.9], [1.0], [1.1]])

    def robust(retain_memory=None):
        s21 = _s21(mesh, solver, retain_memory)
        return jax.value_and_grad(
            lambda eps: jnp.max(jax.vmap(s21)(eps[None] * scales))
        )

    def loop(step, steps=3):
        eps, out = _eps(mesh), []
        for _ in range(steps):
            value, grad = step(eps)
            out.append((float(value), np.asarray(grad)))
            eps = eps + 0.5 * grad / jnp.max(jnp.abs(grad))
        return out

    got = loop(robust())
    assert counts.prepared["backward"] == 0
    assert counts.refactorized["backward"] == 3 * (len(F) - 1)
    cache.ENTRIES.clear()
    want = loop(robust(retain_memory=0))
    for (v, g), (v0, g0) in zip(got, want, strict=True):
        assert v == pytest.approx(v0, rel=1e-12, abs=0)
        np.testing.assert_allclose(g, g0, rtol=0, atol=1e-10 * np.abs(g0).max())


@pytest.mark.parametrize("solver", SOLVERS)
def test_repeated_pullbacks_each_start_at_the_bases_frequency(path, counts, solver):
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    _, vjp = jax.vjp(_s21(mesh, solver), _eps(mesh))
    grads = [vjp(1.0)[0] for _ in range(3)]
    assert counts.prepared["backward"] == 0
    # each pass runs the frequencies from where the last left the base
    assert counts.refactorized["backward"] == 3 * (len(F) - 1)
    for g in grads[1:]:
        np.testing.assert_allclose(g, grads[0], rtol=1e-10, atol=0)
