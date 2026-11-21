"""The eigenmode entry's kept operator, its gap set and the driven memo.

A differentiated eigenmode forward keeps its `EigenOperator` (the factors at
the last shift and the assembled maps) in a kept-operator holder, the
entry's handle: the cache counts it, drops it before any entry over
``retain_memory`` and frees it with a consumed entry, and nothing of it is
pooled. The entry also stores each mode's relative gap to the other
converged modes the selection accepted and to the guard mode.
"""

import importlib.metadata

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from eigen_cases import (
    EPS,
    conductor_line,
    line_mesh,
    line_roots,
    steer_search,
    terminated_line,
)

import gfem
from gfem._eigen.adjoint import deflated_solve, polish
from gfem._eigen.kept import KeptOperator
from gfem._eigen.reach import GapSet
from gfem._eigen.refine import Refined
from gfem._eigen.rule import EigenmodeRule
from gfem._engine import cache
from gfem._linalg.base import host_bytes
from gfem._linalg.direct import DirectHandle

TARGET = 8e9


def _mkl_installed() -> bool:
    try:
        importlib.metadata.distribution("mkl")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"))


def _kept(model, n_modes=1, **options):
    rule = EigenmodeRule(
        model,
        gfem.Eigenmode(target=TARGET, n_modes=n_modes),
        gfem.SolverOptions(**options),
    )
    outputs, entry = rule.forward(model, keep=True)
    return rule, outputs, entry


def test_holder_counts_the_factors_and_maps(mesh):
    _, _, entry = _kept(terminated_line(mesh, 1000.0))
    holder = entry.handle
    assert isinstance(holder, KeptOperator)
    eop = holder.operator
    factors = eop.handle.nbytes
    mass = host_bytes(eop.mass.matrix)
    assert factors > 0
    assert mass > 0
    # the damping of a lumped resistor is applied matrix-free: no bytes
    assert holder.nbytes == factors + mass
    fields = sum(f.nbytes for f in entry.fields.values())
    assert entry.nbytes == fields + factors + mass
    assert holder.shared() == {}  # no material field: no kept fixed parts
    entry.free()
    assert holder.operator is None
    assert holder.nbytes == 0


# --- memory: retain_memory, consumed entries, the bases tier ----------------------


def _field_model(mesh, eps):
    """The terminated line with its permittivity a `gfem.MaterialField`."""
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.MaterialField(eps_r=eps)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z", impedance=1000.0),
            gfem.LumpedPort("p2", "port2", "+z", impedance=1000.0),
        ],
    )


@pytest.fixture
def forwards(monkeypatch):
    """Every eigenmode forward's ``keep`` flag and entry, in order."""
    cache.ENTRIES.clear()
    seen = []
    forward = EigenmodeRule.forward

    def counted(self, model, *, keep, request=None):
        outputs, entry = forward(self, model, keep=keep, request=request)
        seen.append((keep, entry))
        return outputs, entry

    monkeypatch.setattr(EigenmodeRule, "forward", counted)
    yield seen
    cache.ENTRIES.clear()


def _frequency(mesh, **options):
    """``Re ω`` of the mode near the target, as a function of the permittivity."""

    def objective(eps):
        result = gfem.solve(
            _field_model(mesh, eps),
            gfem.Eigenmode(target=TARGET),
            options=gfem.SolverOptions(**options),
        )
        return jnp.real(result.omega[0]) / 1e10

    return objective


def _eps(mesh, value=EPS):
    return jnp.full(len(mesh.elements("dielectric")), value)


def test_over_budget_entry_keeps_its_fields_without_the_operator(mesh, forwards):
    _, _, probe = _kept(
        _field_model(mesh, np.full(len(mesh.elements("dielectric")), EPS))
    )
    fields = sum(f.nbytes for f in probe.fields.values())
    held = probe.handle.nbytes
    probe.free()
    forwards.clear()
    budget = fields + held // 2  # the fields fit, the operator does not
    objective = _frequency(mesh, retain_memory=budget)
    value, pullback = jax.vjp(objective, _eps(mesh))
    ((keep, entry),) = forwards
    assert keep
    assert entry.handle.operator is None  # freed by `put`
    assert entry.fields
    assert len(cache.ENTRIES) == 1
    (grad,) = pullback(jnp.ones_like(value))
    assert len(forwards) == 1  # the frequency gradient ran no second forward
    # the gradient of the kept entry is the one of an unconstrained solve
    np.testing.assert_allclose(grad, jax.grad(_frequency(mesh))(_eps(mesh)), rtol=1e-10)


def test_consumed_entry_frees_the_operator_and_pools_nothing(mesh, forwards):
    step = jax.value_and_grad(_frequency(mesh))
    step(_eps(mesh))
    (_, first) = forwards[0]
    assert first.consumed
    assert first.handle.operator is not None
    step(_eps(mesh, EPS + 0.1))
    assert first.handle is None
    assert first.fields == {}
    assert cache.ENTRIES.pooled == 0
    (_, second) = forwards[-1]
    assert second.handle.operator is not None


def test_bases_tier_frees_eigen_operators_before_entries(mesh):
    model = terminated_line(mesh, 1000.0)
    entries = cache.EntryCache()
    _, _, older = _kept(model)
    _, _, newer = _kept(model)
    budget = older.nbytes + newer.nbytes - 1  # one operator too many
    k_older = entries.put(older, budget)
    k_newer = entries.put(newer, budget)
    assert entries.get(k_older) is older
    assert entries.get(k_newer) is newer
    assert older.handle.operator is None  # the older entry's operator went first
    assert older.fields
    assert newer.handle.operator is not None
    entries.clear()


# --- no factor reuse across designs -----------------------------------------------


@pytest.mark.skipif(
    not _mkl_installed(), reason="the reuse rule is PARDISO's: needs the mkl extra"
)
def test_eigen_refactorizations_never_reuse_factors(mesh, monkeypatch):
    monkeypatch.setattr(DirectHandle, "_reuse_pays", lambda _: True)
    counts = []
    inverse_map = DirectHandle._inverse_map

    def counted(self, *args, **kwargs):
        X, its = inverse_map(self, *args, **kwargs)
        counts.append((its, self._rule.reusing))
        return X, its

    monkeypatch.setattr(DirectHandle, "_inverse_map", counted)
    model = terminated_line(mesh, 1000.0)
    rule, _, entry = _kept(model, linear_solver="direct")
    eop = entry.handle.operator
    handle = eop.handle
    assert isinstance(handle, DirectHandle)
    assert handle._rule is not None
    assert counts
    assert all(c == ((), False) for c in counts)
    forward = len(counts)
    b = np.where(eop.free, 1.0 + 0.0j, 0)
    for shift in (1.001, 1.002, 0.999):
        eop.reshift(eop.shift * shift)
        _, its = eop.handle.solve_counted(np.where(eop.free, b, 0))
        assert its == ()
        assert not eop.handle._rule.reusing
    assert len(counts) == forward + 3
    # the polish and a deflated solve after a reshift (its own, at k (1 +
    # 1e-3)): numeric factorizations of the current values only
    x, k = entry.fields["x"][:, 0], complex(entry.fields["k"][0])
    mode = Refined(k, x / np.linalg.norm(x), 0.0, 0, converged=True)
    polish(eop, mode, np.inf, index=0)
    eop.reshift(k * (1 + 1e-3))
    deflated_solve(eop, x, None, b, k=k, gap=np.inf, mode=0)
    # and the mode-field backward's adjoint solve
    solves = len(counts)
    assert rule.backward(model, entry, {"q_port": np.ones((2, 1))})
    assert len(counts) > solves
    assert len(counts) > forward + 4
    assert all(c == ((), False) for c in counts)
    entry.free()


# --- the gap set ------------------------------------------------------------------


def test_empty_gap_set_takes_the_seed_ritz_values(mesh):
    # no guard mode on the small line: the
    # mode's neighbour is the next root's unrefined Ritz value
    _, _outputs, entry = _kept(terminated_line(mesh, 1000.0))
    first, second = line_roots(1000.0, [1, 2])
    np.testing.assert_allclose(
        entry.fields["gap"], [abs(second - first) / abs(second)], rtol=1e-3
    )
    entry.free()
    # a nonpolynomial operator's frozen seeds are too rough: no neighbour
    _, _, entry = _kept(conductor_line(mesh))
    np.testing.assert_array_equal(entry.fields["gap"], [np.inf])
    entry.free()
    assert (
        GapSet(np.array([12.6 + 0j]), None, np.array([12.6 + 0j])).gap(12.6) == np.inf
    )


@pytest.mark.bitwise
def test_entry_records_a_failed_polish_with_a_zero_gap(mesh, monkeypatch):
    model = terminated_line(mesh, 1000.0)
    steer_search(monkeypatch, polish=lambda _: lambda _eop, mode, _gap, **_: mode)
    unpolished, _ = EigenmodeRule(
        model, gfem.Eigenmode(target=TARGET), gfem.SolverOptions()
    ).forward(model, keep=False)

    def fails(_eop, _mode, _gap, *, index):
        raise gfem.EigenmodeConvergenceError(f"Polishing mode {index} did not converge")

    steer_search(monkeypatch, polish=lambda _: fails)
    _, outputs, entry = _kept(model)
    # the mode as it came from the refinement, recorded as not isolated
    np.testing.assert_array_equal(outputs["omega"], unpolished["omega"])
    np.testing.assert_array_equal(outputs["e_field"], unpolished["e_field"])
    np.testing.assert_array_equal(entry.fields["gap"], [0.0])
    np.testing.assert_array_equal(entry.fields["polish_failed"], [True])
    entry.free()
    # and polished modes record no failure
    monkeypatch.undo()
    _, _, entry = _kept(model)
    np.testing.assert_array_equal(entry.fields["polish_failed"], [False])
    assert entry.fields["gap"][0] > 0
    entry.free()


# --- the driven memo --------------------------------------------------------------


def test_driven_memo_rebuilds_on_a_new_discretization(mesh):
    model = terminated_line(mesh, 1000.0)
    rule = EigenmodeRule(model, gfem.Eigenmode(target=TARGET), gfem.SolverOptions())
    op = rule.driven(model)
    assert rule.driven(model) is op  # one backward builds it once
    # the discretization was evicted (other meshes, other node coordinates)
    cache._DISCRETIZATIONS.clear()
    rebuilt = rule.driven(model)
    assert rebuilt is not op
    assert rebuilt.built is cache.discretization(model.mesh, model._periodic).built


def test_mode_field_backward_without_the_operator_prepares_and_frees_one(mesh):
    # an entry over retain_memory keeps its fields only: the mode-field
    # backward prepares the eigen operator at k (1 + 1e-3) from its own
    # driven operator, solves on it, and frees it; the adjoint is the same
    model = terminated_line(mesh, 1000.0)
    rule, _, entry = _kept(model)
    cotangents = {"q_port": np.array([[1.0], [0.5]]), "bulk": np.ones((1, 1))}
    kept = rule.backward(model, entry, cotangents)
    entry.handle.free()
    built = []
    make = rule.operator

    def operator(op, sigma):
        eop = make(op, sigma)
        built.append((eop, sigma))
        return eop

    rule.operator = operator
    own = rule.backward(model, entry, cotangents)
    ((eop, sigma),) = built
    k = complex(entry.fields["k"][0])
    assert sigma == pytest.approx(k * (1 + 1e-3), rel=1e-15)
    assert eop._handle is None  # freed at the end of the backward
    assert len(own) == len(kept) == 2  # (2 lam, x) and the energy pair
    for a, b in zip(own, kept, strict=True):
        np.testing.assert_allclose(
            a.left, b.left, rtol=1e-7, atol=1e-7 * abs(b.left).max()
        )
        assert a.weights.keys() == b.weights.keys()
    entry.free()
