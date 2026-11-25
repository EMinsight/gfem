r"""The driven port pass: its dense pullback, trace maps, gate and refusals."""

import math
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants
import scipy.sparse as sp
from anisotropy import turned
from gradient_cases import face_line, wave_line, wave_mesh

import gfem
from gfem._engine import port_pass
from gfem._engine.port_pass import FaceForms, ModeValues, RowInputs, pullback_row
from gfem._engine.wave_ports import FaceTrace

MU0 = scipy.constants.mu_0
N_T, N_FREE, WIDTH, SCALE, OMEGA = 11, 9, 3, 1e-3, 2 * math.pi * 12e9


def _forms(rng, excited, lead):
    tangential = np.arange(N_FREE) < 6
    tang_dof = rng.permutation(N_T)[:6]
    select = sp.csr_matrix((np.ones(6), (tang_dof, np.arange(6))), shape=(N_T, N_FREE))
    robin = rng.standard_normal((N_T, N_FREE)) + 1j * rng.standard_normal((N_T, N_FREE))
    robin[:, ~tangential] = 0.0
    coupling = rng.standard_normal((N_T, N_FREE)) + 1j * rng.standard_normal(
        (N_T, N_FREE)
    )
    return FaceForms(
        coupling=sp.csr_matrix(coupling + robin),
        robin=sp.csr_matrix(robin),
        select=select,
        tangential=tangential,
        scale=SCALE,
        lead=lead,
        excited=excited,
        free_t=tang_dof,
    )


def _random(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _psi(forms, modes, row):
    """The port terms' value ``Psi``, written out for the oracle."""
    tang, field, gam = forms.tangential, modes.field, modes.gamma
    a, u = row.adjoint, row.solution
    x = np.where(tang[:, None], field * gam, field)
    h = forms.coupling @ x
    e, s = forms.select @ field, forms.robin @ field
    r = h.T @ e
    pairing = -0.5j * SCALE / (row.omega * MU0)
    path = ~np.isnan(modes.voltage)
    with np.errstate(invalid="ignore"):  # NaN voltages: the other branch
        kappa = np.where(
            path,
            modes.voltage / np.abs(modes.voltage) * np.sqrt(modes.phase.real),
            np.sqrt(pairing * np.diag(r) + 0j),
        )
    f = kappa[:, None] * np.linalg.solve(r, h.T) / SCALE
    gam0 = gam[forms.lead]
    p = np.concatenate([h, gam0 * s], axis=1)
    n = len(gam)
    r_sym = 0.5 * (r + r.T)
    g = gam0 * (s.T @ e)
    pa, pu = p.T @ a, p.T @ u
    modal = np.sum(pa[:n] * np.linalg.solve(r_sym, pu[:n])) - np.sum(
        pa[n:] * np.linalg.solve(g, pu[n:])
    )
    robin = np.sum(a * (forms.robin @ forms.embed(u)))
    total = np.sum(row.weights * (f @ u)) - gam0 * robin - modal
    for k, col in enumerate(forms.excited):
        if col >= 0:
            total += a[:, col] @ (2 * SCALE * h[:, k] / kappa[k])
    z = np.abs(modes.voltage) ** 2 * modes.phase / 2
    total += np.sum(row.gamma * gam)
    total += np.sum(np.where(path, row.impedance * z, 0.0))
    return total.real


def _case(seed, n_modes, excited, lead, with_path):
    rng = np.random.default_rng(seed)
    forms = _forms(rng, excited, lead)
    voltage = np.full(n_modes, complex(math.nan, math.nan))
    if with_path:
        voltage[0] = 0.3 + 0.2j
    modes = ModeValues(
        field=_random(rng, N_FREE, n_modes),
        gamma=_random(rng, n_modes) * 100,
        voltage=voltage,
        phase=np.exp(1j * rng.uniform(-0.6, 0.6, n_modes)),
    )
    row = RowInputs(
        omega=OMEGA,
        adjoint=_random(rng, N_T, WIDTH),
        solution=_random(rng, N_T, WIDTH),
        weights=_random(rng, n_modes, WIDTH),
        gamma=_random(rng, n_modes),
        impedance=_random(rng, n_modes) * with_path,
    )
    return forms, modes, row


def _derivatives(forms, modes, row, name, index):
    """The real derivatives ``(d/dx, d/dy)`` of ``Psi`` in one complex entry."""
    # The central difference errs by its truncation, h^2 |Psi'''| / 6, plus
    # the round-off of the two values of Psi over 2h. Psi reaches 1e4 here and
    # its value moves by a few ulps (1.8e-12 each) with the BLAS/LAPACK build's
    # summation order: k ulps give k * 1.8e-12 / h. At h = 1e-6 that is
    # k * 1.8e-6, past the check's abs tolerance of 1e-6 from k = 1 (one build
    # was 2.4e-6 off). At h = 5e-5 even k = 6 stays near 2e-7, a fifth of it,
    # while the truncation, which grows as h^2, stays under a tenth of the
    # check's tolerance on every entry of these cases.
    h = 5e-5
    out = []
    for step in (1.0, 1j):
        values = []
        for sign in (1, -1):
            shifted = getattr(modes, name).copy()
            shifted[index] += sign * h * step
            moved = ModeValues(**{**vars(modes), name: shifted})
            values.append(_psi(forms, moved, row))
        out.append((values[0] - values[1]) / (2 * h))
    return out


@pytest.mark.parametrize(
    ("n_modes", "excited", "lead", "with_path"),
    [(1, [0], 0, False), (2, [1, -1], 1, False), (2, [0, 2], 0, True)],
    ids=["lone", "shared", "shared-path"],
)
def test_the_pullback_of_the_port_terms_matches_finite_differences(
    n_modes, excited, lead, with_path
):
    forms, modes, row = _case(3, n_modes, excited, lead, with_path)
    bars = pullback_row(forms, modes, row)

    def check(name, bar, index):
        dx, dy = _derivatives(forms, modes, row, name, index)
        # dPsi = Re(bar d) = Re(bar) dx - Im(bar) dy
        assert bar.real == pytest.approx(dx, rel=2e-6, abs=1e-6)
        assert -bar.imag == pytest.approx(dy, rel=2e-6, abs=1e-6)

    for k in range(n_modes):
        check("gamma", bars.gamma[k], k)
    for index in ((0, 0), (5, n_modes - 1), (N_FREE - 1, 0)):
        check("field", bars.field[index], index)
    if with_path:
        check("voltage", bars.voltage[0], 0)
        check("phase", bars.phase[0], 0)
    else:
        assert not bars.voltage.any()
        assert not bars.phase.any()


def test_the_explicit_pairs_cover_the_coupling_and_the_robin_forms():
    forms, modes, row = _case(4, 2, [0, -1], 0, False)
    pairs = pullback_row(forms, modes, row).pairs
    names = [p.form for p in pairs]
    # per mode: both coupling forms on x and the Robin form on the field; the
    # Robin term of every column
    assert names.count("coupling_t") == 2 * 2 + WIDTH
    assert names.count("coupling_g") == 2
    assert not pullback_row(forms, modes, row, explicit=False).pairs


def test_a_gather_is_the_transpose_of_the_lift():
    rng = np.random.default_rng(0)
    n_dof, n_3d = 7, 12
    idx2 = np.array([0, 2, 3, 6])
    idx3 = np.array([5, 1, 8, 10])
    writers = np.ones(n_3d)
    writers[idx3[1]] = 2.0
    trace = FaceTrace(idx3, idx2, writers)
    r = _random(rng, n_dof, 3)
    v = _random(rng, n_3d, 3)
    # <lift(r), v> = <r, gather(v)> for every pair of columns
    lifted = trace.lift(r)
    gathered = trace.gather(v, n_dof)
    np.testing.assert_allclose(
        np.sum(lifted * v), np.sum(r * gathered), rtol=1e-14, atol=1e-14
    )


def _spy(monkeypatch):
    calls = []
    real = port_pass.pullback_row

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(port_pass, "pullback_row", counted)
    return calls


def _objective(model, problem, region):
    def objective(p):
        design = gfem.MaterialField(eps_r=p)
        moved = model.with_material(region, design)
        moved = moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))
        result = gfem.solve(moved, problem)
        return jnp.sum(jnp.abs(result.s.values) ** 2) + jnp.sum(
            jnp.abs(result.port_modes["p1"].gamma)
        )

    return objective


@pytest.fixture(scope="module")
def line():
    return wave_line(wave_mesh(Path(tempfile.mkdtemp())), "lossy")


FREQUENCIES = (12e9, 24e9)


def test_a_design_away_from_the_faces_runs_no_port_pass(line, monkeypatch):
    calls = _spy(monkeypatch)
    problem = gfem.Driven(frequency=FREQUENCIES)
    core = jnp.full(len(line.mesh.elements("core")), 3.0)
    objective = _objective(line, problem, "core")
    grad = jax.grad(objective)(core)
    assert np.abs(np.asarray(grad)).max() > 0
    assert calls == []


def test_a_design_at_a_face_runs_one_pass_per_surface_and_frequency(line, monkeypatch):
    calls = _spy(monkeypatch)
    problem = gfem.Driven(frequency=FREQUENCIES)
    lo = jnp.full(len(line.mesh.elements("lo")), 4.0)
    grad = jax.grad(_objective(line, problem, "lo"))(lo)
    assert np.abs(np.asarray(grad)).max() > 0
    assert len(calls) == 2 * len(FREQUENCIES)  # two surfaces


def test_a_thermal_property_at_a_face_runs_no_pass(line, monkeypatch):
    calls = _spy(monkeypatch)
    problem = gfem.Driven(frequency=FREQUENCIES)
    n = len(line.mesh.elements("lo"))

    def objective(p):
        field = gfem.MaterialField(
            eps_r=jnp.full(n, 4.0), thermal_conductivity=p, tan_delta=jnp.full(n, 0.05)
        )
        moved = line.with_material("lo", field)
        moved = moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))
        return jnp.sum(jnp.abs(gfem.solve(moved, problem).s.values))

    grad = jax.grad(objective)(jnp.full(n, 1.0))
    assert not np.asarray(grad).any()
    assert calls == []


def test_port_mode_gradients_are_exact_zeros_away_from_the_faces(line):
    problem = gfem.Driven(frequency=FREQUENCIES)
    core = jnp.full(len(line.mesh.elements("core")), 3.0)

    def objective(p):
        moved = line.with_material("core", gfem.MaterialField(eps_r=p))
        moved = moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))
        modes = gfem.solve(moved, problem).port_modes
        return jnp.sum(jnp.abs(modes["p1"].gamma)) + jnp.sum(
            jnp.real(modes["p2"].gamma)
        )

    assert not np.asarray(jax.grad(objective)(core)).any()


def test_the_residual_of_a_port_mode_is_refused_beside_a_face_design(line):
    problem = gfem.Driven(frequency=FREQUENCIES)
    lo = jnp.full(len(line.mesh.elements("lo")), 4.0)
    core = jnp.full(len(line.mesh.elements("core")), 3.0)

    def objective(p, region):
        moved = line.with_material(region, gfem.MaterialField(eps_r=p))
        moved = moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))
        return jnp.sum(gfem.solve(moved, problem).port_modes["p1"].residual)

    with pytest.raises(gfem.NonDifferentiableError, match=r"'residual'.*diagnostic"):
        jax.grad(lambda p: objective(p, "lo"))(lo)
    # away from the faces the residual is a constant: an exact zero
    assert not np.asarray(jax.grad(lambda p: objective(p, "core"))(core)).any()


@pytest.mark.parametrize(
    ("kwargs", "where"),
    [
        ({"wave_vector": (1.0, 0.0, 0.0)}, "under a Bloch wave vector"),
        (
            {"wave_vector": (1.0, 0.0, 0.0), "wave_vector_frequency": 12e9},
            "on a fixed-angle sweep",
        ),
    ],
    ids=["bloch", "fixed-angle"],
)
def test_a_bloch_problem_refuses_a_port_face_input(line, kwargs, where):
    rule = _rule(line, gfem.Driven(frequency=FREQUENCIES))
    # the static answer reads the problem's wave vector: swap in a Bloch one
    # (the model has no periodic pair, which the rule's constructor checks)
    rule.problem = gfem.Driven(frequency=FREQUENCIES, **kwargs)
    reason = rule.unsupported_input(("material", "lo", "eps_r"))
    assert reason is not None
    assert where in reason
    assert rule.unsupported_input(("material", "core", "eps_r")) is None


def _rule(model, problem):
    from gfem._engine.driven import DrivenRule

    return DrivenRule(model, problem, gfem.SolverOptions(order=2))


def test_the_driven_rule_no_longer_refuses_a_port_face_input(line):
    rule = _rule(line, gfem.Driven(frequency=FREQUENCIES))
    for kind in (("material", "lo", "eps_r"), ("material", "core", "eps_r")):
        assert rule.unsupported_input(kind) is None
    assert rule.port_design.reading(("material", "lo", "eps_r")) == ("p1", "p2")
    assert rule.port_design.reading(("material", "core", "eps_r")) == ()
    assert rule.port_design.reads({("material", "lo", "eps_r")}, ["p1"])
    assert not rule.port_design.reads({("material", "core", "eps_r")}, ["p1"])


def test_the_adaptive_sweep_no_longer_refuses_a_port_face_input(line):
    rule = _rule(
        line,
        gfem.Driven(
            frequency=(10e9, 12e9, 14e9, 16e9), sweep="adaptive", samples=(10e9, 16e9)
        ),
    )
    for kind in (("material", "lo", "eps_r"), ("material", "core", "eps_r")):
        assert rule.unsupported_input(kind) is None


def test_the_face_variants_are_wave_line_models():
    mesh = wave_mesh(Path(tempfile.mkdtemp()))
    assert face_line(mesh, "face_path").ports[0].voltage_path is not None
    absorbing = face_line(mesh, "face_absorbing").boundaries
    assert any(isinstance(b, gfem.Absorbing) for b in absorbing)


def test_a_per_axis_material_beside_an_absorbing_edge_is_refused_statically():
    # the absorbing edge's admittance has no per-axis derivative: the design of
    # a field with axes on the port face is refused before any solve, one
    # without axes (or away from the face) is not
    mesh = wave_mesh(Path(tempfile.mkdtemp()))
    model = face_line(mesh, "face_absorbing")
    n = len(mesh.elements("lo"))
    axes = turned(0, 30.0)
    field = gfem.MaterialField(eps_r=np.full((n, 3), 4.0), axes=axes)
    rule = _rule(model.with_material("lo", field), gfem.Driven(frequency=(24e9, 30e9)))
    reason = rule.unsupported_input(("material", "lo", "eps_r"))
    assert reason is not None
    assert "per-axis material" in reason
    assert "stop_gradient" in reason
    assert rule.unsupported_input(("material", "core", "eps_r")) is None
    isotropic = model.with_material("lo", gfem.MaterialField(eps_r=np.full(n, 4.0)))
    rule = _rule(isotropic, gfem.Driven(frequency=(24e9, 30e9)))
    assert rule.unsupported_input(("material", "lo", "eps_r")) is None


def test_the_voltage_functional_on_the_free_dofs_reads_the_swept_voltage():
    from gfem._sweep.port_mode import _voltage_row, _voltage_rule, path_voltage

    mesh = wave_mesh(Path(tempfile.mkdtemp()))
    model = face_line(mesh, "face_path")
    path = np.asarray(model.ports[0].voltage_path)
    frequencies = (12e9, 24e9)
    mode = gfem.solve(model, gfem.PortMode("p1", frequencies))
    field = mode.e_field
    space, basis, coords = field._space, field._basis, field._coords
    voltage = path_voltage(space, basis, coords, path, "p1")
    functional = _voltage_row(space, _voltage_rule(space.section, path, "p1"))
    for row, v in enumerate(voltage):
        stored = basis[space.free] @ coords[row]
        assert functional @ stored == pytest.approx(v, rel=1e-12)


def test_a_bloch_wave_vector_holds_the_face_nodes_and_runs_no_node_pass(line):
    # the faces are held under a Bloch wave vector: their nodes are not wanted
    # inputs of the pass, whose face terms nothing there verifies
    rule = _rule(line, gfem.Driven(frequency=FREQUENCIES))
    system, ports = rule.system(line)
    wanted = frozenset({("nodes",)})
    free = port_pass.PortPass(rule, line, system, ports, wanted)
    assert free.wants_nodes
    assert free.active
    rule.problem = gfem.Driven(frequency=FREQUENCIES, wave_vector=(1.0, 0.0, 0.0))
    held = port_pass.PortPass(rule, line, system, ports, wanted)
    assert not held.wants_nodes
    assert not held.active
