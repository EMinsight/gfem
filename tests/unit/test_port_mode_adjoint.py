"""The port mode adjoint core on assembled 2D pencils.

Each derivative is checked against central differences of exact eigenpairs
(re-assembled sections) or a closed form; the objective is scale invariant in
the eigenvector like the normalized mode coordinates are.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from jax.experimental import sparse as jsp

import gfem
from gfem._engine import port_mode_adjoint as adj
from gfem._engine.port_mode import PortModeSpace
from gfem._port_section import port_section
from gfem._sweep.port_mode import _Greedy
from gfem.errors import PortModeError

jax.config.update("jax_enable_x64", True)

DATA = Path(__file__).parent / "data"


def section_of(msh, unit="mm", **material):
    mesh = gfem.Mesh.from_gmsh(DATA / msh, length_unit=unit)
    model = gfem.Model(
        mesh,
        materials=material["materials"],
        boundaries=material["boundaries"],
        ports=[gfem.WavePort("p1", "port1")],
    )
    return port_section(model, model.ports[0])


def guide(eps=1.0):
    return section_of(
        "waveguide.msh",
        materials={"air": gfem.Material(eps_r=eps)},
        boundaries=[gfem.PEC("walls")],
    )


def slab(eps=4.0, tan_delta=0.05):
    return section_of(
        "slab_waveguide.msh",
        materials={
            "slab": gfem.Material(eps_r=eps, tan_delta=tan_delta),
            "air": gfem.Material(),
        },
        boundaries=[gfem.PEC("walls")],
    )


def square():
    return section_of(
        "square_waveguide.msh",
        materials={"air": gfem.Material()},
        boundaries=[gfem.PEC("walls")],
    )


def cpw():
    return section_of(
        "cpw_section.msh",
        unit="um",
        materials={"si": gfem.Material(eps_r=11.9), "air": gfem.Material()},
        boundaries=[gfem.PEC("walls"), gfem.PEC("metal")],
    )


@dataclasses.dataclass
class Pair:
    space: PortModeSpace
    k0: float
    shift: complex
    omega: float
    lam: complex
    x: np.ndarray
    a: sp.csr_matrix
    b: sp.csr_matrix


def exact(section, f, mode=1):
    """The exact eigenpair of the port mode pencil at ``f``, as the sweep solves it."""
    space = PortModeSpace(section, 2)
    g = _Greedy(space, np.array([float(f)]), [mode], 1e-6, [section.port])
    cand = g._solve(0)
    k0, shift = float(g.k0[0]), complex(g.shift[0])
    a, b = space.pencil.at(k0)
    return Pair(
        space,
        k0,
        shift,
        2 * np.pi * f,
        complex(cand.lam[mode - 1]),
        cand.vecs[:, mode - 1].copy(),
        a,
        b,
    )


def objective(pair, seed=0):
    """``L(lam, x)``: a profile functional with the normalization's invariances.

    ``Re r2.c + Im gamma`` with ``c`` the pencil vector scaled to unit
    mass-norm and made ``r.c`` real positive: invariant under ``x -> s x``.
    The mass is a fixed form, so no explicit parameter dependence remains.
    """
    n = len(pair.x)
    rng = np.random.default_rng(seed)
    r = jnp.asarray(rng.standard_normal(n))
    r2 = jnp.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    mass = jsp.BCOO.from_scipy_sparse(pair.space.mass.tocoo())

    def fn(lam, x):
        g = r @ x
        c = x * jnp.abs(g) / g / jnp.sqrt(jnp.real(jnp.conj(x) @ (mass @ x)))
        gamma = 1j * jnp.sqrt(-lam + 0j)
        return jnp.real(r2 @ c) + jnp.imag(gamma)

    return fn


def normalized_objective(pair, seed=0):
    """``Re r.c`` of the sweep's 1 W coordinates ``c`` (power and quadrant gauge)."""
    space = pair.space
    g_form, t_form = (jsp.BCOO.from_scipy_sparse(m.tocoo()) for m in space.power_forms)
    ln, lt = (jnp.asarray(v) for v in space.gauge_forms)
    tangential = jnp.asarray(space.free < space.n_t)
    n = len(pair.x)
    rng = np.random.default_rng(seed)
    r = jnp.asarray(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    mu0 = 4e-7 * np.pi

    def fn(lam, x):
        gamma = 1j * jnp.sqrt(-lam + 0j)
        c = jnp.where(tangential, x / gamma, x)
        p = (
            0.5
            * space.section.scale
            * 1j
            / (pair.omega * mu0)
            * (
                c @ (g_form @ jnp.conj(c))
                + jnp.conj(gamma) * (c @ (t_form @ jnp.conj(c)))
            )
        )
        c = c / jnp.sqrt(jnp.abs(p))
        g = -1j * ((ln + gamma * lt) @ c)
        return jnp.real(r @ (c * jnp.abs(g) / g))

    return fn


def cotangents(fn, pair):
    value, pull = jax.vjp(fn, jnp.asarray(pair.lam + 0j), jnp.asarray(pair.x))
    lam_bar, x_bar = pull(jnp.ones_like(value))
    return complex(lam_bar), np.asarray(x_bar)


def gradient(pair, fn, da, db, **options):
    """``Re q^T (dA - lam dB) x`` through the adjoint, as the kernels contract it."""
    lam_bar, x_bar = cotangents(fn, pair)
    factor = adj.shift_invert_factor(pair.a, pair.b, pair.shift)
    q = adj.pencil_cotangent(
        pair.a,
        pair.b,
        pair.lam,
        pair.x,
        lam_bar=lam_bar,
        x_bar=x_bar,
        factor=factor,
        **options,
    )
    return float(np.real(q @ (da @ pair.x) - pair.lam * (q @ (db @ pair.x))))


def central(fn_of_pair, build, p, h):
    return (fn_of_pair(build(p + h)) - fn_of_pair(build(p - h))) / (2 * h)


def test_filled_guide_gamma_matches_the_closed_form():
    """``d lam / d eps = -k0^2`` for TE modes of a filled PEC guide, with no solve."""
    eps = 2.3
    pair = exact(guide(eps), 7e9)
    pieces = pair.space.pencil
    factor = adj.shift_invert_factor(pair.a, pair.b, pair.shift)
    gamma = 1j * np.sqrt(-pair.lam + 0j)
    # L = Im gamma: cotangent on lam only
    lam_bar = complex(
        jax.grad(lambda lam: jnp.imag(1j * jnp.sqrt(-lam + 0j)))(pair.lam)
    )
    q = adj.pencil_cotangent(
        pair.a, pair.b, pair.lam, pair.x, lam_bar=lam_bar, x_bar=None, factor=factor
    )
    theta = pieces.theta(pair.k0)
    # eps scales the pieces of slot 1 (A_1 and B_1), so d/d eps = theta_1 / eps * (...)
    d_eps = theta[1] / eps * adj.scalar_cotangents(pieces, pair.lam, q, pair.x)[1]
    exact_d = (-(pair.k0**2) / (2 * gamma)).imag
    assert np.real(d_eps) == pytest.approx(exact_d, rel=1e-10)


def test_lossy_slab_profile_matches_exact_differences():
    p0, h, hm = 4.0, 1e-4, 1e-2
    f = 12e9
    pair = exact(slab(p0), f)
    fn = objective(pair)

    def value(section):
        other = exact(section, f)
        return float(fn(other.lam, other.x))

    fd = central(value, slab, p0, h)
    up, dn = exact(slab(p0 + hm), f), exact(slab(p0 - hm), f)
    da, db = (up.a - dn.a) / (2 * hm), (up.b - dn.b) / (2 * hm)
    assert gradient(pair, fn, da, db) == pytest.approx(fd, rel=1e-7)


def sweep_row(td=0.05):
    """A reduced row of the slab's sweep, with its nearest sample's factor."""
    grid = np.linspace(10e9, 20e9, 50)
    section = slab(4.0, td)
    space = PortModeSpace(section, 2)
    g = _Greedy(space, grid, [1], 1e-6, [section.port])
    g.run()
    solved = np.flatnonzero(g.solved)
    i = next(j for j in (10, 38, 20, 30) if not g.solved[j])
    near = solved[np.argmin(np.abs(solved - i))]
    a, b = space.pencil.at(g.k0[near])
    factor = adj.shift_invert_factor(a, b, g.shift[near])
    a, b = space.pencil.at(g.k0[i])
    lam = complex(g.lam[0, i])
    x = np.asarray(g.reduced.lift(g.coords_at(i, 0))).astype(complex)
    return space, grid[i], g.k0[i], factor, a, b, lam, x


def test_polished_reduced_row_matches_the_exact_pair_profile_gradient():
    _space, f, _k0, factor, a, b, lam, x = sweep_row()
    exact_pair = exact(slab(4.0, 0.05), f)
    pair = dataclasses.replace(exact_pair)
    fn = objective(pair)
    hm = 1e-2
    up, dn = exact(slab(4.0 + hm), f), exact(slab(4.0 - hm), f)
    da, db = (up.a - dn.a) / (2 * hm), (up.b - dn.b) / (2 * hm)
    truth = gradient(pair, fn, da, db)
    lam2, x2 = adj.polish(a, b, lam, x, factor=factor)
    rough = dataclasses.replace(pair, lam=lam, x=x)
    polished = dataclasses.replace(pair, lam=lam2, x=x2)
    err = lambda p: abs(gradient(p, fn, da, db) - truth) / abs(truth)  # noqa: E731
    assert err(polished) < 1e-8
    assert err(polished) < err(rough)


def test_gradient_near_a_degenerate_pair_holds_within_the_gap():
    base = square()
    pts = base.points.copy()
    c = pts.mean(axis=0)
    pts[:, 1] = c[1] + (1 + 1e-3) * (pts[:, 1] - c[1])
    base = dataclasses.replace(base, points=pts)
    cent = base.points[base.triangles].mean(axis=1)
    c = base.points.mean(axis=0)
    tri = np.flatnonzero(
        (cent[:, 0] > c[0]) & (cent[:, 1] > c[1] + 0.2 * np.ptp(cent[:, 1]))
    )

    def build(p):
        et, en = base.eps_tt.copy(), base.eps_nn.copy()
        et[tri] = p * np.eye(2)
        en[tri] = p
        return dataclasses.replace(base, eps_tt=et, eps_nn=en)

    f = 20e9
    pair = exact(build(1.0), f)
    fn = objective(pair)
    hm = 1e-3
    up, dn = exact(build(1 + hm), f), exact(build(1 - hm), f)
    da, db = (up.a - dn.a) / (2 * hm), (up.b - dn.b) / (2 * hm)
    got = gradient(pair, fn, da, db)

    def value(p):
        s = exact(build(p), f)
        return float(fn(s.lam, s.x))

    gap = 2e-3
    h = 1e-2 * gap
    fd4 = (
        -value(1 + 2 * h) + 8 * value(1 + h) - 8 * value(1 - h) + value(1 - 2 * h)
    ) / (12 * h)
    assert got == pytest.approx(fd4, rel=1e-6)


def test_a_solve_forced_to_miss_raises_the_degenerate_error():
    pair = exact(slab(), 12e9)
    fn = objective(pair)
    lam_bar, x_bar = cotangents(fn, pair)
    factor = adj.shift_invert_factor(pair.a, pair.b, pair.shift)
    with pytest.raises(PortModeError, match="mirror half model"):
        adj.pencil_cotangent(
            pair.a,
            pair.b,
            pair.lam,
            pair.x,
            lam_bar=lam_bar,
            x_bar=x_bar,
            factor=factor,
            max_iter=1,
            fallback=False,
        )


def test_stagnation_stop_is_bounded_and_keeps_the_gradient():
    """Asking for more than the section's attainable accuracy stops at the floor."""
    pair = exact(cpw(), 100e9)
    # an eigenvector gauge that changes nothing mathematically
    x = pair.x * (0.3 - 2j)
    fn = normalized_objective(pair)
    here = dataclasses.replace(pair, x=x)
    lam_bar, rhs = cotangents(fn, here)
    factor = adj.shift_invert_factor(pair.a, pair.b, pair.shift)
    args = (pair.a, pair.b, pair.lam, x, rhs)
    reference = adj.deflated_solve(*args, factor=factor, rtol=1e-10, accept=None)
    assert reference.iterations < 30
    # rtol 1e-14 lies below the true residual's floor: the residual stalls
    stopped = adj.deflated_solve(*args, factor=factor, rtol=1e-14, max_iter=250)
    assert stopped.iterations < 250
    assert not stopped.direct
    assert stopped.residual < 1e-8
    pieces = pair.space.pencil

    def cot(w):
        q = lam_bar * x / (x @ (pair.b @ x)) - w
        return adj.scalar_cotangents(pieces, pair.lam, q, x)

    ref, got = cot(reference.w), cot(stopped.w)
    assert np.linalg.norm(got - ref) <= 3e-8 * np.linalg.norm(ref)
