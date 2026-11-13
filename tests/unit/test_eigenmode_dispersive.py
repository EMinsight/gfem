"""Eigenmode solves of a model with a permittivity law (ADR 0115).

The frozen seed and residual inverse iteration find the modes of a law
model exactly outside the law's unresolved band. The references:

- a closed PEC box (10 x 8 x 5 mm) filled with the law: each cavity cutoff
  omega_c gives the roots of eps(omega) omega^2 = omega_c^2 (both polariton
  branches of a Lorentz medium, the plasma branch of a Drude medium). With
  the box's analytic cutoffs this is the closed form; with the discrete
  cutoffs of the same space (a vacuum solve of the same box) it is the exact
  spectrum of the discrete nonlinear problem;
- a constant `Material` holding the law's value at a lossless mode.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._engine import terms

GHZ = 1e9
A, B, D = 10.0, 8.0, 5.0  # the box, mm
C_MM = 299792458.0 * 1e3  # mm/s
ORDER_3 = gfem.SolverOptions(order=3, eigen_tolerance=1e-12)
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=25 * GHZ, damping=0.0)],
)
DRUDE = gfem.PermittivityLaw(
    eps_inf=2.0, terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=0.0)]
)
LOSSY_DRUDE = gfem.PermittivityLaw(
    eps_inf=2.0, terms=[gfem.Drude(plasma_frequency=20 * GHZ, damping=1 * GHZ)]
)


def _box(directory, cells):
    path = directory / "box.msh"
    sm.write_msh22(path, sm.plate(A, B, (("box", D),), cells))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    """The box at (5, 4, 3) cells (h_max 3.28 mm).

    At order 3 its cutoffs below the band (kh < pi, 45 GHz in vacuum) are
    good to 5e-5, and eight of them lie there.
    """
    return _box(tmp_path_factory.mktemp("box"), (5, 4, 3))


@pytest.fixture(scope="module")
def discrete(box):
    """The box's discrete cutoffs (Hz) near 30 GHz: 24 to 42 GHz."""
    vacuum = gfem.solve(
        filled(box, 1.0), gfem.Eigenmode(target=30 * GHZ, n_modes=8), options=ORDER_3
    )
    return np.sort(np.real(np.asarray(vacuum.omega))) / (2 * math.pi)


def filled(mesh, eps_r):
    return gfem.Model(
        mesh,
        materials={region: gfem.Material(eps_r=eps_r) for region in mesh.volumes},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side")],
    )


def cutoffs(limit=80 * GHZ):
    """The box's analytic cavity cutoffs (Hz) below ``limit``, with multiplicity."""
    out = []
    n = 12
    for i in range(n):
        for j in range(n):
            for k in range(n):
                zeros = (i == 0) + (j == 0) + (k == 0)
                if zeros >= 2:
                    continue
                fc = 0.5 * C_MM * math.sqrt((i / A) ** 2 + (j / B) ** 2 + (k / D) ** 2)
                if fc < limit:
                    out += [fc] * (1 if zeros else 2)
    return np.sort(out)


def roots(law, fc):
    """The frequencies (Hz) with eps(f) f^2 = fc^2, as polynomial roots.

    Lorentz (lossless): 2 f^4 ... in x = f^2; Drude: eps_inf f^3 - i g
    eps_inf f^2 - (fp^2 + fc^2) f + i g fc^2 = 0 (e^{+i omega t},
    chi = -fp^2 / (f (f - i g))).
    """
    (term,) = law.terms
    e = float(law.eps_inf)
    if isinstance(term, gfem.Lorentz):
        d, f0 = float(term.delta_eps), float(term.resonance_frequency)
        # (e + d f0^2 / (f0^2 - x)) x = fc^2
        x = np.roots([e, -(e * f0**2 + d * f0**2 + fc**2), fc**2 * f0**2])
        return np.sqrt(x.astype(complex))
    fp, g = float(term.plasma_frequency), float(term.damping)
    found = np.roots([e, -1j * g * e, -(fp**2 + fc**2), 1j * g * fc**2])
    return found[found.real > 0]


def closed_form_error(law, f):
    """The relative distance from ``f`` to the nearest closed-form root."""
    candidates = np.concatenate([roots(law, fc) for fc in np.unique(cutoffs())])
    return float(np.min(np.abs(candidates - f)) / abs(f))


def discrete_error(law, f, discrete):
    """``f``'s distance to an exact root on the discrete cutoffs of the space.

    ``eps(f) f^2`` must be the square of a discrete cutoff (real): one of
    the vacuum box's on the same space.
    """
    fc2 = complex(law(f)) * f * f
    return float(np.min(np.abs(fc2 - discrete**2)) / abs(fc2))


@pytest.mark.parametrize(
    ("law", "target"),
    [
        (LORENTZ, 12 * GHZ),  # lower polariton branch, below the 25 GHz pole
        (LORENTZ, 32 * GHZ),  # upper branch, above eps = 0 at 30.6 GHz
        (DRUDE, 22 * GHZ),
        (LOSSY_DRUDE, 22 * GHZ),
    ],
    ids=["lorentz-lower", "lorentz-upper", "drude", "drude-lossy"],
)
def test_a_filled_box_resonates_at_the_closed_form_branches(box, discrete, law, target):
    result = gfem.solve(
        filled(box, law), gfem.Eigenmode(target=target, n_modes=4), options=ORDER_3
    )
    f = np.asarray(result.omega) / (2 * math.pi)
    for mode in f:
        assert closed_form_error(law, mode) < 1e-4, mode
        assert discrete_error(law, mode, discrete) < 1e-10, mode
    if law is LOSSY_DRUDE:
        assert np.all(f.imag != 0)


def test_a_drude_box_returns_no_mode_at_its_plasma_frequency(box):
    # eps(f_B) = 0 at f_B = fp / sqrt(eps_inf) = 14.14 GHz: every gradient
    # field is an eigenvector there, and none is returned; the transverse
    # modes start at the plasma branch of the lowest cutoff (22.09 GHz)
    f_b = 20 * GHZ / math.sqrt(2.0)
    result = gfem.solve(
        filled(box, DRUDE),
        gfem.Eigenmode(target=1.01 * f_b, n_modes=2),
        options=ORDER_3,
    )
    f = np.real(np.asarray(result.omega)) / (2 * math.pi)
    assert np.all(f > 1.5 * f_b)
    for mode in f:
        assert closed_form_error(DRUDE, mode) < 1e-4


@pytest.fixture
def operators(monkeypatch):
    """The eigen operators the solves build."""
    seen = []
    build = eigen_rule.EigenmodeRule.operator

    def spy(self, op, sigma):
        eop = build(self, op, sigma)
        seen.append(eop)
        return eop

    monkeypatch.setattr(eigen_rule.EigenmodeRule, "operator", spy)
    return seen


def test_a_lossless_drude_law_is_solved_as_a_polynomial_problem(
    box, operators, monkeypatch
):
    # theta = -k^2 chi = k_p^2, a constant in K: the real snap applies
    problem = gfem.Eigenmode(target=22 * GHZ, n_modes=2)
    model = filled(box, DRUDE)
    polynomial = np.asarray(gfem.solve(model, problem, options=ORDER_3).omega)
    assert not operators[-1].nonpolynomial
    assert np.all(polynomial.imag == 0)
    # the general route: the same law as a nonpolynomial piece
    monkeypatch.setattr(terms, "polynomial_law", lambda _law: False)
    general = np.asarray(gfem.solve(model, problem, options=ORDER_3).omega)
    assert operators[-1].nonpolynomial
    np.testing.assert_allclose(polynomial, general, rtol=1e-12)


# --- the unresolved band (ADR 0115) ------------------------------------------------


@pytest.fixture(scope="module")
def coarse(tmp_path_factory):
    """The probe-04 box at (6, 5, 5) cells (h_max 2.52 mm), for order 1."""
    return _box(tmp_path_factory.mktemp("coarse"), (6, 5, 5))


@pytest.fixture(scope="module")
def plate(tmp_path_factory):
    """The box at (13, 11, 5 + 5) cells in two layers (h_max 1.22 mm), for order 1.

    Ticket 10's order-1 mesh: there the frozen seed's four nearest roots
    were measured exact. (On the coarse box at order 1, 20 and 22 GHz
    targets skip one nearer root of the crowded lower branch.)
    """
    path = tmp_path_factory.mktemp("plate") / "plate.msh"
    sm.write_msh22(path, sm.plate(A, B, (("host", 3.0), ("core", 2.0)), (13, 11, 5)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


@pytest.fixture
def seed_solves(monkeypatch):
    """The seed solves the eigenmode solves run."""
    count = [0]
    shift_invert = eigen_rule.EigenmodeRule._shift_invert

    def spy(self, *args, **kwargs):
        count[0] += 1
        return shift_invert(self, *args, **kwargs)

    monkeypatch.setattr(eigen_rule.EigenmodeRule, "_shift_invert", spy)
    return count


def lower_branch(law, fc):
    """The lower polariton root (Hz) of a lossless Lorentz law at cutoff ``fc``."""
    return float(np.min(roots(law, fc).real))


def nearest_roots(mesh, law, target, n, options):
    """The ``n`` exact lower-branch roots nearest ``target`` on ``mesh``'s space.

    From the vacuum box's discrete cutoffs around the target's own cutoff
    ``f sqrt(eps(f))``: three times ``n`` of them, so the nearest roots are
    among their images.
    """
    fc = target * math.sqrt(complex(law(target)).real)
    vacuum = gfem.solve(
        filled(mesh, 1.0), gfem.Eigenmode(target=fc, n_modes=3 * n), options=options
    )
    cut = np.real(np.asarray(vacuum.omega)) / (2 * math.pi)
    images = np.array([lower_branch(law, c) for c in cut])
    order = np.argsort(np.abs(images - target))
    # the images beyond the n nearest lie farther than the nth: none skipped
    assert np.abs(images[order[n:]] - target).min() >= np.abs(
        images[order[n - 1]] - target
    )
    return np.sort(images[order[:n]])


@pytest.mark.parametrize(
    ("mesh_name", "order", "target"),
    [("plate", 1, 24 * GHZ), ("box", 3, 12 * GHZ)],
    ids=["order1", "order3"],
)
def test_below_the_band_the_nearest_exact_roots_come_back(
    request, mesh_name, order, target
):
    # kh 2.3 (24 GHz, 4 % below the pole, eps 14.8) and 1.6 (12 GHz)
    mesh = request.getfixturevalue(mesh_name)
    options = gfem.SolverOptions(order=order, eigen_tolerance=1e-12)
    result = gfem.solve(
        filled(mesh, LORENTZ), gfem.Eigenmode(target=target, n_modes=4), options=options
    )
    f = np.sort(np.real(np.asarray(result.omega))) / (2 * math.pi)
    np.testing.assert_allclose(
        f, nearest_roots(mesh, LORENTZ, target, 4, options), rtol=1e-10
    )


@pytest.mark.parametrize(
    ("mesh_name", "order"), [("coarse", 1), ("box", 3)], ids=["order1", "order3"]
)
def test_a_target_in_the_band_raises_before_any_seed_solve(
    request, seed_solves, mesh_name, order
):
    # 24 GHz, 4 % below the lossless 25 GHz pole: eps = 14.8, kh 4.9 / 6.4
    mesh = request.getfixturevalue(mesh_name)
    with pytest.raises(ValueError, match="unresolved band") as raised:
        gfem.solve(
            filled(mesh, LORENTZ),
            gfem.Eigenmode(target=24 * GHZ),
            options=gfem.SolverOptions(order=order),
        )
    message = str(raised.value)
    assert "region 'box'" in message
    assert "law.terms[0] (Lorentz" in message
    assert "25 GHz" in message
    assert "24 GHz" in message
    assert "eps = 14.8" in message
    assert seed_solves[0] == 0


def test_a_target_on_a_lossless_pole_is_in_the_band(coarse, seed_solves):
    with pytest.raises(ValueError, match="unresolved band"):
        gfem.solve(
            filled(coarse, LORENTZ),
            gfem.Eigenmode(target=25 * GHZ),
            options=gfem.SolverOptions(order=1),
        )
    assert seed_solves[0] == 0


def test_a_returned_mode_in_the_band_raises(coarse, seed_solves):
    # a target just below the band (kh 3.09 at 22.3 GHz): the nearest modes
    # reach above it, where kh passes pi from about 22.4 GHz
    with pytest.raises(
        ValueError, match=r"Eigenmode's mode \d at .* GHz lies in the unresolved band"
    ):
        gfem.solve(
            filled(coarse, LORENTZ),
            gfem.Eigenmode(target=22.3 * GHZ, n_modes=6),
            options=gfem.SolverOptions(order=1),
        )
    assert seed_solves[0] >= 1


# --- longitudinal modes (ADR 0115 amendment) -----------------------------------------


def _gradient_pair(model, f_b, options):
    """The rule, its eigen operator at f_B and a gradient field (complex).

    ``x - P x`` for a random ``x``: the gradient part the divergence-free
    projection removes, an exact eigenvector at the plasma zero f_B of a
    lossless Drude fill, where T(k_B) grad phi = (k_p^2 - k_B^2 eps_inf) A
    grad phi = 0.
    """
    from gfem._engine.driven import driven_operator
    from gfem._engine.eigen import projector

    rule = eigen_rule.EigenmodeRule(model, gfem.Eigenmode(target=f_b), options)
    op = driven_operator(model, options.order)
    sigma = terms.wavenumber(2 * math.pi * f_b, op.built.scale)
    eop = rule.operator(op, sigma)
    eop.factorize()
    divfree = projector(model, op)
    rng = np.random.default_rng(0)
    y = np.where(eop.free, rng.standard_normal(len(eop.free)), 0).astype(complex)
    x = y - divfree(y)
    divfree.free()
    return rule, eop, sigma, x / np.linalg.norm(x)


@pytest.mark.parametrize("kh", [1.0, 1e-3])
def test_a_gradient_pair_at_a_plasma_zero_is_dropped(coarse, kh):
    # k_B h_max = kh: f_B = kh c / (2 pi h_max), h_max 2.52 mm
    options = gfem.SolverOptions(order=1, eigen_tolerance=1e-10)
    f_b = kh * 299792458.0 / (2 * math.pi * 2.5166e-3)
    law = gfem.PermittivityLaw(
        eps_inf=2.0, terms=[gfem.Drude(plasma_frequency=math.sqrt(2) * f_b, damping=0)]
    )
    rule, eop, sigma, gradient = _gradient_pair(filled(coarse, law), f_b, options)
    assert (
        np.linalg.norm(eop.apply(sigma, gradient))
        < 1e-8 * np.linalg.norm(eop.parts(gradient)[2]) * abs(sigma) ** 2
    )
    assert eop.curl_ratio(sigma, gradient) < 1e-4
    modes, *_ = rule._select(eop, np.array([sigma + 0j]), gradient[:, None], sigma)
    assert modes == []
    # the box's lowest transverse field at the same k is far above the threshold
    vacuum = gfem.solve(
        filled(coarse, 1.0), gfem.Eigenmode(target=24 * GHZ, n_modes=1), options=options
    )
    transverse = np.asarray(vacuum.e_fields[0].coefficients)
    assert eop.curl_ratio(sigma, transverse) > 0.5
    eop.free_handle()


@pytest.fixture(scope="module")
def core_box(tmp_path_factory):
    """The box at (6, 5, 3 + 3) cells: a 3 mm "host" under a 2 mm "core"."""
    path = tmp_path_factory.mktemp("core") / "box.msh"
    sm.write_msh22(path, sm.plate(A, B, (("host", 3.0), ("core", 2.0)), (6, 5, 3)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def cored(mesh, core):
    """The core box with ``core`` (an eps_r) on "core", eps 4 on "host"."""
    return gfem.Model(
        mesh,
        materials={"host": gfem.Material(eps_r=4.0), "core": gfem.Material(eps_r=core)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side")],
    )


def test_a_drude_core_next_to_its_plasma_zero_returns_transverse_modes(
    core_box, operators
):
    # ticket 04's case: host eps 4, a 2 mm Drude core with f_B = 19.799 GHz,
    # the target 1e-6 above it; hundreds of gradient eigenvalues sit at f_B
    law = gfem.PermittivityLaw(
        eps_inf=2.0, terms=[gfem.Drude(plasma_frequency=28 * GHZ, damping=0)]
    )
    model = cored(core_box, law)
    f_b = 28 * GHZ / math.sqrt(2.0)
    result = gfem.solve(
        model,
        gfem.Eigenmode(target=f_b * (1 + 1e-6), n_modes=4),
        options=gfem.SolverOptions(order=2),
    )
    eop = operators[-1]
    for mode, field in zip(result.omega, result.e_fields, strict=True):
        f = float(np.real(mode)) / (2 * math.pi)
        assert abs(f - f_b) > 1e-3 * f_b
        k = complex(mode) * eop.scale / 299792458.0
        assert eop.curl_ratio(k, np.asarray(field.coefficients)) > 0.5


def test_a_lossless_law_mode_is_the_constant_materials_at_its_frequency(core_box):
    # a lossless Lorentz core: at its mode f*, the core with the constant
    # eps(f*) resonates at f* too
    options = gfem.SolverOptions(order=2, eigen_tolerance=1e-12)
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40 * GHZ, damping=0)],
    )
    result = gfem.solve(
        cored(core_box, law), gfem.Eigenmode(target=15 * GHZ), options=options
    )
    f = float(np.real(result.omega[0])) / (2 * math.pi)
    eps = complex(law(f))
    assert eps.imag == 0
    constant = gfem.solve(
        cored(core_box, eps.real), gfem.Eigenmode(target=f), options=options
    )
    np.testing.assert_allclose(
        np.real(constant.omega[0]) / (2 * math.pi), f, rtol=1e-10
    )


# --- gradients by the law's parameters (ADR 0112) ------------------------------------


def test_omega_differentiates_by_the_law_parameters(core_box):
    # a lossy Lorentz core: the mode is complex, so the weight reads Q too
    import jax
    import jax.numpy as jnp
    from finite_differences import derivative

    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40 * GHZ, damping=2e9)],
    )
    leaves, tree = jax.tree.flatten(law)
    options = gfem.SolverOptions(order=1, eigen_tolerance=1e-12)
    problem = gfem.Eigenmode(target=15 * GHZ)

    def objective(t):
        moved = [p * (1 + t[i]) for i, p in enumerate(leaves)]
        scaled = jax.tree.unflatten(tree, moved)
        omega = gfem.solve(cored(core_box, scaled), problem, options=options).omega[0]
        return jnp.real(omega * (1 + 3j)) * 1e-9

    t0 = jnp.zeros(len(leaves))
    grad = np.asarray(jax.grad(objective)(t0))
    for i in range(len(leaves)):
        e = np.zeros(len(leaves))
        e[i] = 1.0
        fd = derivative(objective, t0, jnp.asarray(e), 1e-3)
        np.testing.assert_allclose(grad[i], fd, rtol=1e-6)


@pytest.mark.parametrize(
    "read",
    [
        lambda r: r.bulk_participation["core"][0],
        lambda r: jnp.real(r.e_fields[0].coefficients[5]),
    ],
    ids=["bulk_participation", "e_fields"],
)
def test_mode_field_outputs_refuse_a_law_parameter(core_box, read):
    import jax

    def objective(eps_inf):
        law = gfem.PermittivityLaw(
            eps_inf=eps_inf, terms=[gfem.Drude(plasma_frequency=5 * GHZ, damping=1e9)]
        )
        return read(gfem.solve(cored(core_box, law), gfem.Eigenmode(target=15 * GHZ)))

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"permittivity law of model\.materials\['core'\].*eigenfrequency or Q",
    ):
        jax.make_jaxpr(jax.grad(objective))(2.0)
