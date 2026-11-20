"""The mode energy of a model with a permittivity law.

A mode's volume energy reads the energy permittivity eps_W = Re[eps +
(omega / 2) eps'] at its frequency on a law region, so that for a lossless
law the mode energy is the eigenproblem's own normalization: the Gram
``x^T T'(k) x`` is ``-k`` times the quasinormal norm ``N``, and ``N = 2 int
eps_W |E|^2``. The ratios then obey the eigenvalue derivatives
(Hellmann-Feynman): the host's share is ``-2 d ln omega / d ln eps_host`` and
the junction's participation ``-2 d ln omega / d ln L``.

The references: a box (a 3 mm host of eps 4 under a 2 mm core
holding a lossless Lorentz law) and the uniform-field line of
`eigen_cases` with a Lorentz region at its junction's end.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.constants as const
import structured_meshes as sm
from eigen_cases import EPS, shunt_mesh
from finite_differences import derivative

import gfem
from gfem._eigen import rule as eigen_rule
from gfem._engine import dispersion
from gfem.conventions import MODE_ENERGY

GHZ = 1e9
#: A lossless Lorentz core: eps 4.46 and eps_W 5.04 at its 17.4 GHz mode
#: (eps_inf alone would weigh it 23 % low).
LORENTZ = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=2.0, resonance_frequency=40 * GHZ, damping=0.0)],
)
TARGET = 18 * GHZ
OPTIONS = gfem.SolverOptions(order=2, eigen_tolerance=1e-12)


@pytest.fixture(scope="module")
def core_box(tmp_path_factory):
    """A 10 x 8 x 5 mm PEC box: a 3 mm "host" under a 2 mm "core"."""
    path = tmp_path_factory.mktemp("core") / "box.msh"
    sm.write_msh22(path, sm.plate(10.0, 8.0, (("host", 3.0), ("core", 2.0)), (6, 5, 3)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def cored(mesh, core, host=4.0):
    return gfem.Model(
        mesh,
        materials={
            "host": gfem.Material(eps_r=host),
            "core": gfem.Material(eps_r=core),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side")],
    )


#: One law per term kind, lossy, evaluated at 7 GHz.
WEIGHT_LAWS = [
    gfem.PermittivityLaw(
        eps_inf=1.5, terms=[gfem.Drude(plasma_frequency=15 * GHZ, damping=2 * GHZ)]
    ),
    gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=9 * GHZ, damping=1e9)],
    ),
    gfem.PermittivityLaw(
        eps_inf=3.0, terms=[gfem.Debye(delta_eps=2.0, relaxation_frequency=5 * GHZ)]
    ),
    gfem.PermittivityLaw(
        eps_inf=3.0,
        terms=[
            gfem.DjordjevicSarkar(delta_eps=1.1, low_frequency=1e3, high_frequency=1e12)
        ],
    ),
]


@pytest.mark.parametrize(
    "law", WEIGHT_LAWS, ids=["drude", "lorentz", "debye", "djordjevic_sarkar"]
)
def test_the_energy_weights_match_the_laws_own_derivatives(law):
    # the law evaluates in jax.numpy: its omega derivatives by jax.jacfwd
    import jax

    def eps(w):
        return law(w / (2 * math.pi))

    def real(f):
        return lambda w: jnp.real(f(w))

    def imag(f):
        return lambda w: jnp.imag(f(w))

    def d(f):
        # d/d omega of a complex function of a real omega
        return lambda w: jax.jacfwd(real(f))(w) + 1j * jax.jacfwd(imag(f))(w)

    w = 2 * math.pi * 7 * GHZ
    e, e1, e2 = complex(eps(w)), complex(d(eps)(w)), complex(d(d(eps))(w))
    np.testing.assert_allclose(
        dispersion.energy_eps(law, w), (e + 0.5 * w * e1).real, rtol=1e-12
    )
    np.testing.assert_allclose(
        dispersion.brillouin_eps(law, w), (e + w * e1).real, rtol=1e-12
    )
    np.testing.assert_allclose(
        dispersion.d_energy_eps(law, w), (1.5 * e1 + 0.5 * w * e2).real, rtol=1e-10
    )


@pytest.fixture
def kept(monkeypatch):
    """The rule and driven operator of the last forward."""
    out = {}
    post = eigen_rule.EigenmodeRule._postprocess

    def wrapped(self, model, op, divfree, modes):
        out.update(rule=self, op=op)
        return post(self, model, op, divfree, modes)

    monkeypatch.setattr(eigen_rule.EigenmodeRule, "_postprocess", wrapped)
    return out


def test_the_normalized_lossless_mode_is_the_grams_quasinormal_norm(core_box, kept):
    result = gfem.solve(
        cored(core_box, LORENTZ), gfem.Eigenmode(target=TARGET), options=OPTIONS
    )
    op, rule = kept["op"], kept["rule"]
    scale = op.built.scale
    per_k = const.c / scale
    omega = complex(result.omega[0])
    k = omega / per_k
    e = np.asarray(result.e_fields[0].coefficients)
    eop = rule.operator(op, k)
    try:
        kx, _, mx, ax = eop._parts(e)
        gram = e @ eop.derivative(k, e)
        (law,) = [t for t in eop.nonpolynomial if t.name == "susceptibility core"]
        (a_x,) = [a for t, a in zip(eop.nonpolynomial, ax, strict=True) if t is law]
    finally:
        eop.free_handle()
    chi = dispersion.susceptibility(LORENTZ, omega)
    d_chi = dispersion.d_susceptibility(LORENTZ, omega) * per_k
    # E-only quasinormal norm, unconjugated: x^T K x / k^2 + x^T d(k eps)/dk M x
    norm = (e @ kx) / k**2 - (e @ mx) + (chi + k * d_chi) * (e @ a_x)
    assert abs(omega.imag) == 0
    np.testing.assert_allclose(gram / (-k * norm), 1.0, rtol=1e-12)
    # 1 J = 1/2 eps0 int eps_W |E|^2 = eps0 scale N / 4
    np.testing.assert_allclose(
        (const.epsilon_0 * scale * norm / 4).real, MODE_ENERGY, rtol=1e-12
    )


def test_the_hosts_share_is_minus_twice_the_log_derivative_in_its_permittivity(
    core_box,
):
    problem = gfem.Eigenmode(target=TARGET)

    def solve(t):
        model = cored(core_box, LORENTZ, host=4.0 * math.exp(float(t)))
        return gfem.solve(model, problem, options=OPTIONS)

    result = solve(0.0)
    slope = derivative(lambda t: jnp.log(jnp.real(solve(t).omega[0])), 0.0, 1.0, 1e-3)
    share = float(result.bulk_participation["host"][0])
    np.testing.assert_allclose(share, -2 * slope, rtol=1e-7)
    # the core's share is eps_W's, not eps_inf's (0.37 against 0.19)
    np.testing.assert_allclose(
        share + float(result.bulk_participation["core"][0]), 1.0, rtol=1e-12
    )


#: The shunt line's junction and the Lorentz law on its "left" half (the
#: junction's end), lossless with its pole far above the 3 GHz mode.
JUNCTION_L = 1e-9
LINE_LAW = gfem.PermittivityLaw(
    eps_inf=EPS - 1.0,
    terms=[gfem.Lorentz(delta_eps=0.9, resonance_frequency=8 * GHZ, damping=0.0)],
)


@pytest.fixture(scope="module")
def halves(tmp_path_factory):
    return shunt_mesh(tmp_path_factory.mktemp("shunt"))


def junction_line(mesh, inductance):
    return gfem.Model(
        mesh,
        materials={
            "left": gfem.Material(eps_r=LINE_LAW),
            "right": gfem.Material(eps_r=EPS),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[gfem.LumpedPort("jj", "port1", "+z", inductance=inductance)],
    )


def test_the_junction_participation_beside_a_law_is_minus_twice_the_log_derivative(
    halves,
):
    # the line's field is uniform over each cross-section (PMC sides), so
    # the port voltage gives the inductor's energy exactly
    def solve(t):
        model = junction_line(halves, JUNCTION_L * math.exp(float(t)))
        return gfem.solve(model, gfem.Eigenmode(target=3 * GHZ), options=OPTIONS)

    result = solve(0.0)
    slope = derivative(lambda t: jnp.log(jnp.real(solve(t).omega[0])), 0.0, 1.0, 1e-2)
    np.testing.assert_allclose(
        abs(float(result.participation["jj"][0])), -2 * slope, rtol=1e-7
    )


# --- refusals -------------------------------------------------------------------------


def damped(f0):
    """A Lorentz law damped at half its resonance: eps_W(f0) = 2 - 4 = -2."""
    return gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=f0, damping=f0 / 2)],
    )


def test_a_mode_where_the_energy_permittivity_is_not_positive_raises(core_box):
    # the box's mode with a constant core of eps_inf is at 18.04 GHz
    f0 = 18.035e9
    with pytest.raises(
        ValueError,
        match=r"mode 0 at .*energy permittivity.* = -1\.\d+ <= 0 in region 'core'.*"
        r"Lorentz.*gfem\.Conductor",
    ):
        gfem.solve(cored(core_box, damped(f0)), gfem.Eigenmode(target=f0))


@pytest.fixture(scope="module")
def strip(tmp_path_factory):
    path = tmp_path_factory.mktemp("strip") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 2, 2), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def strip_line(mesh, core):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=4.0),
            "core": gfem.Material(eps_r=core),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def test_electric_energy_reads_the_brillouin_weight_in_a_law_region(strip):
    # a lossless Lorentz core: Re d(omega eps)/d omega = eps + omega eps'
    f = 6 * GHZ
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=12 * GHZ, damping=0.0)],
    )
    model = strip_line(strip, law)
    field = gfem.solve(model, gfem.Driven(frequency=[f], keep_fields=True)).e_field(
        f, excite="p1"
    )
    energy = gfem.electric_energy(field, model=model)
    # the same field in a model whose core is a constant 1: |E|^2 there
    plain = gfem.electric_energy(field, model=strip_line(strip, 1.0))
    # at f0 / f = 2: eps = 2 + f0^2 / (f0^2 - f^2) = 2 + 4/3 and
    # f eps' = 2 f^2 f0^2 / (f0^2 - f^2)^2 = 8/9
    weight = 2.0 + 4 / 3 + 8 / 9
    np.testing.assert_allclose(energy["core"], weight * plain["core"], rtol=1e-12)
    np.testing.assert_allclose(energy["dielectric"], plain["dielectric"], rtol=1e-15)


def test_electric_energy_where_the_brillouin_weight_is_not_positive_raises(strip):
    f = 6 * GHZ
    model = strip_line(strip, damped(f))
    field = gfem.solve(model, gfem.Driven(frequency=[f], keep_fields=True)).e_field(
        f, excite="p1"
    )
    with pytest.raises(
        ValueError, match=r"field at 6 GHz has Re d\(omega eps\)/d omega = .* 'core'"
    ):
        gfem.electric_energy(field, model=model)


@pytest.fixture(scope="module")
def layered_modes(tmp_path_factory):
    """Modes of a two-layer line ("sub" / "cover", interface "sub__cover").

    The top cells of the cover form a third region "far", which touches no
    face of the interface; the returned law is a lossy Lorentz one.
    """
    path = tmp_path_factory.mktemp("layers") / "line.msh"
    nodes, tets, volumes, tris, surfaces = sm.strip_line(
        10.0, 2.0, 1.0, (10, 2, 2), layers=(("sub", 0.5), ("cover", 0.5))
    )
    centroids = nodes[tets].mean(axis=1)
    volumes = [
        "far" if c[2] > 0.75 else v for v, c in zip(volumes, centroids, strict=True)
    ]
    sm.write_msh22(path, (nodes, tets, volumes, tris, surfaces))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    law = gfem.PermittivityLaw(
        eps_inf=3.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=30 * GHZ, damping=1e9)],
    )

    def solve(cover, far):
        model = gfem.Model(
            mesh,
            materials={
                "sub": gfem.Material(eps_r=4.0),
                "cover": gfem.Material(eps_r=cover),
                "far": gfem.Material(eps_r=far),
            },
            boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
            ports=[gfem.LumpedPort("jj", "port1", "+z", inductance=1e-9)],
        )
        return gfem.solve(model, gfem.Eigenmode(target=3 * GHZ), options=OPTIONS)

    return solve, law


def test_interface_participation_skips_a_law_region_off_the_interface(layered_modes):
    solve, law = layered_modes
    modes = solve(2.0, law)
    measured = gfem.interface_participation(modes, "sub__cover", 3e-9, 10.0)
    assert float(measured.participation[0]) > 0


def test_interface_participation_refuses_a_law_on_a_side(layered_modes):
    solve, law = layered_modes
    modes = solve(law, 2.0)
    with pytest.raises(ValueError, match=r"'sub__cover'.*region 'cover'.*law"):
        gfem.interface_participation(modes, "sub__cover", 3e-9, 10.0)


def test_a_law_as_the_layers_permittivity_is_a_named_type_error(layered_modes):
    solve, law = layered_modes
    modes = solve(2.0, 2.5)
    with pytest.raises(TypeError, match="layer's eps_r must be a number"):
        gfem.interface_participation(modes, "sub__cover", 3e-9, law)
