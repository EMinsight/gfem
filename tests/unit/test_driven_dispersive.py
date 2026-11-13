"""Driven solves of a model with a permittivity law (a dispersive material).

A law region's eps_inf stays in the mass and the rest of the law enters as
one susceptibility piece (ADR 0110). Each sweep must equal, frequency by
frequency, the same model with a constant material holding the law's value
there: a `Material` where Re eps > 0, and where no `Material` admits the
value (Re eps <= 0, a metal below its plasma frequency) the reference
overrides the region's mass coefficient with eps(f).

The line is a 10 x 2 x 1 mm parallel-plate line with lumped ports at both
ends; its 2 mm "core" (x in [4, 6]) holds the law.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem._engine import coefficients, dispersion, terms

GHZ = 1e9
OPTIONS = gfem.SolverOptions(linear_solver="direct")
EPS_HOST = 4.0


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (20, 2, 2), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _model(mesh, core, *, boundaries=(), host=None):
    return gfem.Model(
        mesh,
        materials={
            "dielectric": host or gfem.Material(eps_r=EPS_HOST),
            "core": core,
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *boundaries],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _reference(mesh, eps, f, monkeypatch, *, sigma=0.0, boundaries=()):
    """S at ``f`` of the model with a constant core holding ``eps``."""
    # a name per frequency: no operator cached for another value is reused
    name = f"reference {f:g} {eps!r}"
    if eps.real > 0:
        core = gfem.Material(
            eps_r=eps.real,
            tan_delta=-eps.imag / eps.real,
            conductivity=sigma,
            name=name,
        )
        model = _model(mesh, core, boundaries=boundaries)
        return np.asarray(
            gfem.solve(model, gfem.Driven(frequency=[f]), options=OPTIONS).s.values
        )[0]
    core = gfem.Material(eps_r=1.0, conductivity=sigma, name=name)
    model = _model(mesh, core, boundaries=boundaries)
    mask = np.zeros(len(mesh.tetrahedra), bool)
    mask[mesh.elements("core")] = True
    original = dispersion.high_frequency_eps

    def overridden(m):
        out = np.asarray(original(m)).astype(complex)
        out[mask] = eps
        return out

    with monkeypatch.context() as patch:
        patch.setattr(dispersion, "high_frequency_eps", overridden)
        return np.asarray(
            gfem.solve(model, gfem.Driven(frequency=[f]), options=OPTIONS).s.values
        )[0]


DRUDE = gfem.Drude(plasma_frequency=20 * GHZ, damping=1 * GHZ)
LORENTZ = gfem.Lorentz(delta_eps=3.0, resonance_frequency=8 * GHZ, damping=0.4 * GHZ)
DEBYE = gfem.Debye(delta_eps=5.0, relaxation_frequency=5 * GHZ)

# (law, conductivity, frequencies): every term kind, Re eps < 0 (Drude below
# its plasma frequency, Lorentz just above its pole), and exactly 1 GHz,
# which once was the pieces' real construction probe
CASES = {
    "drude": (gfem.PermittivityLaw(terms=[DRUDE]), 0.0, [1 * GHZ, 5 * GHZ, 25 * GHZ]),
    "lorentz": (
        gfem.PermittivityLaw(eps_inf=2.0, terms=[LORENTZ]),
        0.0,
        [1 * GHZ, 7.6 * GHZ, 8.4 * GHZ, 12 * GHZ],
    ),
    "debye": (
        gfem.PermittivityLaw(eps_inf=3.0, terms=[DEBYE]),
        0.0,
        [1 * GHZ, 10 * GHZ],
    ),
    "djordjevic_sarkar": (
        gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1 * GHZ),
        0.0,
        [1 * GHZ, 5 * GHZ],
    ),
    "drude_lorentz": (
        gfem.PermittivityLaw(eps_inf=1.5, terms=[DRUDE, LORENTZ]),
        0.0,
        [1 * GHZ, 8 * GHZ, 25 * GHZ],
    ),
    "debye_conductivity": (
        gfem.PermittivityLaw(eps_inf=3.0, terms=[DEBYE]),
        0.05,
        [1 * GHZ, 5 * GHZ],
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_law_region_solves_as_a_constant_material_at_each_frequency(
    mesh, case, monkeypatch
):
    law, sigma, freqs = CASES[case]
    model = _model(mesh, gfem.Material(eps_r=law, conductivity=sigma, name=case))
    s = np.asarray(
        gfem.solve(model, gfem.Driven(frequency=freqs), options=OPTIONS).s.values
    )
    for n, f in enumerate(freqs):
        eps = complex(law(f))
        ref = _reference(mesh, eps, f, monkeypatch, sigma=sigma)
        assert np.max(np.abs(s[n] - ref)) < 1e-12, (case, f, eps)


def test_the_reference_covers_a_negative_real_permittivity():
    # the cases above reach Re eps < 0 (so the overridden reference)
    assert complex(CASES["drude"][0](5 * GHZ)).real < 0
    assert complex(CASES["lorentz"][0](8.4 * GHZ)).real < 0


def test_a_lossless_pole_at_the_probe_frequency_builds_and_solves(mesh, monkeypatch):
    # a lossless Lorentz pole at exactly 1 GHz: the pieces never evaluate a
    # law on the real axis at construction
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=1 * GHZ, damping=0.0)],
    )
    model = _model(mesh, gfem.Material(eps_r=law, name="sellmeier"))
    s = np.asarray(
        gfem.solve(model, gfem.Driven(frequency=[2 * GHZ]), options=OPTIONS).s.values
    )
    ref = _reference(mesh, complex(law(2 * GHZ)), 2 * GHZ, monkeypatch)
    assert np.max(np.abs(s[0] - ref)) < 1e-12


def test_an_output_frequency_on_a_lossless_pole_raises_before_assembly(
    mesh, monkeypatch
):
    law = gfem.PermittivityLaw(
        eps_inf=2.0,
        terms=[
            DEBYE,
            gfem.Lorentz(delta_eps=1.0, resonance_frequency=8 * GHZ, damping=0.0),
        ],
    )
    model = _model(mesh, gfem.Material(eps_r=law, name="glass"))

    def no_assembly(*_args, **_kwargs):
        raise AssertionError("assembled")

    monkeypatch.setattr(terms, "pieces", no_assembly)
    with pytest.raises(
        ValueError, match=r"8e\+09 Hz.*terms\[1\].*region 'core'"
    ) as err:
        gfem.solve(model, gfem.Driven(frequency=[5 * GHZ, 8 * GHZ]), options=OPTIONS)
    assert "damping" in str(err.value)


def test_an_absorbing_face_beside_a_law_elsewhere_solves(mesh):
    end = (gfem.Absorbing("port2"),)
    law = CASES["debye"][0]
    model = gfem.Model(
        mesh,
        materials={
            "dielectric": gfem.Material(eps_r=EPS_HOST),
            "core": gfem.Material(eps_r=law, name="absorbing end"),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *end],
        ports=[gfem.LumpedPort("p1", "port1", "+z")],
    )
    s = np.asarray(
        gfem.solve(model, gfem.Driven(frequency=[5 * GHZ]), options=OPTIONS).s.values
    )
    eps = complex(law(5 * GHZ))
    core = gfem.Material(eps_r=eps.real, tan_delta=-eps.imag / eps.real, name="ref end")
    ref_model = gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=EPS_HOST), "core": core},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), *end],
        ports=[gfem.LumpedPort("p1", "port1", "+z")],
    )
    ref = np.asarray(
        gfem.solve(
            ref_model, gfem.Driven(frequency=[5 * GHZ]), options=OPTIONS
        ).s.values
    )
    assert np.max(np.abs(s - ref)) < 1e-12


# (law, frequencies) beside an Absorbing side wall: Re eps > 0 at each
# frequency (a Drude metal above its plasma frequency, Lorentz on both sides
# of its pole), and 1 GHz, the real construction probe of other pieces
ABSORBING_CASES = {
    "debye": (CASES["debye"][0], [1 * GHZ, 5 * GHZ, 10 * GHZ]),
    "lorentz": (CASES["lorentz"][0], [4 * GHZ, 6 * GHZ, 14 * GHZ]),
    "drude": (CASES["drude"][0], [25 * GHZ, 30 * GHZ]),
    "djordjevic_sarkar": (CASES["djordjevic_sarkar"][0], [5 * GHZ]),
}


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("case", list(ABSORBING_CASES))
def test_an_absorbing_face_on_a_law_region_solves_as_a_constant_material(
    mesh, case, order, monkeypatch
):
    # the y walls run the line's whole length: faces of the host and of the
    # law region on one surface
    law, freqs = ABSORBING_CASES[case]
    side = (gfem.Absorbing("side", order=order),)
    model = _model(
        mesh, gfem.Material(eps_r=law, name=f"absorbing {case}"), boundaries=side
    )
    s = np.asarray(
        gfem.solve(model, gfem.Driven(frequency=freqs), options=OPTIONS).s.values
    )
    for n, f in enumerate(freqs):
        eps = complex(law(f))
        assert eps.real > 0
        ref = _reference(mesh, eps, f, monkeypatch, boundaries=side)
        assert np.max(np.abs(s[n] - ref)) < 1e-12, (case, order, f, eps)


@pytest.mark.parametrize("order", [1, 2])
def test_an_absorbing_face_on_a_law_region_with_mu_r_solves(mesh, order):
    law = CASES["debye"][0]
    f = 5 * GHZ
    eps = complex(law(f))
    side = (gfem.Absorbing("side", order=order),)
    model = _model(
        mesh, gfem.Material(eps_r=law, mu_r=1.7, name="mu law"), boundaries=side
    )
    s = np.asarray(
        gfem.solve(model, gfem.Driven(frequency=[f]), options=OPTIONS).s.values
    )[0]
    core = gfem.Material(
        eps_r=eps.real, tan_delta=-eps.imag / eps.real, mu_r=1.7, name="mu ref"
    )
    ref_model = _model(mesh, core, boundaries=side)
    ref = np.asarray(
        gfem.solve(ref_model, gfem.Driven(frequency=[f]), options=OPTIONS).s.values
    )[0]
    assert np.max(np.abs(s - ref)) < 1e-12


@pytest.mark.parametrize("order", [1, 2])
def test_a_non_positive_permittivity_behind_an_absorbing_face_raises(
    mesh, order, monkeypatch
):
    law = CASES["drude"][0]
    model = _model(
        mesh,
        gfem.Material(eps_r=law),
        boundaries=(gfem.Absorbing("side", order=order),),
    )

    def no_assembly(*_args, **_kwargs):
        raise AssertionError("assembled")

    monkeypatch.setattr(terms, "pieces", no_assembly)
    value = complex(law(5 * GHZ)).real
    with pytest.raises(ValueError, match=r"'side'.*region 'core'.*Re eps") as err:
        gfem.solve(model, gfem.Driven(frequency=[25 * GHZ, 5 * GHZ]), options=OPTIONS)
    assert "5e+09 Hz" in str(err.value)
    assert f"{value:.3g}" in str(err.value)
    assert "Re eps > 0" in str(err.value)


def test_an_eigenmode_with_an_absorbing_face_on_a_law_region_raises(mesh):
    model = _model(
        mesh,
        gfem.Material(eps_r=CASES["debye"][0]),
        boundaries=(gfem.Absorbing("side"),),
    )
    with pytest.raises(
        NotImplementedError, match="absorbing boundary on a dispersive region"
    ):
        gfem.solve(model, gfem.Eigenmode(target=5 * GHZ), options=OPTIONS)


def test_element_values_refuses_a_law_region(mesh):
    model = _model(mesh, gfem.Material(eps_r=CASES["drude"][0]))
    with pytest.raises(NotImplementedError, match=r"region 'core'.*high_frequency_eps"):
        coefficients.element_values(model, "eps_r")
    with pytest.raises(NotImplementedError, match="region 'core'"):
        coefficients.lossy_eps(model)
    # other properties of the same region read as before
    assert coefficients.element_values(model, "mu_r")[mesh.elements("core")][0] == 1.0


def _debye_objective(mesh, parameter):
    def objective(x):
        values = {"eps_inf": 3.0, "relaxation_frequency": 5 * GHZ, parameter: x}
        term = gfem.Debye(
            delta_eps=5.0, relaxation_frequency=values["relaxation_frequency"]
        )
        law = gfem.PermittivityLaw(eps_inf=values["eps_inf"], terms=[term])
        model = _model(mesh, gfem.Material(eps_r=law))
        s = gfem.solve(model, gfem.Driven(frequency=[4 * GHZ]), options=OPTIONS)
        return jnp.sum(jnp.abs(s.s.values) ** 2)

    return objective


@pytest.mark.parametrize(
    ("parameter", "value", "where"),
    [
        ("eps_inf", 3.0, r"materials\['core'\]\.eps_r\.eps_inf"),
        (
            "relaxation_frequency",
            5 * GHZ,
            r"materials\['core'\]\.eps_r\.terms\[0\]\.relaxation_frequency",
        ),
    ],
)
def test_a_traced_law_parameter_raises_nondifferentiable(mesh, parameter, value, where):
    objective = _debye_objective(mesh, parameter)
    with pytest.raises(gfem.NonDifferentiableError, match=f"{where}.*permittivity law"):
        jax.grad(objective)(value)
    # a traced law solves: jit runs the forward on the concrete values
    assert float(jax.jit(objective)(value)) == pytest.approx(
        float(objective(value)), rel=1e-12
    )


@pytest.mark.parametrize("order", [0, 1, 2])
def test_an_existing_input_beside_a_law_differentiates(mesh, order):
    # order: of an Absorbing side wall over the host and the law region
    # (0: none); the law's faces carry no host derivative
    law = CASES["debye"][0]
    f = 5 * GHZ
    eps = complex(law(f))
    constant = gfem.Material(eps_r=eps.real, tan_delta=-eps.imag / eps.real)
    side = (gfem.Absorbing("side", order=order),) if order else ()

    def gradient(core):
        def objective(e):
            host = gfem.MaterialField(eps_r=jnp.asarray(e), region="dielectric")
            model = _model(mesh, core, host=host, boundaries=side)
            s = gfem.solve(model, gfem.Driven(frequency=[f]), options=OPTIONS).s.values
            return jnp.sum(jnp.abs(s) ** 2 * jnp.array([[1.0, 0.5], [2.0, -1.0]]))

        return float(jax.grad(objective)(EPS_HOST))

    with_law = gradient(gfem.Material(eps_r=law, name=f"grad law {order}"))
    reference = gradient(constant)
    assert with_law != 0.0
    assert abs(with_law - reference) <= 1e-9 * abs(reference)


@pytest.mark.parametrize("case", list(CASES))
def test_the_helper_reads_the_law_and_its_slope_at_omega(mesh, case):
    law, _, freqs = CASES[case]
    model = _model(mesh, gfem.Material(eps_r=law))
    core = mesh.elements("core")
    host = np.setdiff1d(np.arange(len(mesh.tetrahedra)), core)
    for f in [*freqs, 3.3 * GHZ * (1 - 0.2j)]:
        w = 2 * np.pi * f
        eps = dispersion.eps_at(model, w)
        assert np.allclose(eps[core], complex(law(f)), rtol=1e-13, atol=0)
        assert np.all(eps[host] == EPS_HOST)
        # d chi / d omega against JAX's derivative of the law in hertz
        slope = jax.jacfwd(law, holomorphic=True)(jnp.complex128(f))
        assert dispersion.d_susceptibility(law, w) == pytest.approx(
            complex(slope) / (2 * np.pi), rel=1e-10
        )
    hf = dispersion.high_frequency_eps(model)
    assert np.all(hf[core] == law.eps_inf)
    assert np.all(hf[host] == EPS_HOST)
