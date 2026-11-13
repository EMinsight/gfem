"""Permittivity laws: the value of a `Material`'s eps_r that depends on frequency.

Reference values come from the sign probe of the dispersive-materials design
(e^{+iωt}, angular-frequency forms in rad/s) and its printed output.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _codec

TP = 2 * np.pi


# The probe's forms, in rad/s (the classes take Hz).
def _drude(f, fp, g):
    w = TP * f
    return -((TP * fp) ** 2) / (w * (w - 1j * TP * g))


def _lorentz(f, de, f0, g):
    w, w0 = TP * f, TP * f0
    return de * w0**2 / (w0**2 - w**2 + 1j * TP * g * w)


def _debye(f, de, fr):
    return de / (1 + 1j * f / fr)


def _ds(f, de, f1, f2):
    return de / np.log10(f2 / f1) * np.log10((f2 + 1j * f) / (f1 + 1j * f))


GOLD = gfem.Drude(plasma_frequency=2.183e15, damping=1.28e13)
TERMS = [
    (GOLD, lambda f: _drude(f, 2.183e15, 1.28e13)),
    (
        gfem.Lorentz(delta_eps=2.0, resonance_frequency=1e14, damping=1e12),
        lambda f: _lorentz(f, 2.0, 1e14, 1e12),
    ),
    (
        gfem.Debye(delta_eps=70.0, relaxation_frequency=19e9),
        lambda f: _debye(f, 70.0, 19e9),
    ),
    (
        gfem.DjordjevicSarkar(delta_eps=1.0, low_frequency=1e3, high_frequency=1e12),
        lambda f: _ds(f, 1.0, 1e3, 1e12),
    ),
]
FREQUENCIES = np.logspace(0, 16, 2001)


@pytest.mark.parametrize(("term", "reference"), TERMS, ids=lambda t: type(t).__name__)
def test_each_term_is_the_probe_form_and_passive(term, reference):
    value = np.asarray(term(FREQUENCIES))
    expected = reference(FREQUENCIES)
    # relative to |value|: a vanishing real part carries round-off
    assert np.all(np.abs(value - expected) <= 1e-11 * np.abs(expected))
    assert value.imag.max() < 0  # e^{+iωt}: a passive term has Im eps <= 0


def test_gold_drude_at_600_nm_is_the_probe_value():
    law = gfem.PermittivityLaw(eps_inf=1.0, terms=[GOLD])
    np.testing.assert_allclose(
        complex(law(3e8 / 600e-9)),
        -18.049471738201653 - 0.4876664764979623j,
        rtol=1e-12,
    )


def test_a_law_is_eps_inf_plus_its_terms_and_broadcasts():
    terms = [t for t, _ in TERMS[1:]]
    law = gfem.PermittivityLaw(eps_inf=2.5, terms=terms)
    f = np.array([[1e8], [1e10]]) * np.ones((1, 3))
    value = law(f)
    assert value.shape == (2, 3)
    assert value.dtype == jnp.complex128
    np.testing.assert_allclose(value, 2.5 + sum(r(f) for _, r in TERMS[1:]), rtol=1e-12)
    assert law.terms == tuple(terms)  # stored as a tuple
    assert gfem.PermittivityLaw(terms=terms).eps_inf == 1.0


def test_the_static_permittivity_is_the_law_at_zero():
    lorentz, debye, ds = (t for t, _ in TERMS[1:])
    law = gfem.PermittivityLaw(eps_inf=2.0, terms=[lorentz, debye, ds])
    assert complex(law(0.0)) == pytest.approx(2.0 + 2.0 + 70.0 + 1.0, rel=1e-14)


def test_a_drude_term_has_no_static_permittivity():
    with pytest.raises(ValueError, match="Drude term has no static permittivity"):
        GOLD(0.0)
    law = gfem.PermittivityLaw(terms=[GOLD])
    with pytest.raises(ValueError, match="static permittivity"):
        law(np.array([0.0, 1e9]))


def test_a_complex_frequency_is_the_analytic_continuation():
    # Debye: one pole at f = i f_r under e^{+iωt}; Im f > 0 decays.
    debye = gfem.Debye(delta_eps=1.0, relaxation_frequency=1e9)
    assert complex(debye(1j * 1e9 * 0.5)) == pytest.approx(1 / 0.5)
    # Djordjevic-Sarkar: the principal log of the ratio, finite off the cut
    ds = TERMS[3][0]
    f = 1e9 + 2e8j
    assert complex(ds(f)) == pytest.approx(complex(_ds(f, 1.0, 1e3, 1e12)), rel=1e-12)


def test_laws_differentiate_under_jax():
    def eps_imag(fp):
        law = gfem.PermittivityLaw(
            terms=[gfem.Drude(plasma_frequency=fp, damping=1e13)]
        )
        return jnp.imag(law(2e14))

    fp = 2e15
    expected = (eps_imag(fp * (1 + 1e-6)) - eps_imag(fp * (1 - 1e-6))) / (2e-6 * fp)
    assert float(jax.grad(eps_imag)(fp)) == pytest.approx(float(expected), rel=1e-6)


# --- validation --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: gfem.Drude(plasma_frequency=0.0, damping=1.0), "plasma_frequency"),
        (lambda: gfem.Drude(plasma_frequency=1e15, damping=-1e12), "damping"),
        (
            lambda: gfem.Lorentz(delta_eps=1.0, resonance_frequency=-1.0, damping=0.0),
            "resonance_frequency",
        ),
        (
            lambda: gfem.Lorentz(delta_eps=1.0, resonance_frequency=1e9, damping=-1.0),
            "damping",
        ),
        (lambda: gfem.Debye(delta_eps=1.0, relaxation_frequency=0.0), "relaxation"),
        (
            lambda: gfem.DjordjevicSarkar(
                delta_eps=1.0, low_frequency=1e9, high_frequency=1e3
            ),
            "low_frequency",
        ),
        (
            lambda: gfem.DjordjevicSarkar(
                delta_eps=1.0, low_frequency=0.0, high_frequency=1e3
            ),
            "low_frequency",
        ),
        (lambda: gfem.PermittivityLaw(eps_inf=0.0, terms=[GOLD]), "eps_inf"),
        (lambda: gfem.Debye(delta_eps=np.inf, relaxation_frequency=1.0), "finite"),
        (lambda: gfem.Debye(delta_eps=1.0 + 1j, relaxation_frequency=1.0), "real"),
        (lambda: gfem.Debye(delta_eps="1", relaxation_frequency=1.0), "number"),
    ],
)
def test_out_of_range_parameters_are_refused_by_name(make, match):
    with pytest.raises(ValueError, match=match):
        make()


@pytest.mark.parametrize(
    "make",
    [
        lambda d: gfem.Lorentz(delta_eps=d, resonance_frequency=1e9, damping=0.0),
        lambda d: gfem.Debye(delta_eps=d, relaxation_frequency=1e9),
        lambda d: gfem.DjordjevicSarkar(
            delta_eps=d, low_frequency=1e3, high_frequency=1e12
        ),
    ],
)
def test_a_negative_strength_is_refused_as_a_gain(make):
    with pytest.raises(ValueError, match=r"not passive.*delta_eps=-0\.5") as err:
        make(-0.5)
    assert "gain" in str(err.value)
    with pytest.raises(ValueError, match="delta_eps"):
        make(0.0)


def test_terms_are_keyword_only():
    with pytest.raises(TypeError):
        gfem.Drude(2e15, 1e13)  # ty: ignore[missing-argument, too-many-positional-arguments]
    with pytest.raises(TypeError):
        gfem.PermittivityLaw(1.0, [GOLD])  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_a_law_needs_terms_and_only_terms():
    with pytest.raises(ValueError, match=r"eps_r=<number>"):
        gfem.PermittivityLaw(eps_inf=2.0, terms=[])
    with pytest.raises(TypeError, match="Drude, Lorentz, Debye or DjordjevicSarkar"):
        gfem.PermittivityLaw(terms=[GOLD, 3.0])


# --- Djordjevic-Sarkar from a datasheet -----------------------------------------------


def test_djordjevic_sarkar_fits_a_datasheet_value():
    fr4 = gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1e9)
    (term,) = fr4.terms
    assert isinstance(term, gfem.DjordjevicSarkar)
    assert (term.low_frequency, term.high_frequency) == (1e3, 1e12)
    assert term.delta_eps == pytest.approx(1.135307788970002, rel=1e-12)
    assert fr4.eps_inf == pytest.approx(3.9215640429512675, rel=1e-12)
    assert complex(fr4(1e9)) == pytest.approx(4.3 - 0.086j, rel=1e-12)
    tan = [-complex(fr4(f)).imag / complex(fr4(f)).real for f in (1e8, 1e10)]
    assert tan == pytest.approx([0.01944102762503583, 0.020486327492413216], rel=1e-9)


def test_djordjevic_sarkar_refuses_a_fit_with_nonpositive_eps_inf():
    with pytest.raises(ValueError, match="eps_inf"):
        gfem.PermittivityLaw.djordjevic_sarkar(1.1, 0.5, frequency=1e9)
    with pytest.raises(ValueError, match="tan_delta"):
        gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.0, frequency=1e9)


# --- on a Material -------------------------------------------------------------------


GOLD_LAW = gfem.PermittivityLaw(terms=[GOLD])


def test_a_material_takes_a_law_as_eps_r_beside_a_scalar_conductivity():
    m = gfem.Material(eps_r=GOLD_LAW, conductivity=1.0, name="gold")
    assert m.eps_r is GOLD_LAW
    np.testing.assert_allclose(m.eps_r(1e14), GOLD_LAW(1e14))
    assert m.isotropic
    assert "PermittivityLaw" in repr(m)
    hash(m)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"eps_r": GOLD}, TypeError, r"PermittivityLaw\(eps_inf=\.\.\., terms=\["),
        ({"eps_r": GOLD_LAW, "tan_delta": 0.01}, ValueError, "djordjevic_sarkar"),
        (
            {"eps_r": GOLD_LAW, "axes": np.eye(3)},
            ValueError,
            "anisotropic dispersive materials are not supported",
        ),
        (
            {"eps_r": (GOLD_LAW, GOLD_LAW, GOLD_LAW)},
            ValueError,
            "anisotropic dispersive materials are not supported",
        ),
        (
            {"eps_r": GOLD_LAW, "mu_r": (1.0, 2.0, 3.0)},
            ValueError,
            "anisotropic dispersive materials are not supported",
        ),
        (
            {"eps_r": GOLD_LAW, "conductivity": np.diag([1.0, 2.0, 3.0])},
            ValueError,
            "anisotropic dispersive materials are not supported",
        ),
        ({"mu_r": GOLD_LAW}, ValueError, "only eps_r"),
        ({"mu_r": GOLD}, ValueError, "only eps_r"),
    ],
)
def test_unsupported_combinations_raise_with_a_remedy(kwargs, error, match):
    with pytest.raises(error, match=match):
        gfem.Material(**kwargs, name="m")


def test_a_material_field_does_not_hold_a_law():
    with pytest.raises(TypeError, match="PermittivityLaw"):
        gfem.MaterialField(eps_r=GOLD_LAW)
    gold = gfem.Material(eps_r=GOLD_LAW)
    with pytest.raises(TypeError, match=r"jax\.grad reaches"):
        gfem.MaterialField.uniform("metal", gold)


# --- pytrees ----------------------------------------------------------------------


def test_a_law_is_a_pytree_of_its_numbers():
    law = gfem.PermittivityLaw(eps_inf=2.0, terms=[TERMS[1][0], GOLD])
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(law)
    ]
    assert paths == [
        ".eps_inf",
        ".terms[0].delta_eps",
        ".terms[0].resonance_frequency",
        ".terms[0].damping",
        ".terms[1].plasma_frequency",
        ".terms[1].damping",
    ]
    doubled = jax.tree.map(lambda x: 2 * x, law)
    assert doubled.terms[1].plasma_frequency == 2 * GOLD.plasma_frequency


def test_a_material_s_only_children_are_its_law():
    gold = gfem.Material(eps_r=GOLD_LAW, conductivity=3.0, name="gold")
    leaves = jax.tree_util.tree_leaves_with_path(gold)
    assert [jax.tree_util.keystr(p) for p, _ in leaves] == [
        ".eps_r.eps_inf",
        ".eps_r.terms[0].plasma_frequency",
        ".eps_r.terms[0].damping",
    ]
    back = jax.tree.map(lambda x: x, gold)
    assert back == gold
    assert back.conductivity == 3.0
    assert back.name == "gold"


def test_a_constant_material_stays_leafless():
    for m in (gfem.Material(eps_r=3.9), gfem.Material(eps_r=(2.0, 3.0, 4.0))):
        assert jax.tree.leaves(m) == []
        assert jax.tree.map(lambda x: x, m) == m


def test_jax_grad_reaches_the_law_through_a_material():
    def eps_real(material):
        return jnp.real(material.eps_r(2e14))

    gold = gfem.Material(eps_r=GOLD_LAW)
    g = jax.grad(eps_real)(gold)
    assert isinstance(g, gfem.Material)
    assert float(g.eps_r.eps_inf) == pytest.approx(1.0)
    assert float(g.eps_r.terms[0].plasma_frequency) < 0  # more metal, more negative


def test_a_law_built_under_a_transformation_skips_the_construction_check():
    def f(damping):
        law = gfem.PermittivityLaw(
            terms=[gfem.Drude(plasma_frequency=2e15, damping=damping)]
        )
        m = gfem.Material(eps_r=law)
        return jnp.imag(m.eps_r(1e14))

    assert np.isfinite(float(jax.grad(f)(1e13)))


# --- serialization ------------------------------------------------------------------


def test_every_law_tag_round_trips_without_a_schema_bump():
    law = gfem.PermittivityLaw(eps_inf=1.5, terms=[t for t, _ in TERMS])
    material = gfem.Material(eps_r=law, conductivity=0.5, name="mix")
    enc = _codec.Encoder(_codec.Context(Path()))
    doc = json.loads(json.dumps(enc.encode(material), allow_nan=False))
    assert doc["eps_r"]["$type"] == "PermittivityLaw"
    assert [t["$type"] for t in doc["eps_r"]["terms"]] == [
        "Drude",
        "Lorentz",
        "Debye",
        "DjordjevicSarkar",
    ]
    assert enc.arrays == {}
    back = _codec.Decoder(_codec.Context(Path()), enc.arrays).decode(doc)
    assert back == material
    assert _codec.SCHEMA_VERSION == 2


def test_a_job_with_a_law_round_trips(tmp_path):
    path = sm.write_msh22(
        tmp_path / "stack.msh", sm.plate(4.0, 3.0, (("a", 0.5), ("b", 0.25)), (1, 1, 1))
    )
    fr4 = gfem.Material(
        eps_r=gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1e9),
        name="FR4",
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"a": fr4, "b": gfem.Material()}
    )
    job = gfem.Job(model, gfem.Driven(frequency=[1e9]))
    back = gfem.Job.from_json(job.to_json(tmp_path / "job.json"))
    assert back.model.materials["a"] == fr4


def test_the_conventions_document_the_dispersive_forms():
    doc = gfem.conventions.__doc__
    assert "Dispersive permittivity" in doc
    for name in ("Drude", "Lorentz", "Debye", "DjordjevicSarkar", "hertz"):
        assert name in doc
    for cls in (gfem.PermittivityLaw, gfem.Drude, gfem.Lorentz, gfem.Debye):
        assert "gfem.conventions" in cls.__doc__
