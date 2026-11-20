"""`gfem.thermo_optic_shift` and `gfem.Model.with_temperature_rise`.

A silicon core in oxide under a heater on a small cross-section slab: the
heater is excluded from electromagnetic solves, so the port's section is
the oxide and the core. The shift is first order in the rise from one mode
solve, so it is exactly linear in the rise and in dn/dT; the
re-solve checks live in tests/verification/test_thermo_optic.py.
"""

import jax
import numpy as np
import pytest
import scipy.constants
import thermal_cases as tc

import gfem

WAVELENGTH = 1.55e-6
FREQUENCY = scipy.constants.c / WAVELENGTH
N_SI, N_OX = 3.4777, 1.444
DN_SI, DN_OX = 1.86e-4, 1.0e-5
#: A section: oxide cladding, a 500 x 220 nm silicon core, a heater above.
RECTS = [
    (-1.75, -1.0, 1.75, 1.8, "clad", None),
    (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
    (-0.5, 1.5, 0.5, 1.65, "heater", 0.1),
]
BOTTOM = [(lambda p: abs(p[1] + 1.0) < 1e-9, "bottom")]
#: A permittivity law (lossless here): the shift refuses any law.
LAW = gfem.PermittivityLaw(
    eps_inf=N_OX**2,
    terms=[gfem.Lorentz(delta_eps=0.1, resonance_frequency=1e12, damping=0.0)],
)


def _materials(dn_core=DN_SI, dn_clad=DN_OX, **clad):
    return {
        "clad": gfem.Material(
            eps_r=N_OX**2,
            thermal_conductivity=1.38,
            thermo_optic_coefficient=dn_clad,
            **clad,
        ),
        "core": gfem.Material(
            eps_r=N_SI**2, thermal_conductivity=90.0, thermo_optic_coefficient=dn_core
        ),
        "heater": gfem.Material(thermal_conductivity=28.0),
    }


def _model(mesh, materials):
    return gfem.Model(
        mesh,
        materials=materials,
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
        ports=[gfem.WavePort("p", "front")],
        excluded={"em": ["heater"]},
    )


def _heat(power_per_length):
    return gfem.HeatSource.uniform("heater", power_per_length=power_per_length)


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    path = tmp_path_factory.mktemp("thermo_optic") / "section.msh"
    return tc.section(path, RECTS, 0.3, BOTTOM)


@pytest.fixture(scope="module")
def model(mesh):
    return _model(mesh, _materials())


@pytest.fixture(scope="module")
def steady(model):
    return gfem.solve(model, gfem.SteadyThermal(_heat(1.0)))


@pytest.fixture(scope="module")
def shift(model, steady):
    return gfem.thermo_optic_shift(model, steady.rise, port="p", frequency=FREQUENCY)


def test_the_shift_is_linear_in_the_rise(model, shift):
    doubled = gfem.solve(model, gfem.SteadyThermal(_heat(2.0)))
    twice = gfem.thermo_optic_shift(model, doubled.rise, port="p", frequency=FREQUENCY)
    np.testing.assert_allclose(twice.delta_n_eff, 2 * shift.delta_n_eff, rtol=1e-12)
    np.testing.assert_allclose(
        twice.phase_per_length, 2 * shift.phase_per_length, rtol=1e-12
    )
    np.testing.assert_allclose(twice.n_eff, shift.n_eff, rtol=1e-12)


def test_the_shift_is_linear_in_the_coefficient(mesh, steady, shift):
    def delta(dn_core, dn_clad):
        model = _model(mesh, _materials(dn_core, dn_clad))
        result = gfem.thermo_optic_shift(
            model, steady.rise, port="p", frequency=FREQUENCY
        )
        return float(result.delta_n_eff)

    core, clad = delta(DN_SI, None), delta(None, DN_OX)
    np.testing.assert_allclose(core + clad, float(shift.delta_n_eff), rtol=1e-12)
    np.testing.assert_allclose(delta(3 * DN_SI, None), 3 * core, rtol=1e-12)
    # the core carries almost all of it (0.988 on the femwell heater)
    assert 0.9 < core / float(shift.delta_n_eff) < 1.0


def test_a_positive_coefficient_gives_a_positive_phase(mesh, steady, shift):
    assert float(shift.delta_n_eff) > 0
    assert float(shift.phase_per_length) > 0
    k0 = 2 * np.pi / WAVELENGTH
    np.testing.assert_allclose(
        shift.phase_per_length, k0 * shift.delta_n_eff, rtol=1e-14
    )
    # the mode is the core's: between the oxide and silicon indices
    assert N_OX < float(shift.n_eff) < N_SI
    negative = _model(mesh, _materials(-DN_SI, -DN_OX))
    cooled = gfem.thermo_optic_shift(
        negative, steady.rise, port="p", frequency=FREQUENCY
    )
    assert float(cooled.phase_per_length) < 0


def test_p_pi_is_half_a_wavelength_over_the_slope(steady, shift):
    # pi P' / phase_per_length, the slope s = delta_n_eff per W/m
    p_pi = np.pi * steady.power_per_length / shift.phase_per_length
    slope = shift.delta_n_eff / steady.power_per_length
    np.testing.assert_allclose(p_pi, WAVELENGTH / (2 * slope), rtol=1e-12)


def test_without_a_rise_the_shift_gives_the_mode_and_the_trace(model, steady, shift):
    bare = gfem.thermo_optic_shift(model, None, port="p", frequency=FREQUENCY)
    assert bare.delta_n_eff is None
    assert bare.phase_per_length is None
    np.testing.assert_allclose(bare.n_eff, shift.n_eff, rtol=1e-12)
    trace = bare.trace()
    assert isinstance(trace, gfem.thermal.Trace)
    assert trace.unit == "rad/m"
    assert "'p'" in repr(trace)
    # the trace's value on a rise is the shift's phase per length
    value = trace._sample(None, steady.rise)
    np.testing.assert_allclose(value, shift.phase_per_length, rtol=1e-12)


def test_a_section_without_a_coefficient_raises_naming_the_port(mesh, steady):
    model = _model(mesh, _materials(None, None))
    with pytest.raises(ValueError, match=r"wave port 'p'.*thermo_optic_coefficient"):
        gfem.thermo_optic_shift(model, steady.rise, port="p", frequency=FREQUENCY)


@pytest.mark.parametrize(
    ("loss", "match"),
    [
        ({"tan_delta": 1e-3}, "tan_delta"),
        ({"conductivity": 1e-3}, "conductivity"),
        ({"eps_r": N_OX**2 * (1 - 1e-4j)}, "complex eps_r"),
        ({"eps_r": LAW}, "permittivity law"),
    ],
)
def test_a_lossy_section_raises(mesh, steady, loss, match):
    # a field holds a complex eps_r; a Material any other loss
    kind = gfem.MaterialField if np.iscomplexobj(loss.get("eps_r")) else gfem.Material
    clad = kind(
        **{"eps_r": N_OX**2, **loss},
        thermal_conductivity=1.38,
        thermo_optic_coefficient=DN_OX,
    )
    model = _model(mesh, {**_materials(), "clad": clad})
    with pytest.raises(NotImplementedError, match=rf"'clad'.*{match}.*lossless"):
        gfem.thermo_optic_shift(model, steady.rise, port="p", frequency=FREQUENCY)


def test_a_sequence_of_frequencies_raises(model, steady):
    with pytest.raises(ValueError, match=r"one frequency.*loop"):
        gfem.thermo_optic_shift(
            model, steady.rise, port="p", frequency=[FREQUENCY, 1.01 * FREQUENCY]
        )


def test_an_unknown_port_raises(model, steady):
    with pytest.raises(ValueError, match="'q'"):
        gfem.thermo_optic_shift(model, steady.rise, port="q", frequency=FREQUENCY)


def test_a_rise_that_is_no_field_raises(model):
    with pytest.raises(TypeError, match=r"scalar gfem\.Field"):
        gfem.thermo_optic_shift(model, 1.0, port="p", frequency=FREQUENCY)


def test_a_section_outside_the_rise_raises_naming_the_point(tmp_path, model):
    # a thermal model on a smaller window: the section reaches past it
    rects = [
        (-1.0, -0.5, 1.0, 0.8, "clad", None),
        (-0.25, 0.0, 0.25, 0.22, "core", 0.05),
    ]
    bottom = [(lambda p: abs(p[1] + 0.5) < 1e-9, "bottom")]
    window = tc.section(tmp_path / "window.msh", rects, 0.3, bottom)
    thermal = gfem.Model(
        window,
        materials={k: v for k, v in _materials().items() if k != "heater"},
        boundaries=[gfem.FixedTemperature("bottom", temperature=300.0)],
    )
    rise = gfem.solve(
        thermal, gfem.SteadyThermal(gfem.HeatSource.uniform("core", power_density=1e12))
    ).rise
    with pytest.raises(ValueError, match=r"wave port 'p'.*Point \(.*outside the mesh"):
        gfem.thermo_optic_shift(model, rise, port="p", frequency=FREQUENCY)


def test_the_coefficient_differentiates_and_the_optical_eps_raises(mesh, steady, shift):
    def phase(dn, eps=N_SI**2):
        core = gfem.MaterialField(
            eps_r=eps, thermal_conductivity=90.0, thermo_optic_coefficient=dn
        )
        model = _model(mesh, {**_materials(), "core": core})
        result = gfem.thermo_optic_shift(
            model, steady.rise, port="p", frequency=FREQUENCY
        )
        return result.phase_per_length

    clad_only = float(phase(0.0))
    # linear in the core's dn/dT: the derivative is the core's share per unit
    grad = float(jax.grad(phase)(DN_SI))
    expected = (float(shift.phase_per_length) - clad_only) / DN_SI
    np.testing.assert_allclose(grad, expected, rtol=1e-9)
    with pytest.raises(gfem.NonDifferentiableError, match="thermo-optic shift"):
        jax.grad(lambda eps: phase(DN_SI, eps))(N_SI**2)


def test_with_temperature_rise_sets_the_heated_index(model, steady):
    hot = model.with_temperature_rise(steady.rise)
    for region, n0, dn in (("core", N_SI, DN_SI), ("clad", N_OX, DN_OX)):
        rise = np.asarray(gfem.element_mean(steady.rise, region))
        material = hot.materials[region]
        assert isinstance(material, gfem.MaterialField)
        np.testing.assert_allclose(
            np.asarray(material.eps_r), (n0 + dn * rise) ** 2, rtol=1e-14
        )
        assert float(material.thermal_conductivity) == float(
            model.materials[region].thermal_conductivity
        )
    # a region without a coefficient keeps its material
    assert hot.materials["heater"] is model.materials["heater"]
