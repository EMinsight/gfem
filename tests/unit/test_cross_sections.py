"""Cross sections and the corrected power flow of incident-field columns.

A unit cube of air ("air", um) in a thin perfectly matched layer, with a
one-cell cube scatterer ("scat") off its centre, on a coarse structured
mesh at order 2. The checks are identities of the discrete power balance,
which hold on any mesh, and the measurements' refusals.
"""

import numpy as np
import pml_cases as pc
import pytest

import gfem

ALL = ("-x", "+x", "-y", "+y", "-z", "+z")
AIR = gfem.Material()
LOSSY = gfem.Material(eps_r=4.0, tan_delta=0.25)
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
F = pc.F0
TE = gfem.PlaneWave("te", theta=0.4, phi=0.3, polarization="TE")
TM = gfem.PlaneWave("tm", theta=0.4, phi=0.3, polarization="TM", amplitude=2.0)
ETA0 = 376.730313412


def _inside(c, lo, hi):
    return bool(((c > lo) & (c < hi)).all())


def _scatterer(c):
    if _inside(c, 0.25, 0.5):
        return "scat"
    if _inside(c, 0.25, 0.75):
        return "box"
    return "air" if _inside(c, 0, 1) else "pml"


def _touching(c):
    """A scatterer of the cube's +x quarter, in the truncation layer."""
    if c[0] > 0.75 and _inside(c, 0, 1):
        return "scat"
    return "air" if _inside(c, 0, 1) else "pml"


def _air(c):
    return "air" if _inside(c, 0, 1) else "pml"


def _feed(c):
    """The scatterer's +x face, a lumped port's surface."""
    on = abs(c[0] - 0.5) < 1e-9 and 0.25 < c[1] < 0.5 and 0.25 < c[2] < 0.5
    return "feed" if on else ""


def _box(path, volume):
    return gfem.Mesh.from_gmsh(
        pc.box(path, ALL, thickness=0.25, cells=4, volume=volume)
    )


@pytest.fixture(scope="module")
def meshes(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("xsec")
    return {
        "scatterer": _box(tmp / "s.msh", _scatterer),
        "touching": _box(tmp / "t.msh", _touching),
        "empty": _box(tmp / "e.msh", _air),
        "fed": gfem.Mesh.from_gmsh(
            pc.box(
                tmp / "f.msh",
                ALL,
                thickness=0.25,
                cells=4,
                volume=_scatterer,
                surface=_feed,
            )
        ),
    }


def _model(mesh, scat=None, **kw):
    materials = {"air": AIR, "box": AIR, "pml": AIR}
    if scat is not None:
        materials["scat"] = scat
    materials = {k: v for k, v in materials.items() if k in mesh.volumes}
    return gfem.Model(
        mesh, materials=materials, pml=gfem.PML("pml"), background=AIR, **kw
    )


def _solve(model, *sources, **kw):
    problem = gfem.Driven(F, sources=sources, keep_fields=True, **kw)
    return gfem.solve(model, problem, options=OPTIONS)


@pytest.fixture(scope="module")
def lossy(meshes):
    model = _model(meshes["scatterer"], LOSSY)
    return model, _solve(model, TE, TM)


def _value(x):
    return float(np.asarray(x).reshape(()))


# --- identities -----------------------------------------------------------------


def test_extinction_is_absorption_plus_scattering_exactly(lossy):
    model, result = lossy
    for name in ("te", "tm"):
        xs = gfem.cross_sections(result, name, model=model)
        np.testing.assert_array_equal(xs.extinction, xs.absorption + xs.scattering)
        assert (xs.absorption > 0).all()
        assert (xs.scattering > 0).all()


def test_plane_wave_intensity_is_its_peak_amplitude_over_two_eta(lossy):
    model, result = lossy
    np.testing.assert_allclose(
        gfem.cross_sections(result, "tm", model=model).intensity,
        [4.0 / (2 * ETA0)],
        rtol=1e-9,
    )


def test_corrected_power_flow_is_minus_the_absorbed_power(lossy):
    model, result = lossy
    for name in ("te", "tm"):
        xs = gfem.cross_sections(result, name, model=model)
        total = result.e_field(F, name)
        absorbed = xs.absorption[0] * xs.intensity[0]
        for side in ("scat", "box", ("scat", "box")):
            power = _value(gfem.power_flow(total, side, model=model))
            assert power == pytest.approx(-absorbed, rel=1e-10), side


def test_scattering_is_the_scattered_flux_through_any_side(lossy):
    model, result = lossy
    xs = gfem.cross_sections(result, "te", model=model)
    scattered = result.e_field(F, "te", part="scattered")
    for side in ("scat", "box"):
        power = _value(gfem.power_flow(scattered, side, model=model))
        assert power == pytest.approx(xs.scattering[0] * xs.intensity[0], rel=1e-10)


def test_a_lossless_scatterer_absorbs_nothing(meshes):
    model = _model(meshes["scatterer"], gfem.Material(eps_r=4.0))
    xs = gfem.cross_sections(_solve(model, TE), "te", model=model)
    assert abs(xs.absorption[0]) <= 1e-10 * xs.scattering[0]


def test_without_a_scatterer_every_cross_section_vanishes(meshes, lossy):
    reference = gfem.cross_sections(lossy[1], "te", model=lossy[0])
    model = _model(meshes["empty"])
    xs = gfem.cross_sections(_solve(model, TE), "te", model=model)
    size = reference.extinction[0]
    for value in (xs.extinction, xs.absorption, xs.scattering):
        assert abs(value[0]) <= 1e-12 * size


def _windowed(points, frequency):
    """``TE`` cut off before x = 0.7 um: zero where the slab meets the layer."""
    k = 2 * np.pi * frequency / pc.C0
    x = points[:, 0] * 1e6
    window = np.where(x < 0.7, np.cos(np.clip(x - 0.6, 0, 0.1) * 5 * np.pi) ** 2, 0)
    return (np.exp(-1j * k * points[:, 2]) * window)[:, None] * np.array([0, 1, 0])


def test_an_incident_field_is_normalized_by_its_given_intensity(lossy):
    model, _ = lossy
    wave = gfem.IncidentField("f", _plane_te)
    other = _solve(model, wave)
    i0 = 1.0 / (2 * ETA0)
    xs = gfem.cross_sections(other, "f", model=model, intensity=i0)
    ref = gfem.cross_sections(_solve(model, TE_Z), "z", model=model)
    np.testing.assert_allclose(xs.extinction, ref.extinction, rtol=1e-10)
    np.testing.assert_allclose(xs.intensity, [i0])


BEAM = gfem.GaussianBeam(
    "beam", waist=0.4e-6, focus=(0.4e-6, 0.4e-6, 0.4e-6), theta=0.4, polarization="TE"
)


def test_a_beam_is_normalized_by_its_focal_intensity_at_each_frequency(meshes):
    model = _model(meshes["scatterer"], LOSSY)
    frequencies = [F, 1.2 * F]
    problem = gfem.Driven(frequencies, sources=BEAM, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    xs = gfem.cross_sections(result, "beam", model=model)
    i0 = [BEAM.focal_intensity(f, model=model) for f in frequencies]
    np.testing.assert_allclose(xs.intensity, i0, rtol=1e-14)
    assert i0[0] != pytest.approx(i0[1], rel=1e-3)
    for k, f in enumerate(frequencies):
        absorbed = -_value(
            gfem.power_flow(result.e_field(f, "beam"), "scat", model=model)
        )
        assert xs.absorption[k] * i0[k] == pytest.approx(absorbed, rel=1e-10)
    with pytest.raises(ValueError, match=r"intensity= for an IncidentField only"):
        gfem.cross_sections(result, "beam", model=model, intensity=1.0)


TE_Z = gfem.PlaneWave("z", theta=0.0, phi=0.0, polarization="TE")


def _plane_te(points, frequency):
    """``TE_Z`` as a function: along +z, polarized along +y."""
    k = 2 * np.pi * frequency / pc.C0
    return np.exp(-1j * k * points[:, 2])[:, None] * np.array([0, 1, 0])


# --- refusals ---------------------------------------------------------------------


def test_an_incident_field_needs_its_intensity(lossy):
    model, _ = lossy
    result = _solve(model, gfem.IncidentField("f", _plane_te))
    with pytest.raises(ValueError, match=r"IncidentField 'f' needs intensity="):
        gfem.cross_sections(result, "f", model=model)
    with pytest.raises(ValueError, match=r"positive finite number"):
        gfem.cross_sections(result, "f", model=model, intensity=-1.0)


def test_a_plane_wave_takes_no_intensity(lossy):
    model, result = lossy
    with pytest.raises(ValueError, match=r"intensity= for an IncidentField only"):
        gfem.cross_sections(result, "te", model=model, intensity=1.0)


def test_a_port_column_has_no_cross_sections(meshes):
    model = _model(
        meshes["fed"],
        LOSSY,
        ports=[gfem.LumpedPort("p1", "feed", "+z", resistance=50.0)],
    )
    result = _solve(model, TE)
    with pytest.raises(ValueError, match=r"'p1' is a port column"):
        gfem.cross_sections(result, "p1", model=model)
    with pytest.raises(KeyError, match=r"'nope' is not a column"):
        gfem.cross_sections(result, "nope", model=model)


def test_cross_sections_need_kept_fields(lossy):
    model, _ = lossy
    result = gfem.solve(model, gfem.Driven(F, sources=TE), options=OPTIONS)
    with pytest.raises(ValueError, match=r"needs the kept fields of 'te'"):
        gfem.cross_sections(result, "te", model=model)


def test_a_scatterer_in_the_truncation_layer_is_refused(meshes):
    model = _model(meshes["touching"], LOSSY)
    result = _solve(model, gfem.IncidentField("f", _windowed))
    with pytest.raises(ValueError, match=r"scatterer region 'scat' reaches the trunc"):
        gfem.cross_sections(result, "f", model=model, intensity=1.0)


def test_a_lossy_background_is_not_measured(meshes):
    lossy_air = gfem.Material(tan_delta=0.01)
    model = gfem.Model(
        meshes["scatterer"],
        materials={"air": lossy_air, "box": lossy_air, "pml": lossy_air, "scat": LOSSY},
        pml=gfem.PML("pml"),
        background=lossy_air,
    )
    result = _solve(model, gfem.IncidentField("f", _plane_te))
    with pytest.raises(NotImplementedError, match=r"lossy background"):
        gfem.cross_sections(result, "f", model=model, intensity=1.0)


def test_power_flow_refuses_the_incident_part(lossy):
    model, result = lossy
    incident = result.e_field(F, "te", part="incident")
    with pytest.raises(ValueError, match=r"incident part .* balances no load"):
        gfem.power_flow(incident, "scat", model=model)


def test_power_flow_refuses_a_side_in_the_layer_of_an_incident_column(lossy):
    model, result = lossy
    total = result.e_field(F, "te")
    with pytest.raises(ValueError, match=r"PML layer region 'pml'"):
        gfem.power_flow(total, ("scat", "pml"), model=model)
    with pytest.raises(ValueError, match=r"PML layer region 'pml'"):
        gfem.power_flow(total, lambda c: c[:, 0] > 0.9e-6, model=model)


def test_the_dissipated_power_of_an_incident_column_is_its_absorbed_power(lossy):
    model, result = lossy
    for name in ("te", "tm"):
        xs = gfem.cross_sections(result, name, model=model)
        lost = gfem.dissipated_power(result.e_field(F, name), model=model)
        total = sum(_value(v) for v in lost.values())
        assert total == pytest.approx(xs.absorption[0] * xs.intensity[0], rel=1e-8)


def test_electromagnetic_heating_refuses_a_result_with_no_excited_port(lossy):
    model, result = lossy
    with pytest.raises(ValueError, match=r"no excited port.*dissipated_power"):
        gfem.electromagnetic_heating(
            result, model=model, frequency=F, incident_power=1.0
        )


def _port_walls(c):
    """A wave port patch on the x = 0 wall, the x walls, the y walls."""
    if abs(c[0]) < 1e-9 and 0.25 < c[1] < 0.75 and 0.25 < c[2] < 0.75:
        return "feed"
    if abs(c[0]) < 1e-9 or abs(c[0] - 1) < 1e-9:
        return "xwall"
    return ""


def _away_from_the_port(points, frequency):
    """Along +z, E along x, zero up to x = 0.4 um: none on the port face."""
    k = 2 * np.pi * frequency / pc.C0
    x = points[:, 0] * 1e6
    window = np.where(x > 0.3, np.sin(np.clip(x - 0.3, 0, 0.1) * 5 * np.pi) ** 2, 0)
    return (np.exp(-1j * k * points[:, 2]) * window)[:, None] * np.array([1, 0, 0])


def test_a_side_reaching_a_wave_port_is_refused(tmp_path):
    # the load's rows on a wave port's face hold the port's terms
    path = pc.box(tmp_path / "p.msh", ("-z", "+z"), cells=4, surface=_port_walls)
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": AIR, "pml": AIR},
        boundaries=[gfem.PEC("xwall")],
        ports=[gfem.WavePort("o1", "feed")],
        pml=gfem.PML("pml"),
        background=AIR,
    )
    wave = gfem.IncidentField("f", _away_from_the_port)
    result = _solve(model, wave, excite=())
    total = result.e_field(F, "f")
    near_the_port = lambda c: (c[:, 0] < 0.5e-6) & (c[:, 2] > 0) & (c[:, 2] < 1e-6)  # noqa: E731
    with pytest.raises(ValueError, match=r"reaches wave port 'o1''s surface 'feed'"):
        gfem.power_flow(total, near_the_port, model=model)
    with pytest.raises(ValueError, match=r"reaches wave port 'o1''s surface 'feed'"):
        gfem.cross_sections(result, "f", model=model, intensity=1.0)
