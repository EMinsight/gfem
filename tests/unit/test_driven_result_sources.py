"""A driven result's source columns: labels, port waves, emitted power, saving.

The results here are built by hand, as a solve with sources would return
them; what they hold is checked through their public readers and the codec.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem
from gfem import _codec

F = 5e9
WAVE = gfem.PlaneWave("x", theta=0.0, polarization="TM")
DIPOLE = gfem.ElectricDipole("d", position=(0, 0, 0), moment=(0, 0, 1e-30))
CONTEXT = _codec.Context(Path())


def _s(excited=("p1",)):
    return gfem.NetworkParameters(
        values=jnp.full((1, 2, len(excited)), 0.5 + 0j),
        z_ref=jnp.full((1, 2), 50.0),
        frequency=(F,),
        ports=("p1", "p2"),
        excited=excited,
    )


def _result(excited=("p1",), **kwargs):
    waves = jnp.arange(4.0).reshape(1, 2, 2) * (1 + 1j)
    return gfem.DrivenResult(
        _s(excited),
        sources={"x": WAVE, "d": DIPOLE},
        port_waves=waves,
        emitted_power={"d": jnp.array([2.5e-3])},
        **kwargs,
    )


def _leaves(result):
    return {
        jax.tree_util.keystr(p, simple=True, separator=".")
        for p, _ in jax.tree_util.tree_flatten_with_path(result)[0]
    }


def test_a_port_only_result_has_no_source_leaves():
    result = gfem.DrivenResult(_s())
    assert result.sources == {}
    assert result.port_waves is None
    assert result.emitted_power == {}
    assert _leaves(result) == {"s.values", "s.z_ref"}


def test_source_columns_are_leaves_beside_s():
    result = _result()
    assert _leaves(result) == {
        "s.values",
        "s.z_ref",
        "port_waves",
        "emitted_power.d",
    }
    assert result.sources == {"x": WAVE, "d": DIPOLE}


def test_port_wave_reads_one_port_row_of_one_source_column():
    result = _result()
    np.testing.assert_array_equal(result.port_wave("p2", "x"), [2 + 2j])
    np.testing.assert_array_equal(result.port_wave("p1", "d"), [1 + 1j])
    with pytest.raises(KeyError, match="No port 'p3'; the ports are 'p1', 'p2'"):
        result.port_wave("p3", "x")
    with pytest.raises(KeyError, match="No source 'p1'; the sources are 'x', 'd'"):
        result.port_wave("p2", "p1")


def test_port_wave_without_sources_says_so():
    with pytest.raises(KeyError, match="solved no source"):
        gfem.DrivenResult(_s()).port_wave("p1", "x")


def test_port_waves_must_have_one_column_per_source():
    with pytest.raises(ValueError, match=r"port_waves .* \(1, 2, 2\)"):
        gfem.DrivenResult(
            _s(), sources={"x": WAVE, "d": DIPOLE}, port_waves=jnp.ones((1, 2, 1))
        )


def test_a_result_with_sources_and_no_port_omits_empty_port_waves():
    s = gfem.NetworkParameters(
        values=jnp.zeros((1, 0, 0)),
        z_ref=jnp.zeros((1, 0)),
        frequency=(F,),
        ports=(),
        excited=(),
    )
    result = gfem.DrivenResult(
        s, sources={"d": DIPOLE}, port_waves=jnp.zeros((1, 0, 1))
    )
    assert result.port_waves is None
    assert _leaves(result) == {"s.values", "s.z_ref"}


# --- kept fields by excitation -------------------------------------------------------


def _kept(excited=("p1",)):
    labels = [(F, name) for name in (*excited, "x", "d")]
    fields = [object() for _ in labels]
    return _result(excited, e_fields=fields, kept=labels), fields


def test_a_kept_field_is_labelled_by_its_port_or_source():
    result, fields = _kept()
    assert result.e_field(F, "p1") is fields[0]
    assert result.e_field(F, "x") is fields[1]
    assert result.e_field(F, "d", part="total") is fields[2]


def test_one_excitation_needs_no_name():
    result, fields = _kept(excited=())
    one = gfem.DrivenResult(
        _s(()),
        sources={"d": DIPOLE},
        e_fields=fields[1:],
        kept=[(F, "d")],
    )
    assert one.e_field(F) is fields[1]
    with pytest.raises(ValueError, match="'x', 'd' were solved; pass excite="):
        result.e_field(F)


def test_an_unknown_excitation_lists_ports_and_sources():
    result, _ = _kept()
    with pytest.raises(KeyError, match=r"'p2' was not excited.*'p1', 'x', 'd'"):
        result.e_field(F, "p2")


def test_part_is_total_scattered_or_incident():
    result, _ = _kept()
    with pytest.raises(ValueError, match="part must be 'total', 'scattered' or"):
        result.e_field(F, "x", part="reflected")


@pytest.mark.parametrize("name", ["p1", "d"])
@pytest.mark.parametrize("part", ["scattered", "incident"])
def test_a_port_or_dipole_column_has_only_a_total_field(name, part):
    result, _ = _kept()
    with pytest.raises(ValueError, match=f"{name!r} .* has no incident field"):
        result.e_field(F, name, part=part)


# --- the codec ------------------------------------------------------------------------


def _round_trip(value):
    enc = _codec.Encoder(CONTEXT)
    doc = json.loads(json.dumps(enc.encode(value), allow_nan=False))
    return doc, _codec.Decoder(CONTEXT, enc.arrays).decode(doc)


def test_source_columns_round_trip():
    result = _result()
    _, back = _round_trip(result)
    assert back.sources == result.sources
    np.testing.assert_array_equal(back.port_waves, result.port_waves)
    np.testing.assert_array_equal(back.emitted_power["d"], [2.5e-3])


def test_a_port_only_result_is_written_as_before():
    doc, back = _round_trip(gfem.DrivenResult(_s()))
    assert not {"sources", "port_waves", "emitted_power"} & set(doc)
    assert back.port_waves is None


@pytest.mark.parametrize(
    "value",
    [
        WAVE,
        gfem.PlaneWave("y", theta=2.5, phi=-1.0, polarization=(1, 2j), amplitude=3j),
        gfem.GaussianBeam(
            "b", waist=2e-6, focus=(0, 1e-6, 0), theta=3.0, polarization="LCP"
        ),
        DIPOLE,
        gfem.MagneticDipole("m", position=(1, 2, 3), moment=(1j, 0, 0)),
        gfem.Background(gfem.Material()),
        gfem.Background(
            [gfem.Material(eps_r=2.1), gfem.Material(eps_r=1.0, tan_delta=0.01)],
            interfaces=[0.0],
            ground=-1e-6,
        ),
    ],
)
def test_every_declaration_round_trips(value):
    _, back = _round_trip(value)
    assert back == value
    assert type(back) is type(value)


def test_a_driven_problem_round_trips_its_sources_and_omits_none():
    problem = gfem.Driven(F, excite=(), sources=[WAVE, DIPOLE])
    doc, back = _round_trip(problem)
    assert back == problem
    doc, _ = _round_trip(gfem.Driven(F))
    assert "sources" not in doc


def _field(points, _frequency):
    return np.zeros((len(points), 3), complex)


def test_an_incident_field_cannot_be_written():
    problem = gfem.Driven(F, sources=gfem.IncidentField("mine", _field))
    with pytest.raises(TypeError, match=r"IncidentField 'mine' .*_field.* cannot be"):
        _codec.Encoder(CONTEXT).encode(problem)
    result = gfem.DrivenResult(_s(), sources={"mine": problem.sources[0]})
    with pytest.raises(TypeError, match=r"\.sources\['mine'\].*IncidentField 'mine'"):
        _codec.Encoder(CONTEXT).encode(result)


def test_a_job_with_an_incident_field_cannot_be_written(tmp_path):
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path / "b.msh", sm.plate(1.0, 1.0, (("air", 1.0),), (1, 1, 1))
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path),
        materials={"air": gfem.Material()},
        background=gfem.Material(),
    )
    problem = gfem.Driven(F, sources=gfem.IncidentField("mine", _field))
    with pytest.raises(TypeError, match=r"IncidentField 'mine'.*cannot be written"):
        gfem.Job(model, problem).to_json(tmp_path / "job.json")


def test_a_model_background_round_trips_through_a_job(tmp_path):
    import structured_meshes as sm

    path = sm.write_msh22(
        tmp_path / "b.msh", sm.plate(1.0, 1.0, (("air", 1.0),), (1, 1, 1))
    )
    stack = gfem.Background(
        [gfem.Material(eps_r=2.1), gfem.Material()], interfaces=[0.5]
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path), materials={"air": gfem.Material()}, background=stack
    )
    job = gfem.Job(model, gfem.Driven(F, sources=WAVE))
    back = gfem.Job.from_json(job.to_json(tmp_path / "job.json"))
    assert back.model.background == stack
    assert back.problem == job.problem
    bare = gfem.Job(model.replace(background=None), gfem.Driven(F))
    text = bare.to_json(tmp_path / "bare.json").read_text()
    assert "background" not in text
    assert "sources" not in text


def test_the_parts_of_an_incident_field_come_with_its_solve():
    result, _ = _kept()
    with pytest.raises(NotImplementedError, match="scattered part of the incident"):
        result.e_field(F, "x", part="scattered")
