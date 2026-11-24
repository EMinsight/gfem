"""A permittivity law's fitted range, and the warning of a solve that reads it outside.

`PermittivityLaw.frequency_range` holds the frequencies over which a law's
parameters were matched to data or to a source's formula. Outside it the law
stays passive, so a solve runs on, but its accuracy is unknown: the driven,
eigenmode, port mode and electrostatic solves warn `gfem.FittedRangeWarning`
once per solve. A law without a range behaves as before.
"""

import dataclasses
import json
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _codec

DATA = Path(__file__).parents[2] / "docs" / "tutorials" / "data"
THZ = 1e12
LORENTZ = gfem.Lorentz(delta_eps=1.5, resonance_frequency=300 * THZ, damping=1 * THZ)


def _law(frequency_range=None, **kwargs):
    return gfem.PermittivityLaw(
        eps_inf=2.0, terms=[LORENTZ], frequency_range=frequency_range, **kwargs
    )


# --- the field ---------------------------------------------------------------------


def test_the_range_is_validated():
    assert gfem.PermittivityLaw(terms=[LORENTZ]).frequency_range is None
    law = _law([10 * THZ, np.float32(200e12)])
    assert law.frequency_range == (10e12, float(np.float32(200e12)))
    assert type(law.frequency_range) is tuple
    assert all(type(x) is float for x in law.frequency_range)
    for bad in [(2e12, 1e12), (1e12, 1e12), (0.0, 1e12), (-1e12, 1e12),
                (1e12, np.inf), (np.nan, 1e12)]:
        with pytest.raises(ValueError, match="frequency_range"):
            _law(bad)
    for bad in [np.array([1e12, 2e12]), jnp.array([1e12, 2e12]), 1e12,
                (1e12, 2e12, 3e12), (1e12, 2e12 + 1j), ("a", "b")]:
        with pytest.raises(TypeError, match="frequency_range takes two numbers in Hz"):
            _law(bad)

    @jax.jit
    def traced(lo):
        return _law((lo, 2e12)).eps_inf

    with pytest.raises(TypeError, match="frequency_range takes two numbers in Hz"):
        traced(1e12)


# --- the file ----------------------------------------------------------------------

CONTEXT = _codec.Context(Path())

#: A law with no range as gfem wrote it before the field existed.
TODAY_S_LAW = {
    "$type": "PermittivityLaw",
    "eps_inf": 2.0,
    "terms": [
        {
            "$type": "Lorentz",
            "delta_eps": 1.5,
            "resonance_frequency": 300000000000000.0,
            "damping": 1000000000000.0,
        }
    ],
}


def _encoded(value):
    return json.loads(json.dumps(_codec.Encoder(CONTEXT).encode(value)))


def _decoded(doc):
    return _codec.Decoder(CONTEXT, {}).decode(doc)


def test_a_law_without_a_range_writes_today_s_json():
    assert json.dumps(_encoded(_law())) == json.dumps(TODAY_S_LAW)
    old = _decoded(TODAY_S_LAW)
    assert old.frequency_range is None
    assert old == _law()


def _stack(tmp_path, law):
    path = sm.write_msh22(
        tmp_path / "stack.msh", sm.plate(4.0, 3.0, (("oxide", 0.5), ("core", 0.25)), (1, 1, 1))
    )
    return gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={"oxide": gfem.Material(eps_r=3.9), "core": gfem.Material(eps_r=law)},
    )


def test_a_law_with_a_range_round_trips(tmp_path):
    law = _law((10 * THZ, 200 * THZ))
    doc = _encoded(law)
    assert doc["frequency_range"] == [10e12, 200e12]
    back = _decoded(doc)
    assert back == law
    assert back.frequency_range == (10e12, 200e12)
    assert type(back.frequency_range) is tuple

    job = gfem.Job(_stack(tmp_path, law), gfem.Electrostatic(terminals={"top": "top"}))
    path = job.to_json(tmp_path / "job.json")
    assert json.loads(path.read_text())["schema_version"] == 2
    read = gfem.Job.from_json(path)
    assert read.model.materials["core"].eps_r.frequency_range == (10e12, 200e12)


def test_the_range_is_static_pytree_data():
    law = _law((10 * THZ, 200 * THZ))
    leaves = jax.tree.leaves(law)
    assert 10e12 not in leaves
    assert 200e12 not in leaves
    assert len(leaves) == len(jax.tree.leaves(_law()))
    grad = jax.grad(lambda x: jnp.real(x(100 * THZ)))(law)
    assert grad.frequency_range == law.frequency_range

    traces = []

    @jax.jit
    def value(x):
        traces.append(1)
        return jnp.real(x(100 * THZ))

    value(law)
    value(dataclasses.replace(law, eps_inf=2.5))
    assert len(traces) == 1
    value(dataclasses.replace(law, frequency_range=(1 * THZ, 300 * THZ)))
    assert len(traces) == 2


# --- the solves --------------------------------------------------------------------

GHZ = 1e9
OPTIONS = gfem.SolverOptions(linear_solver="direct")
#: A dielectric law fitted over 2-6 GHz, under the name a library entry gives.
FITTED = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=3.0, resonance_frequency=8 * GHZ, damping=0.4 * GHZ)],
    frequency_range=(2 * GHZ, 6 * GHZ),
)
NAME = "test glass (Doe 2026)"


def _fitted_warnings(caught):
    return [w for w in caught if issubclass(w.category, gfem.FittedRangeWarning)]


def _solve_noting(model, problem, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = gfem.solve(model, problem, options=OPTIONS, **kwargs)
    return result, [str(w.message) for w in _fitted_warnings(caught)]


@pytest.fixture(scope="module")
def line(tmp_path_factory):
    path = tmp_path_factory.mktemp("line") / "line.msh"
    sm.write_msh22(path, sm.strip_line(10.0, 2.0, 1.0, (10, 1, 1), core=(4.0, 6.0)))
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _line(mesh, core):
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=4.0), "core": core},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


@pytest.mark.parametrize("sweep", ["uniform", "adaptive"])
def test_a_driven_sweep_outside_warns_once(line, sweep):
    model = _line(line, gfem.Material(eps_r=FITTED, name=NAME))
    grid = [1 * GHZ, 3 * GHZ, 5 * GHZ, 7 * GHZ, 12 * GHZ]
    _, messages = _solve_noting(model, gfem.Driven(frequency=grid, sweep=sweep))
    assert len(messages) == 1
    (message,) = messages
    assert message.startswith(
        f"region 'core' ({NAME}): its permittivity law was fitted over 2-6 GHz, "
        "and this driven solve reads it at 3 of 5 frequencies, 1-12 GHz. The law "
        "stays passive there, but its accuracy is unknown."
    )
    assert "Narrow the frequencies or fit data that cover them" in message
    assert "dataclasses.replace(law, frequency_range=None)" in message

    inside = gfem.Driven(frequency=[2 * GHZ, 4 * GHZ, 6 * GHZ], sweep=sweep)
    _, messages = _solve_noting(model, inside)
    assert messages == []


def test_a_2d_driven_sweep_outside_warns():
    mesh = gfem.Mesh.from_gmsh(DATA / "line2d.msh", length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"feed": gfem.Material(), "design": gfem.Material(eps_r=FITTED)},
        ports=(gfem.WavePort("p1", "port1"), gfem.WavePort("p2", "port2")),
    )
    _, messages = _solve_noting(model, gfem.Driven(frequency=[1 * GHZ, 4 * GHZ], polarization="Ez"))
    assert messages == [
        "region 'design': its permittivity law was fitted over 2-6 GHz, and this "
        "driven solve reads it at 1 of 2 frequencies, 1 GHz. The law stays passive "
        "there, but its accuracy is unknown. Narrow the frequencies or fit data "
        "that cover them, or clear the range with "
        "dataclasses.replace(law, frequency_range=None) once checked."
    ]


def _box_law(frequency_range):
    # lossless, eps about 2 near 20 GHz: a closed box's lowest modes there
    return gfem.PermittivityLaw(
        eps_inf=1.0,
        terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=200 * GHZ, damping=0.0)],
        frequency_range=frequency_range,
    )


def test_an_eigenmode_target_and_a_mode_outside_warn_once(tmp_path):
    path = sm.write_msh22(
        tmp_path / "box.msh",
        sm.plate(10.0, 8.0, (("lower", 2.5), ("upper", 2.5)), (5, 4, 2)),
    )
    model = gfem.Model(
        gfem.Mesh.from_gmsh(path, length_unit="mm"),
        materials={
            # the target (20 GHz) lies outside the lower law's range
            "lower": gfem.Material(eps_r=_box_law((1 * GHZ, 10 * GHZ))),
            # the target lies inside the upper law's, its two modes outside
            "upper": gfem.Material(eps_r=_box_law((19 * GHZ, 21 * GHZ))),
        },
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.PEC("side")],
    )
    result, messages = _solve_noting(model, gfem.Eigenmode(target=20 * GHZ, n_modes=2))
    found = np.sort(np.real(np.asarray(result.omega))) / (2 * np.pi)
    assert found[0] < 19 * GHZ and found[1] > 21 * GHZ
    target, modes = messages
    assert target.startswith(
        "region 'lower': its permittivity law was fitted over 1-10 GHz, and this "
        "eigenmode target reads it at 20 GHz."
    )
    # the lower region is not repeated
    assert "'lower'" not in modes
    assert modes.startswith(
        "region 'upper': its permittivity law was fitted over 19-21 GHz, and this "
        "eigenmode reads it at 2 of 2 modes, "
    )
