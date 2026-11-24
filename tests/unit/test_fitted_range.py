"""A permittivity law's fitted range, and the warning of a solve that reads it outside.

`PermittivityLaw.frequency_range` holds the frequencies over which a law's
parameters were matched to data or to a source's formula. Outside it the law
stays passive, so a solve runs on, but its accuracy is unknown: the driven,
eigenmode, port mode and electrostatic solves warn `gfem.FittedRangeWarning`
once per solve. A law without a range behaves as before.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm

import gfem
from gfem import _codec

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
