"""Sweep outputs at their interface: rows, join, the seam's names and specs.

Small arrays and duck-typed wave port couplings; no mesh, no solve. The
eigenmode and electrostatic rules have no such encoding: their forward
outputs are checked against their own ``output_specs`` at the end, on the
smallest meshes.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import structured_meshes as sm
from eigen_cases import F1, line, line_mesh

import gfem
from gfem._eigen.rule import EigenmodeRule
from gfem._engine.electrostatic import ElectrostaticRule
from gfem._engine.sweep_outputs import DIFFERENTIABLE, SweepOutputs

N_FREQ, N_PORTS, N_EXC, NDOF, M = 4, 3, 2, 7, 5
HOST_ONLY = {"iterations", "reused", "port_sweeps", "sweep"}


def _complex(rng, *shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _wave(rng, name):
    """A wave port coupling with only its port mode sweep over the grid."""
    space = SimpleNamespace(section=SimpleNamespace(port=SimpleNamespace(name=name)))
    sweep = SimpleNamespace(
        space=space,
        basis=rng.standard_normal((NDOF, M)),
        coords=_complex(rng, N_FREQ, M),
        gamma=_complex(rng, N_FREQ),
        z_pv=rng.random(N_FREQ),
        residual=rng.random(N_FREQ),
        solved=rng.random(N_FREQ) > 0.5,
    )
    return SimpleNamespace(sweep=sweep)


def _uniform(rng, rows, *, waves=(), fields=True):
    return SweepOutputs.uniform(
        rows,
        s=_complex(rng, N_FREQ, N_PORTS, N_EXC),
        z_ref=rng.random((N_FREQ, N_PORTS)),
        iterations=[(3, 4)] * len(rows),
        reused=[False] + [True] * (len(rows) - 1),
        e_field=_complex(rng, NDOF, N_EXC) if fields else None,
        waves=list(waves),
    )


def _adaptive(rng, *, waves=(), fields=True):
    n = 6
    reduced = (rng.standard_normal((NDOF, n)), _complex(rng, N_EXC, n))
    return SweepOutputs.adaptive(
        s=_complex(rng, N_FREQ, N_PORTS, N_EXC),
        z_ref=rng.random((N_FREQ, N_PORTS)),
        iterations=[(5, 5), (6, 7)],
        reused=[False, False],
        sweep={"samples": [0, 3], "errors": [1.0], "converged": True, "basis_size": 4},
        reduced=reduced if fields else None,
        waves=list(waves),
    )


def test_uniform_trims_every_array_and_the_port_mode_coordinates_together():
    rng = np.random.default_rng(0)
    waves = [_wave(rng, "w1"), _wave(rng, "w2")]
    s = _complex(rng, N_FREQ, N_PORTS, N_EXC)
    z_ref = rng.random((N_FREQ, N_PORTS))
    out = SweepOutputs.uniform(
        (0, 1),
        s=s,
        z_ref=z_ref,
        iterations=[(1,), (2,)],
        reused=[False, True],
        e_field=None,
        waves=waves,
    )
    assert out.rows == (0, 1)
    np.testing.assert_array_equal(out.s, s[:2])
    np.testing.assert_array_equal(out.z_ref, z_ref[:2])
    ports = out.ports
    assert ports is not None
    for name in ("gamma", "z_pv", "residual", "solved"):
        expected = np.stack([getattr(w.sweep, name) for w in waves], axis=1)[:2]
        np.testing.assert_array_equal(getattr(ports, name), expected)
    for k, wave in enumerate(waves):
        field = ports.field(k)
        assert field is not None
        space, basis, coords = field._parts
        assert space is wave.sweep.space
        assert basis is wave.sweep.basis
        np.testing.assert_array_equal(coords, wave.sweep.coords[[0, 1]])


def test_every_row_leaves_the_port_mode_coordinates_unsliced():
    rng = np.random.default_rng(1)
    waves = [_wave(rng, "w1")]
    out = _uniform(rng, tuple(range(N_FREQ)), waves=waves)
    field = out.ports.field(0)
    assert field._parts[2] is waves[0].sweep.coords


def test_no_wave_port_has_no_port_modes():
    out = _uniform(np.random.default_rng(2), (0, 1, 2))
    assert out.ports is None
    assert not any(name.startswith("port_") for name in out.outputs())


def test_join_maps_each_blocks_rows_to_grid_indices():
    rng = np.random.default_rng(3)
    first = SweepOutputs.uniform(
        (0, 1),
        s=_complex(rng, 3, N_PORTS, N_EXC),
        z_ref=rng.random((3, N_PORTS)),
        iterations=[(1,), (2,)],
        reused=[False, True],
        e_field=_complex(rng, NDOF, N_EXC),
        waves=[],
    )
    second = SweepOutputs.uniform(
        (0, 1, 2),
        s=_complex(rng, 3, N_PORTS, N_EXC),
        z_ref=rng.random((3, N_PORTS)),
        iterations=[(3,), (4,), (5,)],
        reused=[False, True, True],
        e_field=_complex(rng, NDOF, 2 * N_EXC),
        waves=[],
    )
    out = SweepOutputs.join([((0, 1, 2), first), ((3, 4, 5), second)])
    assert out.rows == (0, 1, 3, 4, 5)
    np.testing.assert_array_equal(out.s, np.concatenate([first.s, second.s]))
    np.testing.assert_array_equal(
        out.z_ref, np.concatenate([first.z_ref, second.z_ref])
    )
    assert out.iterations == [(1,), (2,), (3,), (4,), (5,)]
    assert out.reused == [False, True, False, True, True]
    np.testing.assert_array_equal(
        out.e_field, np.concatenate([first.e_field, second.e_field], axis=1)
    )
    assert out.ports is None
    assert out.sweep is None


def test_join_keeps_no_field_when_no_block_kept_any():
    rng = np.random.default_rng(4)
    blocks = [
        ((0, 1), _uniform(rng, (0,), fields=False)),
        ((2, 3), _uniform(rng, (0, 1), fields=False)),
    ]
    out = SweepOutputs.join(blocks)
    assert out.rows == (0, 2, 3)
    assert out.e_field is None
    assert "e_field" not in out.outputs()


def test_join_takes_the_whole_grids_port_modes_at_the_joined_rows():
    # workers return their blocks without port modes: the parent's sweeps
    # over the whole grid give the rows the blocks solved
    rng = np.random.default_rng(5)
    sweeps = [_wave(rng, "w1").sweep, _wave(rng, "w2").sweep]
    blocks = [
        ((0, 1), _uniform(rng, (0,), fields=False)),
        ((2, 3), _uniform(rng, (0, 1), fields=False)),
    ]
    out = SweepOutputs.join(blocks, sweeps=sweeps)
    assert out.rows == (0, 2, 3)
    ports = out.ports
    assert ports is not None
    assert ports.rows == out.rows
    for name in ("gamma", "z_pv", "residual", "solved"):
        expected = np.stack([getattr(w, name) for w in sweeps], axis=1)[[0, 2, 3]]
        np.testing.assert_array_equal(getattr(ports, name), expected)
    np.testing.assert_array_equal(ports.field(1)._parts[2], sweeps[1].coords[[0, 2, 3]])
    assert out.outputs()["port_sweeps"] == sweeps


def test_join_refuses_a_blocks_own_port_modes():
    # a block's own port mode sweep is not the whole grid's
    rng = np.random.default_rng(6)
    block = _uniform(rng, (0, 1), waves=[_wave(rng, "w1")])
    with pytest.raises(ValueError, match="port mode"):
        SweepOutputs.join([((0, 1, 2, 3), block)])


def _check_specs(out, specs):
    outputs = out.outputs()
    assert set(specs) <= set(outputs)
    for name, spec in specs.items():
        np.asarray(outputs[name], spec.dtype).reshape(spec.shape)
    host = {"iterations", "reused", "sweep"} & set(outputs)
    if out.ports is not None:
        host.add("port_sweeps")
    assert set(outputs) - set(specs) == host
    assert host <= HOST_ONLY


def test_a_uniform_encoding_matches_its_specs():
    rng = np.random.default_rng(6)
    out = _uniform(rng, tuple(range(N_FREQ)), waves=[_wave(rng, "w1")])
    specs = SweepOutputs.specs(
        n_freq=N_FREQ,
        n_ports=N_PORTS,
        n_excited=N_EXC,
        n_wave=1,
        ndof=NDOF,
        n_fields=N_EXC,
    )
    assert list(specs) == [
        "s",
        "z_ref",
        "port_gamma",
        "port_z_pv",
        "port_residual",
        "port_solved",
        "e_field",
    ]
    _check_specs(out, specs)


def test_an_adaptive_encoding_matches_its_specs():
    rng = np.random.default_rng(7)
    out = _adaptive(rng, waves=[_wave(rng, "w1")])
    specs = SweepOutputs.specs(
        n_freq=N_FREQ,
        n_ports=N_PORTS,
        n_excited=N_EXC,
        n_wave=1,
        ndof=NDOF,
        n_fields=N_EXC,
        reduced=6,
    )
    assert list(specs)[-2:] == ["e_basis", "e_coordinates"]
    assert "sweep" in out.outputs()
    _check_specs(out, specs)


def test_a_fieldless_encoding_matches_its_specs():
    rng = np.random.default_rng(8)
    out = _uniform(rng, tuple(range(N_FREQ)), fields=False)
    specs = SweepOutputs.specs(
        n_freq=N_FREQ, n_ports=N_PORTS, n_excited=N_EXC, n_wave=0
    )
    assert list(specs) == ["s", "z_ref"]
    _check_specs(out, specs)


def test_the_differentiable_outputs_are_spec_names():
    specs = SweepOutputs.specs(
        n_freq=1,
        n_ports=1,
        n_excited=1,
        n_wave=1,
        ndof=1,
        n_fields=1,
        n_sources=1,
        n_dipoles=1,
    )
    reduced = SweepOutputs.specs(
        n_freq=1, n_ports=1, n_excited=1, n_wave=0, ndof=1, n_fields=1, reduced=1
    )
    assert set(specs) | set(reduced) >= DIFFERENTIABLE
    assert "e_basis" not in DIFFERENTIABLE


@pytest.mark.parametrize("kind", ["uniform", "adaptive"])
def test_reading_the_encoding_gives_back_every_field(kind):
    rng = np.random.default_rng(9)
    waves = [_wave(rng, "w1"), _wave(rng, "w2")]
    if kind == "uniform":
        out = _uniform(rng, tuple(range(N_FREQ)), waves=waves)
    else:
        out = _adaptive(rng, waves=waves)
    back = SweepOutputs.read(out.outputs())
    assert back.rows == out.rows
    for name in ("s", "z_ref", "iterations", "reused", "e_field", "sweep"):
        assert getattr(back, name) is getattr(out, name)
    if out.reduced is None:
        assert back.reduced is None
    else:
        assert all(a is b for a, b in zip(back.reduced, out.reduced, strict=True))
    for name in ("gamma", "z_pv", "residual", "solved"):
        assert getattr(back.ports, name) is getattr(out.ports, name)
    assert back.ports.rows == out.ports.rows
    assert all(a is b for a, b in zip(back.ports.sweeps, out.ports.sweeps, strict=True))
    assert back.ports.field(1)._parts[2] is waves[1].sweep.coords


def test_reading_the_spec_names_only_gives_no_host_only_field():
    rng = np.random.default_rng(10)
    out = _adaptive(rng, waves=[_wave(rng, "w1")])
    specs = SweepOutputs.specs(
        n_freq=N_FREQ,
        n_ports=N_PORTS,
        n_excited=N_EXC,
        n_wave=1,
        ndof=NDOF,
        n_fields=N_EXC,
        reduced=6,
    )
    outputs = out.outputs()
    back = SweepOutputs.read({name: outputs[name] for name in specs})
    assert back.rows == tuple(range(N_FREQ))
    assert back.iterations is None
    assert back.reused is None
    assert back.sweep is None
    assert back.ports.sweeps is None
    assert back.ports.field(0) is None
    np.testing.assert_array_equal(back.s, out.s)
    assert back.reduced[0] is out.reduced[0]


def test_reading_outputs_without_reuse_flags_gives_none():
    outputs = _uniform(np.random.default_rng(11), (0, 1)).outputs()
    del outputs["reused"]
    back = SweepOutputs.read(outputs)
    assert back.reused is None
    assert back.iterations == [(3, 4), (3, 4)]


# --- the rules without a sweep: forward outputs against output_specs -------------


def _check_forward(rule, model):
    """Every forward output has its spec's name, dtype and shape."""
    specs = rule.output_specs(model)
    outputs, _ = rule.forward(model, keep=False)
    assert set(outputs) == set(specs)
    for name, spec in specs.items():
        value = np.asarray(outputs[name])
        assert value.shape == spec.shape, name
        assert value.dtype == spec.dtype, name
    return specs


@pytest.fixture(scope="module")
def line_grid(tmp_path_factory):
    return line_mesh(tmp_path_factory.mktemp("line"))


def test_eigenmode_outputs_match_their_specs_without_ports(line_grid):
    model = line(line_grid, tan_delta=1e-3)
    problem = gfem.Eigenmode(1.6 * F1, n_modes=2)
    specs = _check_forward(EigenmodeRule(model, problem, gfem.SolverOptions()), model)
    assert specs["q_port"].shape == (0, 2)
    assert specs["participation"].shape == (0, 2)


def test_eigenmode_outputs_match_their_specs_with_both_kinds_of_port(line_grid):
    ports = [
        gfem.LumpedPort("load", "port1", "+z", impedance=1000.0),
        gfem.LumpedPort("jj", "port2", "+z", inductance=1e-9),
    ]
    model = line(line_grid, ports=ports)
    problem = gfem.Eigenmode(8e9, n_modes=2)
    specs = _check_forward(EigenmodeRule(model, problem, gfem.SolverOptions()), model)
    assert specs["q_port"].shape == (1, 2)
    assert specs["participation"].shape == (1, 2)


@pytest.mark.parametrize(
    "terminals", [{"top": "top"}, {"top": "top", "mid": "oxide__nitride"}]
)
def test_electrostatic_outputs_match_their_specs(tmp_path, terminals):
    path = tmp_path / "stack.msh"
    layers = (("oxide", 0.5), ("nitride", 0.25))
    sm.write_msh22(path, sm.plate(4.0, 3.0, layers, (1, 1, 1)))
    materials = {"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)}
    model = gfem.Model(gfem.Mesh.from_gmsh(path), materials=materials)
    problem = gfem.Electrostatic(terminals=terminals, ground="bottom")
    rule = ElectrostaticRule(model, problem, gfem.SolverOptions())
    specs = _check_forward(rule, model)
    n = len(terminals)
    assert specs["capacitance"].shape == (n, n)
    assert specs["energy"].shape == (2, n)


def _with_port_waves(rng, rows, n_freq):
    return SweepOutputs.uniform(
        rows,
        s=_complex(rng, n_freq, N_PORTS, N_EXC),
        z_ref=rng.random((n_freq, N_PORTS)),
        iterations=[],
        reused=[],
        e_field=None,
        waves=[],
        port_waves=_complex(rng, n_freq, N_PORTS, 2),
    )


def test_source_port_waves_follow_the_rows_the_seam_and_the_join():
    rng = np.random.default_rng(9)
    out = _with_port_waves(rng, (0, 2), 3)
    assert out.port_waves.shape == (2, N_PORTS, 2)
    back = SweepOutputs.read(out.outputs())
    np.testing.assert_array_equal(back.port_waves, out.port_waves)
    specs = SweepOutputs.specs(
        n_freq=2, n_ports=N_PORTS, n_excited=N_EXC, n_wave=0, n_sources=2
    )
    assert specs["port_waves"].shape == out.port_waves.shape
    second = _with_port_waves(rng, (0,), 1)
    joined = SweepOutputs.join([([0, 1, 2], out), ([3], second)])
    np.testing.assert_array_equal(
        joined.port_waves, np.concatenate([out.port_waves, second.port_waves])
    )


def test_emitted_power_follows_the_rows_the_seam_and_the_join():
    rng = np.random.default_rng(3)
    power = rng.standard_normal((N_FREQ, 1))
    rows = (0, 2)
    out = SweepOutputs.uniform(
        rows,
        s=_complex(rng, N_FREQ, N_PORTS, N_EXC),
        z_ref=rng.random((N_FREQ, N_PORTS)),
        iterations=[],
        reused=[],
        e_field=None,
        waves=[],
        emitted_power=power,
    )
    np.testing.assert_array_equal(out.emitted_power, power[list(rows)])
    back = SweepOutputs.read(out.outputs())
    np.testing.assert_array_equal(back.emitted_power, power[list(rows)])
    joined = SweepOutputs.join([((0, 1, 2), out), ((3, 4, 5), out)])
    np.testing.assert_array_equal(
        joined.emitted_power, np.concatenate([power[list(rows)]] * 2)
    )
