"""`DrivenSystemBuilder`: a driven solve's system operator, port network and kept state.

The couplings and their port networks are kept on the discretization
(`KeptPorts`) and reused while their key holds; the design guess is kept and
used for a ``bddc_gmres`` sweep on one process only.
"""

from types import SimpleNamespace

from gradient_cases import wave_line, wave_mesh

import gfem
from gfem._engine import cache, wave_ports
from gfem._engine.driven import DrivenRule
from gfem._engine.driven_system import DrivenSystemBuilder, KeptPorts

FREQUENCIES = (20e9, 25e9)
OPTIONS = gfem.SolverOptions(order=1, linear_solver="direct", threads=1)


def _builder(**fields):
    settings = {
        "frequency": (1e9,),
        "columns": (0,),
        "excite": ("p1",),
        "port_mode_tolerance": 1e-6,
        "floquet_order_tolerance": 1e-6,
        "orders": {},
        "order_frequency": (1e9,),
        "wave_vector": None,
        "problem_wave_vector": None,
        "wave_vector_frequency": None,
        "swept_spaces": False,
        "comm": None,
        "linear": SimpleNamespace(kind="bddc_gmres", tolerance=1e-8),
        "adjoint_tolerance": 1e-8,
        "retain_memory": None,
        "gauge": False,
    }
    return DrivenSystemBuilder(**{**settings, **fields})


def test_kept_ports_keep_four_networks_and_drop_the_oldest():
    kept = KeptPorts((), [])
    networks = [object() for _ in range(5)]
    for k, network in enumerate(networks):
        kept.keep((k,), network)
    assert list(kept.networks) == [(1,), (2,), (3,), (4,)]
    assert list(kept.networks.values()) == networks[1:]


def test_a_design_guess_is_kept_for_a_bddc_sweep_on_one_process_only(monkeypatch):
    built = object()
    disc = cache.Discretization(built=built, topology=None)
    monkeypatch.setitem(cache._DISCRETIZATIONS, "guess test", disc)
    system = SimpleNamespace(
        operator=SimpleNamespace(built=built, space_key="space", symmetry=None),
        free_digest=lambda: "free",
    )
    guess = _builder().design_guess(system)
    assert guess.disc is disc
    assert guess.key == ("design guess", ("p1",), 0, None, None)
    assert guess.match == ("space", "free", None, 1e-8)
    inert = [
        _builder(linear=SimpleNamespace(kind="direct", tolerance=1e-8)),
        _builder(comm=object()),
        _builder(retain_memory=0),
    ]
    for builder in inert:
        assert builder.design_guess(system).disc is None
    # a discretization no longer cached keeps nothing
    off = SimpleNamespace(operator=SimpleNamespace(built=object()))
    assert _builder().design_guess(off).disc is None


def test_a_second_build_reuses_the_couplings_and_adds_a_network(tmp_path, monkeypatch):
    model = wave_line(wave_mesh(tmp_path), "lossless")
    sweeps = []
    real = wave_ports.swept_once

    def spy(*args, **kwargs):
        sweeps.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(wave_ports, "swept_once", spy)
    cache._DISCRETIZATIONS.clear()
    rule = DrivenRule(model, gfem.Driven(FREQUENCIES), OPTIONS)
    _, network = rule.system(model)
    disc = cache.cached(rule.operator(model).built)
    kept = disc.ports
    assert isinstance(kept, KeptPorts)
    assert len(sweeps) == 2  # one per wave port surface
    couplings = list(kept.couplings)
    _, again = rule.system(model)
    assert disc.ports is kept
    assert all(a is b for a, b in zip(kept.couplings, couplings, strict=True))
    assert again is network
    assert len(sweeps) == 2
    # another excited set: a second network on the same couplings
    one = DrivenRule(model, gfem.Driven(FREQUENCIES, excite=("p1",)), OPTIONS)
    _, other = one.system(model)
    assert disc.ports is kept
    assert other is not network
    assert len(kept.networks) == 2
    assert len(sweeps) == 2
    assert other.columns == [0]
