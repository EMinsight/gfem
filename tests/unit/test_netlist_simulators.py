r"""The exported netlists run in real circuit simulators.

`RationalModel.to_spice` in ngspice 45 and `RationalModel.to_verilog_a`,
compiled by openvaf-r, in ngspice 45 and VACASK, must give the model's S in
an AC sweep: :math:`\max|S_\mathrm{netlist} - S_\mathrm{model}| \le 10^{-9}`.
A transient run of the SPICE subcircuit follows the model's closed-form ramp
response to the simulator's tolerance.

The simulators come from the ``netlist`` dependency group (``uv sync --group
netlist``); without it these tests skip.
"""

import itertools
import math
import subprocess

import numpy as np
import pytest

import gfem

pytestmark = pytest.mark.netlist

sedes = pytest.importorskip(
    "sedes", reason="needs the netlist group: uv sync --group netlist"
)

#: Highest frequency of the synthetic models, Hz.
F_MAX = 10e9
W_MAX = 2 * math.pi * F_MAX
#: Reference impedances, ohm, of the first ports.
Z_REF = (50.0, 75.0, 30.0, 100.0)
#: The AC sweep: 301 points from 10 MHz to F_MAX.
F_START, N_FREQ = 1e7, 301


def _build(n_ports, poles, residues, d):
    return gfem.RationalModel(
        poles=poles,
        residues=residues,
        d=d,
        z_ref=np.array(Z_REF[:n_ports]),
        ports=tuple(f"p{i + 1}" for i in range(n_ports)),
        frequency_range=(0.0, 3 * F_MAX),
    )


def _model(n_ports, seed=0):
    """A passive model: 2 real poles and 3 pairs, largest singular value 0.8."""
    rng = np.random.default_rng(seed)
    real = -np.array([0.1, 1.3]) * W_MAX
    wr = np.array([0.2, 0.55, 0.9]) * W_MAX
    cplx = -wr * np.array([0.03, 0.05, 0.02]) + 1j * wr
    poles = list(real)
    residues = [rng.standard_normal((n_ports, n_ports)) * abs(p) for p in real]
    for p in cplx:
        m = rng.standard_normal((n_ports, n_ports)) + 1j * rng.standard_normal(
            (n_ports, n_ports)
        )
        m *= abs(p) * 0.1
        poles += [p, np.conj(p)]
        residues += [m, np.conj(m)]
    d = 0.3 * rng.standard_normal((n_ports, n_ports))
    model = _build(n_ports, np.array(poles), np.array(residues), d)
    s = np.asarray(model.network(np.linspace(0, 3 * F_MAX, 30001)).values)
    scale = 0.8 / np.linalg.svd(s, compute_uv=False)[:, 0].max()
    return _build(n_ports, model.poles, model.residues * scale, d * scale)


def _short_at_infinity(n_ports):
    r""":math:`D = -I` plus positive real terms, a short circuit at high frequency.

    Real symmetric positive semidefinite residues on real poles and pairs make
    :math:`\sum_n R_n/(s - p_n)` positive real, so a small multiple of it keeps
    :math:`-I` plus it passive.
    """
    rng = np.random.default_rng(3)
    poles, residues = [], []
    for p in (-0.3 * W_MAX, complex(-0.05, 0.6) * W_MAX):
        a = rng.standard_normal((n_ports, n_ports))
        m = a @ a.T * abs(p) * 0.01
        poles += [p] if np.imag(p) == 0 else [p, np.conj(p)]
        residues += [m] * (1 if np.imag(p) == 0 else 2)
    return _build(n_ports, np.array(poles), np.array(residues), -np.eye(n_ports))


#: The models of the AC checks, by test id.
MODELS = {
    "2 ports": lambda: _model(2),
    "3 ports": lambda: _model(3),
    "4 ports": lambda: _model(4),
    "D = -I": lambda: _short_at_infinity(3),
}


def _num(x):
    return f"{x:.17e}"


def _terminations(z_ref, excited):
    r"""SPICE cards that drive pin ``t<j>`` with a unit incident wave.

    Port :math:`j` sees a source :math:`2\sqrt{z_j}` behind :math:`z_j`, so
    its incident power wave is 1; every other port ends on its reference.
    """
    cards = []
    for i, z in enumerate(z_ref, start=1):
        if i == excited:
            cards += [
                f"vsrc t{i}x 0 dc 0 ac {_num(2 * math.sqrt(z))}",
                f"rt{i} t{i}x t{i} {_num(z)}",
            ]
        else:
            cards.append(f"rt{i} t{i} 0 {_num(z)}")
    return cards


def _s_column(voltages, z_ref, excited):
    r""":math:`S_{ij} = V_i/\sqrt{z_i} - \delta_{ij}` for a unit incident wave."""
    return np.stack(
        [
            v / math.sqrt(z) - (1.0 if i == excited else 0.0)
            for i, (v, z) in enumerate(zip(voltages, z_ref, strict=True), start=1)
        ],
        axis=-1,
    )


@pytest.fixture(scope="module")
def pool():
    return sedes.SimulationPool(workers=1)


def _ngspice_s(pool, model, head, instance):
    """S from one ngspice AC sweep per excited port."""
    n = len(model.ports)
    pins = " ".join(f"t{i} 0" for i in range(1, n + 1))
    columns = []
    for j in range(1, n + 1):
        deck = [
            "* AC check",
            *head,
            f"{instance} {pins} dut",
            *_terminations(model.z_ref, j),
            f".ac lin {N_FREQ} {_num(F_START)} {_num(F_MAX)}",
            ".end",
        ]
        result = pool.run_simulation("\n".join(deck) + "\n", ["run"])
        assert result.success, result.error
        voltages = [result.complex_vector(f"v(t{i})") for i in range(1, n + 1)]
        columns.append(_s_column(voltages, model.z_ref, j))
        f = result.frequency_vector()
    return f, np.stack(columns, axis=-1)


def _error(model, f, s):
    return np.abs(s - np.asarray(model.network(f).values)).max()


@pytest.mark.parametrize("case", MODELS)
def test_the_spice_subcircuit_gives_the_model_s_in_ngspice(case, pool, tmp_path):
    model = MODELS[case]()
    path = model.to_spice(tmp_path / "dut.sp")

    f, s = _ngspice_s(pool, model, [f".include {path}"], "xdut")

    assert f.size == N_FREQ
    assert _error(model, f, s) <= 1e-9


@pytest.fixture(scope="module")
def vacask_bin():
    return pytest.importorskip(
        "vacask_bin", reason="needs the netlist group: uv sync --group netlist"
    )


#: Module names: an ngspice worker keeps the first OSDI device of a name.
_MODULES = itertools.count()


def _osdi(vacask_bin, model, tmp_path):
    """The model's Verilog-A module, compiled by openvaf-r; its name."""
    name = f"dut{next(_MODULES)}"
    source = model.to_verilog_a(tmp_path / f"{name}.va")
    target = tmp_path / f"{name}.osdi"
    run = subprocess.run(
        [vacask_bin.OPENVAF_CMD, str(source), "-o", str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return target, name


@pytest.mark.parametrize("case", MODELS)
def test_the_verilog_a_module_gives_the_model_s_in_ngspice(
    case, pool, vacask_bin, tmp_path
):
    model = MODELS[case]()
    osdi, name = _osdi(vacask_bin, model, tmp_path)
    head = [".control", f"pre_osdi {osdi}", ".endc", f".model dut {name}"]

    f, s = _ngspice_s(pool, model, head, "ndut")

    assert f.size == N_FREQ
    assert _error(model, f, s) <= 1e-9


def _read_raw(path):
    """The vectors of a binary SPICE raw file holding one complex analysis."""
    head, _, body = path.read_bytes().partition(b"Binary:\n")
    lines = head.decode().splitlines()
    field = {
        key.strip(): value.strip()
        for key, _, value in (line.partition(":") for line in lines)
    }
    assert field["Flags"] == "complex"
    n_vars, n_points = int(field["No. Variables"]), int(field["No. Points"])
    first = lines.index("Variables:") + 1
    names = [line.split("\t")[2] for line in lines[first : first + n_vars]]
    values = np.frombuffer(body, "<c16", n_vars * n_points).reshape(n_points, n_vars)
    return dict(zip(names, values.T, strict=True))


def _vacask_s(vacask_bin, model, tmp_path):
    """S from one VACASK AC sweep per excited port."""
    osdi, name = _osdi(vacask_bin, model, tmp_path)
    n = len(model.ports)
    columns = []
    for j in range(1, n + 1):
        cards = []
        for i, z in enumerate(model.z_ref, start=1):
            if i == j:
                cards += [
                    f"vs (t{i}x 0) v dc=0 mag={_num(2 * math.sqrt(z))}",
                    f"rt{i} (t{i}x t{i}) r r={_num(z)}",
                ]
            else:
                cards.append(f"rt{i} (t{i} 0) r r={_num(z)}")
        deck = [
            "AC check",
            "ground 0",
            f'load "{osdi}"',
            'load "resistor.osdi"',
            "model v vsource",
            "model r resistor",
            f"model dut {name}",
            "x (" + " ".join(f"t{i} 0" for i in range(1, n + 1)) + ") dut",
            *cards,
            "control",
            (
                f"  analysis ac{j} ac from={_num(F_START)} to={_num(F_MAX)} "
                f'mode="lin" points={N_FREQ - 1}'
            ),
            "endc",
            "",
        ]
        (tmp_path / f"ac{j}.sim").write_text("\n".join(deck))
        run = subprocess.run(
            [vacask_bin.VACASK_CMD, "-qp", f"ac{j}.sim"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        vectors = _read_raw(tmp_path / f"ac{j}.raw")
        voltages = [vectors[f"t{i}"] for i in range(1, n + 1)]
        columns.append(_s_column(voltages, model.z_ref, j))
        f = vectors["frequency"].real
    return f, np.stack(columns, axis=-1)


def test_the_verilog_a_module_gives_the_model_s_in_vacask(vacask_bin, tmp_path):
    model = _model(2)

    f, s = _vacask_s(vacask_bin, model, tmp_path)

    assert f.size == N_FREQ
    assert _error(model, f, s) <= 1e-9


#: The transient check: a 20 ps ramp to a unit incident wave at port 1, run
#: for 5 ns (about 6 time constants of the slowest pole) in 0.2 ps steps.
RISE, STOP, STEP = 20e-12, 5e-9, 0.2e-12


def _step_integral(model, time):
    r""":math:`\int_0^t` of the model's step response from port 1.

    ``(n_time, n_ports)``, by Gauss-Legendre quadrature of
    `RationalModel.step_response` on a 1 ps grid refined at the given times,
    so exact to round-off.
    """
    time = np.maximum(time, 0.0)
    edges = np.unique(np.concatenate([time, np.linspace(0.0, time.max(), 5001)]))
    x, w = np.polynomial.legendre.leggauss(8)
    a, b = edges[:-1], edges[1:]
    nodes = (a + b)[:, None] / 2 + (b - a)[:, None] / 2 * x
    step = np.asarray(model.step_response(nodes.ravel()).values)[:, :, 0]
    pieces = np.einsum("k,ikp->ip", w, step.reshape(*nodes.shape, -1))
    pieces *= ((b - a) / 2)[:, None]
    total = np.concatenate([np.zeros((1, step.shape[-1])), np.cumsum(pieces, 0)])
    return total[np.searchsorted(edges, time)]


def test_a_transient_of_the_spice_subcircuit_follows_the_closed_form_ramp(
    pool, tmp_path
):
    model = _model(2)
    path = model.to_spice(tmp_path / "dut.sp")
    z1, z2 = model.z_ref
    deck = [
        "* transient check",
        f".include {path}",
        "xdut t1 0 t2 0 dut",
        f"vsrc t1x 0 pwl(0 0 {_num(RISE)} {_num(2 * math.sqrt(z1))})",
        f"rt1 t1x t1 {_num(z1)}",
        f"rt2 t2 0 {_num(z2)}",
        ".options reltol=1e-6",
        f".tran {_num(STEP)} {_num(STOP)}",
        ".end",
    ]

    result = pool.run_simulation("\n".join(deck) + "\n", ["run"])

    assert result.success, result.error
    t = result.time_vector()
    incident = np.clip(t / RISE, 0.0, 1.0)
    b = np.stack(
        [
            result.vector("v(t1)") / math.sqrt(z1) - incident,
            result.vector("v(t2)") / math.sqrt(z2),
        ],
        axis=-1,
    )
    # a ramp is the difference of two integrated steps, RISE apart
    ramp = (_step_integral(model, t) - _step_integral(model, t - RISE)) / RISE
    assert t[-1] == pytest.approx(STOP)
    assert np.abs(b - ramp).max() <= 1e-4
