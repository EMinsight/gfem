r"""`RationalModel.to_spice` and `RationalModel.to_verilog_a`.

The SPICE subcircuit is read back by the test helper `spice_mna`, a small
AC solver of the R, C, V, G and F cards, and must give the model's S:
:math:`\max|S_\mathrm{netlist} - S_\mathrm{model}| \le 10^{-9}`.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest
import spice_mna

import gfem

#: Highest frequency of the synthetic models, Hz.
F_MAX = 10e9
W_MAX = 2 * math.pi * F_MAX
#: Reference impedances, ohm, of the first ports.
Z_REF = (50.0, 75.0, 30.0, 100.0)


def _sigma_max(model, f):
    s = np.asarray(model.network(f).values)
    return np.linalg.svd(s, compute_uv=False)[:, 0].max()


def _model(n_ports, *, ports=None, seed=0, **kwargs):
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
    model = _build(n_ports, ports, np.array(poles), np.array(residues), d, **kwargs)
    scale = 0.8 / _sigma_max(model, np.linspace(0, 3 * F_MAX, 30001))
    return _build(
        n_ports, ports, model.poles, model.residues * scale, d * scale, **kwargs
    )


def _build(n_ports, ports, poles, residues, d, **kwargs):
    kwargs.setdefault("frequency_range", (0.0, 3 * F_MAX))
    return gfem.RationalModel(
        poles=poles,
        residues=residues,
        d=d,
        z_ref=np.array(Z_REF[:n_ports]),
        ports=ports or tuple(f"p{i + 1}" for i in range(n_ports)),
        **kwargs,
    )


#: A shunt resistor at port 1, and port 2 driven by a G source on port 1's
#: voltage and an F source on its current: the reader's conventions in a
#: two-port whose S is known in closed form.
_HAND_NETLIST = """\
* a hand-written two-port
.subckt hand a a_ref b
+ b_ref
vs a m 0
r1 m a_ref 40
g1 b b_ref a a_ref 0.02
f1 b b_ref vs 3
.ends hand
"""


def test_the_reader_gives_a_hand_netlist_its_closed_form_s():
    z1, z2, r, gm, k = 50.0, 75.0, 40.0, 0.02, 3.0
    # driven at port 1: V1 = 2 sqrt(z1) r / (r + z1) and the pin current
    # V1 / r; each controlled source pulls its current out of pin b
    v1 = 2 * math.sqrt(z1) * r / (r + z1)
    v2 = -(gm * v1 + k * v1 / r) * z2
    expected = [[(r - z1) / (r + z1), 0.0], [v2 / math.sqrt(z2), 1.0]]

    s = spice_mna.s_parameters(_HAND_NETLIST, [1e6, 1e9], [z1, z2])

    assert np.abs(s - np.asarray(expected)[None]).max() <= 1e-14


def _spice_error(model, tmp_path):
    path = model.to_spice(tmp_path / "model.sp")
    f = np.linspace(1e7, F_MAX, 301)
    s_netlist = spice_mna.s_parameters(path.read_text(), f, list(model.z_ref))
    s_model = np.asarray(model.network(f).values)
    return np.abs(s_netlist - s_model).max()


@pytest.mark.parametrize("n_ports", [2, 3, 4])
def test_the_spice_subcircuit_reproduces_the_model_s(n_ports, tmp_path):
    model = _model(n_ports)
    assert model.passive

    assert _spice_error(model, tmp_path) <= 1e-9


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
    return _build(n_ports, None, np.array(poles), np.array(residues), -np.eye(n_ports))


def test_a_model_that_is_a_short_at_infinite_frequency_is_reproduced(tmp_path):
    model = _short_at_infinity(3)
    assert model.passive
    assert np.array_equal(model.d, -np.eye(3))

    assert _spice_error(model, tmp_path) <= 1e-9


#: A number in exponent form, as both writers print them.
_EXPONENT = re.compile(r"-?\d(\.\d+)?e[+-]\d+")


def _cards(text):
    """The element cards of a SPICE netlist, split into words."""
    return spice_mna.subcircuit(text)[1]


def _values(card):
    """The numeric value fields of an R, C, V, G or F card."""
    return {"r": card[3:], "c": card[3:], "v": card[3:], "g": card[5:], "f": card[4:]}[
        card[0][0]
    ]


@pytest.mark.parametrize("writer", ["to_spice", "to_verilog_a"])
def test_pins_are_sanitized_port_names_and_their_ref_pins_in_port_order(
    writer, tmp_path
):
    model = _model(3, ports=("Out-1", "IN", "2nd"))

    text = getattr(model, writer)(tmp_path / "m.txt").read_text()

    pins = ["out_1", "out_1_ref", "in", "in_ref", "p_2nd", "p_2nd_ref"]
    if writer == "to_spice":
        assert spice_mna.subcircuit(text)[0] == pins
    else:
        assert f"module m({', '.join(pins)});" in text


@pytest.mark.parametrize("writer", ["to_spice", "to_verilog_a"])
@pytest.mark.parametrize(
    ("ports", "pin"),
    [(("P1", "p1"), "p1"), (("a", "a_ref"), "a_ref"), (("x.y", "x_y"), "x_y")],
)
def test_two_ports_that_give_one_pin_are_refused(writer, ports, pin, tmp_path):
    model = _model(2, ports=ports)

    with pytest.raises(
        ValueError, match=rf"{ports[0]!r} and {ports[1]!r}.*{pin!r}.*rename"
    ):
        getattr(model, writer)(tmp_path / "m.sp")
    assert not (tmp_path / "m.sp").exists()


@pytest.mark.parametrize("writer", ["to_spice", "to_verilog_a"])
@pytest.mark.parametrize("port", ["GND", "gnd"])
def test_a_pin_named_gnd_is_refused(writer, port, tmp_path):
    model = _model(2, ports=(port, "p2"))

    with pytest.raises(ValueError, match=r"'gnd'.*ground.*rename"):
        getattr(model, writer)(tmp_path / "m.sp")


@pytest.mark.parametrize("writer", ["to_spice", "to_verilog_a"])
def test_a_model_that_is_not_passive_is_refused(writer, tmp_path):
    model = _model(2)
    active = _build(2, None, model.poles, model.residues * 1.5, model.d * 1.5)
    assert not active.passive

    with pytest.raises(ValueError, match=r"not passive.*enforce_passivity=True"):
        getattr(active, writer)(tmp_path / "m.sp")
    assert not (tmp_path / "m.sp").exists()


@pytest.mark.parametrize("writer", ["to_spice", "to_verilog_a"])
def test_an_empty_name_is_refused(writer, tmp_path):
    with pytest.raises(ValueError, match="name must not be empty"):
        getattr(_model(2), writer)(tmp_path / "m.sp", name="")


@pytest.mark.parametrize("writer", ["to_spice", "to_verilog_a"])
def test_the_name_defaults_to_the_sanitized_file_stem_and_the_path_is_returned(
    writer, tmp_path
):
    model = _model(2)
    target = tmp_path / "Low-pass 2.lib"

    path = getattr(model, writer)(str(target))
    named = getattr(model, writer)(tmp_path / "other.inc", name="7 Up")

    assert isinstance(path, Path)
    assert path == target
    head = ".subckt" if writer == "to_spice" else "module"
    assert f"{head} low_pass_2" in path.read_text()
    assert f"{head} p_7_up" in named.read_text()


def test_the_subcircuit_has_only_r_c_v_g_f_cards_and_exponent_numbers(tmp_path):
    text = _model(3).to_spice(tmp_path / "m.sp").read_text()

    cards = _cards(text)
    assert {card[0][0] for card in cards} == set("rcvgf")
    values = [value for card in cards for value in _values(card)]
    assert len(values) == len(cards)
    assert all(_EXPONENT.fullmatch(value) for value in values), values


def test_the_subcircuit_is_lower_case_with_full_line_star_comments(tmp_path):
    text = _model(2, ports=("In", "Out")).to_spice(tmp_path / "m.sp").read_text()

    lines = text.splitlines()
    assert lines[0].startswith("*")
    for line in lines:
        if line.startswith("*"):
            continue
        assert line == line.lower()
        assert line[0] in "+.rcvgf"
        assert not set(";$") & set(line)


@pytest.mark.parametrize(
    ("writer", "mark"), [("to_spice", "*"), ("to_verilog_a", "//")]
)
def test_the_header_holds_ports_references_range_error_version_and_convention(
    writer, mark, tmp_path
):
    model = _model(2, ports=("In", "Out-2"), error=1.25e-4)
    nominal = _model(2, nominal_reference=True)

    text = getattr(model, writer)(tmp_path / "m.txt").read_text()
    labelled = getattr(nominal, writer)(tmp_path / "n.txt").read_text()

    header = "\n".join(line for line in text.splitlines() if line.startswith(mark))
    assert f"gfem {gfem.__version__}" in header
    assert "exp(+j omega t)" in header
    assert "0-30 GHz" in header
    assert "error: 0.000125" in header
    assert "in in_ref: port 'In', z_ref 50 ohm" in header
    assert "out_2 out_2_ref: port 'Out-2', z_ref 75 ohm" in header
    assert "nominal label" not in text
    assert "z_ref is a nominal label" in labelled


def _code(text):
    """Verilog-A text without its ``//`` comments."""
    return "\n".join(line.split("//")[0] for line in text.splitlines())


def test_the_verilog_a_module_uses_ddt_states_and_scalar_code_only(tmp_path):
    text = _model(3).to_verilog_a(tmp_path / "m.va").read_text()

    code = _code(text)
    statements = [s.strip() for s in code.split(";")]
    ddt = [s for s in statements if "ddt(" in s]
    # 2 real poles and 3 pairs, driven from each of 3 ports
    assert len(ddt) == (2 + 2 * 3) * 3
    assert all(re.fullmatch(r"I\(\w+\) <\+ ddt\([^()]*\(\w+\)\)", s) for s in ddt)
    assert "laplace_" not in code
    assert "[" not in code
    assert not re.search(r"\b(genvar|parameter|ground|for)\b", code)
    assert "<+" in code
    assert "<=" not in code
    numbers = re.findall(r"(?<![\w.])\d[\w.+-]*", code)
    assert numbers
    assert all(_EXPONENT.fullmatch(x) for x in numbers), numbers


@pytest.mark.parametrize(
    ("ports", "name", "match"),
    [
        (("input", "p2"), None, r"Port 'input' gives the pin 'input'.*reserved"),
        (("p1", "Output"), None, r"Port 'Output' gives the pin 'output'.*reserved"),
        (("p1", "p2"), "module", r"module name 'module'.*reserved.*name="),
    ],
)
def test_a_verilog_a_reserved_word_as_pin_or_module_name_is_refused(
    ports, name, match, tmp_path
):
    model = _model(2, ports=ports)

    with pytest.raises(ValueError, match=match):
        model.to_verilog_a(tmp_path / "m.va", name=name)
    assert model.to_spice(tmp_path / "m.sp", name=name).exists()
