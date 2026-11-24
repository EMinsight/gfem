r"""SPICE and Verilog-A text of a rational model of S.

One realization for both formats: the input-wise state-space form of
:math:`S(s) = D + \sum_n R_n / (s - p_n)`, with port :math:`j` driving one state
per real pole and two per conjugate pair,

.. math::

    \frac{s}{|p|}\, x = \frac{p}{|p|}\, x + a_j, \qquad
    b_i = \sum_j D_{ij} a_j + \sum_{n, j} \frac{R_{n,ij}}{|p_n|}\, x_{n,j},

scaled so that each state has unit DC gain and the element values stay near
one whatever the frequency band. A pair's complex state is split into its
real and imaginary parts, which gives real elements. Each port is a Norton
circuit on its reference impedance :math:`z`, built from the power waves
:math:`a = (V + zI)/(2\sqrt{z})` and :math:`b = (V - zI)/(2\sqrt{z})`:

.. math::

    I = \frac{V}{z} - \frac{2 b}{\sqrt{z}},

with :math:`I` flowing into the port's pin. The incident wave :math:`a` is an
internal node, so :math:`D = -I` (a short at infinite frequency) needs no
inversion.

Internal nodes and variables start with ``_``, which a sanitized pin never
does (it starts with a letter), so they never collide with a pin.
"""

from __future__ import annotations

import math
import re
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np

from gfem._text import hertz_band

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gfem.broadband import RationalModel

#: Verilog-AMS reserved words (lower case), refused as pin or module names.
_VERILOG_KEYWORD_TEXT = """
    above abs absdelay absdelta abstol access acos acosh ac_stim aliasparam
    always analog analysis and asin asinh assert assign atan atan2 atanh
    automatic begin branch buf bufif0 bufif1 case casex casez ceil cell cmos
    config connect connectmodule connectrules continuous cos cosh cross ddt
    ddt_nature ddx deassign default defparam design disable discipline
    discrete domain driver_update edge else end endcase endconfig
    endconnectrules enddiscipline endfunction endgenerate endmodule endnature
    endparamset endprimitive endspecify endtable endtask event exclude exp
    final_step flicker_noise floor flow for force forever fork from function
    generate genvar ground highz0 highz1 hypot idt idtmod idt_nature if ifnone
    incdir include inf initial initial_step inout input instance integer join
    laplace_nd laplace_np laplace_zd laplace_zp large last_crossing liblist
    library limexp ln localparam log macromodule max medium merged min module
    nand nature negedge net_resolution nmos noise_table noise_table_log nor
    noshowcancelled not notif0 notif1 or output parameter paramset pmos
    posedge potential pow primitive pull0 pull1 pulldown pullup
    pulsestyle_ondetect pulsestyle_onevent rcmos real realtime reg release
    repeat resolveto rnmos rpmos rtran rtranif0 rtranif1 scalared sin sinh
    showcancelled signed slew small specify specparam split sqrt string
    strong0 strong1 supply0 supply1 table tan tanh task time timer tran
    tranif0 tranif1 transition tri tri0 tri1 triand trior trireg units
    unsigned use uwire vectored wait wand weak0 weak1 white_noise while wire
    wor wreal xnor xor zi_nd zi_np zi_zd zi_zp
"""
_VERILOG_KEYWORDS = frozenset(_VERILOG_KEYWORD_TEXT.split())


def sanitize(name: str) -> str:
    """Lower case, ``[a-z0-9_]`` only, ``p_`` first when not a letter."""
    out = re.sub(r"[^a-z0-9_]", "_", name.lower())
    return out if re.match(r"[a-z]", out) else "p_" + out


def pins(ports: tuple[str, ...]) -> list[tuple[str, str]]:
    """``(pin, reference pin)`` of each port, in port order.

    Raises:
        ValueError: Two pins share a name, or a pin is named ``gnd``.
    """
    owner: dict[str, str] = {}
    out = []
    for port in ports:
        pin = sanitize(port)
        pair = (pin, pin + "_ref")
        for p in pair:
            if p == "gnd":
                raise ValueError(
                    f"Port {port!r} gives the pin 'gnd', which netlists read as "
                    "ground; rename the port."
                )
            if p in owner:
                raise ValueError(
                    f"Ports {owner[p]!r} and {port!r} both give the pin {p!r} "
                    "(pins are <port> and <port>_ref, lower case, characters "
                    "outside a-z, 0-9 and _ replaced by _); rename one of the "
                    "ports."
                )
            owner[p] = port
        out.append(pair)
    return out


def number(x: float) -> str:
    """``x`` in exponent form, the fewest digits that read back exactly.

    No SI suffix: ``M`` is milli to SPICE and mega to everyone else.
    """
    x = float(x)
    for digits in range(17):
        text = f"{x:.{digits}e}"
        if float(text) == x:
            return text
    return f"{x:.16e}"


def _sections(model: RationalModel) -> Iterator[tuple[int, complex, np.ndarray]]:
    """``(k, pole, residue)``: each real pole and one pole of each pair."""
    k = 0
    for p, r in zip(model.poles, model.residues, strict=True):
        if p.imag >= 0:
            k += 1
            yield k, complex(p), r


def _header(model: RationalModel, name: str) -> list[str]:
    """The header comment lines, without the comment mark."""
    n = len(model.ports)
    band = (
        "none" if model.frequency_range is None else hertz_band(*model.frequency_range)
    )
    error = (
        "not recorded"
        if math.isnan(model.error)
        else f"{model.error:.3g} (largest per-element RMS of the S error)"
    )
    lines = [
        (
            f"gfem {version('gfem')}: rational model {name} of S, {n} ports, "
            f"{model.n_poles} poles, passive"
        ),
        "S(s) = D + sum_n R_n / (s - p_n), s = j omega",
        "Time convention exp(+j omega t)",
        "Power waves (Kurokawa) on each port's real reference impedance z_ref",
        f"Fitted frequency range: {band}; error: {error}",
        "Pins <port> <port>_ref in port order; tie each *_ref pin to its return",
    ]
    for port, (pin, ref), z in zip(
        model.ports, pins(model.ports), model.z_ref, strict=True
    ):
        lines.append(f"  {pin} {ref}: port {port!r}, z_ref {z:.12g} ohm")
    if model.nominal_reference:
        lines.append(
            "No port had a reference impedance (wave ports without "
            "voltage_path): S is normalized to unit modal power and was not "
            "renormalized; z_ref is a nominal label"
        )
    return lines


def _wrap(head: str, words: list[str], width: int = 78) -> list[str]:
    """``head`` and ``words`` on lines of about ``width``, ``+`` continued."""
    lines = [head]
    for word in words:
        if len(lines[-1]) + 1 + len(word) > width and lines[-1] != "+":
            lines.append("+")
        lines[-1] += " " + word
    return lines


def spice(model: RationalModel, name: str) -> str:
    """The SPICE subcircuit ``name``: R, C, V, G and F cards only."""
    n = len(model.ports)
    pin_pairs = pins(model.ports)
    lines = [f"* {line}" if line else "*" for line in _header(model, name)]
    lines += _wrap(f".subckt {name}", [p for pair in pin_pairs for p in pair])
    for j, ((pin, ref), z) in enumerate(zip(pin_pairs, model.z_ref, strict=True)):
        k, sz = j + 1, math.sqrt(z)
        lines += [
            f"* port {k}: sense the pin current, Norton source, incident wave",
            f"vs{k} {pin} _s{k} {number(0.0)}",
            f"rz{k} _s{k} {ref} {number(z)}",
            f"gb{k} _s{k} {ref} _b{k} 0 {number(-2 / sz)}",
            f"rb{k} _b{k} 0 {number(1.0)}",
            f"ra{k} _a{k} 0 {number(1.0)}",
            f"gav{k} 0 _a{k} {pin} {ref} {number(1 / (2 * sz))}",
            f"fai{k} 0 _a{k} vs{k} {number(sz / 2)}",
        ]
    lines.append("* constant term D")
    for i in range(n):
        for j in range(n):
            if model.d[i, j] != 0:
                lines.append(
                    f"gd{i + 1}_{j + 1} 0 _b{i + 1} _a{j + 1} 0 {number(model.d[i, j])}"
                )
    for k, p, r in _sections(model):
        ap, sig, om = abs(p), p.real, p.imag
        lines.append(f"* pole {k}: {number(sig)} {number(om)}j rad/s")
        for j in range(n):
            if om == 0:
                x = f"_x{k}_{j + 1}"
                lines += [
                    f"cx{k}_{j + 1} {x} 0 {number(1 / ap)}",
                    f"rx{k}_{j + 1} {x} 0 {number(ap / -sig)}",
                    f"gx{k}_{j + 1} 0 {x} _a{j + 1} 0 {number(1.0)}",
                ]
                gains = [(x, f"{k}_", r[:, j].real / ap)]
            else:
                xr, xi = f"_x{k}r_{j + 1}", f"_x{k}i_{j + 1}"
                lines += [
                    f"cx{k}r_{j + 1} {xr} 0 {number(1 / ap)}",
                    f"rx{k}r_{j + 1} {xr} 0 {number(ap / -sig)}",
                    f"gx{k}r_{j + 1} 0 {xr} _a{j + 1} 0 {number(1.0)}",
                    f"gw{k}r_{j + 1} 0 {xr} {xi} 0 {number(-om / ap)}",
                    f"cx{k}i_{j + 1} {xi} 0 {number(1 / ap)}",
                    f"rx{k}i_{j + 1} {xi} 0 {number(ap / -sig)}",
                    f"gw{k}i_{j + 1} 0 {xi} {xr} 0 {number(om / ap)}",
                ]
                gains = [
                    (xr, f"{k}r_", 2 * r[:, j].real / ap),
                    (xi, f"{k}i_", -2 * r[:, j].imag / ap),
                ]
            for x, tag, column in gains:
                for i in range(n):
                    if column[i] != 0:
                        element = f"gc{tag}{i + 1}_{j + 1}"
                        lines.append(f"{element} 0 _b{i + 1} {x} 0 {number(column[i])}")
    lines.append(f".ends {name}")
    return "\n".join(lines) + "\n"


def verilog_keywords(
    pin_pairs: list[tuple[str, str]], ports: tuple[str, ...], name: str
) -> None:
    """Refuse a pin or module name that is a Verilog-A reserved word.

    Raises:
        ValueError: A pin or ``name`` is a reserved word.
    """
    for port, pair in zip(ports, pin_pairs, strict=True):
        for pin in pair:
            if pin in _VERILOG_KEYWORDS:
                raise ValueError(
                    f"Port {port!r} gives the pin {pin!r}, a Verilog-A reserved "
                    "word; rename the port."
                )
    if name in _VERILOG_KEYWORDS:
        raise ValueError(
            f"The module name {name!r} is a Verilog-A reserved word; pass "
            "another name=."
        )


def verilog_a(model: RationalModel, name: str) -> str:
    """The Verilog-A module ``name``: explicit ``ddt`` states, scalar code."""
    n = len(model.ports)
    pin_pairs = pins(model.ports)
    verilog_keywords(pin_pairs, model.ports, name)
    flat = [p for pair in pin_pairs for p in pair]
    internal = [f"_a{j + 1}" for j in range(n)]
    states: list[str] = []
    outputs: list[list[str]] = [
        [
            f"{number(model.d[i, j])} * V(_a{j + 1})"
            for j in range(n)
            if model.d[i, j] != 0
        ]
        for i in range(n)
    ]
    for k, p, r in _sections(model):
        ap, sig, om = abs(p), p.real, p.imag
        c, g, w = number(1 / ap), number(-sig / ap), om / ap
        for j in range(n):
            a = f"V(_a{j + 1})"
            if om == 0:
                x = f"_x{k}_{j + 1}"
                internal.append(x)
                states += [
                    f"    I({x}) <+ ddt({c} * V({x}));",
                    f"    I({x}) <+ {g} * V({x}) - {a};",
                ]
                gains = [(x, r[:, j].real / ap)]
            else:
                xr, xi = f"_x{k}r_{j + 1}", f"_x{k}i_{j + 1}"
                internal += [xr, xi]
                states += [
                    f"    I({xr}) <+ ddt({c} * V({xr}));",
                    f"    I({xr}) <+ {g} * V({xr}) + {number(w)} * V({xi}) - {a};",
                    f"    I({xi}) <+ ddt({c} * V({xi}));",
                    f"    I({xi}) <+ {g} * V({xi}) - {number(w)} * V({xr});",
                ]
                gains = [(xr, 2 * r[:, j].real / ap), (xi, -2 * r[:, j].imag / ap)]
            for x, column in gains:
                for i in range(n):
                    if column[i] != 0:
                        outputs[i].append(f"{number(column[i])} * V({x})")
    lines = [f"// {line}" for line in _header(model, name)]
    lines += [
        '`include "disciplines.vams"',
        "",
        f"module {name}({', '.join(flat)});",
        f"  inout {', '.join(flat)};",
        f"  electrical {', '.join(flat)};",
        *(f"  electrical {node};" for node in internal),
        *(f"  real _b{i + 1};" for i in range(n)),
        "  analog begin",
        "    // reflected waves b = D a + the states' outputs",
    ]
    for i, terms in enumerate(outputs):
        lines.append(
            f"    _b{i + 1} = " + "\n      + ".join(terms or [number(0.0)]) + ";"
        )
    lines.append("    // Norton ports; each incident wave a = V / sqrt(z) - b")
    for j, ((pin, ref), z) in enumerate(zip(pin_pairs, model.z_ref, strict=True)):
        sz, v = math.sqrt(z), f"V({pin}, {ref})"
        lines += [
            (
                f"    I({pin}, {ref}) <+ {number(1 / z)} * {v}"
                f" - {number(2 / sz)} * _b{j + 1};"
            ),
            f"    I(_a{j + 1}) <+ V(_a{j + 1}) + _b{j + 1} - {number(1 / sz)} * {v};",
        ]
    lines.append("    // pole states, one ddt per statement")
    lines += states
    lines += ["  end", "endmodule"]
    return "\n".join(lines) + "\n"
