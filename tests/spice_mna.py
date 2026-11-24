r"""A small AC solver for SPICE netlists of R, C, V, G and F cards.

Test-only: it reads the subcircuit a `gfem.RationalModel` writes, so its
realization is checked on every platform without a simulator. Modified nodal
analysis with SPICE's sign conventions: a V source's current flows from its
``+`` node through the source to its ``-`` node; a G source
``g n+ n- c+ c- gm`` drives :math:`g_m V(c_+, c_-)` from ``n+`` through the
source to ``n-``, and an F source ``f n+ n- vname k`` drives
:math:`k\, I(\mathrm{vname})` the same way. Values are plain numbers: a SI
suffix fails ``float``. Any other card fails.
"""

import math

import numpy as np

GROUND = ("0", "gnd")


def subcircuit(text):
    """``(pins, cards)`` of the single ``.subckt`` in ``text``.

    Full-line ``*`` comments and blank lines are skipped, ``+`` lines continue
    the previous card.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            lines[-1] += " " + line[1:]
        else:
            lines.append(line)
    head, *cards, tail = (line.split() for line in lines)
    assert head[0] == ".subckt", head
    assert tail[0] == ".ends", tail
    return head[2:], cards


def s_parameters(text, frequency, z_ref):
    r"""S ``(n_freq, n, n)`` of the subcircuit at its pin pairs ``<p>``, ``<p>_ref``.

    Each ``*_ref`` pin is grounded. Port ``j`` is driven by
    :math:`2\sqrt{z_j}` behind :math:`z_j`, the others are terminated in
    :math:`z_i`, and :math:`S_{ij} = V_i / \sqrt{z_i} - \delta_{ij}`.
    """
    pins, cards = subcircuit(text)
    ports = pins[0::2]
    ground = {*GROUND, *pins[1::2]}
    n = len(ports)
    out = np.empty((len(frequency), n, n), complex)
    for j in range(n):
        extra = [
            ["rterm", p, "0", repr(float(z_ref[i]))]
            for i, p in enumerate(ports)
            if i != j
        ]
        sz = math.sqrt(z_ref[j])
        extra += [
            ["vdrive", "drive_", "0", "ac", repr(float(2 * sz))],
            ["rdrive", "drive_", ports[j], repr(float(z_ref[j]))],
        ]
        v = solve(cards + extra, frequency, ground)
        for i, p in enumerate(ports):
            out[:, i, j] = v[p] / math.sqrt(z_ref[i]) - (i == j)
    return out


def solve(cards, frequency, ground=GROUND):
    """Node voltages ``{node: (n_freq,)}`` of an AC analysis of ``cards``."""
    nodes = {}
    for card in cards:
        for node in card[1:3] + (card[3:5] if card[0][0] == "g" else []):
            if node not in ground:
                nodes.setdefault(node, len(nodes))
    sources = {
        c[0]: len(nodes) + k for k, c in enumerate(c for c in cards if c[0][0] == "v")
    }
    size = len(nodes) + len(sources)
    g = np.zeros((size, size))
    c = np.zeros((size, size))
    rhs = np.zeros(size, complex)

    def at(node):
        return nodes.get(node)

    def stamp(m, rows, cols, value):
        # current ``value * x[col]`` leaving rows[0], entering rows[1]
        for r, sr in zip(rows, (1, -1), strict=True):
            for col, sc in zip(cols, (1, -1), strict=True):
                if at(r) is not None and col is not None:
                    m[at(r), col] += sr * sc * value

    for card in cards:
        kind, a, b = card[0][0], card[1], card[2]
        if kind in "rc":
            value = float(card[3])
            m, y = (g, 1 / value) if kind == "r" else (c, value)
            stamp(m, (a, b), (at(a), at(b)), y)
        elif kind == "v":
            k = sources[card[0]]
            stamp(g, (a, b), (k, None), 1.0)
            for node, sign in ((a, 1), (b, -1)):
                if at(node) is not None:
                    g[k, at(node)] += sign
            words = card[3:]
            rhs[k] = float(words[words.index("ac") + 1]) if "ac" in words else 0.0
        elif kind == "g":
            stamp(g, (a, b), (at(card[3]), at(card[4])), float(card[5]))
        elif kind == "f":
            stamp(g, (a, b), (sources[card[3]], None), float(card[4]))
        else:
            raise ValueError(f"unsupported card {card[0]!r}")
    s = 2j * math.pi * np.asarray(frequency, float)
    a = g[None] + s[:, None, None] * c[None]
    x = np.linalg.solve(a, np.broadcast_to(rhs[:, None], (len(s), size, 1)))[..., 0]
    return {node: x[:, k] for node, k in nodes.items()}
