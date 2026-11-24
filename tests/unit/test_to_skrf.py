"""`NetworkParameters.to_skrf`: S as a scikit-rf `Network` (``skrf`` extra).

Every network is built by hand, on complex per-port references and
frequencies out of order. The expected impedance matrix is the closed form
of a transmission line's ABCD matrix, converted by Z11 = A / C,
Z12 = det / C, Z21 = 1 / C, Z22 = D / C.
"""

import math
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import gfem

rf = pytest.importorskip("skrf")

F = (3e9, 1e9, 2e9)


def _line(f, z0=80.0, eps=4.0, length=0.01):
    """ABCD matrix of a lossless line of impedance ``z0``."""
    beta = 2 * math.pi * f / 299_792_458.0 * math.sqrt(eps)
    c, s = math.cos(beta * length), math.sin(beta * length)
    return np.array([[c, 1j * z0 * s], [1j * s / z0, c]])


def _z_from_abcd(m):
    (a, b), (c, d) = m
    return np.array([[a / c, (a * d - b * c) / c], [1 / c, d / c]])


def _s_from_z(z, z_ref):
    """Pseudo-wave S of an impedance matrix on per-port references ``z_ref``."""
    k = np.diag(np.sqrt(z_ref.real) / np.abs(z_ref))
    g = np.diag(z_ref)
    return k @ (z - g) @ np.linalg.inv(z + g) @ np.linalg.inv(k)


# complex, frequency-dependent references, as a WavePort's Z_PV
Z_REF = np.array([[40 - 5j, 55 + 8j], [42 - 3j, 52 + 6j], [45 - 1j, 50 + 4j]])


@pytest.fixture
def line():
    s = np.array(
        [_s_from_z(_z_from_abcd(_line(f)), z) for f, z in zip(F, Z_REF, strict=True)]
    )
    return gfem.NetworkParameters(
        values=jnp.asarray(s),
        z_ref=jnp.asarray(Z_REF),
        frequency=F,
        ports=("in", "out"),
        excited=("in", "out"),
    )


def test_network_has_the_lines_impedance_on_the_ports_own_references(line):
    nw = line.to_skrf(name="line")
    order = np.argsort(F)
    np.testing.assert_array_equal(nw.f, np.sort(F))
    want = np.array([_z_from_abcd(_line(f)) for f in np.sort(F)])
    np.testing.assert_allclose(nw.z, want, rtol=1e-12)
    np.testing.assert_array_equal(nw.z0, Z_REF[order])
    np.testing.assert_array_equal(nw.s, np.asarray(line.values)[order])
    assert nw.s_def == "pseudo"
    assert nw.name == "line"
    assert list(nw.port_names) == ["in", "out"]


def test_a_sub_network_leaves_an_internal_load_inside(line):
    # p3 has only a capacitance: never excited, no reference, NaN row
    s3 = np.full((3, 3, 2), np.nan, complex)
    s3[:, :2, :] = np.asarray(line.values)
    net = gfem.NetworkParameters(
        values=jnp.asarray(s3),
        z_ref=jnp.asarray(np.column_stack([Z_REF, np.full(3, np.nan)])),
        frequency=F,
        ports=("in", "out", "c"),
        excited=("in", "out"),
    )
    nw = net.to_skrf(ports=["out", "in"])
    assert list(nw.port_names) == ["out", "in"]
    want = np.array([_z_from_abcd(_line(f)) for f in np.sort(F)])
    np.testing.assert_allclose(nw.z, want[:, ::-1, ::-1], rtol=1e-12)
    with pytest.raises(ValueError, match=r"'c' was not excited"):
        net.to_skrf()


def test_a_port_without_a_reference_is_refused(line):
    modal = gfem.NetworkParameters(
        values=line.values,
        z_ref=jnp.full((3, 2), jnp.nan),
        frequency=F,
        ports=line.ports,
        excited=line.excited,
    )
    with pytest.raises(
        ValueError, match=r"Port 'in', 'out' has no reference.*to_touchstone\(z0=\)"
    ):
        modal.to_skrf()


@pytest.mark.parametrize(
    ("ports", "error", "match"),
    [
        (["in", "x"], ValueError, r"No port 'x'; the ports are 'in', 'out'"),
        (["in", "in"], ValueError, r"Port 'in' is listed twice"),
        ([], ValueError, r"selects no port"),
        ("in", TypeError, r"e\.g\. \['in'\]"),
    ],
)
def test_bad_port_selections_are_refused(line, ports, error, match):
    with pytest.raises(error, match=match):
        line.to_skrf(ports=ports)


def test_traced_values_are_refused(line):
    def export(a):
        net = gfem.NetworkParameters(
            values=a * line.values,
            z_ref=line.z_ref,
            frequency=F,
            ports=line.ports,
            excited=line.excited,
        )
        return net.to_skrf().s.sum().real

    with pytest.raises(TypeError, match="outside the transformed function"):
        jax.grad(export)(1.0)


def test_without_scikit_rf_the_error_names_the_extra(line, monkeypatch):
    monkeypatch.setitem(sys.modules, "skrf", None)
    with pytest.raises(ImportError, match=r'pip install "gfem\[skrf\]"'):
        line.to_skrf()
