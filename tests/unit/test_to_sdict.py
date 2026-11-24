"""`NetworkParameters.to_sdict`: S as a SAX dictionary, without importing SAX.

Every network is built by hand. Expected values are entries of those literal
arrays (conjugated for an optical circuit) or closed forms: a load's
pseudo-wave reflection and a transmission line's ABCD matrix.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from circuit_cases import F, line_abcd

import gfem

# S[f, to, from] of a lossy, non-reciprocal 2-port, one matrix per frequency
S2 = np.array(
    [
        [[0.1 + 0.2j, 0.7 - 0.1j], [0.6 + 0.3j, -0.2 + 0.05j]],
        [[0.3 - 0.1j, 0.5 + 0.4j], [0.4 - 0.5j, 0.1 + 0.1j]],
        [[-0.2 + 0.1j, 0.2 - 0.6j], [0.3 + 0.6j, 0.25 - 0.3j]],
    ]
)


def _net(values, z_ref, ports=("p1", "p2"), excited=None):
    values = jnp.asarray(values, complex)
    n_f = values.shape[0]
    return gfem.NetworkParameters(
        values=values,
        z_ref=jnp.asarray(np.broadcast_to(z_ref, (n_f, len(ports)))),
        frequency=F[:n_f],
        ports=ports,
        excited=ports if excited is None else excited,
    )


# --- keys and values ---------------------------------------------------------------


def test_optical_gives_every_pair_conjugated_keyed_input_then_output():
    net = _net(S2, np.nan)
    sd = net.to_sdict(circuit="optical", ports={"p1": "o1", "p2": "o2"})
    assert list(sd) == [("o1", "o1"), ("o1", "o2"), ("o2", "o1"), ("o2", "o2")]
    # (input, output): sd["o1", "o2"] is S21, the wave out of p2 fed at p1
    np.testing.assert_array_equal(sd["o1", "o2"], np.conj(S2[:, 1, 0]))
    np.testing.assert_array_equal(sd["o2", "o1"], np.conj(S2[:, 0, 1]))
    np.testing.assert_array_equal(sd["o1", "o1"], np.conj(S2[:, 0, 0]))
    np.testing.assert_array_equal(sd["o2", "o2"], np.conj(S2[:, 1, 1]))
    assert all(isinstance(v, jax.Array) and v.shape == (3,) for v in sd.values())


def _s_from_abcd(m, r):
    """S of an ABCD matrix on a real reference ``r`` at both ports."""
    (a, b), (c, d) = m
    den = a + b / r + c * r + d
    return np.array(
        [
            [(a + b / r - c * r - d) / den, 2 * (a * d - b * c) / den],
            [2 / den, (-a + b / r - c * r + d) / den],
        ]
    )


def test_rf_gives_s_on_50_ohm_by_default_unconjugated():
    # the line is solved on 30 ohm ports; the circuit sees it on 50 ohm
    net = _net([_s_from_abcd(line_abcd(f), 30.0) for f in F], 30.0)
    sd = net.to_sdict(circuit="rf", ports={"p1": "in", "p2": "out"})
    want = np.array([_s_from_abcd(line_abcd(f), 50.0) for f in F])
    np.testing.assert_allclose(sd["in", "out"], want[:, 1, 0], rtol=1e-12)
    np.testing.assert_allclose(sd["in", "in"], want[:, 0, 0], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sd["out", "in"], want[:, 0, 1], rtol=1e-12)
    np.testing.assert_allclose(sd["out", "out"], want[:, 1, 1], rtol=1e-12, atol=1e-15)


def test_rf_renormalizes_to_z0_from_each_ports_own_complex_reference():
    # a load's pseudo-wave reflection on reference z: (Z_L - z) / (Z_L + z)
    loads = np.array([20.0 + 5j, 75.0 - 30j, 1e3 + 0j])
    z_ref = np.array([30.0 + 10j, 45.0 - 5j, 60.0 + 0j])
    net = _net(
        ((loads - z_ref) / (loads + z_ref)).reshape(-1, 1, 1), z_ref[:, None], ("p",)
    )
    sd = net.to_sdict(circuit="rf", ports={"p": "o1"}, z0=75.0)
    np.testing.assert_allclose(
        sd["o1", "o1"], (loads - 75.0) / (loads + 75.0), rtol=1e-12
    )


def test_rf_takes_the_exposed_sub_network_before_renormalizing():
    # p3 is an internal lumped load: never excited, no reference (NaN row)
    s3 = np.zeros((3, 3, 2), complex)
    s3[:, :2, :] = S2
    s3[:, 2, :] = np.nan
    z_ref = np.array([25.0, 40.0, np.nan])
    net = _net(s3, z_ref, ports=("p1", "p2", "p3"), excited=("p1", "p2"))
    sd = net.to_sdict(circuit="rf", ports={"p2": "b", "p1": "a"}, z0=50.0)
    # the sub-matrix with p3 terminated is the network of p1 and p2
    sub = _net(S2, z_ref[:2]).renormalize(50.0)
    assert list(sd) == [("b", "b"), ("b", "a"), ("a", "b"), ("a", "a")]
    for (c_in, p_in), (c_out, p_out) in [
        (("a", "p1"), ("b", "p2")),
        (("b", "p2"), ("a", "p1")),
        (("a", "p1"), ("a", "p1")),
        (("b", "p2"), ("b", "p2")),
    ]:
        np.testing.assert_allclose(sd[c_in, c_out], sub[p_out, p_in], rtol=1e-12)


def test_ports_none_exposes_every_port_under_its_own_name():
    sd = _net(S2, np.nan).to_sdict(circuit="optical")
    assert list(sd) == [("p1", "p1"), ("p1", "p2"), ("p2", "p1"), ("p2", "p2")]
    np.testing.assert_array_equal(sd["p1", "p2"], np.conj(S2[:, 1, 0]))


def test_multimode_circuit_names_are_kept():
    ports = {"p1": "o1@TE0", "p2": "o1@TM0"}
    sd = _net(S2, np.nan).to_sdict(circuit="optical", ports=ports)
    np.testing.assert_array_equal(sd["o1@TE0", "o1@TM0"], np.conj(S2[:, 1, 0]))


# --- refusals ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ports", "match"),
    [
        ({"p3": "o1"}, r"No port 'p3'; the ports are 'p1', 'p2'"),
        ({"p1": "o-1"}, r"'o-1' is not an identifier or name@mode"),
        ({"p1": "o1@"}, r"'o1@' is not an identifier or name@mode"),
        ({"p1": "o1", "p2": "o1"}, r"'o1' is given to more than one port"),
        ({"p1": "o1@TE0", "p2": "o2"}, r"every circuit port name is name@mode or none"),
        ({}, r"exposes no port"),
    ],
)
def test_bad_port_names_are_refused(ports, match):
    with pytest.raises(ValueError, match=match):
        _net(S2, np.nan).to_sdict(circuit="optical", ports=ports)


def test_a_port_name_sax_cannot_use_asks_for_ports():
    net = _net(S2, np.nan, ports=("port-1", "p2"))
    with pytest.raises(ValueError, match=r"ports=\{'port-1': 'o1'\}"):
        net.to_sdict(circuit="optical")


def test_ports_is_a_mapping():
    with pytest.raises(TypeError, match=r"\{'p1': 'o1'\}; got list"):
        _net(S2, np.nan).to_sdict(circuit="optical", ports=["p1", "p2"])


def test_an_exposed_port_must_have_been_excited():
    net = _net(S2[:, :, :1], np.nan, excited=("p1",))
    with pytest.raises(
        ValueError, match=r"'p2' was not excited.*Driven\(excite=None\)"
    ):
        net.to_sdict(circuit="optical")
    sd = net.to_sdict(circuit="optical", ports={"p1": "o1"})
    np.testing.assert_array_equal(sd["o1", "o1"], np.conj(S2[:, 0, 0]))


def test_circuit_is_optical_or_rf():
    with pytest.raises(ValueError, match="circuit must be 'optical' or 'rf'"):
        _net(S2, 50.0).to_sdict(circuit="photonic")


def test_optical_refuses_z0():
    with pytest.raises(ValueError, match="drop z0= or use circuit='rf'"):
        _net(S2, 50.0).to_sdict(circuit="optical", z0=50.0)


@pytest.mark.parametrize(
    "z0", [50.0 + 1j, [50.0, 50.0], 0.0, -50.0, np.nan, np.inf, "50", True]
)
def test_rf_z0_is_one_real_positive_value(z0):
    with pytest.raises(ValueError, match="z0 must be one real reference impedance > 0"):
        _net(S2, 50.0).to_sdict(circuit="rf", z0=z0)


def test_rf_refuses_an_exposed_port_without_a_reference():
    net = _net(S2, np.array([50.0, np.nan]))
    with pytest.raises(
        ValueError, match=r"Port 'p2' has no reference impedance.*ports="
    ):
        net.to_sdict(circuit="rf")
    # the same network is fine for an optical circuit
    net.to_sdict(circuit="optical")


def test_rf_refuses_a_traced_complex_or_vector_z0():
    net = _net(S2, 50.0)
    with pytest.raises(ValueError, match="z0 must be one real"):
        jax.jit(lambda z: net.to_sdict(circuit="rf", z0=z)["p1", "p2"])(50.0 + 0j)
    with pytest.raises(ValueError, match="z0 must be one real"):
        jax.jit(lambda z: net.to_sdict(circuit="rf", z0=z)["p1", "p2"])(jnp.ones(2))


# --- gradients ---------------------------------------------------------------------


def test_grad_through_rf_reaches_z0_and_s():
    # S11 of a load on z0 is (Z_L - z0) / (Z_L + z0),
    # so d S11 / d z0 = -2 Z_L / (Z_L + z0)^2
    load = 20.0 + 5j
    net = _net(np.reshape((load - 30.0) / (load + 30.0), (1, 1, 1)), [[30.0]], ("p",))

    def reflection(z0):
        return net.to_sdict(circuit="rf", ports={"p": "o1"}, z0=z0)["o1", "o1"][0]

    want = -2 * load / (load + 60.0) ** 2
    assert jax.grad(lambda z: jnp.real(reflection(z)))(60.0) == pytest.approx(want.real)
    assert jax.jit(jax.grad(lambda z: jnp.imag(reflection(z))))(60.0) == pytest.approx(
        want.imag
    )


def test_grad_through_optical_conjugates_s():
    # sd = conj(a S), so d Im(sd) / d a = -Im(S)
    def imag_s21(a):
        net = _net(a * jnp.asarray(S2), np.nan)
        return jnp.imag(net.to_sdict(circuit="optical")["p1", "p2"][0])

    assert jax.grad(imag_s21)(1.5) == pytest.approx(-S2[0, 1, 0].imag)
