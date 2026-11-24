"""Aperture fields and their angular-spectrum propagation (analytic, no solve)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import hankel2

import gfem
from gfem import diffraction as dif

jax.config.update("jax_enable_x64", True)

C0 = 299792458.0
UM = 1e-6
LAM = 1.0 * UM
F0 = C0 / LAM
K0 = 2 * np.pi / LAM


def _centres(n, a):
    return (np.arange(n) - (n - 1) / 2) * a


def _field(u, x, y=None, *, freq=F0, index=1.0, comp=1):
    """Analytic one-component field (component ``comp``) as an ApertureField."""
    u = np.asarray(u, dtype=complex)
    values = np.zeros((1, 2, *u.shape), dtype=complex)
    values[0, comp] = u
    return dif.ApertureField(values, [freq], x, y, index, 0.0)


# --- constructor and propagate messages ---------------------------------------------


def _ok(**kw):
    args = {
        "values": np.ones((1, 2, 4), complex),
        "frequency": [F0],
        "x": _centres(4, 0.5 * UM),
        "y": None,
    }
    args.update(kw)
    return dif.ApertureField(**args)


@pytest.mark.parametrize(
    ("kw", "error", "match"),
    [
        (
            {"values": np.ones((1, 3, 4))},
            ValueError,
            (
                r"ApertureField: values must have shape \(n_freq, 2, n1\) or "
                r"\(n_freq, 2, n1, n2\), got \(1, 3, 4\)\."
            ),
        ),
        (
            {"x": _centres(5, 0.5 * UM)},
            ValueError,
            (
                r"values of shape \(1, 2, 4\) need x of length 4 and y=None, got x of "
                r"length 5 and y=None\."
            ),
        ),
        (
            {"y": _centres(3, 0.5 * UM)},
            ValueError,
            (
                r"values of shape \(1, 2, 4\) need x of length 4 and y=None, got x of "
                r"length 4 and y of length 3\."
            ),
        ),
        (
            {"values": np.ones((1, 2, 4, 3))},
            ValueError,
            (
                r"values of shape \(1, 2, 4, 3\) need x of length 4 and y of length 3, "
                r"got x of length 4 and y=None\."
            ),
        ),
        (
            {"x": np.array([0.0, 1.0, 3.0, 4.0]) * UM},
            ValueError,
            (
                r"x must be evenly spaced and increasing .* spacings range over "
                r"1e-06 to 2e-06 m\."
            ),
        ),
        (
            {"x": _centres(4, 0.5 * UM)[::-1]},
            ValueError,
            r"x must be evenly spaced and increasing",
        ),
        (
            {"frequency": [F0, F0]},
            ValueError,
            (
                r"frequency must hold 1 frequencies > 0 Hz, one per leading entry of "
                r"values"
            ),
        ),
        ({"frequency": [-1.0]}, ValueError, r"frequency must hold 1 frequencies"),
        (
            {"index": 0.0},
            ValueError,
            r"ApertureField: index must be a finite number > 0, got 0\.0\.",
        ),
        (
            {"distance": -1e-6},
            ValueError,
            r"ApertureField: distance must be a finite length >= 0 in m, got -1e-06\.",
        ),
    ],
    ids=[
        "c1",
        "c2-x",
        "c2-y-given",
        "c2-y-missing",
        "c3-uneven",
        "c3-decreasing",
        "c4-count",
        "c4-sign",
        "c5",
        "c6",
    ],
)
def test_constructor_messages(kw, error, match):
    with pytest.raises(error, match=match):
        _ok(**kw)


@pytest.mark.parametrize("name", ["frequency", "x", "index", "distance"])
def test_constructor_c7_traced_constants(name):
    base = {
        "values": np.ones((1, 2, 4), complex),
        "frequency": [F0],
        "x": _centres(4, 0.5 * UM),
        "index": 1.0,
        "distance": 0.0,
    }

    def build(t):
        kw = dict(base)
        kw[name] = np.asarray(kw[name]) * t
        dif.ApertureField(**kw)
        return 0.0

    with pytest.raises(
        gfem.NonDifferentiableError,
        match=rf"ApertureField: {name} must be fixed numbers: sample positions, "
        r"frequencies, index and distance are constants of the field\. Build them "
        r"outside jax\.grad, jax\.jit or jax\.vmap\.",
    ):
        jax.grad(build)(1.0)


def test_propagate_messages():
    field = _ok()
    with pytest.raises(
        TypeError,
        match=r"propagate needs a gfem\.diffraction\.ApertureField, got ndarray\.",
    ):
        dif.propagate(np.ones(3), 1e-6)
    for bad in (-1.0, float("nan"), "x"):
        with pytest.raises(
            ValueError,
            match=r"propagate: distance must be a finite length >= 0 in m, got ",
        ):
            dif.propagate(field, bad)
    for bad in (0.5, float("inf")):
        with pytest.raises(
            ValueError,
            match=r"propagate: padding must be a finite number >= 1 \(the FFT window "
            r"as a multiple of the aperture's width\), got ",
        ):
            dif.propagate(field, 1e-6, padding=bad)
    for fn in (
        lambda t: dif.propagate(field, t),
        lambda t: dif.propagate(field, 1e-6, padding=t),
    ):
        with pytest.raises(
            gfem.NonDifferentiableError,
            match=r"propagate: distance and padding must be fixed numbers: the "
            r"transfer function is built once, outside the trace\. Build them "
            r"outside jax\.grad, jax\.jit or jax\.vmap\.",
        ):
            jax.grad(lambda t, fn=fn: fn(t).intensity.sum())(2.0)


def test_intensity_and_distance_bookkeeping():
    x = _centres(8, 0.5 * UM)
    f = dif.ApertureField(
        np.stack([np.full((1, 8), 1 + 1j), np.full((1, 8), 2.0)], axis=1),
        [F0],
        x,
        index=1.5,
        distance=1e-6,
    )
    assert f.intensity.shape == (1, 8)
    np.testing.assert_allclose(f.intensity, 2.0 + 4.0)
    g = dif.propagate(f, 3e-6)
    assert g.distance == pytest.approx(4e-6)
    assert g.index == 1.5
    assert g.values.shape == f.values.shape


# --- L2: exact plane-wave factors ---------------------------------------------------


def test_l2_uniform_field_padding_one():
    n, a, d = 16, 0.4 * UM, 7.3e-6
    for index in (1.0, 1.7):
        f = _field(np.full(n, 0.3 - 0.8j), _centres(n, a), index=index)
        g = dif.propagate(f, d, padding=1.0)
        want = f.values * np.exp(-1j * K0 * index * d)
        np.testing.assert_allclose(g.values, want, atol=1e-12, rtol=0)


def test_l2_grid_plane_wave_and_evanescent_component():
    n, a, d = 40, 0.25 * UM, 2.1e-6
    x = _centres(n, a)
    big = n  # padding 1: the window is the aperture
    for m, kind in ((3, "propagating"), (12, "evanescent")):
        q = 2 * np.pi * m / (big * a)
        k = K0
        if kind == "propagating":
            assert q < k
            kz = np.sqrt(k**2 - q**2)
            want_factor = np.exp(-1j * kz * d)
        else:
            assert q > k
            want_factor = np.exp(-np.sqrt(q**2 - k**2) * d)
        u = np.exp(-1j * q * x)
        g = dif.propagate(_field(u, x), d, padding=1.0)
        np.testing.assert_allclose(g.values[0, 1], u * want_factor, atol=1e-12, rtol=0)


# --- L3: 1D lens --------------------------------------------------------------------


def _rs1d(u0, x0, x, z, k):
    a = x0[1] - x0[0]
    r = np.sqrt((x[:, None] - x0[None, :]) ** 2 + z**2)
    return (-1j * k * z / 2) * (hankel2(1, k * r) / r) @ (u0 * a)


@pytest.mark.parametrize("na", [0.1, 0.5, 0.8])
def test_l3_lens_1d(na):
    lam, a, n = 1.0, 0.5, 2000  # um
    k = 2 * np.pi / lam
    # a grid with a sample on the axis: the focal amplitude is read there
    x0 = _centres(n, a) + a / 2
    half = n * a / 2
    f = half / np.tan(np.arcsin(na))
    u0 = np.exp(1j * k * (np.sqrt(x0**2 + f**2) - f))
    field = _field(u0, x0 * UM, freq=C0 / (lam * UM))
    out = dif.propagate(field, f * UM, padding=2.0)
    u = np.asarray(out.values[0, 1])
    # focal amplitude against the stationary-phase closed form
    closed = np.sqrt(1 / lam) * np.sum(a * f * (x0**2 + f**2) ** -0.75)
    i0 = int(np.argmin(np.abs(x0)))
    assert x0[i0] == 0.0
    centre = abs(u[i0])
    assert centre == pytest.approx(closed, rel=5e-4)
    # focal spot against the Rayleigh-Sommerfeld sum of the samples
    sel = np.arange(i0 - 20, i0 + 21)
    ref = _rs1d(u0, x0, x0[sel], f, k)
    peak = np.max(np.abs(ref))
    assert np.max(np.abs(u[sel] - ref)) / peak < 2e-4


# --- L4: 2D lens --------------------------------------------------------------------


def test_l4_lens_2d():
    lam, a, n, na = 1.0, 0.5, 400, 0.5
    k = 2 * np.pi / lam
    x0 = _centres(n, a) + a / 2
    f = (n * a / 2) / np.tan(np.arcsin(na))
    X, Y = np.meshgrid(x0, x0, indexing="ij")
    u0 = np.exp(1j * k * (np.sqrt(X**2 + Y**2 + f**2) - f))
    # a circular pupil of the same NA
    inside = X**2 + Y**2 <= (n * a / 2) ** 2
    u0 = np.where(inside, u0, 0.0)
    field = _field(u0, x0 * UM, x0 * UM, freq=C0 / (lam * UM))
    out = dif.propagate(field, f * UM, padding=2.0)
    r2 = X**2 + Y**2 + f**2
    sampled = np.sum(np.where(inside, k * f * a**2 / (2 * np.pi * r2), 0.0))
    i = int(np.argmin(np.abs(x0)))
    centre = abs(np.asarray(out.values[0, 1])[i, i])
    assert centre == pytest.approx(sampled, rel=5e-4)


# --- L5: deflector ------------------------------------------------------------------


@pytest.mark.parametrize("deg", [10.0, 30.0, 45.0])
def test_l5_deflector(deg):
    th = np.radians(deg)
    a, n, z = 0.5, 1000, 300.0  # um; D = 500 um
    k = K0 * UM
    xa = _centres(n, a)
    u = np.zeros(3 * n, complex)
    u[n : 2 * n] = np.exp(-1j * k * np.sin(th) * xa)
    x = _centres(3 * n, a)
    out = dif.propagate(_field(u, x * UM), z * UM, padding=2.0)
    inten = np.abs(np.asarray(out.values[0, 1])) ** 2
    # the angular spectrum of the samples peaks at asin(lambda / Lambda)
    kx = 2 * np.pi * np.fft.fftfreq(3 * n * 8, a)
    spec = np.abs(np.fft.ifft(np.pad(u, (0, 3 * n * 7))))
    peak = np.degrees(np.arcsin(kx[np.argmax(spec)] / k))
    assert abs(peak - deg) < 0.01
    # +z tan(theta): the sign pins the Floquet convention (exp(-i kx x))
    centroid = np.sum(x * inten) / np.sum(inten)
    assert centroid == pytest.approx(z * np.tan(th), rel=1e-3)


# --- smoke: jit, grad and vmap -------------------------------------------------------


def test_propagate_jit_grad_vmap():
    n, a = 32, 0.4 * UM
    x = _centres(n, a)
    base = _field(np.exp(1j * np.arange(n) / 5.0), x)

    def focal(v):
        f = dif.ApertureField(v, base.frequency, base.x, None, 1.0, 0.0)
        return dif.propagate(f, 5e-6).intensity[0, n // 2]

    v = base.values
    g = jax.grad(focal)(v)
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(jax.jit(focal)(v), focal(v), rtol=1e-12)
    batch = jnp.stack([v, 2 * v, 3 * v])
    np.testing.assert_allclose(
        jax.vmap(focal)(batch), [focal(w) for w in batch], rtol=1e-12
    )
    # a pytree flatten round trip
    leaves, tree = jax.tree.flatten(base)
    assert len(leaves) == 1
    assert jax.tree.unflatten(tree, leaves).distance == 0.0
