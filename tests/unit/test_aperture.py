"""Aperture fields and their angular-spectrum propagation (analytic, no solve)."""

import dataclasses
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
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
        "values-shape",
        "x-length",
        "y-given",
        "y-missing",
        "x-uneven",
        "x-decreasing",
        "frequency-count",
        "frequency-sign",
        "index",
        "distance",
    ],
)
def test_constructor_messages(kw, error, match):
    with pytest.raises(error, match=match):
        _ok(**kw)


@pytest.mark.parametrize("name", ["frequency", "x", "index", "distance"])
def test_constructor_traced_constants(name):
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


# --- exact plane-wave factors ---------------------------------------------------


def test_uniform_field_padding_one():
    n, a, d = 16, 0.4 * UM, 7.3e-6
    for index in (1.0, 1.7):
        f = _field(np.full(n, 0.3 - 0.8j), _centres(n, a), index=index)
        g = dif.propagate(f, d, padding=1.0)
        want = f.values * np.exp(-1j * K0 * index * d)
        np.testing.assert_allclose(g.values, want, atol=1e-12, rtol=0)


def test_grid_plane_wave_and_evanescent_component():
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


# --- 1D lens --------------------------------------------------------------------


def _rs1d(u0, x0, x, z, k):
    a = x0[1] - x0[0]
    r = np.sqrt((x[:, None] - x0[None, :]) ** 2 + z**2)
    return (-1j * k * z / 2) * (hankel2(1, k * r) / r) @ (u0 * a)


@pytest.mark.parametrize("na", [0.1, 0.5, 0.8])
def test_lens_1d(na):
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
    # the documented accuracy: the whole footprint within 1.1e-3 of the peak at
    # padding 2 (1.0e-3 measured at NA 0.1), the focal spot within 4e-5 at padding 3
    foot = _rs1d(u0, x0, x0, f, k)
    assert np.max(np.abs(u - foot)) / np.max(np.abs(foot)) < 1.1e-3
    wide = np.asarray(dif.propagate(field, f * UM, padding=3.0).values[0, 1])
    assert np.max(np.abs(wide[sel] - ref)) / peak < 4e-5


# --- 2D lens --------------------------------------------------------------------


def test_lens_2d():
    lam, a, n, na = 1.0, 0.5, 400, 0.5
    k = 2 * np.pi / lam
    x0 = _centres(n, a) + a / 2
    f = (n * a / 2) / np.tan(np.arcsin(na))
    X, Y = np.meshgrid(x0, x0, indexing="ij")
    u0 = np.exp(1j * k * (np.sqrt(X**2 + Y**2 + f**2) - f))
    # a circular pupil of the same NA
    radius = n * a / 2
    inside = np.hypot(X, Y) <= radius
    u0 = np.where(inside, u0, 0.0)
    field = _field(u0, x0 * UM, x0 * UM, freq=C0 / (lam * UM))
    out = dif.propagate(field, f * UM, padding=2.0)
    r2 = X**2 + Y**2 + f**2
    sampled = np.sum(np.where(inside, k * f * a**2 / (2 * np.pi * r2), 0.0))
    i = int(np.argmin(np.abs(x0)))
    centre = abs(np.asarray(out.values[0, 1])[i, i])
    assert centre == pytest.approx(sampled, rel=5e-4)


# --- deflector ------------------------------------------------------------------


@pytest.mark.parametrize("deg", [10.0, 30.0, 45.0])
def test_deflector(deg):
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
    # the propagated samples peak at the same angle, and the peak coefficient
    # has gained the propagation phase exp(-i kz z)
    got = np.fft.ifft(np.pad(np.asarray(out.values[0, 1]), (0, 3 * n * 7)))
    assert abs(np.degrees(np.arcsin(kx[np.argmax(np.abs(got))] / k)) - deg) < 0.01
    i = int(np.argmax(spec))
    kz = np.sqrt(k**2 - kx[i] ** 2)
    ratio = got[i] / np.fft.ifft(np.pad(u, (0, 3 * n * 7)))[i]
    assert abs(ratio - np.exp(-1j * kz * z)) < 0.05
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


# --- aperture_field: the locally periodic approximation ---------------------------

SQ = np.sqrt(0.5)
#: (polarization, incident face, incident Jones pair as seen in (TM, TE)).
E_IN = [
    ("TE", "z_min", (0, 1)),
    ("TM", "z_min", (1, 0)),
    ("RCP", "z_min", (SQ, -1j * SQ)),
    ("LCP", "z_min", (SQ, 1j * SQ)),
    ("RCP", "z_max", (SQ, 1j * SQ)),
    ((0.6, 0.8j), "z_min", (0.6, 0.8j)),
]


@pytest.mark.parametrize(("pol", "face", "e_in"), E_IN)
def test_a_uniform_map_gives_the_jones_block_times_the_polarization(pol, face, e_in):
    r = pc.analytic_response()
    cols = [0, 1] if face == "z_min" else [2, 3]
    out = "z_max"
    for shape in ((5,), (3, 4)):
        p0 = 0.2731
        f = dif.aperture_field(r, np.full(shape, p0), out, face, polarization=pol)
        j = np.asarray(r(p0).values)[0][2:4][:, cols]
        want = j @ np.asarray(e_in)
        got = np.asarray(f.values)
        assert got.shape == (1, 2, *shape)
        np.testing.assert_allclose(
            got,
            np.broadcast_to(want[None, :, *([None] * len(shape))], got.shape),
            rtol=0,
            atol=1e-14,
        )
        assert f.index == 1.0
        assert f.distance == 0.0
        assert (f.y is None) == (len(shape) == 1)


def test_at_the_knots_the_field_is_the_stored_entries_and_the_amplitude_scales():
    r = pc.analytic_response()
    f = dif.aperture_field(r, r.parameter, "z_max", "z_min", polarization="TE")
    want = np.asarray(r.values)[:, 0, 2:4, 1]  # (n_p, 2): air rows, sub TE
    np.testing.assert_allclose(np.asarray(f.values)[0].T, want, rtol=0, atol=1e-15)
    amp = np.exp(1j * np.arange(len(r.parameter)))
    g = dif.aperture_field(
        r, r.parameter, "z_max", "z_min", polarization="TE", amplitude=amp
    )
    np.testing.assert_allclose(
        np.asarray(g.values), np.asarray(f.values) * amp, rtol=0, atol=1e-15
    )
    # a (n1, n2) map with an amplitude along one axis only
    p2 = np.tile(r.parameter[:, None], (1, 3))
    h = dif.aperture_field(
        r, p2, "z_max", "z_min", polarization="TE", amplitude=amp[:, None]
    )
    np.testing.assert_allclose(
        np.asarray(h.values)[0, :, :, 1], np.asarray(g.values)[0], atol=1e-15
    )
    # positions run from the aperture centre; the pitch is the lattice's
    np.testing.assert_allclose(f.x, _centres(len(r.parameter), 0.5 * UM), atol=1e-18)
    np.testing.assert_allclose(h.y, _centres(3, 0.16 * UM), atol=1e-18)


def test_a_reflection_reads_the_incident_face_block():
    r = pc.analytic_response()
    f = dif.aperture_field(r, np.full(4, 0.31), "z_max", "z_max", polarization="TM")
    want = np.asarray(r(0.31).values)[0][2:4, 2]
    np.testing.assert_allclose(np.asarray(f.values)[0, :, 0], want, atol=1e-14)


def _ap(r, **kw):
    args = {"parameter": [0.3, 0.31], "outgoing": "z_max", "incident": "z_min"}
    args |= {"polarization": "TE"} | kw
    return dif.aperture_field(
        r, args.pop("parameter"), args.pop("outgoing"), args.pop("incident"), **args
    )


def _lat(a2=(0.0, 0.16 * UM, 0.0), a1=(0.5 * UM, 0.0, 0.0)):
    return {"z_min": (a1, a2), "z_max": (a1, a2)}


def test_the_aperture_messages():
    r = pc.analytic_response()
    with pytest.raises(
        TypeError,
        match=(r"^aperture_field needs a gfem\.diffraction\.CellResponse, got str\.$"),
    ):
        dif.aperture_field("x", [0.3], "z_max", "z_min", polarization="TE")
    for shape in ((), (2, 2, 2), (0,), (2, 0), (1,), (1, 3)):
        with pytest.raises(
            ValueError,
            match=(
                r"^aperture_field: the parameter map must have one axis \(cells "
                r"along ê1\) or two \(cells along ê1, ê2\), with at least two "
                r"cells along each, "
                rf"got shape {re.escape(str(shape))}\.$"
            ),
        ):
            _ap(r, parameter=np.full(shape, 0.3))
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: amplitude of shape \(3,\) does not broadcast to the "
            r"parameter map's shape \(2,\)\.$"
        ),
    ):
        _ap(r, amplitude=np.ones(3))
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: 2 of 3 cells have a parameter outside the sampled "
            r"range \[0\.2, 0\.4\] \(min 0\.1, max 0\.5\); the cell response does not "
            r"extrapolate\. Map an unbounded design variable onto the range, e\.g\. "
            r"lo \+ \(hi - lo\) \* jax\.nn\.sigmoid\(x\)\.$"
        ),
    ):
        _ap(r, parameter=[0.1, 0.3, 0.5])
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: the cell response was sampled at oblique incidence "
            r"\(wave vector \(1e\+06, 0, 0\) rad/m\); the aperture field "
            r"assumes normal incidence, where every cell's TM and TE references are "
            r"ê1 and ê2\.$"
        ),
    ):
        _ap(pc.analytic_response(wave_vector=(1e6, 0.0, 0.0)))
    # a Bloch vector is tangential: a component along z is oblique as well
    with pytest.raises(ValueError, match=r"wave vector \(0, 0, 2e\+06\) rad/m"):
        _ap(pc.analytic_response(wave_vector=(0.0, 0.0, 2e6)))
    hexagonal = _lat(
        a1=(0.16 * UM, 0.0, 0.0), a2=(0.08 * UM, 0.08 * math.sqrt(3) * UM, 0.0)
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: the lattice vectors of 'z_max' make 60°; the aperture "
            r"places cells on a rectangular grid along ê1 and ê2\.$"
        ),
    ):
        _ap(pc.analytic_response(lattice=hexagonal))
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: the cells diffract: order \(-1, 0\) propagates behind "
            r"'z_max' at 299\.792 THz, so a cell does not send out one plane wave and "
            r"the "
            r"locally periodic approximation does not hold\. Keep the pitch below the "
            r"wavelength in that half-space \(1e-06 m at 299\.792 THz\)\.$"
        ),
    ):
        _ap(pc.analytic_response(pitch=1.2 * UM, lattice=_lat(a1=(1.2 * UM, 0.0, 0.0))))
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: polarization='unpolarized' is an incoherent mean, "
            r"not one field; build the aperture field for 'TE' and for 'TM', "
            r"propagate both and average their intensities\.$"
        ),
    ):
        _ap(r, polarization="unpolarized")


def test_inherited_messages_name_aperture_field():
    r = pc.analytic_response()
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: 'nope' carries no Floquet port of the result; its "
            r"Floquet surfaces are 'z_min', 'z_max'\.$"
        ),
    ):
        _ap(r, outgoing="nope")
    # the aperture variant: the outgoing order needs both polarizations
    ports = ("sub TM", "sub TE", "air TM")
    one_row = dataclasses.replace(
        r,
        ports=ports,
        values=r.values[:, :, :3],
        derivative=r.derivative[:, :, :3],
        z_ref=np.asarray(r.z_ref)[:, :3],
        offsets=dict.fromkeys(ports, 0.3 * UM),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: no Floquet port reads order \(0, 0\) in TE on "
            r"'z_max' \(an aperture field needs both polarizations of the outgoing "
            r"order\)\. Add gfem\.FloquetPort\(name, 'z_max', order=\(0, 0\), "
            r"polarization='TE'\) to the model\.$"
        ),
    ):
        _ap(one_row)
    # the incident column was not excited
    sub_te = dataclasses.replace(
        r,
        excited=("sub TE",),
        values=r.values[..., 1:2],
        derivative=r.derivative[..., 1:2],
    )
    _ap(sub_te)
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: Floquet port 'sub TM' \(order \(0, 0\), TM on 'z_min'\) "
            r"was not excited, so S has no column for it\."
        ),
    ):
        _ap(sub_te, polarization="RCP")
    # the outgoing TM and TE ports read at two offsets
    skew = pc.analytic_response(
        offsets={
            "sub TM": 0.3 * UM,
            "sub TE": 0.3 * UM,
            "air TM": 0.35 * UM,
            "air TE": 0.3 * UM,
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^aperture_field: Floquet ports 'air TM' and 'air TE' read order "
            r"\(0, 0\) on 'z_max' at different offsets"
        ),
    ):
        _ap(skew)
    # a bad polarization
    with pytest.raises(
        ValueError, match=r"^aperture_field: polarization must be 'TE', 'TM'"
    ):
        _ap(r, polarization="diagonal")
    # a traced polarization pair
    with pytest.raises(
        gfem.NonDifferentiableError,
        match=r"^aperture_field: polarization must be fixed numbers",
    ):
        jax.grad(lambda a: jnp.sum(jnp.abs(_ap(r, polarization=(a, 1.0 + 0j)).values)))(
            1.0
        )


def test_a_traced_map_outside_the_range_is_nan_not_an_error():
    r = pc.analytic_response()
    p = jnp.array([0.25, 0.3, 0.5])

    def field(q):
        return _ap(r, parameter=q).values

    for f in (jax.jit(field), lambda q: jax.vmap(field)(q[None])[0]):
        out = np.asarray(f(p))
        assert np.all(np.isfinite(out[..., :2]))
        assert np.all(np.isnan(out[..., 2]))
    g = jax.grad(lambda q: jnp.sum(jnp.abs(field(q))))(jnp.array([0.25, 0.5]))
    assert np.isnan(np.asarray(g)[1])


def test_jit_and_vmap_over_designs_match_eager():
    r = pc.analytic_response()
    base = jnp.linspace(0.21, 0.39, 6)

    def intensity(p):
        return dif.propagate(_ap(r, parameter=p, polarization="RCP"), 3e-6).intensity[
            0, 2
        ]

    np.testing.assert_allclose(jax.jit(intensity)(base), intensity(base), rtol=1e-12)
    batch = jnp.stack([base, base[::-1]])
    np.testing.assert_allclose(
        jax.vmap(intensity)(batch), [intensity(b) for b in batch], rtol=1e-12
    )
