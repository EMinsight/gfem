"""Network parameters of a piezoelectric result, and the strain of its displacement."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import piezo_cases as pc
import pytest

import gfem
from gfem._field import FieldMap

jax.config.update("jax_enable_x64", True)

FREQUENCIES = (0.85e9, 0.86e9, 0.87e9)


def _result(n_terminals=2, seed=0):
    """A result holding a random symmetric admittance per length."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(3, n_terminals, n_terminals))
    b = rng.normal(size=(3, n_terminals, n_terminals))
    y = (a + 1j * b) * 30.0
    y = 0.5 * (y + np.swapaxes(y, 1, 2))
    names = tuple("ab"[:n_terminals])
    return gfem.PiezoelectricResult(
        admittance_per_length=jnp.asarray(y),
        clamped_capacitance_per_length=jnp.eye(n_terminals) * 1e-10,
        terminals=names,
        frequencies=FREQUENCIES,
    )


def test_the_series_resistance_is_a_matrix_in_the_lead_of_each_terminal():
    result = _result()
    net = result.network_parameters(
        aperture=300e-6, series_resistance=[7.0, 3.0], reference_impedance=50.0
    )
    assert isinstance(net, gfem.NetworkParameters)
    assert net.ports == net.excited == ("a", "b")
    assert net.frequency == FREQUENCIES
    assert net.z_ref.shape == (3, 2)
    y = 300e-6 * np.asarray(result.admittance_per_length)
    r = np.diag([7.0, 3.0])
    expected = np.stack([yf @ np.linalg.inv(np.eye(2) + r @ yf) for yf in y])
    assert np.abs(np.asarray(net.y()) - expected).max() / np.abs(expected).max() < 1e-12


def test_without_resistance_the_network_is_the_admittance_times_the_aperture():
    result = _result()
    net = result.network_parameters(aperture=2e-4)
    y = 2e-4 * np.asarray(result.admittance_per_length)
    assert np.abs(np.asarray(net.y()) - y).max() / np.abs(y).max() < 1e-12
    half = result.network_parameters(aperture=1e-4)
    # the aperture scales Y, so the wave changes but the admittance halves exactly
    assert np.abs(np.asarray(half.y()) - y / 2).max() / np.abs(y).max() < 1e-12


def test_a_one_port_reflection_is_the_load_formula_on_the_reference():
    result = _result(1)
    net = result.network_parameters(aperture=400e-6, series_resistance=7.46)
    y = 400e-6 * np.asarray(result.admittance_per_length)[:, 0, 0]
    load = 1 / y + 7.46
    expected = (load - 50.0) / (load + 50.0)
    got = np.asarray(net["a", "a"])
    assert np.abs(got - expected).max() < 1e-12
    db = np.asarray(net.db("a", "a"))
    assert np.abs(db - 20 * np.log10(np.abs(expected))).max() < 1e-10


def test_a_complex_reference_per_terminal_round_trips_through_renormalize():
    result = _result()
    net = result.network_parameters(
        aperture=2e-4, series_resistance=5.0, reference_impedance=[50.0, 75.0 + 5j]
    )
    y = np.asarray(net.y())
    again = net.renormalize(50.0).renormalize([50.0, 75.0 + 5j])
    assert np.abs(np.asarray(again.y()) - y).max() / np.abs(y).max() < 1e-10


def test_the_network_writes_a_touchstone_file(tmp_path):
    net = _result().network_parameters(aperture=2e-4, series_resistance=5.0)
    path = net.to_touchstone(tmp_path / "idt.s2p")
    assert path.read_text().count("\n") > 3


def test_the_inputs_are_checked():
    result = _result()
    with pytest.raises(ValueError, match="aperture"):
        result.network_parameters(aperture=0.0)
    with pytest.raises(ValueError, match="series_resistance"):
        result.network_parameters(aperture=1e-4, series_resistance=-1.0)
    with pytest.raises(ValueError, match="series_resistance"):
        result.network_parameters(aperture=1e-4, series_resistance=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="reference_impedance"):
        result.network_parameters(aperture=1e-4, reference_impedance=-50.0)


def _s11_db(result, k):
    def f(aperture, resistance):
        net = result.network_parameters(aperture=aperture, series_resistance=resistance)
        return net.db("a", "a")[k]

    return f


@pytest.mark.parametrize("argnum", [0, 1])
def test_the_reflection_differentiates_in_the_aperture_and_the_resistance(argnum):
    result = _result(1)
    f = _s11_db(result, 1)
    x = (400e-6, 7.46)
    grad = jax.grad(f, argnums=argnum)(*x)
    step = 1e-6 * x[argnum]

    def moved(sign):
        a = list(x)
        a[argnum] += sign * step
        return float(f(*a))

    fd = (moved(1) - moved(-1)) / (2 * step)
    assert abs(float(grad) - fd) / abs(fd) < 1e-6


def test_a_two_terminal_reflection_differentiates_in_the_resistance():
    result = _result()

    def f(r):
        net = result.network_parameters(aperture=2e-4, series_resistance=r)
        return jnp.abs(net["a", "b"][2]) ** 2

    r0 = jnp.array([7.0, 3.0])
    grad = jax.grad(f)(r0)
    for i in range(2):
        e = jnp.zeros(2).at[i].set(1e-5)
        fd = (f(r0 + e) - f(r0 - e)) / 2e-5
        assert abs(float(grad[i]) - float(fd)) / abs(float(fd)) < 1e-6


@pytest.fixture(scope="module")
def displacement(tmp_path_factory):
    path = tmp_path_factory.mktemp("strain") / "idt.msh"
    mesh = pc.mesh(path, "lines")
    model = pc.model(mesh, "lines")
    result = gfem.solve(model, pc.problem(keep_fields=True))
    return result.displacement(pc.FREQUENCY)


def _affine(field, gradient, offset=0.0):
    """``field`` with the coefficients of ``offset + gradient . (x, y)`` (SI)."""
    import ngsolve as ngs

    from gfem._engine.fields import fe_space

    built, fes, _ = fe_space(
        field.mesh, field.support, field._excluded, "h1", field.order
    )
    gf = ngs.GridFunction(fes)
    # engine coordinates are in mesh length units, `scale` metres each
    s = built.scale
    gf.Set(offset + s * (gradient[0] * ngs.x + gradient[1] * ngs.y))
    return dataclasses.replace(field, coefficients=jnp.asarray(gf.vec.FV().NumPy()))


def test_the_strain_of_an_affine_displacement_is_exact_with_engineering_shear(
    displacement,
):
    gx, gy, gz = (1e-3, -2e-3), (5e-4, 3e-3), (7e-4, -1e-3)
    fields = {
        "x": _affine(displacement["x"], gx, 1e-9),
        "y": _affine(displacement["y"], gy),
        "z": _affine(displacement["z"], gz, -2e-9),
    }
    moved = FieldMap(fields, label="component")
    points = np.array([[0.7e-6, -1.1e-6], [-2.3e-6, -3.5e-6], [1.9e-6, -0.4e-6]])
    s = np.asarray(gfem.strain(moved, points))
    assert s.shape == (3, 6)
    expected = [gx[0], gy[1], 0.0, gz[1], gz[0], gx[1] + gy[0]]
    assert np.abs(s - expected).max() < 1e-9 * np.abs(expected).max()
    one = gfem.strain(moved, points[0])
    assert one.shape == (6,)


def test_the_strain_of_a_kept_displacement_differentiates_in_its_coefficients(
    displacement,
):
    point = np.array([[0.7e-6, -1.1e-6]])
    zx = displacement["z"]
    c0 = jnp.asarray(zx.coefficients)
    imag = 1j * jnp.imag(c0)

    def f(a):
        moved = FieldMap(
            {
                "x": displacement["x"],
                "y": displacement["y"],
                "z": dataclasses.replace(zx, coefficients=a + imag),
            },
            label="component",
        )
        return jnp.abs(gfem.strain(moved, point)[0, 3]) ** 2

    a = jnp.real(c0)
    grad = jax.grad(f)(a)
    direction = jax.random.normal(jax.random.PRNGKey(0), a.shape)
    h = 1e-2 * jnp.linalg.norm(a) / jnp.linalg.norm(direction)
    fd = (f(a + h * direction) - f(a - h * direction)) / (2 * h)
    assert abs(float(jnp.vdot(grad, direction)) - float(fd)) <= 1e-6 * abs(float(fd))


def test_the_strain_refuses_what_is_not_a_displacement(displacement):
    with pytest.raises(TypeError, match=r"res\.displacement"):
        gfem.strain(displacement["x"], np.zeros((1, 2)))
    with pytest.raises(ValueError, match="outside"):
        gfem.strain(displacement, np.array([[1.0, 1.0]]))
