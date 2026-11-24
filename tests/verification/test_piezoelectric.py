r"""Piezoelectric plates, a finger array and a damping frame against closed forms.

All sections are 2D, solved through the public surface (`gfem.Piezoelectric`),
and reach the closed forms without periodic pairs.

- Thickness-extensional and thickness-shear plates of a hexagonal (6mm)
  crystal, electrodes on the two faces, free side edges: the one-dimensional
  thickness solution satisfies the side edges exactly, so the admittance is
  :math:`Y' = i\omega C_0/(1 - k^2\tan X/X)`, with :math:`X = \omega h/(2v)`,
  :math:`\bar c = c + e^2/\varepsilon`, :math:`v = \sqrt{\bar c/\rho}`,
  :math:`k^2 = e^2/(\varepsilon\bar c)` and :math:`C_0 = \varepsilon w/h` (the
  thickness-extensional crystal has :math:`c_{13} = e_{31} = 0`). At order 3,
  lossless and with :math:`Q_m = 200`, at :math:`f/f_a` = 0.05, 0.3, 0.9 and
  1.05 within :math:`10^{-6}` (:math:`5\times10^{-8}` measured). At order 2
  the error falls at least tenfold from :math:`h = 0.25` to 0.125 µm
  (about 20 measured). The clamped capacitance is :math:`C_0` to
  :math:`10^{-12}`.
- The thickness-shear plate's fields at :math:`f/f_a = 0.3`: :math:`u_z(y)` at
  points and :math:`S_{yz} = \partial_y u_z`, read by differencing the
  displacement field inside one element; the other strains vanish. Within
  :math:`10^{-6}`, :math:`10^{-5}` and :math:`10^{-5}` of the amplitude
  (:math:`3\times10^{-8}`, :math:`1.5\times10^{-6}`, :math:`4\times10^{-7}`
  measured).
- A finite array of line electrodes of equal width on a dielectric under air:
  the capacitance gained by two more fingers is that of the interior ones,
  :math:`\varepsilon_0(\varepsilon_a + \varepsilon_b) K(\sin\frac{\pi\eta}{2}) /
  K(\cos\frac{\pi\eta}{2})` per unit length of the array, within
  :math:`3\times10^{-3}` (:math:`1.3\times10^{-3}` measured: the field is
  singular at the electrode edges).
- A damping frame terminating a shear-wave strip driven by a piezoelectric
  block: the reflection :math:`|R|` of the frame, fitted from the two
  counter-propagating waves in front of it, is within a factor 2 of
  :math:`e^{-2\pi\eta (L/\lambda)/3} + 3.4\times10^{-3}\,\eta(\lambda/L)^2` from
  four wavelengths on, and below :math:`10^{-3}` at four wavelengths for the
  default :math:`\eta = 1.2` (an observation: the law is the frame's design
  rule, not a convergence result).

Runtime about 45 s on one thread (plates 6 s, fingers 18 s, frame 17 s).
"""

import functools

import numpy as np
import piezo_cases as pc
import pytest
from scipy.special import ellipk

import gfem

PLATE_FRACTIONS = (0.05, 0.3, 0.9, 1.05)
QUALITIES = (None, 200.0)
THICKNESS = pc.PLATE_THICKNESS * 1e-6


def _options(order):
    return gfem.SolverOptions(order=order, linear_solver="direct")


@pytest.fixture(scope="module")
def plate_errors(tmp_path_factory):
    """``errors(kind, order, maxh, quality)`` -> (C0 error, error per fraction)."""
    directory = tmp_path_factory.mktemp("plates")

    @functools.cache
    def errors(kind, order, maxh, quality):
        form = pc.plate_form(kind, quality)
        fa = np.real(form["v"]) / (2 * THICKNESS)
        mesh = pc.plate_mesh(directory / f"{kind}_{maxh}.msh", maxh)
        model = pc.plate_model(mesh, kind, quality=quality)
        problem = pc.plate_problem([f * fa for f in PLATE_FRACTIONS])
        result = gfem.solve(model, problem, options=_options(order))
        c0 = float(result.clamped_capacitance_per_length[0, 0])
        out = []
        for k, fraction in enumerate(PLATE_FRACTIONS):
            omega = 2 * np.pi * fraction * fa
            x = omega * THICKNESS / (2 * form["v"])
            exact = 1j * omega * form["c0"] / (1 - form["k2"] * np.tan(x) / x)
            y = complex(result.admittance_per_length[k, 0, 0])
            out.append(abs(y - exact) / abs(exact))
        return abs(c0 - form["c0"]) / form["c0"], tuple(out)

    return errors


@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("kind", ["TE", "TS"])
def test_a_plate_admittance_follows_its_closed_form(plate_errors, kind, quality):
    c0_error, errors = plate_errors(kind, 3, 0.125, quality)
    assert c0_error < 1e-12
    assert max(errors) < 1e-6, dict(zip(PLATE_FRACTIONS, errors, strict=True))


@pytest.mark.parametrize("kind", ["TE", "TS"])
def test_a_plate_admittance_converges_at_fourth_order(plate_errors, kind):
    _, coarse = plate_errors(kind, 2, 0.25, 200.0)
    _, fine = plate_errors(kind, 2, 0.125, 200.0)
    for k, fraction in enumerate(PLATE_FRACTIONS):
        if fraction >= 0.9:
            assert coarse[k] / fine[k] >= 10, (fraction, coarse[k], fine[k])


#: Points (x, y) in micrometres across the thickness-shear plate.
FIELD_POINTS = ((0.37, 0.11), (1.21, 0.5), (0.9, 0.83), (1.63, 0.97))
#: Differencing step (m): well inside one element (about 0.1 um).
STEP = 1e-9


def _derivative(field, x, y, axis):
    """The fourth-order central difference of a scalar field at a point, per m."""
    offsets = np.array([-2.0, -1.0, 1.0, 2.0]) * STEP
    weights = np.array([1.0, -8.0, 8.0, -1.0]) / (12 * STEP)
    shift = np.zeros((4, 2))
    shift[:, axis] = offsets
    points = np.array([x, y]) * 1e-6 + shift
    return complex(weights @ np.asarray(field.at(points)))


def test_the_shear_plate_fields_follow_the_thickness_solution(tmp_path):
    form = pc.plate_form("TS", 200.0)
    frequency = 0.3 * np.real(form["v"]) / (2 * THICKNESS)
    mesh = pc.plate_mesh(tmp_path / "plate.msh", 0.125)
    model = pc.plate_model(mesh, "TS", quality=200.0)
    problem = pc.plate_problem(frequency, keep_fields=True)
    result = gfem.solve(model, problem, options=_options(3))
    u = result.displacement(frequency)

    omega = 2 * np.pi * frequency
    k = omega * np.sqrt(pc.HEX["rho"] / form["cbar"])
    x = k * THICKNESS / 2
    d0 = -(form["eps"] / THICKNESS) / (1 - form["k2"] * np.tan(x) / x)
    amplitude = form["e"] * d0 / (form["eps"] * form["cbar"])  # S_yz, per V
    errors = np.zeros(3)
    for px, py in FIELD_POINTS:
        phase = k * (py * 1e-6 - THICKNESS / 2)
        u_exact = amplitude / k * np.sin(phase) / np.cos(x)
        s_exact = amplitude * np.cos(phase) / np.cos(x)
        value = complex(u["z"].at(np.array([[px, py]]) * 1e-6)[0])
        errors[0] = max(errors[0], abs(value - u_exact) / abs(amplitude / k))
        errors[1] = max(
            errors[1], abs(_derivative(u["z"], px, py, 1) - s_exact) / abs(amplitude)
        )
        other = (
            _derivative(u["x"], px, py, 0),
            _derivative(u["y"], px, py, 1),
            _derivative(u["x"], px, py, 1) + _derivative(u["y"], px, py, 0),
            _derivative(u["z"], px, py, 0),
        )
        errors[2] = max(errors[2], max(abs(v) for v in other) / abs(amplitude))
    assert errors[0] < 1e-6, errors
    assert errors[1] < 1e-5, errors
    assert errors[2] < 1e-5, errors


def test_a_finger_array_adds_the_capacitance_of_its_interior_fingers(tmp_path):
    eta, eps_a, eps_b = 0.5, 1.0, 44.0
    exact = (
        pc.EPS0
        * (eps_a + eps_b)
        * ellipk(np.sin(np.pi * eta / 2) ** 2)
        / ellipk(np.cos(np.pi * eta / 2) ** 2)
    )
    capacitance = []
    for n in (10, 12):
        mesh = pc.finger_array_mesh(tmp_path / f"{n}.msh", n, eta)
        model = pc.finger_array_model(mesh, eps_a, eps_b)
        problem = gfem.Piezoelectric(1e8, terminals=["hot"], ground=["gnd"])
        result = gfem.solve(model, problem, options=_options(3))
        capacitance.append(float(result.clamped_capacitance_per_length[0, 0]))
    assert abs((capacitance[1] - capacitance[0]) / exact - 1) < 3e-3


def _frame_reflection(directory, frame, loss_factor):
    """|R| of a frame of ``frame`` wavelengths, fitted in front of it."""
    lam = pc.strip_wavelength()
    mesh = pc.strip_mesh(directory / f"strip_{frame}.msh", frame)
    problem = gfem.Piezoelectric(
        pc.STRIP_FREQUENCY,
        terminals=["top"],
        ground=["bot"],
        damping_frame=gfem.DampingFrame(["frame"], loss_factor=loss_factor),
        keep_fields=True,
    )
    result = gfem.solve(pc.strip_model(mesh), problem, options=_options(3))
    # Beyond one wavelength from the source the evanescent modes of the strip are gone.
    x = np.linspace(pc.STRIP_SOURCE + lam, pc.STRIP_SOURCE + 2.5 * lam, 200)
    points = np.c_[x, np.full_like(x, lam / 8)] * 1e-6
    u = np.asarray(result.displacement(pc.STRIP_FREQUENCY)["z"].at(points))
    k = 2 * np.pi / lam
    basis = np.c_[np.exp(-1j * k * x), np.exp(1j * k * x)]
    (forward, backward), *_ = np.linalg.lstsq(basis, u, rcond=None)
    assert np.linalg.norm(basis @ [forward, backward] - u) < 1e-4 * np.linalg.norm(u)
    return abs(backward / forward)


@pytest.mark.parametrize(
    ("loss_factor", "frame"), [(1.2, 4.0), (1.2, 5.7), (1.2, 8.0), (2.0, 5.7)]
)
def test_a_damping_frame_reflects_as_its_design_rule_says(tmp_path, loss_factor, frame):
    reflection = _frame_reflection(tmp_path, frame, loss_factor)
    law = np.exp(-2 * np.pi * loss_factor * frame / 3) + 3.4e-3 * loss_factor / frame**2
    assert law / 2 < reflection < 2 * law, (reflection, law)
    if (loss_factor, frame) == (1.2, 4.0):
        assert reflection < 1e-3
