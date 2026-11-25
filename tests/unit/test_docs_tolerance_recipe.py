"""The tolerance recipe runs, prints what the page shows, and its numbers hold.

docs/recipes/tolerance-report.md runs nothing at docs build time. This runs its
blocks on the getting-started line and checks what the page states: the nominal
transmissions against the textbook cascade of three line sections, the
permittivity entry of the Jacobian against its derivative, the edge-bias entry
against central differences of the same solves, and the report's yields, Cpk,
effects and shares against their closed forms.
"""

import contextlib
import shutil
import urllib.request

import docs_pages as dp
import numpy as np
import pytest
from scipy.stats import norm

PAGE = dp.DOCS / "recipes" / "tolerance-report.md"
ETA0 = 376.730313668  # ohm
C0 = 299792458.0  # m/s
HEIGHT, WIDTH = 1e-3, 6e-3  # m, the plates' gap and the line's width
Q5, Q8 = "|S21|² at 5 GHz", "|S21|² at 8 GHz"


def cascade(f, eps, width=WIDTH):
    """|S21|^2: 10 mm of air, 20 mm of ceramic, 10 mm of air, 62.8 ohm ports."""

    def section(z, k, length):
        return np.array(
            [
                [np.cos(k * length), 1j * z * np.sin(k * length)],
                [1j * np.sin(k * length) / z, np.cos(k * length)],
            ]
        )

    k0 = 2 * np.pi * f / C0
    z_air = ETA0 * HEIGHT / width
    z_port = 376.73 / 6
    a, b, c, d = (
        section(z_air, k0, 10e-3)
        @ section(z_air / np.sqrt(eps), k0 * np.sqrt(eps), 20e-3)
        @ section(z_air, k0, 10e-3)
    ).ravel()
    return abs(2 / (a + b / z_port + c * z_port + d)) ** 2


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    return dp.run_page(PAGE, tmp_path_factory.mktemp("tolerance_recipe"))


@pytest.fixture(scope="module")
def script(ran, tmp_path_factory):
    """The first block run again in a namespace of its own: report and quantities."""
    workdir = tmp_path_factory.mktemp("tolerance_script")

    def copy(url, filename):
        shutil.copy(dp.MESHES[url], workdir / filename)

    namespace = {"__name__": "tolerance_script"}
    with (
        pytest.MonkeyPatch.context() as patch,
        contextlib.chdir(workdir),
        contextlib.redirect_stdout(None),
    ):
        patch.setattr(urllib.request, "urlretrieve", copy)
        exec(compile(ran[0].code, str(PAGE), "exec"), namespace)  # noqa: S102
    return namespace


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_nominal_transmissions_are_the_cascade_of_three_sections(script):
    values = script["report"].values
    # The lumped ports and the coarse mesh differ from the closed form by well
    # under 1e-3 here (get-started quotes 0.901 against 0.9013 in |S21|).
    assert values[Q5] == pytest.approx(cascade(5e9, 10.0), abs=2e-3)
    assert values[Q8] == pytest.approx(cascade(8e9, 10.0), abs=2e-3)


def test_the_permittivity_entry_is_the_cascades_derivative(script):
    report = script["report"]
    j = report.parameters.index("eps_r")
    for q, f in ((Q5, 5e9), (Q8, 8e9)):
        i = report.quantities.index(q)
        h = 1e-4  # central difference: truncation ~ h^2 f''', round-off ~ 1e-16 / h
        exact = (cascade(f, 10.0 + h) - cascade(f, 10.0 - h)) / (2 * h)
        assert report.jacobian[i, j] == pytest.approx(exact, abs=2e-3)


def test_the_edge_bias_entry_is_a_central_difference_of_the_solves(script):
    report, quantities = script["report"], script["quantities"]
    j = report.parameters.index("edge bias")
    nominal = dict(report.nominal)

    def values(b):
        out = quantities({**nominal, "edge bias": b})
        return np.array([float(out[q]) for q in report.quantities])

    # FD(h) has error c h^2; FD(h/2) is off by (FD(h) - FD(h/2)) / 3. The bound
    # takes that estimate three times over plus round-off of 1e-10 / h for the
    # iterative solves' 1e-8-relative tolerance (taken 100x loose).
    h = 5e-5
    coarse = (values(h) - values(-h)) / (2 * h)
    fine = (values(h / 2) - values(-h / 2)) / h
    bound = 3 * np.abs(coarse - fine) + 1e-10 / (h / 2)
    assert (np.abs(report.jacobian[:, j] - fine) <= bound).all(), (
        report.jacobian[:, j],
        fine,
    )


def test_the_printed_check_is_the_derivative(ran):
    (line,) = [e.printed for e in ran if e.section == "Check your results"]
    report_value, difference = dp.numbers(line.split("report")[1])
    assert report_value == pytest.approx(difference, abs=2e-5)


def test_effects_shares_and_yields_are_their_closed_forms(script):
    r = script["report"]
    t = np.array([r.tolerance[p] for p in r.parameters])
    np.testing.assert_allclose(r.effect, r.jacobian * t, rtol=1e-12)
    np.testing.assert_allclose(r.share.sum(axis=1), 1.0, atol=1e-12)
    for q, low in ((Q5, 0.75), (Q8, 0.3)):
        i = r.quantities.index(q)
        s = np.sqrt(np.sum((r.jacobian[i] * t / 3) ** 2))
        assert r.std[q] == pytest.approx(s, rel=1e-12)
        assert r.yields[q] == pytest.approx(
            norm.cdf((r.values[q] - low) / s), abs=1e-12
        )
        assert r.cpk[q] == pytest.approx((r.values[q] - low) / (3 * s), rel=1e-12)
        assert r.worst_case[q][0] == pytest.approx(
            r.values[q] - np.abs(r.effect[i]).sum(), rel=1e-12
        )


def test_the_page_states_the_ranking_and_the_joint_yield(script):
    r = script["report"]
    i5, i8 = r.quantities.index(Q5), r.quantities.index(Q8)
    p_bias, p_eps = r.parameters.index("edge bias"), r.parameters.index("eps_r")
    assert r.share[i5, p_eps] > r.share[i5, p_bias]  # 5 GHz: the permittivity
    assert r.share[i8, p_bias] > r.share[i8, p_eps]  # 8 GHz: the width
    low, high = r.joint_yield_bounds
    slack = 4 * r.joint_yield_error  # the Monte Carlo estimate, not the exact value
    assert low - slack <= r.joint_yield <= high + slack
    # Fréchet: the product of independent marginals is not claimed, the bounds are
    assert low == pytest.approx(max(0.0, sum(r.yields.values()) - 1), abs=1e-12)
    assert high == pytest.approx(min(r.yields.values()), abs=1e-12)
    # the two quantities share both parameters, so the product is only a loose guide
    assert abs(r.joint_yield - r.yields[Q5] * r.yields[Q8]) < 1e-2
    assert r.nonlinear == ()


def test_the_limits_name_their_errors():
    text = PAGE.read_text(encoding="utf-8")
    limits = dp._prose(dp.section(text, "Limits"))
    for name in ("gfem.NonDifferentiableError", "jax.jit", "gfem.solve"):
        assert name in limits, name
