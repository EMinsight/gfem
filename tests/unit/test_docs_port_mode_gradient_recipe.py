"""The port-mode gradient recipe runs and its gradients meet their closed forms.

docs/recipes/port-mode-gradient.md runs nothing at docs build time. This runs
its blocks on the thermo-optic recipe's strip section and the getting-started
line: the effective-index gradient against the identity that the
permittivity-weighted gradient is half the group index, a line width found for
50 ohm by Newton steps against the closed form eta_0 h / w, and a driven
objective whose design runs into both port faces against a central difference.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "port-mode-gradient.md"
ETA0 = 376.730313668  # ohm
HEIGHT = 1e-3  # m, the plates' gap


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("port_mode_gradient")
    return dp.run_page(PAGE, workdir)


def _lines(ran, section: str) -> list:
    examples = [example for example in ran if example.section == section]
    return [
        dp.numbers(line)
        for example in examples
        for line in example.printed.splitlines()
    ]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_effective_index_and_its_gradient(ran):
    value, grads = _lines(ran, "The effective index of a strip waveguide")[:2]
    assert value[0] == pytest.approx(2.447236, abs=1e-6)
    assert (grads[-2:] > 0).all()


def test_the_weighted_gradient_is_half_the_group_index(ran):
    lines = _lines(ran, "The effective index of a strip waveguide")
    weighted, half_group = lines[2][0], lines[2][-1]
    assert weighted == pytest.approx(half_group, rel=1e-6)


def test_each_newton_line_is_its_closed_form(ran):
    lines = _lines(ran, "Size a line for 50 Ω")
    *steps, last = lines
    for width, z in (line[-2:] for line in steps):
        assert z == pytest.approx(ETA0 * HEIGHT / (width * 1e-3), rel=1e-6)
    assert last[-1] == pytest.approx(7.5346, abs=1e-4)


def test_the_driven_gradient_is_a_central_difference(ran):
    _, (grad, difference) = _lines(ran, "A design that reaches the ports")
    assert grad == pytest.approx(difference, rel=1e-5)


def test_the_limits_name_their_errors():
    text = PAGE.read_text(encoding="utf-8")
    limits = dp._prose(dp.section(text, "Limits"))
    for name in (
        "gfem.PortGeometryError",
        "gfem.NonDifferentiableError",
        "sliding_ports",
        "stop_gradient",
    ):
        assert name in limits, name
