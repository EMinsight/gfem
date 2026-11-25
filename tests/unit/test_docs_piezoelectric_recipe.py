"""The piezoelectric admittance recipe runs and prints what the page shows.

docs/recipes/piezoelectric-admittance.md runs nothing at docs build time. This
runs its blocks in order on a small section built by the repository's own
docs/tutorials/transducer_section.py (the page downloads it) and checks that the
output blocks are what the code prints, that the radiation conductance is
positive, and that the strain at the two arms is the same.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "piezoelectric-admittance.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("piezoelectric_admittance")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _example(ran, section: str) -> dp.Example:
    (example,) = [example for example in ran if example.section == section]
    return example


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_conductance_is_positive(ran):
    assert _example(ran, "The result").printed.splitlines()[-1].endswith("True")


def test_the_two_arms_strain_the_same(ran):
    line = _example(ran, "Fields and strain").printed.splitlines()[1]
    first, second = dp.numbers(line)[-2:]
    assert first == pytest.approx(second, rel=1e-2)
