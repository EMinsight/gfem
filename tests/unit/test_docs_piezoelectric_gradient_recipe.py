"""The piezoelectric gradient recipe runs and prints what the page shows.

docs/recipes/piezoelectric-gradient.md runs nothing at docs build time. This
runs its blocks in order on a small section built by the repository's own
docs/tutorials/transducer_section.py (the page downloads it) and checks that the
output blocks are what the code prints, that every gradient agrees with its
central difference, and that a mirrored node map raises.
"""

import docs_pages as dp
import pytest

PAGE = dp.DOCS / "recipes" / "piezoelectric-gradient.md"


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("piezoelectric_gradient")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(workdir))  # a script imports from its folder
        return dp.run_page(PAGE, workdir)


def _printed(ran, section: str) -> list[str]:
    return [
        line
        for example in ran
        if example.section == section
        for line in example.printed.splitlines()
    ]


def test_every_output_block_is_what_the_code_prints(ran):
    assert not dp.output_problems(ran), "\n".join(dp.output_problems(ran))


def test_the_density_and_the_pitch_gradients_match_their_differences(ran):
    for section in (
        "The conductance as a function of its inputs",
        "The pitch, through a node map",
    ):
        for line in _printed(ran, section):
            if "gradient" in line:
                gradient, difference = dp.numbers(line)[-2:]
                assert difference == pytest.approx(gradient, rel=1e-4), line


def test_the_thickness_gradient_matches_a_difference_at_one_percent(ran):
    (line,) = [x for x in _printed(ran, "The pitch, through a node map") if "tau" in x]
    gradient, difference = dp.numbers(line)[-2:]
    assert difference == pytest.approx(gradient, rel=1e-3)


def test_the_shares_name_the_coupling_and_the_density(ran):
    shares = {
        line.split(":")[0]: float(line.split(":")[1])
        for line in _printed(ran, "The conductance as a function of its inputs")
        if line.split(":")[0] in {"piezoelectric tensor", "density", "stiffness"}
    }
    assert shares["piezoelectric tensor"] > 1.0  # the conductance grows as e^2
    assert shares["density"] > 0.5
    assert shares["stiffness"] < 0.0


def test_a_mirrored_node_map_raises(ran):
    assert "MeshQualityError" in _printed(ran, "The pitch, through a node map")


def test_a_point_reading_differentiates_through_the_nodes(ran):
    (line,) = _printed(ran, "A reading of the field at a point")
    strain, derivative = dp.numbers(line)[-3:][0], dp.numbers(line)[-1]
    assert strain > 0.0
    assert derivative != 0.0
