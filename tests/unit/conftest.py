"""Fixtures shared by the unit tests of one family."""

import piezo_cases as pc
import pytest


@pytest.fixture(scope="module", params=("lines", "meshed"))
def device(request, tmp_path_factory):
    """``(variant, mesh, model)`` of the piezoelectric transducer section."""
    variant = request.param
    path = tmp_path_factory.mktemp(variant) / "idt.msh"
    mesh = pc.mesh(path, variant)
    return variant, mesh, pc.model(mesh, variant)
