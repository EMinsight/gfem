"""A `SolveRequest`'s reply dict: fresh per request, shared by its replacements."""

import dataclasses

from gfem._engine.solve_request import SolveRequest


def test_two_requests_never_share_adjoints():
    first, second = SolveRequest(), SolveRequest()
    first.adjoints["key"] = 1.0
    assert first.adjoints is not second.adjoints
    assert second.adjoints == {}


def test_replace_keeps_the_reply_dict():
    parent = SolveRequest()
    child = dataclasses.replace(parent, keeps_partial=True, rerun=True)
    assert child.adjoints is parent.adjoints
    child.adjoints["key"] = 1.0
    assert parent.adjoints == {"key": 1.0}
