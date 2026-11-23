"""Count preconditioner setups, BDDC's coarse columns and Krylov iterations in a test.

Usage::

    from preconditioners import count_builds, count_iterations

    builds = count_builds(monkeypatch, "bddc_gmres")
    iterations = count_iterations(monkeypatch)
    ...  # solves
    assert len(builds) == 2
    assert max(iterations) < 60
"""

import ngsolve as ngs

from gfem._linalg import bddc, pmg  # noqa: F401  (each handle class, imported)
from gfem._linalg.krylov_handle import KrylovHandle


def count_preconditioners(monkeypatch) -> list[int]:
    """Spy on BDDC setups: one entry per preconditioner built."""
    calls = []
    build = ngs.Preconditioner

    def spy(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(ngs, "Preconditioner", spy)
    return calls


def count_iterations(monkeypatch) -> list[int]:
    """Spy on preconditioned Krylov solves: one entry per converged right-hand side."""
    calls = []
    solve_counted = KrylovHandle.solve_counted

    def spy(self, *args, **kwargs):
        X, its = solve_counted(self, *args, **kwargs)
        calls.extend(its)
        return X, its

    monkeypatch.setattr(KrylovHandle, "solve_counted", spy)
    return calls


def _handle_class(kind: str) -> type[KrylovHandle]:
    """The `KrylovHandle` subclass of ``kind``."""
    found, classes = {}, [KrylovHandle]
    while classes:
        cls = classes.pop()
        found[getattr(cls, "kind", None)] = cls
        classes.extend(cls.__subclasses__())
    if kind not in found:
        raise ValueError(f"no preconditioned Krylov handle of kind {kind!r}")
    return found[kind]


def count_builds(monkeypatch, kind: str) -> list[int]:
    """Spy on setups of ``kind``'s preconditioner: one entry per setup.

    Counts the first setup of a handle and every one after it (a rebuild, a
    capped column, a cold solve), through the handle's `_set_up`.
    """
    calls = []
    cls = _handle_class(kind)
    set_up = cls._set_up

    def spy(self, *args, **kwargs):
        calls.append(1)
        return set_up(self, *args, **kwargs)

    monkeypatch.setattr(cls, "_set_up", spy)
    return calls


def coarse_columns(monkeypatch) -> list[int]:
    """Spy on BDDC setups: the columns each one is expected to serve.

    One entry per coarse factorization chosen (`bddc._coarse_inverse`).
    """
    calls = []
    choose = bddc._coarse_inverse

    def spy(*args, **kwargs):
        calls.append(kwargs["columns"])
        return choose(*args, **kwargs)

    monkeypatch.setattr(bddc, "_coarse_inverse", spy)
    return calls
