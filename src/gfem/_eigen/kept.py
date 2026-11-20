"""The kept-operator holder: an eigenmode entry's handle.

A differentiated eigenmode forward keeps its `EigenOperator` as it leaves it
(the factors at the last shift, the mass, damping and nonpolynomial maps,
the free mask), only to map vectors in its backward pass. The holder is the
cache entry's ``handle`` (`gfem._engine.cache.Entry`): the cache counts its
bytes and, through `free_base`, drops the operator of an entry over
``SolverOptions.retain_memory`` before it drops any entry, which keeps its
fields. Its factorization is the forward's own (`linear.prepare`), never
the plan pool's, and nothing of it returns there.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from gfem._eigen.operator import EigenOperator

__all__ = ["KeptOperator"]


@dataclasses.dataclass(eq=False)
class KeptOperator:
    """An eigenmode forward's operator, kept for its backward pass.

    Attributes:
        operator: The forward's `EigenOperator`, or ``None`` once freed.
        store: The kept fixed parts of the driven operator ``operator``
            assembles from (its ``fixed.store``), or ``None``: its closure
            keeps the driven operator, so the store lives as long as it.
    """

    operator: EigenOperator | None
    store: Any = None

    @property
    def nbytes(self) -> int:
        """The factors' and assembled maps' bytes (0 once freed), but `shared`."""
        return 0 if self.operator is None else self.operator.nbytes

    def shared(self) -> dict[int, int]:
        """The kept fixed parts' bytes, by identity (empty without or once freed).

        The entry cache counts them once with the discretization's.
        """
        if self.operator is None or self.store is None:
            return {}
        return self.store.shared()

    def free_base(self, comm: Any, *, walk: bool = True) -> None:  # noqa: ARG002
        """Free the operator; the entry keeps its fields.

        ``comm`` is always ``None`` on the eigenmode path, and nothing goes
        to the plan pool, so there is no walk to run.
        """
        self.free()

    def free(self, *, return_plans: bool = False) -> None:  # noqa: ARG002
        """Free the operator's factors and drop it; nothing returns to the pool."""
        if self.operator is not None:
            self.operator.free_handle()
            self.operator = None
            self.store = None
