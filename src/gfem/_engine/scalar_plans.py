"""The plan pool of the scalar conduction solves: thermal and DC conduction.

A serial CPU ``direct`` solve of a real symmetric operator on an H1 space
takes its handle from the plan pool (`gfem._engine.cache.EntryCache.take_plan`)
and refactorizes it, as a driven base does (`gfem._engine.system_operator`):
a design loop pays the numeric factorization alone, on the kept analysis, and
a design that leaves the operator's values unchanged solves on the held
factors. The handle returns to the pool when its solve is over, or when its
cache entry is consumed, and ``retain_memory`` bounds it with the rest of the
cache.

The key holds what fixes the operator's pattern and its factors' meaning: the
mesh topology, the space key, a digest of the free DOFs, the symmetry, the
operator's tag (the physics, and what changes its values beyond the
design: the heat-capacity weight of a stepping operator) and the thread
count of a plan made now. Node positions are not in it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem._engine.cache import ENTRIES, retain_budget
from gfem._engine.electrostatic import KeptHandle
from gfem._linalg import (
    Handle,
    Operator,
    Refactorable,
    ResolvedLinearSolver,
    refactorize_or_prepare,
)
from gfem._linalg.base import free_mask
from gfem.options import SolverOptions

__all__ = [
    "PlanSite",
    "PooledKeptHandle",
    "digest",
    "plan_site",
    "prepare_pooled",
    "release",
]

# the handle's own attribute: where it returns when released
_SITE = "_plan_site"


@dataclasses.dataclass(frozen=True, eq=False)
class PlanSite:
    """Where a handle pools: its key, the topology its plan lies on, the budget.

    Attributes:
        key: The pool key, thread count included.
        topology: The mesh topology (the pool frees its handles once no
            discretization of it is left).
        budget: ``retain_memory``'s bytes, the walk's budget on return.
    """

    key: tuple
    topology: Any
    budget: int


def digest(mask: np.ndarray | None) -> str:
    """A digest of a boolean mask (``""`` for ``None``): a key's part."""
    if mask is None:
        return ""
    packed = np.packbits(np.asarray(mask, dtype=bool))
    return hashlib.blake2b(packed.tobytes(), digest_size=16).hexdigest()


def plan_site(
    linear: ResolvedLinearSolver,
    options: SolverOptions,
    *,
    topology: Any,
    comm: Any,
    space: ngs.FESpace,
    space_key: tuple,
    freedofs: ngs.BitArray,
    tag: tuple,
) -> PlanSite | None:
    """Where an operator's handle pools, or ``None`` off the pool.

    Serial CPU ``direct`` solves only (BDDC, PETSc, cuDSS and ranks keep
    preparing a handle per operator).
    """
    if (
        comm is not None
        or topology is None
        or (linear.kind, linear.device) != ("direct", "cpu")
    ):
        return None
    free = digest(free_mask(freedofs, space.ndof))
    key = (id(topology), space_key, free, "hermitian", tag, max(1, ngs.GetNumThreads()))
    return PlanSite(key, topology, retain_budget(options.retain_memory))


def prepare_pooled(
    linear: ResolvedLinearSolver,
    site: PlanSite | None,
    operator: Operator,
    *,
    width: int = 1,
    tolerance: float | None = None,
) -> Handle:
    """A handle of the real symmetric ``operator``: a pooled one refactorized.

    A handle of ``site``'s key takes ``operator``'s values and the
    solves' ``tolerance`` (a refactorization on the factors it holds solves
    to it); else a new one is prepared. With no ``site`` it always is.
    """

    def prepare(op: Operator) -> Handle:
        return linear.prepare(
            op, symmetry="hermitian", width=width, tolerance=tolerance
        )

    if site is None:
        return prepare(operator)
    pooled = ENTRIES.take_plan(site.key)
    if pooled is not None and not isinstance(pooled, Refactorable):
        pooled.free()
        pooled = None
    if pooled is not None and tolerance is not None:
        pooled.tolerance = tolerance
    handle = refactorize_or_prepare(pooled, operator, prepare, frequency=0.0)
    setattr(handle, _SITE, site)
    return handle


def release(handle: Handle, *, walk: bool = True) -> None:
    """Return a handle to the plan pool, with its factors, or free it.

    A handle `prepare_pooled` made returns to the pool the key it was taken
    under; any other is freed. The pool walks unless not to ``walk``
    (`gfem._engine.cache.EntryCache.give_plan`).
    """
    site: PlanSite | None = getattr(handle, _SITE, None)
    if site is None:
        handle.free()
        return
    delattr(handle, _SITE)
    ENTRIES.give_plan(site.key, site.topology, handle, site.budget, walk=walk)


@dataclasses.dataclass(eq=False)
class PooledKeptHandle(KeptHandle):
    """A forward's prepared operator, kept for its backward pass, from the pool.

    As `KeptHandle`, but the operator returns to the plan pool where it came
    from one: when its entry is consumed, and when the cache gives it up for
    ``retain_memory`` (its factors may stay pooled, released first under
    pressure).
    """

    def free_base(self, comm: Any, *, walk: bool = True) -> None:  # noqa: ARG002
        """Return the operator to the plan pool; the entry keeps its fields."""
        self._give(walk=walk)

    def free(self, *, return_plans: bool = False) -> None:
        """Drop the operator: back to the pool with ``return_plans``, else freed."""
        if return_plans:
            self._give(walk=True)
        else:
            super().free()

    def _give(self, *, walk: bool) -> None:
        if self.handle is not None:
            release(self.handle, walk=walk)
            self.handle = None
