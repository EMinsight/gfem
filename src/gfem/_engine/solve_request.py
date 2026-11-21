"""`SolveRequest`: what a caller asks of one rule forward beyond the model.

A rule's ``forward(model, *, keep, request=None)`` reads its per-call inputs
from the request, never from attributes a caller wrote on the rule before
the call: the adjoint loads to solve along with the forward, whether a
partial result will be read, which inputs a gradient will read, whether the
forward reruns for a backward, and a driven block's rows of its parent's
port mode sweep. The forward hands back the duals it solved in the
request's ``adjoints``, a dict made fresh per request.

A child rule (a fixed-angle sweep's per-frequency rule) gets
``dataclasses.replace(request, ...)`` of its parent's request, which keeps
the same ``adjoints`` dict: what the child solves lands in the parent's
reply.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from gfem._eigen.rule import DualRequest
    from gfem._engine.driven import AdjointLoads
    from gfem._engine.electrostatic import ElectrostaticLoads

__all__ = ["SolveRequest"]


@dataclasses.dataclass(frozen=True)
class SolveRequest:
    """The per-call inputs of one rule forward, and the duals it solved.

    A rule takes ``request=None`` and builds a fresh one: a module-level
    default instance would share its ``adjoints`` across unrelated calls.
    Requests derived with `dataclasses.replace` share it on purpose.

    Attributes:
        adjoint_loads: The adjoint loads the forward also solves on its own
            factorization: a driven sweep's per-frequency `AdjointLoads`, an
            electrostatic objective's `ElectrostaticLoads`, an eigenmode
            search's participation duals (`DualRequest`); ``None`` for none.
            `gfem.refine_mesh` sets them for an estimate's solved duals, never
            on a differentiated solve.
        keeps_partial: Whether the caller reads ``partial_result``: a uniform
            sweep whose linear solve does not converge then keeps what it
            solved on its error. Set by `gfem.solve`'s eager path and for a
            driven sweep's child rules, never under JAX.
        wanted: The wanted input kinds of a differentiated solve, so that a
            forward keeps only what its backward reads; ``None`` off JAX.
        rerun: Whether this forward reruns for a backward that lost its
            entry (it keeps the least it can).
        port_modes: Per wave port surface, a driven block's rows of its
            parent's port mode sweep at the block's frequencies, so that the
            block sweeps no port mode itself; ``None`` anywhere else.
        adjoints: The reply: the duals of ``adjoint_loads`` the forward
            solved, keyed as the rule keys them.
    """

    adjoint_loads: AdjointLoads | ElectrostaticLoads | Sequence[DualRequest] | None = (
        None
    )
    keeps_partial: bool = False
    wanted: frozenset | None = None
    rerun: bool = False
    port_modes: list[dict[str, np.ndarray]] | None = None
    adjoints: dict = dataclasses.field(default_factory=dict)
