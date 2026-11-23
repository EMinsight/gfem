"""The source columns of a driven solve: right-hand sides beside the excited ports.

Each source (`gfem.Driven(sources=...)`) is one more column of the block
solved on the one factorization per frequency, after the excited ports, in
the problem's order. Each kind of source has one builder here
(`COLUMN_KINDS`): given the forward's concrete model and system operator
and its sources of that kind, it returns their loads as a function of ω,
``(N, n)`` on the solve space. `SourceColumns` places every kind's
columns in problem order.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem._sources import INCIDENT, Source

if TYPE_CHECKING:
    from gfem._engine.system_operator import SystemOperator
    from gfem._pml import Stretch
    from gfem.model import Model

__all__ = ["COLUMN_KINDS", "ColumnContext", "SourceColumns"]


@dataclasses.dataclass(frozen=True, eq=False)
class ColumnContext:
    """What a builder of source columns reads of one forward.

    Attributes:
        model: The model, with concrete leaves.
        system: The system operator of the solve (its operator's space is
            the solve space).
        order: The Nédélec order.
        stretch: The perfectly matched layer's stretch, or ``None``.
        frequencies: The frequencies the forward solves, Hz.
    """

    model: Model
    system: SystemOperator
    order: int
    stretch: Stretch | None
    frequencies: tuple[float, ...]


#: ``context, sources -> (omega -> (N, len(sources)) loads)``.
ColumnBuilder = Callable[[ColumnContext, Sequence[Any]], Callable[[float], np.ndarray]]


def _incident(
    context: ColumnContext, sources: Sequence[Any]
) -> Callable[[float], np.ndarray]:
    from gfem._engine.incident import incident_columns

    return incident_columns(
        context.model,
        context.system,
        sources,
        order=context.order,
        stretch=context.stretch,
    )


#: Per kind of source (its classes), the builder of its columns.
COLUMN_KINDS: tuple[tuple[tuple[type, ...], ColumnBuilder], ...] = (
    (INCIDENT, _incident),
)


@dataclasses.dataclass(eq=False)
class SourceColumns:
    """The source columns of one forward, in the problem's source order.

    Attributes:
        names: The sources' names, in column order.
        ndof: The solve space's size.
        kinds: Per kind, the column positions it fills and its loads.
    """

    names: tuple[str, ...]
    ndof: int
    kinds: list[tuple[list[int], Callable[[float], np.ndarray]]]

    @classmethod
    def build(cls, sources: Sequence[Source], context: ColumnContext) -> SourceColumns:
        """Every kind's builder on its sources.

        Raises:
            NotImplementedError: A source of a kind without a builder.
        """
        kinds = []
        placed: set[int] = set()
        for classes, builder in COLUMN_KINDS:
            positions = [k for k, s in enumerate(sources) if isinstance(s, classes)]
            if positions:
                loads = builder(context, [sources[k] for k in positions])
                kinds.append((positions, loads))
                placed.update(positions)
        for k, source in enumerate(sources):
            if k not in placed:
                raise NotImplementedError(
                    f"The {type(source).__name__} {source.name!r} cannot be solved "
                    "in this gfem version."
                )
        ndof = int(context.system.operator.space.ndof)
        return cls(tuple(s.name for s in sources), ndof, kinds)

    def rhs(self, omega: float) -> np.ndarray:
        """The loads at ω, ``(N, n_sources)``."""
        out = np.zeros((self.ndof, len(self.names)), complex)
        for positions, loads in self.kinds:
            out[:, positions] = loads(omega)
        return out
