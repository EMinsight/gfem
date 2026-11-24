r"""`DrivenSystemBuilder`: the system operator and port network of a driven solve.

A driven rule builds its forward operator (`gfem._engine.driven.DrivenOperator`)
and hands it here with the model. The builder adds the wave and Floquet
port couplings, wires the transposed operator at :math:`-\mathbf{k}`, the
plan-pool budget and the gauge basis into a `SystemOperator`, and builds
the `PortNetwork` on the same couplings.

It owns every piece of state that driven solves keep on a discretization
across designs, and every key of it: the couplings and their port networks
(`KeptPorts`, behind `FloquetKey` and the network key) and the design
initial guesses (`DesignGuess`). Its fields are the rule's resolved
settings, fixed at construction; the per-call data (a frequency block's
rows of its parent's port mode sweep) is an argument.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, NamedTuple

from gfem._boundaries import lead_port, wave_port_surfaces
from gfem._engine.cache import Discretization, cached, retain_budget
from gfem._engine.floquet_ports import FloquetSurfaceCoupling, floquet_couplings
from gfem._engine.gauge import gauge_basis
from gfem._engine.port_network import PortNetwork
from gfem._engine.system_operator import SystemOperator, SystemSweep
from gfem._engine.wave_ports import PortSurfaceCoupling, wave_port_couplings
from gfem._linalg import KRYLOV_KINDS
from gfem._linalg.layout import ONE_PROCESS
from gfem._port_section import port_section_key

if TYPE_CHECKING:
    import numpy as np

    from gfem import _floquet
    from gfem._engine.driven import DrivenOperator
    from gfem._linalg import ResolvedLinearSolver
    from gfem._pml import Stretch
    from gfem.model import Model
    from gfem.problems import Polarization

__all__ = ["DesignGuess", "DrivenSystemBuilder", "FloquetKey", "KeptPorts"]


@dataclasses.dataclass(frozen=True, eq=False)
class DrivenSystemBuilder:
    """The system operator and port network of a driven rule's solves.

    Built once per rule from its resolved settings; holds no mutable state,
    so a shallow copy of the rule shares it.

    Attributes:
        frequency: The problem's output grid, Hz.
        columns: The excited ports' indices into the model's ports.
        excite: The excited ports' names.
        port_mode_tolerance: ``SolverOptions.port_mode_tolerance``.
        floquet_order_tolerance: ``SolverOptions.floquet_order_tolerance``.
        orders: Per Floquet surface, its order table over ``order_frequency``.
        order_frequency: The grid of the Floquet order table, Hz.
        wave_vector: The operator's Bloch wave vector, rad/m, or ``None``.
        problem_wave_vector: The problem's ``wave_vector`` (``None`` at a
            fixed angle): the network and design guess keys read it.
        wave_vector_frequency: A fixed-angle sweep's reference frequency,
            Hz, or ``None``.
        swept_spaces: Whether the Bloch spaces sit in the discretization's
            swept group.
        comm: The communicator of a distributed solve, or ``None``.
        linear: The resolved linear solver.
        adjoint_tolerance: The adjoint solves' relative residual.
        retain_memory: ``SolverOptions.retain_memory``, read into a budget
            at each build.
        gauge: Whether the solve runs through the tree-cotree gauge.
        polarization: A 2D model's polarization, or ``None``.
        stretch: The perfectly matched layer's stretch (the rule's), which
            the port sections and their key take; ``None`` without a layer.
    """

    frequency: tuple[float, ...]
    columns: tuple[int, ...]
    excite: tuple[str, ...]
    port_mode_tolerance: float
    floquet_order_tolerance: float
    orders: dict[str, _floquet.SurfaceOrders]
    order_frequency: tuple[float, ...]
    wave_vector: tuple[float, float, float] | None
    problem_wave_vector: tuple[float, float, float] | None
    wave_vector_frequency: float | None
    swept_spaces: bool
    comm: Any
    linear: ResolvedLinearSolver
    adjoint_tolerance: float
    retain_memory: int | None
    gauge: bool
    polarization: Polarization | None = None
    stretch: Stretch | None = None

    def build(
        self,
        model: Model,
        op: DrivenOperator,
        *,
        handed: list[dict[str, np.ndarray]] | None = None,
    ) -> tuple[SystemOperator, PortNetwork]:
        """The system operator and the port network of one driven solve on ``op``.

        Both from one set of wave port couplings: the system operator adds
        their Robin pieces and modal corrections, and the port network holds
        the excitations and S functionals. Runs each wave port surface's
        port mode sweep over the output grid.

        The couplings and the network are kept on the discretization
        (`KeptPorts`) and reused while their key holds: a design away from
        the port faces runs no port mode sweep. A frequency block's
        couplings stand on ``handed``, its rows of its parent's sweep per wave
        port surface (`SolveRequest.port_modes`).
        """
        from gfem._engine.driven import driven_operator

        columns = list(self.columns)
        frequency = self.frequency
        tolerance = self.port_mode_tolerance
        disc = cached(op.built)
        key = KeptPorts.key_of(
            model,
            op,
            frequency,
            tolerance,
            FloquetKey.of(
                self.orders,
                tolerance=self.floquet_order_tolerance,
                grid=self.order_frequency,
                wave_vector=self.wave_vector,
                wave_vector_frequency=self.wave_vector_frequency,
            ),
            polarization=self.polarization,
            stretch=self.stretch,
        )
        kept = disc.ports if disc is not None else None
        if not isinstance(kept, KeptPorts) or kept.key != key:
            couplings = wave_port_couplings(
                model,
                op.built,
                op.space,
                order=op.order,
                frequency=frequency,
                tolerance=tolerance,
                layout=op.layout or ONE_PROCESS,
                swept=handed,
                polarization=self.polarization,
                stretch=self.stretch,
            )
            floquet = floquet_couplings(
                model, op, self.orders, frequency, grid=self.order_frequency
            )
            kept = KeptPorts(key, couplings, floquet)
            if disc is not None:
                disc.ports = kept
        couplings = kept.couplings
        transposed = None
        if self.wave_vector is not None:
            # A(k)^T = A(-k): the adjoint solves' operator
            kx, ky, kz = self.wave_vector
            minus_k = (-kx, -ky, -kz)
            transposed = functools.partial(
                driven_operator,
                model,
                op.order,
                wave_vector=minus_k,
                comm=self.comm,
                swept=self.swept_spaces,
            )
        system = SystemOperator(
            op,
            [*couplings, *kept.floquet],
            self.linear,
            width=len(columns),
            adjoint_tolerance=self.adjoint_tolerance,
            transposed=transposed,
            pool_budget=retain_budget(self.retain_memory),
            gauge=gauge_basis(op) if self.gauge else None,
        )
        network_key = (
            tuple(columns),
            len(frequency),
            self.problem_wave_vector,
            self.wave_vector_frequency,
        )
        ports = kept.networks.get(network_key)
        if ports is None:
            ports = PortNetwork.build(
                model,
                system.operator,
                couplings,
                columns=columns,
                n_freq=len(frequency),
                floquet=kept.floquet,
            )
            kept.keep(network_key, ports)
        return system, ports

    def design_guess(self, system: SystemOperator) -> DesignGuess:
        """Where ``system``'s uniform sweep finds the previous design's solution."""
        return DesignGuess.of(self, system)


@dataclasses.dataclass(frozen=True, eq=False)
class DesignGuess:
    """Where a uniform Krylov sweep's cold solves find the previous design's solution.

    On one process, a ``bddc_gmres`` or ``pmg_gmres`` sweep's first
    frequency starts from the previous design's solution there, kept with
    the discretization (`Discretization.guesses`, dropped first by the
    walk). It is used only for the same excited ports and wave vector, free
    DOFs, symmetry and right-hand-side shape. The guess carries the
    ``n_ref`` its design's preconditioner measured
    (`gfem._linalg.krylov_handle.KrylovHandle.assume_reference`): a solve
    from it takes too few iterations to measure one.
    """

    disc: Discretization | None
    key: tuple = ()
    match: tuple = ()

    @classmethod
    def of(cls, builder: DrivenSystemBuilder, system: SystemOperator) -> DesignGuess:
        """``builder``'s guesses on ``system``'s discretization.

        None kept or used (no ``disc``) but for ``bddc_gmres`` and
        ``pmg_gmres`` on one process with a ``retain_memory`` above 0 and a
        cached discretization.
        """
        op = system.operator
        disc = cached(op.built)
        if (
            builder.linear.kind not in KRYLOV_KINDS
            or builder.comm is not None
            or builder.retain_memory == 0
            or disc is None
        ):
            return cls(None)
        key = (
            "design guess",
            tuple(builder.excite),
            builder.problem_wave_vector,
            builder.wave_vector_frequency,
        )
        match = (
            op.space_key,
            system.free_digest(),
            op.symmetry,
            builder.linear.tolerance,
        )
        return cls(disc, key, match)

    def solve(
        self,
        sweep: SystemSweep,
        omega: float,
        rhs: np.ndarray,
        previous: np.ndarray | None,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """`SystemSweep.solve` from the ``previous`` frequency's solution.

        Without one (the sweep's first frequency), from the previous design's
        solution at ``ω`` where one fits; this design's is kept for the next.
        """
        if previous is not None or self.disc is None:
            return sweep.solve(omega, rhs, x0=previous)
        key = (*self.key, float(omega))
        kept = self.disc.guess(key, self.match)
        if isinstance(kept, _Solution) and kept.block.shape == rhs.shape:
            x, its = sweep.solve(
                omega,
                rhs,
                x0=kept.block,
                guess="previous design",
                reference=kept.reference,
            )
        else:
            x, its = sweep.solve(omega, rhs)
        reference = sweep.reference_iterations
        if reference is not None:
            self.disc.keep_guess(key, self.match, _Solution(x, reference))
        return x, its


@dataclasses.dataclass(frozen=True, eq=False)
class _Solution:
    """A design's solution block and the reference iterations its BDDC measured."""

    block: np.ndarray
    reference: int

    @property
    def nbytes(self) -> int:
        """The block's bytes."""
        return int(self.block.nbytes)


#: Port networks kept per set of wave port couplings (one per excited set).
_MAX_NETWORKS = 4


class FloquetKey(NamedTuple):
    """What the Floquet surfaces' couplings depend on beyond the discretization.

    Attributes:
        tolerance: The order table's tolerance (``floquet_order_tolerance``).
        grid: The order table's grid, Hz.
        wave_vector: The operator's Bloch wave vector, rad/m, or ``None``.
        wave_vector_frequency: A fixed-angle sweep's reference frequency,
            Hz, or ``None``.
        media: Per Floquet surface, its half-space medium
            ``(surface, eps_r, mu_r)``: a new medium on the same mesh
            (`gfem.Model.with_material`) gives new orders and impedances.
    """

    tolerance: float
    grid: tuple[float, ...]
    wave_vector: tuple[float, float, float] | None
    wave_vector_frequency: float | None
    media: tuple[tuple[str, float, float], ...]

    @classmethod
    def of(
        cls,
        table: dict[str, _floquet.SurfaceOrders],
        *,
        tolerance: float,
        grid: tuple[float, ...],
        wave_vector: tuple[float, float, float] | None,
        wave_vector_frequency: float | None,
    ) -> FloquetKey:
        """The key of the order table ``table`` and its solve's settings."""
        media = tuple(
            (surface, orders.face.eps_r, orders.face.mu_r)
            for surface, orders in table.items()
        )
        return cls(tolerance, grid, wave_vector, wave_vector_frequency, media)


@dataclasses.dataclass(eq=False)
class KeptPorts:
    """Wave port couplings kept across driven solves, and their port networks.

    One per discretization (`gfem._engine.cache.Discretization.ports`),
    replaced when a solve's key differs: the next design of an optimization
    reuses them, a design touching a port face or a node move (a new
    discretization) rebuilds them.

    Attributes:
        key: `key_of` the solve that built them.
        couplings: The wave port surfaces' couplings.
        floquet: The Floquet surfaces' couplings.
        networks: Port networks on them by ``(excited columns, number of
            frequencies, Bloch wave vector)``.
    """

    key: tuple
    couplings: list[PortSurfaceCoupling]
    floquet: list[FloquetSurfaceCoupling] = dataclasses.field(default_factory=list)
    networks: dict[tuple, PortNetwork] = dataclasses.field(default_factory=dict)

    @staticmethod
    def key_of(
        model: Model,
        op: DrivenOperator,
        frequency: tuple[float, ...],
        tolerance: float,
        floquet: FloquetKey | None = None,
        *,
        polarization: Polarization | None = None,
        stretch: Stretch | None = None,
    ) -> tuple:
        """What the couplings of ``model`` on ``op``'s discretization depend on.

        The space (a discretization keeps its spaces, so its identity is
        stable, and a Bloch-phased space is its wave vector's), the order,
        the output grid, the port mode tolerance, the Floquet surfaces'
        `FloquetKey` (order table settings and half-space media), the
        ports, per wave port surface
        `port_section_key` with the layer's ``stretch``
        (never the port section itself), and a 2D model's ``polarization``.
        """
        sections = tuple(
            port_section_key(model, lead_port(group), stretch)
            for group in wave_port_surfaces(model.ports).values()
        )
        key = (
            id(op.space),
            op.order,
            tuple(frequency),
            tolerance,
            floquet,
            repr(model.ports),
            sections,
        )
        return key if polarization is None else (*key, polarization)

    def shared(self) -> dict[int, int]:
        """The couplings' and the networks' arrays' bytes, by identity."""
        held = {
            id(c.vectors): int(c.vectors.nbytes)
            for c in (*self.couplings, *self.floquet)
        }
        for network in self.networks.values():
            held.update(network.shared())
        return held

    def keep(self, key: tuple, network: PortNetwork) -> None:
        """Keep ``network`` under ``key``, the oldest out past `_MAX_NETWORKS`."""
        self.networks[key] = network
        while len(self.networks) > _MAX_NETWORKS:
            del self.networks[next(iter(self.networks))]
