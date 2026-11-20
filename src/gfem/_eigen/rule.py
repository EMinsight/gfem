"""Eigenmode rule: resonant modes of a model in complex Nédélec elements.

The operator is the driven operator's pieces (`gfem._engine.driven`) sorted
by power of the frequency (`gfem._eigen.operator`): ``T(k) = K + k D +
k^2 M + A(k)``, ``k = ω scale / c``. Without damping the problem is linear
in ``k^2``; with lumped or impedance resistance, conductivity or a
first-order absorbing boundary it is quadratic; a `Conductor` or a
second-order absorbing boundary adds the nonpolynomial ``A(k)``, which the
seed freezes at the target. One factorization of ``T(sigma)`` at the
target serves the shift-invert Arnoldi of the seed (a retry with a wider
subspace, after refinement reshifted, first refactorizes at the target)
(`gfem._eigen.arpack` or `gfem._eigen.slepc`, chosen by
`SolverOptions.eigen_solver`, on the shared `gfem._eigen.seed` operator,
with the divergence-free projection), the Rayleigh functional and the
residual inverse iteration that refines every seed on the true ``T(k)``
(`gfem._eigen.refine`). A `WavePort` raises: its propagation constant would
have to be solved at the complex eigenfrequency. Everything that touches
NGSolve (the assembly of ``T(k)``, the divergence-free projection, the
energy per element) is in `gfem._engine.eigen`.

Gradients (``omega``, so frequency and Q) need no solve: a simple mode's
``dk/dp = -x^T (dT/dp) x / x^T T'(k) x``, ``T'(k) = D + 2 k M + Σ θ_p'(k)
A_p``, at the refined pair; ``dT/dp`` is a material, sheet-fill or
node-coordinate derivative of the driven operator's pieces (the shape
terms are the driven rule's, port surfaces held). The modes of a
degenerate cluster (frequencies within ``SolverOptions.cluster_rtol``) are
an arbitrary basis of their eigenspace: only an objective symmetric in them
differentiates, through the basis-invariant
``d Σ k = -tr((X^T T' X)^-1 X^T (dT/dp) X)``. A differentiated solve
refines one guard mode beyond ``n_modes`` from the seed solve's remaining
pairs, so a cluster cut by the request is detected when its next member is
among them (best-effort: no Arnoldi attempt is added for it).

The ratios of the mode fields (``q_port``, ``participation``, the region,
principal and sheet shares) differentiate for isolated modes (ADR 0102) with
respect to materials, sheet fills and node coordinates (port surfaces
held), and the normalized mode fields themselves (through the
normalization multipliers of `gfem._eigen.ratios`) with respect to
materials and sheet fills, never the node coordinates; on one CPU process
with the ``direct`` linear solver: per mode with a cotangent, one deflated
GMRES solve of the singular system of the mode-field adjoint, ``T(k) lam0
= -w``, ``t^T lam0 = 0``, on the live factors the forward kept
(`gfem._eigen.adjoint`, ADR 0104), the load ``(w, kappa, explicit terms)``
of `gfem._eigen.ratios`, and the pairs
``(2 lam, x)`` and ``(conj(x), x)`` on the mode energy's ``Re(eps_r)``
terms (ADR 0103) and the sheets' L and C pieces. What does not
differentiate is refused by name: at trace time from the model, the
options and the wanted inputs (`EigenmodeRule.nondifferentiable_output`),
in the backward from the modes (a cluster, a near pair, a defective mode,
an infinite ``q_port``).

Post-processing follows Palace: electric energy ``E = 1/2 Re ∫ D* . E`` per
volume region plus ``C |V|^2 / 2`` per capacitive lumped port and ``1/2 ∫
w C_s |E_t|^2`` per capacitive `Impedance` sheet (the quadratic form of
its C piece, ADR 0103); a port's voltage is its driven voltage functional.
Each mode is scaled to ``E = 1 J`` with ``∫ D . E`` real and positive. An
anisotropic region's energy is also split along its principal axes (``1/2
ε0 eps_k ∫ |a_k . E|^2``, ADR 0055).

On a model with `gfem.Periodic` pairs the modes are solved on the periodic
space and their fields expanded to the unpaired space (``e = P x``,
`DrivenOperator.expand`); the backward pass's pairs are expanded too
(``P X``, and ``conj(P) Y`` on the left). With ``Eigenmode.wave_vector``
(Bloch) ``T(k)`` is general (ADR 0042): the refinement is two-sided, its
left eigenvectors ``Y`` replace ``X`` on the left of every gradient (``G =
Y^T T'(k_c) X``), only the ``direct`` linear solver runs it (transposed
solves on its factors), the real snap of a lossless cell tests that ``K``
and ``M`` are Hermitian, and the phase of a mode makes its largest
solve-space coefficient real and positive (``∫ E . E`` over a Bloch cell can
vanish).
"""

from __future__ import annotations

import cmath
import contextlib
import dataclasses
import math
from collections.abc import Callable, Generator, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem import _mpi, _petsc, _timing
from gfem._boundaries import (
    FloquetPort,
    Impedance,
    LumpedPort,
    WavePort,
    wave_port_surfaces,
)
from gfem._eigen import slepc
from gfem._eigen.adjoint import deflated_solve, polish, reshift_offset
from gfem._eigen.arpack import shift_invert
from gfem._eigen.kept import KeptOperator
from gfem._eigen.operator import EigenOperator
from gfem._eigen.ratios import (
    Ratios,
    energy_name,
    sheet_capacitors,
    sheet_inductors,
)
from gfem._eigen.refine import TRUSTED, Refined, refine
from gfem._eigen.seed import MIN_SUBSPACE, SeedBoundReached
from gfem._engine import dispersion, terms
from gfem._engine.cache import Entry, discretization
from gfem._engine.coefficients import anisotropic_regions
from gfem._engine.dispersion import energy_eps
from gfem._engine.domain import em_region
from gfem._engine.driven import (
    LOW_FREQUENCY_FLOOR,
    DrivenOperator,
    LowFrequencyFloor,
    driven_operator,
    lumped_terms,
    operator_terms,
    owned_dofs,
)
from gfem._engine.eigen import (
    DivFreeProjector,
    eigen_operator,
    element_energy,
    law_scale,
    principal_energy,
    projector,
    scale_per_tetrahedron,
)
from gfem._engine.fields import dof_shape, fe_space, one_process_dofs
from gfem._engine.runtime import task_manager
from gfem._engine.sensitivity import FieldPair, Term, is_lumped
from gfem._linalg import Kind, ResolvedLinearSolver
from gfem._linalg.layout import ONE_PROCESS, DofLayout, RankLayout
from gfem._materials import Material, PermittivityLaw
from gfem._periodic import check_bloch, check_wave_vector, pair_names
from gfem._sign import largest_entry, largest_value
from gfem._text import hertz
from gfem.conventions import MODE_ENERGY
from gfem.errors import (
    DegenerateModeError,
    EigenmodeConvergenceError,
    LinearSolverConvergenceError,
    LowFrequencyWarning,
    SolverUnavailableError,
)
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Eigenmode
from gfem.results import EigenmodeResult, Field

__all__ = ["DEFAULT_SOLVER", "EigenmodeRule"]

#: The linear solver for ``linear_solver="auto"``: shift-invert wants exact
#: inner solves.
DEFAULT_SOLVER: Kind = "direct"
#: Inner solves of an iterative backend run this much tighter than the
#: eigen tolerance.
_INNER = 1e-2
#: ... but never below this: the true preconditioned residual of the
#: shift-invert operators stays near 1e-13, a double-precision limit of the
#: operator, and BDDC + GMRES ends within 0.1 % of 1e-14, too thin a margin
#: to ask for. It also bounds PETSc Krylov inner solves, whose ``ksp_rtol``
#: floor is no lower.
_INNER_FLOOR = 1e-13
#: Seeds with Re k at or below this fraction of the shift are the curl-curl
#: null space, a lossy medium's relaxation modes (Re k = 0), mirror roots or
#: a conductor's linearization cluster (Re k near -sigma), not resonances. A
#: second-order absorbing wall's cluster lies near Re k = 2 sigma: this test
#: keeps it, and only the distance to the target ranks the modes ahead of it.
_STATIC = 1e-6
#: A converged pair of a law model whose curl energy is at most this fraction
#: of its electric energy (`EigenOperator.curl_ratio`) is a gradient field at
#: a zero of eps(omega), a longitudinal mode: dropped like a static pair
#: (ADR 0115 amendment; round-off reaches it only below k h_max ~ 3e-6,
#: transverse modes measured 0.55 and up).
_LONGITUDINAL = 1e-4
#: A refined mode closer than this many eigen tolerances (relative, and at
#: least ``SolverOptions.cluster_rtol``) to accepted modes, with this much of
#: its norm in their span, is one of them again.
_DUPLICATE = 10.0
#: The modal overlap from which a seed solve's error names the region: every
#: case below it converged in 47 shift solves or fewer, every hang had 70 or
#: more (ADR 0122).
_CROWDED = 10.0
_OVERLAP = 0.9
#: A duplicate with more than this fraction of its norm outside the span
#: landed on a degenerate eigenspace: that part is refined as the twin the
#: seed solve may not have returned (a frozen seed solve gives one Ritz
#: value for a pair, ticket 17 of quantum-metrics).
_TWIN = 1e-2
#: Ritz values this close (relative) are one multiple eigenvalue. The seed
#: solver returns an arbitrary basis of its eigenspace, as nearly parallel as
#: round-off makes it: the seeds are orthonormalized, or two members would
#: overlap like one mode refined twice. Round-off also splits the multiple
#: Ritz value: by up to 3e-8 for a frozen nonlinear seed at its 1e-3 seed
#: tolerance.
_MULTIPLE = 1e-6
#: A seed that needs correction steps is refined on a new factorization at
#: its own Ritz value when the current shift would contract its error by
#: less than this per step (the shift is nearer another Ritz value: residual
#: inverse iteration would drift to that mode).
_CONTRACTION = 0.5
#: Arnoldi runs, each asking for twice as many pairs as the last, before
#: too few modes near the target is an error.
_ATTEMPTS = 3
#: Arnoldi tolerance of a nonlinear problem's frozen seeds. Their roots are
#: off the true ones by the freeze error anyway (small for a conductor,
#: whose seeds reach backward error 1e-8 to 1e-4 now that the projection
#: fixes its potential on the wall, ADR 0097; up to 18 % for a second-order
#: absorbing wall) and every seed is refined, so converging them further
#: only spends shift solves. The linearization
#: cluster's solves are avoided by asking for few pairs (see the nev comment
#: in the forward pass, ADR 0035), not by this tolerance.
_SEED_TOLERANCE = 1e-3
#: Cotangents on the members of a cluster that agree to this (relative to
#: the largest) make an objective symmetric in the cluster to first order
#: (a smooth symmetric function's cotangents differ by O(gap)).
_SYMMETRIC_RTOL = 1e-3
#: The smallest relative gap of a mode in mode-field reach (with
#: ``SolverOptions.cluster_rtol`` if larger): an isolated mode (ADR 0102),
#: polished in the forward and given mode-field gradients (ADR 0104).
_ISOLATED = 1e-4
#: A Bloch cluster's ``G = Y^T T' X`` with a condition number above this is
#: singular: a defective eigenvalue, whose frequency has no derivative.
_DEFECTIVE = 1e8
#: A mode with ``|y^T T'(k) x|`` below this times ``|y| |T'(k) x|`` is
#: defective: the singular system of its mode-field adjoint has no solution
#: (ADR 0102).
_DEFECTIVE_FIELD = 1e-8
#: The largest ``SolverOptions.eigen_tolerance`` of a solve whose mode
#: fields differentiate: it keeps the forward polish's Newton start close.
_FIELD_TOLERANCE = 1e-10
#: The outputs that are quantities of the mode fields (mode-field outputs),
#: with their names on `EigenmodeResult` for messages.
_MODE_FIELD = {
    "q_port": "q_port",
    "participation": "participation",
    "bulk": "bulk_participation",
    "principal": "principal_participation",
    "sheet": "sheet_participation",
    "e_field": "e_fields (the normalized mode fields)",
}


@dataclasses.dataclass(frozen=True)
class _InputRefusal:
    """Why a mode-field output does not differentiate by a wanted input.

    Attributes:
        kind: ``"nodes"`` (no node gradient of the normalized field),
            ``"lumped"`` (a lumped value the output has no gradient by yet)
            or ``"law"`` (a permittivity law's parameter: the mode-field
            outputs read the mode energy, ADR 0112).
        message: The reason, starting "with respect to".
    """

    kind: Literal["nodes", "lumped", "law"]
    message: str


#: ``|v^H A u - conj(u^H A v)|`` below this times ``|A| |u| |v|`` for ``K``
#: and ``M``: a Hermitian (lossless) Bloch problem, whose eigenvalues are real.
_HERMITIAN = 1e-12


@dataclasses.dataclass(frozen=True)
class GapSet:
    """The modes a mode's relative gap is measured to (ADR 0104).

    Attributes:
        pool: The ``k`` of every converged, distinct pair the selection
            accepted, those cut beyond ``n_modes`` included, and of a mode
            found after the count was met: the modes themselves among them.
        guard: The guard mode's ``k``, or ``None`` when none was found.
        ritz: The seed solve's unrefined Ritz values (polynomial operators
            only): the gap set when the pool and the guard leave it empty.
    """

    pool: np.ndarray
    guard: complex | None
    ritz: np.ndarray

    def gap(self, k: complex) -> float:
        """``min |k - k_j| / max(|k|, |k_j|)`` over ``k``'s neighbours.

        The neighbours are the pool and the guard, ``k``'s own entry (the
        pool's nearest to it) left out; without any, the Ritz values but
        the nearest to ``k`` (its own seed). ``inf`` when there are none:
        the mode is isolated.
        """
        near = _others(self.pool, k)
        if self.guard is not None:
            near = np.append(near, self.guard)
        if not len(near):
            near = _others(self.ritz, k)
        if not len(near):
            return math.inf
        return float(np.min(np.abs(near - k) / np.maximum(abs(k), np.abs(near))))


def _others(values: np.ndarray, k: complex) -> np.ndarray:
    """``values`` without the one nearest ``k`` (``k``'s own)."""
    if not len(values):
        return values
    return np.delete(values, int(np.argmin(np.abs(values - k))))


@dataclasses.dataclass(frozen=True)
class DualRequest:
    """A participation dual `gfem.refine_mesh` asks the forward to solve.

    Attributes:
        start: The mode's index on the first mesh (the key it is kept by).
        omega: Its omega on the previous mesh, rad/s: the forward takes the
            mode nearest it.
        regions: The regions whose bulk participation the objective reads.
    """

    start: int
    omega: complex
    regions: tuple[str, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class ParticipationDual:
    """The mode-field dual of one region's bulk participation (ADR 0142).

    Attributes:
        mode: The mode's index on the result it was solved for.
        omega: The mode's omega, rad/s.
        lam: ``lam = lam0 - kappa u / (u^T t)`` on the solve space, for the
            result's normalized mode ``u``.
        weights: The load ``w = sum_r weights[r] conj(A_r u)``, ``A_r`` the
            region's ``Re eps_r`` mass form (`energy_parts`), by region.
        applies: The deflated solve's operator applies.
    """

    mode: int
    omega: complex
    lam: np.ndarray
    weights: dict[str, float]
    applies: int


class EigenmodeRule:
    """Host-side forward of an `Eigenmode` problem on a `Model`."""

    physics = "em"
    #: The operator family (a key of `sensitivity.PROPERTIES`).
    form = "eigenmode"
    #: Material properties the eigenmode forms read.
    reads = frozenset({"eps_r", "mu_r", "tan_delta", "conductivity"})
    #: Outputs whose cotangents the backward pass can handle: the eigenvalues
    #: (``cluster`` holds integer labels: its cotangent is always zero) and
    #: the mode-field outputs, within the reach `nondifferentiable_output`
    #: checks for the wanted inputs (ADR 0102).
    differentiable_outputs = frozenset({"omega", "cluster", *_MODE_FIELD})
    #: Outputs that are a rank's own piece under ranks (ADR 0063): the ranks
    #: compare their cotangents' shapes, not their values. The mode fields'
    #: ratios are values.
    pieces = frozenset({"e_field"})
    #: `nondifferentiable_output` takes the wanted input kinds and may
    #: return ``None`` (`gfem._adjoint.solve_vjp`).
    input_aware = True

    def __init__(
        self,
        model: Model,
        problem: Eigenmode,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.Eigenmode`).

        ``comm`` is the resolved communicator of a distributed solve: its seed
        solver is SLEPc, also for ``eigen_solver="auto"`` (ADR 0502).

        Raises:
            NotImplementedError: A feature this version cannot solve.
            ValueError: ``wave_vector`` is set without a periodic pair, or
                outside the span of the pairs' translations, or with a
                linear solver other than ``"direct"``.
            gfem.SolverUnavailableError: ``linear_solver="petsc"`` or
                ``eigen_solver="slepc"`` names a library that cannot be used,
                or under ``comm`` the MPI PETSc stack with SLEPc is missing.
        """
        if not isinstance(problem, Eigenmode):
            raise TypeError(f"Expected gfem.Eigenmode, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        # before the Bloch refusals: a wave vector without a pair is the
        # first thing to fix
        check_wave_vector(model, problem.wave_vector, "Eigenmode")
        if problem.wave_vector is not None:
            check_bloch(
                "Eigenmode",
                device=options.device,
                linear_solver=options.linear_solver,
                solvers=("direct",),
                reason=(
                    "cannot solve the transposed system that two-sided "
                    "refinement solves on the shift's factors"
                ),
            )
        # before any assembly
        self.linear = ResolvedLinearSolver.resolve(
            options, default=DEFAULT_SOLVER, problem="eigenmode", comm=comm
        )
        if comm is not None:
            _require_rank_slepc(comm)
        elif options.eigen_solver == "slepc":
            _petsc.require_slepc("eigen_solver")
        _check_supported(model)
        lumped = [p for p in model.ports if isinstance(p, LumpedPort)]
        #: Ports with a resistance (coupling Q) and with an inductance
        #: (participation), in model order.
        self.resistive = tuple(p.name for p in lumped if p.rlc[0] is not None)
        self.inductive = tuple(p.name for p in lumped if p.rlc[1] is not None)
        #: The surfaces of the `Impedance` sheets with an inductance (sheet
        #: participation), in model order.
        self.kinetic = tuple(
            b.surface
            for b in model.boundaries
            if isinstance(b, Impedance) and b.inductance is not None
        )
        mesh, active = model.mesh, model._domain.active  # noqa: SLF001
        #: Volume regions with tetrahedra in the solve domain: the field support.
        self.regions = tuple(r for r in mesh.volumes if active[mesh.elements(r)].any())
        #: The regions with an anisotropic material: the principal split.
        self.anisotropic = anisotropic_regions(model, self.regions)
        self._mesh = model.mesh
        self._materials = dict(model.materials)
        self._periodic_surfaces = pair_names(model._periodic)  # noqa: SLF001
        self._excluded = model._domain.excluded  # noqa: SLF001
        #: Why the node coordinates cannot be differentiated, or None.
        self.nodes_unsupported = terms.absorbing_shape_unsupported(model)
        #: The last backward's explicit terms in the lumped ports' values.
        self._explicit: dict[tuple, float] = {}
        #: Whether a region holds a permittivity law: only then can a
        #: converged pair be a longitudinal mode (`_longitudinal`).
        self._laws = bool(dispersion.law_regions(model))
        #: Participation duals for `gfem.refine_mesh` to solve on the
        #: forward's live factors (ADR 0142), and the duals solved.
        self.adjoint_loads: Sequence[DualRequest] | None = None
        self.adjoints: dict[tuple[int, str], ParticipationDual] = {}

    # --- forward -----------------------------------------------------------------

    def forward(
        self, model: Model, *, keep: bool
    ) -> tuple[dict[str, object], Entry | None]:
        """Find the modes.

        Args:
            model: The model, with concrete leaves.
            keep: Keep what the backward pass needs (the modes, their
                ``x^T T' x`` per cluster, whether a cluster reaches the
                guard mode, refined beyond ``n_modes`` for this, each mode's
                gap, and the eigen operator as the search leaves it, in a
                `KeptOperator`, instead of freeing it).

        Returns:
            ``{"omega": (n,) complex rad/s, "error": (n,), "q_port":
            (n_resistive, n), "participation": (n_inductive, n), "bulk":
            (n_regions, n), "principal": (n_anisotropic, n, 3), "sheet":
            (n_kinetic, n), "e_field": (ndof, n) complex, "cluster": (n,)
            int}``, modes nearest the
            target first, and the kept `Entry` (``None`` unless ``keep``).

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes
                converged, the modes found would skip a pair that nearly
                converged, or an isolated mode's polish did not converge.
            LinearSolverConvergenceError: A shift solve of an iterative
                linear solver did not converge (the message names
                ``SolverOptions.eigen_tolerance``), or a polish step's
                deflated solve was not accepted.
        """
        opts = self.options
        order = opts.order
        comm = self.comm
        # before any assembly: a target in a law's unresolved band (ADR 0115)
        with _agreed(ValueError):
            dispersion.check_band(model, self.problem.target, "target")
        with task_manager(opts.threads, comm):
            with _timing.phase("discretize") as span:
                op = driven_operator(
                    model, order, wave_vector=self.problem.wave_vector, comm=comm
                )
                built = op.built
                sigma = terms.wavenumber(2 * math.pi * self.problem.target, built.scale)
                eop = self.operator(op, sigma)
                span.set(ndof=owned_dofs(op))
            divfree = None
            entry = None
            try:
                eop.factorize()  # before the projector's setup
                divfree = projector(model, op, self.problem.wave_vector)
                # the whole search is one solve: its shift solves are internal
                with (
                    _timing.phase("solve", iterations=()),
                    _agreed(EigenmodeConvergenceError),
                ):
                    modes, guard, gaps, unpolished = self._find(
                        eop, divfree, sigma, guard=keep or self.polishable
                    )
                per_hz = const.c / built.scale / (2 * math.pi)
                for j, mode in enumerate(modes):
                    f = mode.k.real * per_hz
                    with _agreed(ValueError):
                        dispersion.check_band(model, f, f"mode {j} at")
                        # the mode energy's weight (ADR 0109)
                        dispersion.check_energy(model, f, f"mode {j} at")
                outputs, x = self._postprocess(model, op, divfree, modes)
                ks = [m.k for m in modes]
                lowest = min(k.real for k in ks) * const.c / built.scale
                _check_low_frequency(
                    model,
                    self.problem.target,
                    lowest / (2 * math.pi),
                    stacklevel=2,
                    comm=comm,
                )
                outputs["cluster"] = clusters(ks, opts.cluster_rtol)
                if self.adjoint_loads:
                    self._ride_duals(
                        model,
                        op,
                        eop,
                        x=x,
                        ks=ks,
                        found=(outputs["cluster"], guard, gaps, unpolished),
                    )
                if keep:
                    y = None
                    if eop.general:
                        # every pair of a general operator carries its left one
                        y = np.stack(
                            [m.x if m.y is None else m.y for m in modes], axis=1
                        )
                    # the backward maps vectors with the operator as it is
                    # left (ADR 0104): the entry's holder owns it from here
                    store = None if op.fixed is None else op.fixed.store
                    entry = self._entry(
                        KeptOperator(eop, store),
                        ks,
                        x,
                        y,
                        labels=outputs["cluster"],
                        guard=guard,
                        gaps=gaps,
                        polish_failed=unpolished,
                    )
            except LinearSolverConvergenceError as err:
                raise self._inner_failure(err) from err
            finally:
                if entry is None:
                    eop.free_handle()
                if divfree is not None:
                    divfree.free()
        return outputs, entry

    def _ride_duals(
        self,
        model: Model,
        op: DrivenOperator,
        eop: EigenOperator,
        *,
        x: np.ndarray,
        ks: list[complex],
        found: tuple[np.ndarray, Refined | None, GapSet, np.ndarray],
    ) -> None:
        """Solve `adjoint_loads`' participation duals on the live factors.

        Each request takes the mode nearest its previous omega, as
        `gfem.refine_mesh` follows it; a mode out of mode-field reach
        (`isolated`) gets none, and the estimate raises for it. ``found``
        holds the modes' cluster labels, the guard mode, the gap set and
        the failed polishes (`_find`).
        """
        labels, guard, gaps, unpolished = found
        rtol = self.options.cluster_rtol
        guard_k = None if guard is None else guard.k
        gap = np.array(
            [0.0 if f else gaps.gap(z) for z, f in zip(ks, unpolished, strict=True)]
        )
        reach = isolated(labels, _incomplete(ks, labels, guard_k, rtol), gap, rtol)
        per_k = const.c / op.built.scale
        omegas = np.array(ks, dtype=complex) * per_k
        self.adjoints = {}
        for request in self.adjoint_loads or ():
            j = int(np.argmin(np.abs(omegas - request.omega)))
            if not reach[j]:
                continue
            duals = self.participation_duals(
                model,
                op,
                eop,
                u=x[:, j],
                k=ks[j],
                gap=gaps.gap(ks[j]),
                mode=j,
                regions=request.regions,
            )
            for region, dual in duals.items():
                self.adjoints[request.start, region] = dual

    def participation_duals(
        self,
        model: Model,
        op: DrivenOperator,
        eop: EigenOperator,
        *,
        u: np.ndarray,
        k: complex,
        gap: float,
        mode: int,
        regions: Sequence[str],
    ) -> dict[str, ParticipationDual]:
        """The dual of each region's bulk participation of mode ``(k, u)`` (ADR 0142).

        Per region, the load of a unit cotangent on its participation
        (`Ratios.load`), the deflated solve ``T(k) lam0 = -w``, ``t^T lam0 =
        0`` on ``eop``'s factors (`deflated_solve`), and ``lam = lam0 - kappa
        u / (u^T t)``: the backward's own adjoint. ``u`` is the mode on the
        solve space at any scale (the ratios are invariant). The load's
        volume weights go with it, for the dual flux's equilibration.
        """
        ratios = Ratios(
            model,
            op,
            regions=self.regions,
            anisotropic=self.anisotropic,
            resistive=self.resistive,
            inductive=self.inductive,
            kinetic=self.kinetic,
        )
        ut = u @ eop.derivative(k, u)
        omega = complex(k * const.c / op.built.scale)
        out = {}
        for region in regions:
            unit = np.zeros(len(self.regions))
            unit[self.regions.index(region)] = 1.0
            load = ratios.load(u, k, {"bulk": unit}, mode=mode)
            solved = deflated_solve(eop, u, None, -load.w, k=k, gap=gap, mode=mode)
            weights = {
                r: load.energy[energy_name(r)]
                for r in self.regions
                if energy_name(r) in load.energy
            }
            out[region] = ParticipationDual(
                mode=mode,
                omega=omega,
                lam=solved.z - load.kappa * u / ut,
                weights=weights,
                applies=solved.applies,
            )
        return out

    def _inner_failure(
        self, err: LinearSolverConvergenceError
    ) -> LinearSolverConvergenceError:
        """``err`` of a shift solve, with the remedy that acts on this path.

        The linear solver's message offers "a looser tolerance", but the
        inner solves run at `_inner_tolerance`, set by
        ``SolverOptions.eigen_tolerance`` unless ``tolerance`` is tighter.
        """
        remedy = (
            "a looser SolverOptions.eigen_tolerance (the inner solves run at "
            "min(tolerance, 0.01 * eigen_tolerance), here "
            f"{self._inner_tolerance():g})"
        )
        message = str(err)
        if "a looser tolerance" in message:
            message = message.replace("a looser tolerance", remedy)
        else:
            message = f"{message} For an eigenmode solve, try {remedy}."
        return LinearSolverConvergenceError(message)

    def _find(
        self, eop: EigenOperator, divfree, sigma: float, *, guard: bool
    ) -> tuple[list[Refined], Refined | None, GapSet, np.ndarray]:
        """The ``n_modes`` modes, the guard mode, the gap set, and failed polishes.

        Seed solves at the target (each attempt asking for twice as many
        pairs as the last) and the selection of their refined pairs
        (`_select`). A seed solve that stops at its restart bound ends the
        attempts: its converged pairs are selected, and a wider retry would
        converge none (ADR 0122). ``guard``: refine one mode beyond
        ``n_modes`` (``None`` when no seed gave one). Where the forward
        polishes (`polishable`), each isolated mode is polished (`_polish`)
        on the factors the search leaves, before a lossless problem's ``k``
        are snapped real; the last return flags the modes whose polish
        failed (kept unpolished).

        Raises:
            EigenmodeConvergenceError: Fewer than ``n_modes`` modes
                converged, or the modes found would skip a pair that nearly
                converged (`_skipped`).
        """
        opts = self.options
        n_modes = self.problem.n_modes
        seed_tolerance = opts.eigen_tolerance
        # the companion problem also holds each mode's mirror root
        # -conj(k), |k + sigma| from the shift (behind its mode, about
        # 2 sigma away for a mode near the target), and pairs near
        # k = 0 about sigma away; they take slots in front of wanted
        # modes farther than that, so a damped linear seed asks for
        # twice as many pairs. A nonpolynomial seed asks for one pair
        # beyond n_modes, the guard's slot: behind its modes lies the
        # linearization cluster, which costs shift solves to converge
        # and holds no mode (ADR 0035).
        if eop.nonpolynomial:
            nev = n_modes + 1
            seed_tolerance = max(seed_tolerance, _SEED_TOLERANCE)
        else:
            nev = max(2 * n_modes, n_modes + 2)
            nev *= 2 if eop.seed_damped else 1
        found = None
        bound = None
        for _ in range(_ATTEMPTS):
            if eop.shift != sigma:
                # a retry is a seed solve: at the target, on a
                # factorization counted for the seed (refinement
                # reshifted it)
                eop.reshift(sigma, solves=MIN_SUBSPACE)
            try:
                k, vectors = self._shift_invert(eop, divfree, nev, tol=seed_tolerance)
            except SeedBoundReached as err:
                bound = err
                k, vectors = err.k, err.vectors
            modes, failed, found, pool, ritz = self._select(
                eop, k, vectors, sigma, guard=guard
            )
            if len(modes) == n_modes or failed or bound is not None:
                break
            nev *= 2  # too few candidates near the target: look wider
        if bound is not None and len(modes) < n_modes:
            raise EigenmodeConvergenceError(self._seed_message(bound))
        skipped = self._skipped(modes, failed, sigma, layout=eop.layout)
        if len(modes) < n_modes or skipped:
            raise EigenmodeConvergenceError(
                self._convergence_message(eop, modes, failed, skipped)
            )
        guard_k = None if found is None else found.k
        unpolished = np.zeros(len(modes), dtype=bool)
        if self.polishable:
            gap_set = GapSet(pool, guard_k, ritz)
            modes, unpolished = self._polish(eop, modes, gap_set, guard_k)
        lossless = _is_hermitian if eop.general else _is_real
        if not eop.seed_damped and not eop.nonpolynomial and lossless(eop):
            modes = [dataclasses.replace(m, k=m.k.real) for m in modes]
            if found is not None:
                found = dataclasses.replace(found, k=found.k.real)
            pool = pool.real.astype(complex)
        gaps = GapSet(pool, None if found is None else found.k, ritz)
        return modes, found, gaps, unpolished

    @property
    def polishable(self) -> bool:
        """Whether the forward polishes its isolated modes (ADR 0104).

        On the CPU with ``linear_solver="direct"``, without a periodic pair
        or a Bloch wave vector: where mode-field gradients reach (ADR 0102).
        """
        return (
            self.options.device == "cpu"
            and self.linear.kind == "direct"
            and self.problem.wave_vector is None
            and not self._periodic_surfaces
        )

    def _polish(
        self,
        eop: EigenOperator,
        modes: list[Refined],
        gaps: GapSet,
        guard: complex | None,
    ) -> tuple[list[Refined], np.ndarray]:
        """``modes`` with each isolated one polished (`polish`), and which failed.

        A mode is isolated (`isolated`) when it is alone in its cluster, the
        guard mode is not in its cluster, and its relative gap is at least
        `_ISOLATED` (``cluster_rtol`` if larger). The guard mode is never
        polished. A polish that does not converge (or whose solve is not
        accepted) is evidence of a neighbour the gap set missed, such as the
        twin of a degenerate pair the seed solve never returned: the mode
        is kept as mode refinement left it and flagged, and its entry
        records it as not isolated (ADR 0104, amendment of ticket 04). A
        solve never raises because of polishing.
        """
        ks = [m.k for m in modes]
        rtol = self.options.cluster_rtol
        labels = clusters(ks, rtol)
        gap = np.array([gaps.gap(k) for k in ks])
        reach = isolated(labels, _incomplete(ks, labels, guard, rtol), gap, rtol)
        failed = np.zeros(len(modes), dtype=bool)
        out = []
        for i, m in enumerate(modes):
            if reach[i]:
                try:
                    m = polish(eop, m, float(gap[i]), index=i)  # noqa: PLW2901
                except (EigenmodeConvergenceError, LinearSolverConvergenceError):
                    failed[i] = True
            out.append(m)
        return out, failed

    def _shift_invert(
        self, op: EigenOperator, project, nev: int, *, tol: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """The seed solve on ``SolverOptions.eigen_solver`` (see `shift_invert`).

        Above one rank ``"auto"`` is SLEPc (ADR 0502).
        """
        if self.options.eigen_solver == "slepc" or self.comm is not None:
            options = self.options.petsc_options
            return slepc.shift_invert(op, project, nev, tol=tol, options=options)
        # the module's name, looked up now: tests replace it
        return shift_invert(op, project, nev, tol=tol)

    def operator(self, op: DrivenOperator, sigma: float) -> EigenOperator:
        """The eigen operator ``T(k)`` of the driven operator, shifted at ``sigma``.

        Nothing is factorized until the handle is first used
        (`gfem._engine.eigen.eigen_operator`).
        """
        return eigen_operator(
            op,
            sigma,
            linear=self.linear,
            tolerance=self._inner_tolerance(),
            numbers=self._start_numbers(op),
        )

    def _numbers(self) -> np.ndarray | None:
        """Each field row's one-process number under ranks; ``None`` on one process.

        `gfem._engine.fields.one_process_dofs`, memoized per topology: the rows
        of a result field, which are the solve space's without periodic pairs.
        """
        if self.comm is None:
            return None
        return one_process_dofs(
            self._mesh, self.regions, self._excluded, self.options.order, self.comm
        )

    def _start_numbers(self, op: DrivenOperator) -> np.ndarray | None:
        """A number per solve-space DOF for the norm start; ``None`` on one process.

        Without periodic pairs each DOF's one-process number (`_numbers`). On a
        periodic space, whose DOFs no one-process number names (a swapped
        partition changes the seam's basis), the layout's global numbers
        (``ParallelDofs.EnumerateGlobally``): they depend on the partition, and
        only the norm estimates read them.
        """
        if self.comm is None or op.expansion is None:
            return self._numbers()
        numbers, _ = _layout(op).pardofs.EnumerateGlobally()  # ty: ignore[unresolved-attribute]
        return np.asarray(numbers, dtype=np.int64)

    def _inner_tolerance(self) -> float:
        """The tolerance of the shift solves of an iterative linear solver."""
        opts = self.options
        return min(opts.tolerance, max(_INNER * opts.eigen_tolerance, _INNER_FLOOR))

    def _entry(
        self,
        holder: KeptOperator,
        ks: list[complex],
        x: np.ndarray,
        y: np.ndarray | None,
        *,
        labels: np.ndarray,
        guard: Refined | None,
        gaps: GapSet,
        polish_failed: np.ndarray | None = None,
    ) -> Entry:
        """What the backward pass needs: the kept operator, the modes and their T'.

        ``x`` holds the modes on the solve space, ``y`` their left
        eigenvectors (``None``: ``x``, complex symmetric). ``gram`` is block
        diagonal by cluster: ``y_i^T T'(k_c) x_j`` for modes i, j of cluster
        c, with ``k_c`` the cluster's mean; under ranks a pairing of loads
        and states (`DofLayout.pair`), the same on every rank. ``incomplete``
        flags the modes of a cluster the guard mode belongs to; ``gap`` each
        mode's relative gap to its gap set (`GapSet.gap`, ``inf`` without a
        neighbour), 0 for a mode whose polish failed; ``polish_failed`` flags
        those modes (`isolated` is then false for them: the backward names the
        failure).
        """
        left = x if y is None else y
        labels = np.asarray(labels)
        k = np.array(ks, dtype=complex)
        gram = np.zeros((len(k), len(k)), dtype=complex)
        rtol = self.options.cluster_rtol
        incomplete = _incomplete(ks, labels, None if guard is None else guard.k, rtol)
        op = cast("EigenOperator", holder.operator)
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            mean = k[members].mean()
            tx = np.stack([op.derivative(mean, x[:, j]) for j in members], axis=1)
            if self.comm is None:
                gram[np.ix_(members, members)] = left[:, members].T @ tx
            else:
                # the loads T'(k_c) x paired with the states y: agreed values
                gram[np.ix_(members, members)] = op.layout.pair(tx, left[:, members]).T
        fields = {
            "x": x,
            "k": k,
            "cluster": labels,
            "gram": gram,
            "incomplete": incomplete,
            "gap": np.array([gaps.gap(z) for z in k]),
            "polish_failed": np.zeros(len(k), dtype=bool)
            if polish_failed is None
            else np.asarray(polish_failed, dtype=bool),
        }
        fields["gap"][fields["polish_failed"]] = 0.0
        if y is not None:
            fields["y"] = y
        return Entry(self.physics, handle=holder, fields=fields)

    def _select(
        self,
        op: EigenOperator,
        k: np.ndarray,
        vectors: np.ndarray,
        sigma: float,
        *,
        guard: bool = False,
    ) -> tuple[list[Refined], list[Refined], Refined | None, np.ndarray, np.ndarray]:
        """Refine the seeds nearest ``sigma`` until ``n_modes`` distinct ones.

        Seeds (and refined pairs) with ``Re k`` near zero or negative are the
        curl-curl null space, a lossy medium's relaxation modes, mirror
        roots or a conductor's linearization cluster, not resonances: they
        are dropped, and so is a converged longitudinal mode of a law model
        (`_longitudinal`).
        A second-order absorbing wall's cluster has ``Re k`` near ``2
        sigma``, so it is not dropped; the modes nearer the target come
        first. A seed far from the current shift whose refinement gives
        nothing to accept as its mode (see `_lost`) is refined again on a
        factorization at its own Ritz value. A converged pair that is not
        static and not in the span of the accepted modes (`_in_span`) is
        never discarded: the first refinement of such a seed is accepted
        too when it is one (ADR 0097), so the accepted modes can outnumber
        ``n_modes``; they are sorted by distance to the target and cut. A
        pair in that span with more than `_TWIN` of its norm outside it
        landed on a degenerate eigenspace: its outside part is refined and,
        converged, is the twin the seed solve did not give.
        With ``guard``, the remaining seeds are refined (without
        refactorizing) until one more distinct mode converges: the guard
        mode, or the nearest mode cut off when there is one. It is
        best-effort: none is found when no remaining seed converges to a
        mode.

        Returns:
            The accepted modes, nearest ``sigma`` first (at most
            ``n_modes``), the seeds that did not converge, the guard
            mode (``None`` without ``guard`` or when no seed gave one), the
            pool of the gap set (`GapSet`): the ``k`` of every converged,
            distinct pair accepted, those cut beyond ``n_modes`` included,
            and of a mode found after the count was met, nearest ``sigma``
            first; and the seeds' unrefined Ritz values (none for a
            nonpolynomial operator, whose frozen seeds are too rough).
        """
        n_modes = self.problem.n_modes
        tolerance = self.options.eigen_tolerance
        seeds = [j for j in range(len(k)) if k[j].real > _STATIC * sigma]
        seeds.sort(key=lambda j: abs(k[j].real - sigma))
        layout = op.layout
        vectors = _orthonormal_multiples(k, vectors, layout=layout)
        accepted: list[Refined] = []
        failed: list[Refined] = []
        extra = None
        window = self._window()
        steps = self.options.max_refine_iterations

        def distinct(mode: Refined) -> Refined | None:
            # a converged pair in the span of the accepted modes is a duplicate
            # unless its part outside the span is more than round-off: then it
            # landed on a degenerate eigenspace whose twin the seed solve
            # missed, and that part, refined, is the twin (ticket 17)
            if not _in_span(mode, accepted, window, layout=layout):
                return mode
            rest = _outside(mode, accepted, window, layout=layout)
            if rest is None:
                return None
            twin = refine(
                op,
                mode.k,
                rest,
                tolerance=tolerance,
                max_iterations=steps,
                reshift=False,
            )
            if (
                twin.converged
                and twin.k.real > _STATIC * sigma
                and not self._longitudinal(op, twin)
                and not _in_span(twin, accepted, window, layout=layout)
            ):
                return twin
            return None

        for j in seeds:
            full = len(accepted) >= n_modes
            if full and not guard:
                break
            seed = complex(k[j])
            far = _contraction(seed, op.shift, k) > _CONTRACTION
            mode = refine(
                op,
                seed,
                vectors[:, j],
                tolerance=tolerance,
                max_iterations=steps,
                reshift=not full,
                reshift_first=far,
            )
            if (
                far
                and not full
                and (
                    _lost(
                        mode,
                        seed,
                        k,
                        sigma,
                        accepted=accepted,
                        window=window,
                        layout=layout,
                    )
                    or self._longitudinal(op, mode)
                )
            ):
                # a frozen seed rougher than TRUSTED, refined on a shift nearer
                # another mode, wanders where round-off takes it (onto an
                # accepted mode, a mirror root, or another seed's mode): refine
                # it again on a factorization at its own Ritz value. Where it
                # wandered may be a mode no other seed gives: it is kept
                if (
                    mode.converged
                    and mode.k.real > _STATIC * sigma
                    and not self._longitudinal(op, mode)
                ):
                    kept = distinct(mode)
                    if kept is not None:
                        accepted.append(kept)
                op.reshift(seed)
                mode = refine(
                    op, seed, vectors[:, j], tolerance=tolerance, max_iterations=steps
                )
            if mode.k.real <= _STATIC * sigma or self._longitudinal(op, mode):
                continue
            if not mode.converged:
                if not full:
                    failed.append(mode)
                continue
            mode = distinct(mode)
            if mode is None:
                continue
            if full:
                extra = mode
                break
            accepted.append(mode)
        accepted.sort(key=lambda m: abs(m.k.real - sigma))
        pool = [m.k for m in accepted] + ([] if extra is None else [extra.k])
        pool.sort(key=lambda z: abs(z.real - sigma))
        ritz = [complex(k[j]) for j in seeds] if not op.nonpolynomial else []
        if guard and len(accepted) > n_modes:
            # nearer the target than a guard found after the count was met
            extra = accepted[n_modes]
        return (
            accepted[:n_modes],
            failed,
            extra,
            np.array(pool, dtype=complex),
            np.array(ritz, dtype=complex),
        )

    def _longitudinal(self, op: EigenOperator, mode: Refined) -> bool:
        """Whether ``mode`` is a converged longitudinal mode of a law model.

        A gradient field at a zero of a law's eps(omega): its curl energy
        is at most `_LONGITUDINAL` of its electric energy
        (`EigenOperator.curl_ratio`, ADR 0115). One ``K`` apply; never on a
        model without a law, where eps has no zero.
        """
        if not self._laws or not mode.converged:
            return False
        return op.curl_ratio(mode.k, mode.x) <= _LONGITUDINAL

    def _window(self) -> float:
        """Relative distance in ``k`` within which two refined pairs can be one mode."""
        return max(_DUPLICATE * self.options.eigen_tolerance, self.options.cluster_rtol)

    def _skipped(
        self,
        accepted: list[Refined],
        failed: list[Refined],
        sigma: float,
        *,
        layout: DofLayout = ONE_PROCESS,
    ) -> list[Refined]:
        """The unconverged pairs that the accepted modes may have skipped.

        A pair that stopped within `TRUSTED` of an eigenpair, nearer the
        target than the farthest accepted mode and outside the span of the
        accepted modes, is a mode the set would silently leave out. Rougher
        pairs (members of a linearization cluster) are not modes.
        """
        if len(accepted) < self.problem.n_modes:
            return []  # the count already raises
        reach = max(abs(m.k.real - sigma) for m in accepted)
        window = self._window()
        return [
            m
            for m in failed
            if m.error <= TRUSTED
            and abs(m.k.real - sigma) < reach
            and not _in_span(m, accepted, window, layout=layout)
        ]

    def _convergence_message(
        self,
        op: EigenOperator,
        accepted: list[Refined],
        failed: list[Refined],
        skipped: list[Refined] | None = None,
    ) -> str:
        """Why the solve raises: too few modes, or ``skipped`` (see `_skipped`)."""
        c = const.c / op.scale / (2 * math.pi)
        target = hertz(self.problem.target)
        tolerance = (
            f"backward error {self.options.eigen_tolerance:g} "
            "(SolverOptions.eigen_tolerance)"
        )
        if skipped:
            farthest = hertz(accepted[-1].k.real * c)  # sorted by distance
            head = (
                f"Eigenmode found {len(accepted)} mode(s) near {target} to "
                f"{tolerance}, the farthest at {farthest}, but a pair "
                "nearer the target did not converge: the modes found would "
                "skip it."
            )
        else:
            head = (
                f"Eigenmode found {len(accepted)} of the {self.problem.n_modes} "
                f"requested mode(s) near {target} to {tolerance}."
            )
        lines = [
            head,
            *(
                f"  not converged: {hertz(m.k.real * c)}, best backward "
                f"error {m.error:.3g} after {m.iterations} refinement steps"
                for m in (skipped or failed)
            ),
        ]
        lines.append(
            "Remedies: move Eigenmode.target closer to the wanted modes; loosen "
            "SolverOptions.eigen_tolerance or raise max_refine_iterations; for a "
            "mode with Q below about 2, find the resonance from a gfem.Driven "
            "sweep instead."
        )
        return "\n".join(lines)

    def _seed_message(self, bound: SeedBoundReached) -> str:
        """Why the solve raises after a seed solve stopped at its restart bound.

        Names the lossy region with the largest modal overlap at the target
        (`_modal_overlap`) when it is `_CROWDED` or more, and for a law
        region its term nearest the target: a diagnosis only.
        """
        target = self.problem.target
        head = (
            f"Eigenmode's seed solve near {hertz(target)} did not converge in "
            f"{bound.restarts} restarts ({bound.applies} shift solves): the modes "
            "there are closer together than their linewidth, so the nearest ones "
            "cannot be told apart."
        )
        lines = [head]
        overlaps = {
            region: _modal_overlap(self._mesh, region, material, target)
            for region, material in self._materials.items()
            if region in self.regions and isinstance(material, Material)
        }
        if overlaps:
            region = max(overlaps, key=lambda r: overlaps[r])
            if overlaps[region] >= _CROWDED:
                eps_r = self._materials[region].eps_r
                term = (
                    f" ({dispersion.nearest_term(eps_r, target)})"
                    if isinstance(eps_r, PermittivityLaw)
                    else ""
                )
                lines.append(
                    f"Region {region!r} holds about {overlaps[region]:.0f} modes "
                    f"within one linewidth of the target{term}."
                )
        lines.append(
            "Remedies: move Eigenmode.target where the modes are separated (away "
            "from a lossy resonance); find overlapping resonances from a "
            "gfem.Driven sweep; raise eps_max_it in SolverOptions.petsc_options "
            "(SLEPc) only if the modes are known to be separated."
        )
        return "\n".join(lines)

    # --- post-processing ---------------------------------------------------------

    def _postprocess(
        self,
        model: Model,
        op,
        divfree: DivFreeProjector,
        modes: list[Refined],
    ) -> tuple[dict[str, object], np.ndarray]:
        """The outputs, and the modes scaled to them on the solve space.

        ``e_field`` holds the modes expanded to the unpaired space on a
        periodic model (``P x``). A mode is a state, an energy form's or a
        port's functional a load: their products are the layout's pairings.
        Under ranks a rank integrates its own tetrahedra's energies, summed over
        the ranks per region (`_volume_energies`), the sign follows one process's
        largest entry (`gfem._sign.largest_value`), and ``e_field`` is the rank's
        piece, padded (`_piece`).
        """
        built, space, order = op.built, op.space, op.order
        layout, numbers = _layout(op), self._numbers()
        scale = built.scale
        ports = {data.port.name: data for data in op.ports}
        voltage = {
            name: terms.port_vectors(space, built, order, data)[1]
            for name, data in ports.items()
        }
        mesh = model.mesh
        density = element_energy(model, op)
        sheets = _sheet_energy(op)
        split = (
            principal_energy(model, op, self.anisotropic) if self.anisotropic else None
        )
        n = len(modes)
        omega = np.array([m.k * const.c / scale for m in modes], dtype=complex)
        solve = np.zeros((space.ndof, n), dtype=complex)
        bulk = np.zeros((len(self.regions), n))
        principal = np.zeros((len(self.anisotropic), n, 3))
        q_port = np.zeros((len(self.resistive), n))
        participation = np.zeros((len(self.inductive), n))
        for j, mode in enumerate(modes):
            x = self._phase(mode.x, op, divfree, numbers)
            # electric energy per element (gfem order) at this amplitude
            per_tet = np.zeros(len(mesh.tetrahedra))
            per_tet[built.tetrahedron_order] = (
                0.5 * const.epsilon_0 * scale * np.real(density(x))
            )
            per_tet *= self._law_factor(model, mode.k, scale)
            volts = {name: complex(layout.pair(v, x)) for name, v in voltage.items()}
            capacitors = sum(
                0.5 * data.port.rlc[2] * abs(volts[name]) ** 2
                for name, data in ports.items()
                if data.port.rlc[2] is not None
            )
            per_axis = None
            if split is not None:
                # the same energy along each principal axis
                per_axis = np.zeros((len(mesh.tetrahedra), 3))
                per_axis[built.tetrahedron_order] = (
                    0.5 * const.epsilon_0 * scale * np.real(split(x))
                )
            volume, regions, axes = self._volume_energies(mesh, per_tet, per_axis)
            # + 0.0 without a capacitive sheet leaves the sum bit-identical
            energy = volume + capacitors + (0.0 if sheets is None else sheets(x))
            amplitude = math.sqrt(MODE_ENERGY / energy)
            solve[:, j] = x * amplitude
            for r in range(len(self.regions)):
                bulk[r, j] = regions[r] / energy
            for r in range(len(self.anisotropic)):
                principal[r, j] = axes[r] / energy
            w = omega[j].real
            for p, name in enumerate(self.resistive):
                resistance = ports[name].port.rlc[0]
                kappa = abs(volts[name]) ** 2 / (2 * resistance * energy)
                q_port[p, j] = w / kappa if kappa > 0 else math.inf
            for p, name in enumerate(self.inductive):
                inductance = ports[name].port.rlc[1]
                current = volts[name] / (1j * w * inductance)
                ratio = inductance * abs(current) ** 2 / (2 * energy)
                participation[p, j] = math.copysign(ratio, volts[name].real)
        outputs = {
            "omega": omega,
            "error": np.array([m.error for m in modes]),
            "q_port": q_port,
            "participation": participation,
            "bulk": bulk,
            "principal": principal,
            "sheet": self._sheet_shares(op, solve, omega),
            "e_field": self._piece(model, op.expand(solve)),
        }
        return outputs, solve

    def _phase(
        self,
        x: np.ndarray,
        op: DrivenOperator,
        divfree: DivFreeProjector,
        numbers: np.ndarray | None,
    ) -> np.ndarray:
        """``x`` with the phase and sign of the mode convention (`gfem.conventions`).

        ``∫ E . E`` real and positive, then the largest entry's real part
        positive (one process's entry under ranks, `largest_value`). On a
        periodic model that entry is the expanded field's (``P x`` on the field
        rows, ``numbers`` theirs), on one process and under ranks alike: a
        partition can change the periodic basis on the seam, never ``P x``.
        """
        if self.problem.wave_vector is not None:
            # a Bloch mode's ∫ E . E depends on the cell and can vanish:
            # its largest coefficient is made real and positive instead
            # (set exactly: the product's imaginary part is round-off)
            i = largest_entry(x)
            largest = x[i]
            x = x * (np.conj(largest) / abs(largest))
            x[i] = abs(largest)
            return x
        layout = _layout(op)
        s = layout.pair(divfree.energy(x), x)
        x = x * np.exp(-0.5j * np.angle(s))
        if op.expansion is None:
            reference = largest_value(x, layout, numbers)
        else:
            reference = largest_value(op.expand(x), self._field_layout(), numbers)
        if reference.real < 0:
            x = -x
        return x

    def _field_layout(self) -> DofLayout:
        """The field rows' layout across ranks; `ONE_PROCESS` on one process.

        The plain mesh's H(curl) space of `fe_space`, whose rows are a result
        field's (`gfem._engine.driven.Expansion.field_rows`), memoized with it.
        """
        if self.comm is None:
            return ONE_PROCESS
        still = Mesh(self._mesh.imported_nodes, self._mesh._topology)  # noqa: SLF001
        regions, excluded, order = self.regions, self._excluded, self.options.order
        _, space, _ = fe_space(still, regions, excluded, "hcurl", order, comm=self.comm)
        disc = discretization(still, (), self.comm)
        key = (("field", tuple(regions), tuple(excluded)), "hcurl", order, "layout")
        return disc.space(key, lambda: RankLayout(space, self.comm))

    def _volume_energies(
        self, mesh: Mesh, per_tet: np.ndarray, per_axis: np.ndarray | None
    ) -> tuple[float, list, list]:
        """The volume energy, each region's, and each anisotropic region's per axis.

        ``per_tet`` and ``per_axis`` hold this rank's tetrahedra (gfem order;
        zero elsewhere). Under ranks the sums are rank sums of the local ones, one
        reduction of ``1 + n_regions + 3 n_anisotropic`` values, never of a whole
        array.
        """
        volume = per_tet.sum()
        regions = [per_tet[mesh.elements(region)].sum() for region in self.regions]
        axes = []
        if per_axis is not None:
            axes = [
                per_axis[mesh.elements(region)].sum(axis=0)
                for region in self.anisotropic
            ]
        if self.comm is None:
            return volume, regions, axes
        local = np.concatenate([[volume], regions, np.ravel(axes)])
        total = _mpi.reduce_bcast(self.comm, local)
        n = len(self.regions)
        return (
            float(total[0]),
            list(total[1 : 1 + n]),
            list(total[1 + n :].reshape(-1, 3)),
        )

    def _piece(self, model: Model, columns: np.ndarray) -> np.ndarray:
        """The mode columns padded with zero rows to one shape on all ranks.

        A rank's piece of a field (ADR 0063).
        """
        if self.comm is None:
            return columns
        piece = np.zeros((self._field_width(model), columns.shape[1]), complex)
        piece[: columns.shape[0]] = columns
        return piece

    def _field_width(self, model: Model) -> int:
        """A mode field's rows on every rank: `dof_shape`'s width."""
        return dof_shape(
            model.mesh,
            self.regions,
            self._excluded,
            "hcurl",
            self.options.order,
            comm=self.comm,
        )[0]

    def _law_factor(self, model: Model, k: complex, scale: float) -> np.ndarray | float:
        """The mode energy's factor per tetrahedron at the mode ``k`` (ADR 0109).

        eps_W / eps_inf on a law region (its energy permittivity at ``Re
        omega`` over the eps_inf that `element_energy` integrates), 1
        elsewhere; the number 1 without a law (bit-identical).
        """
        if not self._laws:
            return 1.0
        omega = k.real * const.c / scale
        return scale_per_tetrahedron(model, law_scale(model, omega, energy_eps))

    def _sheet_shares(
        self, op: DrivenOperator, solve: np.ndarray, omega: np.ndarray
    ) -> np.ndarray:
        """``(n_kinetic, n)`` sheet participations of the modes scaled to 1 J.

        A sheet's L piece is ``mu0 scale / L_s ∫ w u_t . v_t`` (the fill
        ``w`` as in the operator, ADR 0013): ``∫ w |E_t|^2 / (2 omega^2
        L_s)`` is its form over ``2 mu0 scale omega^2``.
        """
        out = np.zeros((len(self.kinetic), len(omega)))
        layout = _layout(op)
        for s, piece in enumerate(sheet_inductors(op, self.kinetic)):
            applied = op.apply(piece, solve)
            for j in range(len(omega)):
                form = float(np.real(layout.pair(applied[:, j], np.conj(solve[:, j]))))
                out[s, j] = form / (
                    2 * const.mu_0 * op.built.scale * omega[j].real ** 2 * MODE_ENERGY
                )
        return out

    # --- the solve_vjp interface ---------------------------------------------------

    def output_specs(self, model: Model) -> dict[str, jax.ShapeDtypeStruct]:
        """Shape and dtype of each forward output."""
        n = self.problem.n_modes
        if self.comm is None:
            mesh = model.mesh
            still = Mesh(mesh.imported_nodes, mesh._topology)  # noqa: SLF001
            with task_manager(self.options.threads):
                _, space, _ = fe_space(
                    still, self.regions, self._excluded, "hcurl", self.options.order
                )
            rows = space.ndof
        else:
            with task_manager(self.options.threads, self.comm):
                rows = self._field_width(model)
        return {
            "omega": jax.ShapeDtypeStruct((n,), jnp.complex128),
            "error": jax.ShapeDtypeStruct((n,), jnp.float64),
            "q_port": jax.ShapeDtypeStruct((len(self.resistive), n), jnp.float64),
            "participation": jax.ShapeDtypeStruct(
                (len(self.inductive), n), jnp.float64
            ),
            "bulk": jax.ShapeDtypeStruct((len(self.regions), n), jnp.float64),
            "principal": jax.ShapeDtypeStruct(
                (len(self.anisotropic), n, 3), jnp.float64
            ),
            "sheet": jax.ShapeDtypeStruct((len(self.kinetic), n), jnp.float64),
            "e_field": jax.ShapeDtypeStruct((rows, n), jnp.complex128),
            "cluster": jax.ShapeDtypeStruct((n,), jnp.int64),
        }

    def role(self, kind: tuple) -> str:
        """``"read"`` if the eigenmode solve depends on input ``kind``."""
        if kind[0] == "material":
            return "read" if kind[2] in self.reads else "unread"
        return "read"

    def unsupported_input(self, kind: tuple) -> str | None:
        """Why a read input cannot be differentiated here, or ``None``.

        The node coordinates, next to a tensor absorbing admittance
        (`terms.absorbing_shape_unsupported`); a permittivity law's
        parameter on a periodic model (no verification cell, ADR 0112).
        """
        if kind == ("nodes",):
            return self.nodes_unsupported
        if kind[0] == "law" and (
            self._periodic_surfaces or self.problem.wave_vector is not None
        ):
            return (
                "gradients with respect to a permittivity law's parameters are not "
                "verified on a periodic eigenmode model in this version"
            )
        return None

    def nondifferentiable_output(
        self, name: str, kinds: frozenset[tuple] | set[tuple] = frozenset()
    ) -> str | None:
        """Why a cotangent on output ``name`` cannot be pulled back, or ``None``.

        Input-aware (ADR 0102): ``kinds`` are the wanted input kinds. The
        eigenvalues differentiate with respect to every read input. The
        mode-field outputs (``q_port``, ``participation``,
        ``bulk_participation``, ``principal_participation``,
        ``sheet_participation`` and the normalized fields ``e_fields``)
        differentiate on one CPU process with ``linear_solver="direct"``,
        ``eigen_tolerance`` at most 1e-10, without periodic pairs or a Bloch
        wave vector, with respect to materials, sheet fills, node
        coordinates (port surfaces held) and lumped values; the normalized
        fields never with respect to the node coordinates, the principal
        shares not with respect to a lumped value, and none of them with
        respect to a permittivity law's parameter. What depends on the modes
        (a cluster, a near pair, a port without voltage) raises in the
        backward.

        Raises:
            DistributedUnsupportedError: Above one rank, any mode-field output
                (key ``mode_field_gradient``). It reads replicated values
                only, so every rank raises it at the backward's trace.
        """
        if name in ("omega", "cluster"):
            return None
        if name in _MODE_FIELD and self.comm is not None:
            # replicated values only, so every rank raises it at the backward's
            # trace (ADR 0072 amendment, decision 08 Raised-by-01 item 4)
            _mpi.refuse_feature("mode_field_gradient")
        public = _MODE_FIELD.get(name, name)
        head = f"gfem cannot differentiate the eigenmode output {public!r}"
        opt_out = (
            "Wrap it in jax.lax.stop_gradient to use its value without its derivative."
        )
        if name not in _MODE_FIELD:
            return (
                f"{head}: it is the backward error of the solve, which has no "
                "derivative with respect to anything. " + opt_out
            )
        why = self._solve_refusal()
        refusal = None if why is not None else self._input_refusal(name, kinds)
        if refusal is not None and refusal.kind == "law":
            return (
                f"{head} {refusal.message}: the mode fields and the ratios of a "
                "dispersive model have no verified gradient by its permittivity "
                "law in this version. Differentiate the eigenfrequency or Q "
                "(omega) instead, or wrap the value in jax.lax.stop_gradient to "
                "use it without its derivative."
            )
        if refusal is not None and refusal.kind == "lumped":
            return (
                f"{head} {refusal.message} in this version; the eigenfrequencies "
                "differentiate with respect to it. Wrap the value in "
                "jax.lax.stop_gradient to hold it."
            )
        if refusal is not None:
            why = refusal.message
        if why is None:
            return None
        return (
            f"{head} {why}. The eigenfrequencies (omega, so frequency and q) "
            "keep their gradients. " + opt_out
        )

    def _solve_refusal(self) -> str | None:
        """Why no mode-field output of this solve differentiates, or ``None``."""
        options = self.options
        reasons = (
            (
                self.problem.wave_vector is not None,
                (
                    "at a Bloch wave vector: mode-field gradients are for "
                    "non-periodic models only"
                ),
            ),
            (
                bool(self._periodic_surfaces),
                (
                    "of a periodic model: mode-field gradients are for "
                    "non-periodic models only"
                ),
            ),
            (
                options.device != "cpu",
                (
                    f'on device="{options.device}": its mode-field adjoint runs on '
                    'the CPU with linear_solver="direct" only'
                ),
            ),
            (
                self.linear.kind != "direct",
                (
                    f'on linear_solver="{self.linear.kind}": its mode-field adjoint '
                    'runs on linear_solver="direct" only'
                ),
            ),
            (
                options.eigen_tolerance > _FIELD_TOLERANCE,
                (
                    f"solved with SolverOptions.eigen_tolerance = "
                    f"{options.eigen_tolerance:g}: mode-field gradients need "
                    f"eigen_tolerance <= {_FIELD_TOLERANCE:g} (the default)"
                ),
            ),
        )
        return next((why for refused, why in reasons if refused), None)

    @staticmethod
    def _input_refusal(
        name: str, kinds: frozenset[tuple] | set[tuple]
    ) -> _InputRefusal | None:
        """Why output ``name`` does not differentiate by ``kinds``, or ``None``.

        Every mode-field output by a permittivity law's parameter (ADR 0112),
        the normalized field by the node coordinates, and the principal
        shares by a lumped value.
        """
        laws = [k for k in kinds if k[0] == "law"]
        if laws:
            region = min(laws, key=str)[1]
            return _InputRefusal(
                "law",
                f"with respect to the permittivity law of model.materials[{region!r}]",
            )
        if name == "e_field" and ("nodes",) in kinds:
            return _InputRefusal(
                "nodes",
                "with respect to the mesh node coordinates: the normalized mode "
                "field (and the measurements of it: Field.at, "
                "interface_participation) has no node gradient; it "
                "differentiates with respect to materials and sheet fills. For "
                "a shape derivative take finite differences of the measurement "
                "over meshes moved by gfem.shape.boundary_morph",
            )
        lumped = sorted(k for k in kinds if is_lumped(k))
        # the principal shares have no finite-difference cell by a lumped
        # value yet (ADR 0102 amendment); every other output does
        if lumped and name == "principal":
            kind = lumped[0]
            where = "ports" if kind[0] == "port" else "boundaries"
            return _InputRefusal(
                "lumped", f"with respect to model.{where}[{kind[1]}].{kind[2]}"
            )
        return None

    def backward(
        self, model: Model, entry: Entry, cotangents: dict[str, np.ndarray]
    ) -> list[FieldPair]:
        """Fold the cotangents into field pairs: per cluster, and per mode.

        **Eigenvalues.** With ``ω = k c / scale``, cluster c's modes ``X``,
        their left eigenvectors ``Y`` (``X`` unless Bloch) and ``G = Y^T
        T'(k_c) X``: ``ct_c d(Σ ω)/dp = -(c / scale) ct_c tr(G^-1 Y^T
        (dT/dp) X)``, the pair ``(-(c / scale) ct_c Y, X G^-1)`` weighted with
        the pieces' θ at ``k_c``. A simple mode is a cluster of one (``G =
        y^T T' x``). No solve. On a periodic model the pair is expanded to
        the unpaired space, ``(conj(P) Y, P X G^-1)``: ``y^T P^H a P x =
        (conj(P) y)^T a (P x)`` for the unpaired forms ``a``
        (`DrivenOperator.pair`). Under ranks ``gram`` is agreed and the
        pairs hold each rank's state columns, whose element passes the
        caller sums over ranks (ADR 0065).

        **Mode-field outputs** (`_field_pairs`): per mode with a cotangent
        on its ratios, one adjoint solve on the kept eigen operator and two
        pairs, ``(2 lam, x)`` on the pieces and the explicit energy pair;
        the explicit terms of the lumped ports' values are kept for
        `explicit_lumped`.

        Raises:
            DegenerateModeError: The ``omega`` cotangents on a cluster's
                modes differ (an objective not symmetric in them), a cluster
                with a nonzero cotangent extends beyond the ``n_modes``
                modes, a Bloch cluster's ``G`` is singular (a defective
                eigenvalue), or a mode with a mode-field cotangent is not
                isolated or is defective.
            NonDifferentiableError: A nonzero cotangent on an infinite
                ``q_port``.
            LinearSolverConvergenceError: An adjoint solve was not accepted.
        """
        fields = entry.fields
        x, k, labels = fields["x"], fields["k"], fields["cluster"]
        y = fields.get("y", x)
        gram, incomplete = fields["gram"], fields["incomplete"]
        ct = cotangents.get("omega")
        pairs = []
        self._explicit = {}
        with task_manager(self.options.threads, self.comm):
            op = self.driven(model)
            domain = em_region(model, op.built)
            per_k = const.c / op.built.scale
            if ct is not None and np.any(ct):
                ct = np.asarray(ct, dtype=complex)
                for label in np.unique(labels):
                    members = np.flatnonzero(labels == label)
                    c = ct[members]
                    if not c.any():
                        continue
                    spread = np.abs(c - c.mean()).max()
                    symmetric = spread <= _SYMMETRIC_RTOL * np.abs(c).max()
                    # from agreed values (the cotangents, the clusters, gram):
                    # every rank raises it together
                    if incomplete[members].any() or not symmetric:
                        raise _mpi.agreed(
                            DegenerateModeError(
                                self._degenerate_message(
                                    k, members, c, incomplete, per_k
                                )
                            )
                        )
                    g = gram[np.ix_(members, members)]
                    if "y" in fields and np.linalg.cond(g) > _DEFECTIVE:
                        raise _mpi.agreed(
                            DegenerateModeError(
                                self._defective_message(
                                    k, members, np.linalg.cond(g), per_k
                                )
                            )
                        )
                    pairs.append(
                        op.pair(
                            -per_k * c.mean() * y[:, members],
                            x[:, members] @ np.linalg.inv(g),
                            weights=op.weights(k[members].mean() * per_k),
                            domain=domain,
                        )
                    )
            pairs += self._field_pairs(model, entry, cotangents, op, domain)
        return pairs

    def _field_pairs(
        self,
        model: Model,
        entry: Entry,
        cotangents: dict[str, np.ndarray],
        op: DrivenOperator,
        domain,
    ) -> list[FieldPair]:
        """The mode-field outputs' pairs (quantum-metrics ticket 03, ADR 0104).

        Per mode ``(k, x)`` with a nonzero cotangent on its ratios or on its
        normalized field: the load ``(w, kappa, explicit terms)``
        (`Ratios.load`, with the field's normalization multipliers ``mu1``,
        ``mu2``), the deflated solve ``T(k) lam0 = -w``, ``t^T lam0 = 0``
        (``t = T'(k) x``) on the kept operator's live factors
        (`deflated_solve`), ``lam = lam0 - kappa x / (x^T t)``, and the pairs
        ``(2 lam, x)`` with the pieces' θ at the mode (``dG/dp = 2 Re(lam^T
        dT/dp x)``), ``(conj(x), x)`` with the explicit energy terms' weights
        and ``(x, x)`` with the phase rule's. The load's explicit terms in
        the lumped ports' values (`Load.lumped`) add up in ``_explicit``. An
        entry without its operator (over ``retain_memory``) gets one from
        ``op``, prepared at ``k (1 + d)`` of its first such mode, and freed
        at the end.
        """
        names = ("q_port", "participation", "bulk", "principal", "sheet")
        cts = {
            n: np.real(np.asarray(cotangents[n]))
            for n in names
            if n in cotangents and np.any(cotangents[n])
        }
        # the normalized field's cotangent, complex (non-periodic: the solve
        # space is the unpaired one)
        field = cotangents.get("e_field")
        if field is not None and not np.any(field):
            field = None
        if not cts and field is None:
            return []
        fields = entry.fields
        x, k = fields["x"], fields["k"]
        n = len(k)

        def column(c: np.ndarray, j: int) -> np.ndarray:
            return c[:, j] if c.ndim == 2 else c[:, j, :]

        wanted = [
            j
            for j in range(n)
            if any(column(c, j).any() for c in cts.values())
            or (field is not None and field[:, j].any())
        ]
        if not wanted:
            return []
        per_k = const.c / op.built.scale
        self._check_reach(entry, wanted, per_k)
        holder = entry.handle
        eop = getattr(holder, "operator", None)
        own = eop is None
        gap = fields["gap"]
        if own:
            first = wanted[0]
            eop = self.operator(op, k[first] * (1 + reshift_offset(float(gap[first]))))
        pairs = []
        try:
            # every mode's guard before any form is built
            gram = {}
            for j in wanted:
                u, kj = x[:, j], complex(k[j])
                t = eop.derivative(kj, u)
                gram[j] = u @ t
                size = np.linalg.norm(u) * np.linalg.norm(t)
                if abs(gram[j]) < _DEFECTIVE_FIELD * size:
                    raise DegenerateModeError(
                        self._defective_message(
                            k, np.array([j]), size / max(abs(gram[j]), 1e-300), per_k
                        )
                    )
            ratios = Ratios(
                model,
                op,
                regions=self.regions,
                anisotropic=self.anisotropic,
                resistive=self.resistive,
                inductive=self.inductive,
                kinetic=self.kinetic,
            )
            index = {port.name: i for i, port in enumerate(model.ports)}
            for j in wanted:
                u, kj, ut = x[:, j], complex(k[j]), gram[j]
                load = ratios.load(
                    u,
                    kj,
                    {name: column(c, j) for name, c in cts.items()},
                    mode=j,
                    field=None
                    if field is None
                    else np.where(eop.free, np.asarray(field[:, j], complex), 0),
                )
                lam = (
                    deflated_solve(
                        eop, u, None, -load.w, k=kj, gap=float(gap[j]), mode=j
                    ).z
                    - load.kappa * u / ut
                )
                pairs.append(
                    op.pair(
                        2 * lam[:, None],
                        u[:, None],
                        weights=op.weights(kj * per_k),
                        domain=domain,
                    )
                )
                if load.energy:
                    pairs.append(
                        op.pair(
                            np.conj(u)[:, None],
                            u[:, None],
                            weights=load.energy,
                            domain=domain,
                        )
                    )
                if load.phase:
                    pairs.append(
                        op.pair(
                            u[:, None], u[:, None], weights=load.phase, domain=domain
                        )
                    )
                for (port, value), c in load.lumped.items():
                    kind = ("port", index[port], value)
                    self._explicit[kind] = self._explicit.get(kind, 0.0) + c
        finally:
            if own:
                eop.free_handle()
        return pairs

    def _check_reach(self, entry: Entry, modes: list[int], per_k: float) -> None:
        """Raise unless every mode of ``modes`` is isolated (`isolated`).

        Raises:
            DegenerateModeError: A mode's polish failed in the forward, it
                belongs to a cluster of several modes or one cut by
                ``n_modes``, or its relative gap is below `_ISOLATED`
                (``cluster_rtol`` if larger): nearly degenerate.
        """
        fields = entry.fields
        k, labels = fields["k"], fields["cluster"]
        incomplete, gap = fields["incomplete"], fields["gap"]
        failed = fields.get("polish_failed", np.zeros(len(k), dtype=bool))
        rtol = self.options.cluster_rtol
        threshold = max(_ISOLATED, rtol)
        reach = isolated(labels, incomplete, gap, rtol)
        for j in modes:
            if reach[j]:
                continue
            members = np.flatnonzero(labels == labels[j])
            hz = k * per_k / (2 * math.pi)
            named = ", ".join(
                f"{int(i)} ({hertz(hz[i].real, 9)})" for i in members if i != j
            )
            mode = f"mode {j} ({hertz(hz[j].real, 9)})"
            if failed[j]:
                reason = (
                    "its polish in the forward did not converge (a sign of a "
                    "neighbour the solve did not return)"
                )
            elif len(members) > 1:
                reason = (
                    f"it belongs to a degenerate cluster with mode(s) {named} "
                    f"(relative gap at most SolverOptions.cluster_rtol = {rtol:g})"
                )
            elif incomplete[j]:
                reason = (
                    "its cluster extends beyond the "
                    f"{self.problem.n_modes} mode(s) asked for (Eigenmode.n_modes)"
                )
            else:
                reason = (
                    f"it is nearly degenerate: its relative gap to the nearest "
                    f"mode found is {gap[j]:.3g}, below {threshold:g} (the "
                    f"isolation threshold 1e-4, or SolverOptions.cluster_rtol = "
                    f"{rtol:g} if larger)"
                )
            raise DegenerateModeError(
                f"DegenerateModeError: a quantity of the field of {mode} "
                f"(participation, port Q, a field) is not differentiable: "
                f"{reason}. Mode-field gradients are for isolated modes only "
                "(ADR 0102): near another mode the field is a mix of the two "
                "that any perturbation rotates. Remedies: separate the modes "
                "(break the design's symmetry, move them apart); differentiate "
                "the frequencies only; or wrap the field quantity in "
                "jax.lax.stop_gradient."
            )

    def driven(self, model: Model) -> DrivenOperator:
        """The model's driven operator (the pieces of ``T(k)``) at its nodes.

        The last model's operator is kept, so that a backward pass and its
        terms build it once, while its discretization is the model's cached
        one: an operator on an evicted discretization would pair with
        kernel objects of the new one (ADR 0104).
        """
        last = getattr(self, "_last_driven", None)
        if (
            last is not None
            and last[0] is model
            and last[1].built
            is discretization(model.mesh, model._periodic, self.comm).built  # noqa: SLF001
        ):
            return last[1]
        op = driven_operator(
            model,
            self.options.order,
            wave_vector=self.problem.wave_vector,
            comm=self.comm,
        )
        self._last_driven = (model, op)
        return op

    def terms(self, model: Model, *, nodes: bool) -> list[Term]:
        """The pieces of ``T(k)`` as the kernel sees them: one `Term` per piece.

        The driven operator's (`gfem._engine.driven.operator_terms`): a
        mode's ``dk/dp = -x^T (dT/dp) x / x^T T'(k) x`` contracts the same
        pieces, weighted with their θ at the mode, for materials, fills and
        node coordinates. Port pieces are left out: shape gradients hold
        every port surface.

        The explicit energy terms of the mode-field outputs follow
        (`Ratios.terms`, ADR 0103): ``Re(eps_r)`` per region and per
        principal axis, which only their energy pairs weight. They are
        ``"value"`` terms on the element-wise ``Re(eps_r)`` masked to their
        part, so the node pass transports them as it does the mass piece; the
        port voltages read only the held port surfaces, so a node gradient
        of the ratios needs no other term.

        Raises:
            PortGeometryError: ``nodes`` and a node of a port surface has
                moved (with an ``omega`` or a mode-field cotangent alike).
        """
        op = self.driven(model)
        ratios = Ratios(
            model,
            op,
            regions=self.regions,
            anisotropic=self.anisotropic,
            resistive=(),
            inductive=(),
        )
        # a density blend never reaches an eigenmode solve (ADR 0113): its
        # mass derivative is the driven family's only
        pieces = [
            dataclasses.replace(
                t, derivatives={k: d for k, d in t.derivatives.items() if k != "blend"}
            )
            if "blend" in t.derivatives
            else t
            for t in operator_terms(model, op, nodes=nodes)
        ]
        return pieces + ratios.terms()

    def lumped_terms(self, model: Model) -> list[Term]:
        """The pieces of ``T(k)`` that move with a lumped value (ADR 0100).

        The driven operator's (`gfem._engine.driven.lumped_terms`), weighted
        with their θ at the mode like the other pieces.
        """
        return lumped_terms(self.driven(model))

    def explicit_lumped(self) -> dict[tuple, float]:
        """The last backward's explicit terms in the lumped values, by input kind.

        The lumped pass contracts the pairs with the pieces that move with
        a lumped value: the implicit part ``2 Re(lam^T dT/dp x)``, and a
        sheet's L and C through the explicit energy pair. A lumped port's L
        and C enter the mode-field outputs outside any piece, at fixed mode:
        ``-p / L`` in its participation, ``|V|^2 / 2`` in the mode energy
        (and so in every ratio and, through ``mu1``, the normalized field).
        These are scalars, by ``("port", index, "inductance" |
        "capacitance")``, which `solve_vjp` adds to the lumped pass's values
        (quantum-metrics ticket 21, ADR 0102 amendment).
        """
        return dict(self._explicit)

    def _degenerate_message(
        self,
        k: np.ndarray,
        members: np.ndarray,
        ct: np.ndarray,
        incomplete: np.ndarray,
        per_k: float,
    ) -> str:
        hz = k[members] * per_k / (2 * math.pi)
        gap = np.ptp(np.abs(k[members])) / np.abs(k[members]).max()
        modes = ", ".join(
            f"{int(j)} ({hertz(f.real, 9)})" for j, f in zip(members, hz, strict=True)
        )
        if incomplete[members].any():
            reason = (
                "the cluster extends beyond the "
                f"{self.problem.n_modes} mode(s) asked for (Eigenmode.n_modes), so "
                "no objective of these modes is symmetric in the whole cluster"
            )
        else:
            reason = (
                "the objective is not symmetric in them (cotangents "
                + ", ".join(f"{complex(v):.4g}" for v in ct)
                + ")"
            )
        return (
            f"DegenerateModeError: mode(s) {modes} belong to a degenerate cluster "
            f"(relative gap {gap:.2g}, SolverOptions.cluster_rtol = "
            f"{self.options.cluster_rtol:g}): the modes of a cluster are an "
            "arbitrary basis of their eigenspace, so one mode's frequency is not "
            f"differentiable, and {reason}. Remedies: differentiate a function "
            "symmetric in the cluster's modes (their sum or mean, a smooth "
            "aggregate such as log-sum-exp instead of max), with n_modes "
            "covering the whole cluster; break the symmetry of the design; or "
            "use a gfem.Driven sweep as a surrogate."
        )

    def _defective_message(
        self, k: np.ndarray, members: np.ndarray, condition: float, per_k: float
    ) -> str:
        hz = k[members] * per_k / (2 * math.pi)
        modes = ", ".join(
            f"{int(j)} ({hertz(f.real, 9)})" for j, f in zip(members, hz, strict=True)
        )
        return (
            f"DegenerateModeError: mode(s) {modes} have a singular Y^T T'(k) X "
            f"(condition number {condition:.3g}): the eigenvalue is defective "
            "(an exceptional point of a lossy problem), where its left and "
            "right eigenvectors are orthogonal, and neither its frequency nor "
            "a quantity of its field is differentiable. Remedies: move the "
            "design (or the wave vector) off the exceptional point, or use a "
            "gfem.Driven sweep as a surrogate."
        )

    def result(self, outputs: dict) -> EigenmodeResult:
        """The public result from the forward outputs."""
        omega = outputs["omega"]
        fields = [
            Field(
                coefficients=outputs["e_field"][:, j],
                mesh=self._mesh,
                quantity="E",
                unit="V/m",
                space="hcurl",
                order=self.options.order,
                support=self.regions,
                omega=omega[j],
                _excluded=self._excluded,
                _comm=_mpi.static_comm(self.comm),
            )
            for j in range(self.problem.n_modes)
        ]
        return EigenmodeResult(
            omega=jnp.asarray(omega),
            error=jnp.asarray(outputs["error"]),
            q_port={
                name: jnp.asarray(outputs["q_port"][p])
                for p, name in enumerate(self.resistive)
            },
            participation={
                name: jnp.asarray(outputs["participation"][p])
                for p, name in enumerate(self.inductive)
            },
            bulk_participation={
                region: jnp.asarray(outputs["bulk"][r])
                for r, region in enumerate(self.regions)
            },
            principal_participation={
                region: jnp.asarray(outputs["principal"][r])
                for r, region in enumerate(self.anisotropic)
            },
            e_fields=tuple(fields),
            sheet_participation={
                surface: jnp.asarray(outputs["sheet"][s])
                for s, surface in enumerate(self.kinetic)
            },
            materials=self._materials,
            periodic_surfaces=self._periodic_surfaces,
            cluster=jnp.asarray(outputs["cluster"]),
        )


def _close(a: complex, b: complex, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def _incomplete(
    ks: list[complex], labels: np.ndarray, guard: complex | None, rtol: float
) -> np.ndarray:
    """Whether each mode's cluster reaches the guard mode ``guard`` (cut short)."""
    incomplete = np.zeros(len(ks), dtype=bool)
    if guard is None:
        return incomplete
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        if any(_close(guard, ks[j], rtol) for j in members):
            incomplete[members] = True
    return incomplete


def isolated(
    labels: np.ndarray, incomplete: np.ndarray, gap: np.ndarray, rtol: float
) -> np.ndarray:
    """Which modes are in mode-field reach (ADR 0102, ADR 0104).

    A mode is when it is the only member of its cluster (``labels``), its
    cluster is not cut short by the guard mode (``incomplete``), and its
    relative gap to its gap set is at least `_ISOLATED`, or ``rtol``
    (``SolverOptions.cluster_rtol``) when that is larger. An entry stores a
    gap of 0 for a mode whose forward polish failed (its ``polish_failed``
    field), so such a mode is never isolated.
    """
    labels = np.asarray(labels)
    single = np.array([np.count_nonzero(labels == v) == 1 for v in labels], dtype=bool)
    return (
        single
        & ~np.asarray(incomplete, dtype=bool)
        & (np.asarray(gap) >= max(_ISOLATED, rtol))
    )


def clusters(ks: list[complex], rtol: float) -> np.ndarray:
    """A label per eigenvalue: chains of eigenvalues within ``rtol`` share one.

    Labels count from 0 in order of first appearance.
    """
    labels = list(range(len(ks)))
    for i in range(len(ks)):
        for j in range(i):
            if _close(ks[i], ks[j], rtol) and labels[i] != labels[j]:
                old, new = labels[i], labels[j]
                labels = [new if v == old else v for v in labels]
    renumber: dict[int, int] = {}
    for v in labels:
        renumber.setdefault(v, len(renumber))
    return np.array([renumber[v] for v in labels], dtype=np.int64)


def _modal_overlap(
    mesh: Mesh, region: str, material: Material, frequency: float
) -> float:
    """How many of ``region``'s modes lie within one linewidth of ``frequency``.

    Weyl's density of cavity modes times the linewidth, ``M = V k0^3 |Im
    eps| sqrt(Re eps) / pi^2``, with ``V`` the region's volume and ``eps``
    its complex permittivity at ``frequency`` (the mean of its principal
    values, or its permittivity law's value; ``tan_delta`` and
    ``conductivity`` give the loss). 0 where ``Re eps <= 0``. It matched the
    counted modes to 10-30 % (ADR 0122).
    """
    omega = 2 * math.pi * frequency
    sigma = np.mean(np.asarray(material.conductivity, dtype=float))
    if isinstance(material.eps_r, PermittivityLaw):
        law = material.eps_r
        eps = complex(law.eps_inf) + dispersion.susceptibility(law, omega)
    else:
        # a scalar eps_r may sit beside three principal loss tangents
        eps_r, tan = np.broadcast_arrays(
            np.asarray(material.eps_r, dtype=float),
            np.asarray(material.tan_delta, dtype=float),
        )
        eps = complex(np.mean(eps_r * (1 - 1j * tan)))
    eps -= 1j * sigma / (omega * const.epsilon_0)
    if not cmath.isfinite(eps):
        return 0.0
    if eps.real <= 0:
        return 0.0
    nodes = np.asarray(mesh.nodes, dtype=float)[mesh.tetrahedra[mesh.elements(region)]]
    edges = nodes[:, 1:] - nodes[:, :1]
    volume = float(np.abs(np.linalg.det(edges)).sum()) / 6
    k0 = omega / const.c
    return volume * k0**3 * abs(eps.imag) * math.sqrt(eps.real) / math.pi**2


def _contraction(seed: complex, shift: complex, ritz: np.ndarray) -> float:
    """Residual inverse iteration's error ratio per step for ``seed``.

    ``|seed - shift|`` over the distance from ``shift`` to the nearest other
    Ritz value: the component of that mode grows unless the ratio is below 1.
    """
    others = ritz[np.abs(ritz - seed) > _MULTIPLE * abs(seed)]
    if not len(others):
        return 0.0
    nearest = float(np.min(np.abs(others - shift)))
    return abs(seed - shift) / nearest if nearest > 0 else math.inf


def _orthonormal_multiples(
    k: np.ndarray, vectors: np.ndarray, *, layout: DofLayout = ONE_PROCESS
) -> np.ndarray:
    """``vectors`` with each multiple Ritz value's vectors orthonormalized.

    Any basis of a multiple eigenvalue's eigenspace is as good a set of
    seeds; an orthonormal one keeps its members apart under `_in_span`.
    The vectors are states, orthonormal over ``layout``'s owned DOFs.
    """
    out = vectors
    done = np.zeros(len(k), dtype=bool)
    for j in range(len(k)):
        if done[j]:
            continue
        members = np.flatnonzero(np.abs(k - k[j]) <= _MULTIPLE * abs(k[j]))
        done[members] = True
        if len(members) > 1:
            if out is vectors:
                out = vectors.copy()
            out[:, members] = layout.state_qr(vectors[:, members])
    return out


def _same(
    a: Refined, b: Refined, window: float, *, layout: DofLayout = ONE_PROCESS
) -> bool:
    """Whether ``a`` and ``b`` are one mode: near in ``k``, parallel in ``x``."""
    if abs(a.k - b.k) > window * abs(b.k):
        return False
    norm = layout.state_norm
    overlap = abs(layout.state_inner(a.x, b.x)) / (norm(a.x) * norm(b.x))
    return overlap > _OVERLAP


def _in_span(
    mode: Refined,
    accepted: list[Refined],
    window: float,
    *,
    layout: DofLayout = ONE_PROCESS,
) -> bool:
    """Whether ``mode`` is a combination of accepted modes: a duplicate.

    The norm of its projection on the span of the accepted modes within
    ``window`` (relative) of its ``k`` exceeds `_OVERLAP` of its own. Against
    the span, not each mode: any vector of a degenerate cluster's eigenspace
    is a duplicate, also one at 45 degrees from two accepted members.
    The modes are states, their products ``layout``'s.
    """
    near = [a.x for a in accepted if abs(mode.k - a.k) <= window * abs(a.k)]
    if not near:
        return False
    basis = layout.state_qr(np.stack(near, axis=1))
    inside = np.linalg.norm(layout.state_gram(basis, mode.x))
    return bool(inside > _OVERLAP * layout.state_norm(mode.x))


def _outside(
    mode: Refined,
    accepted: list[Refined],
    window: float,
    *,
    layout: DofLayout = ONE_PROCESS,
) -> np.ndarray | None:
    """``mode``'s part outside the span of the accepted modes near its ``k``.

    ``None`` when it is at most `_TWIN` of its norm: round-off of a duplicate
    of a simple mode (1e-11 to 2e-9 measured), not a twin (0.36 to 0.94 on the
    lossy-wall mirrored cavity, quantum-metrics ticket 17). The modes are
    states, their products ``layout``'s.
    """
    near = [a.x for a in accepted if abs(mode.k - a.k) <= window * abs(a.k)]
    if not near:
        return None
    basis = layout.state_qr(np.stack(near, axis=1))
    rest = mode.x - basis @ layout.state_gram(basis, mode.x)
    norm = layout.state_norm(rest)
    if norm <= _TWIN * layout.state_norm(mode.x):
        return None
    return rest / norm


def _lost(
    mode: Refined,
    seed: complex,
    ritz: np.ndarray,
    sigma: float,
    *,
    accepted: list[Refined],
    window: float,
    layout: DofLayout = ONE_PROCESS,
) -> bool:
    """Whether ``seed``'s refinement gave nothing to accept as its mode.

    It did not converge, converged to a static or mirror root, converged
    nearer another of the ``ritz`` values than ``seed`` (that seed's mode,
    which would take this one's place), or converged to an accepted mode
    the seed is not near.
    """
    if not mode.converged or mode.k.real <= _STATIC * sigma:
        return True
    others = ritz[np.abs(ritz - seed) > _MULTIPLE * abs(seed)]
    if len(others) and np.min(np.abs(others - mode.k)) < abs(mode.k - seed):
        return True
    return any(
        _same(mode, other, window, layout=layout) and not _close(seed, other.k, window)
        for other in accepted
    )


def _check_low_frequency(
    model: Model, target: float, lowest: float, *, stacklevel: int, comm: Any = None
) -> None:
    """Warn if the target or the lowest mode is below the low-frequency floor.

    The floor is the driven one, ``k0 h_min = LOW_FREQUENCY_FLOOR`` with
    h_min the shortest edge of the solve domain: below it the E-field
    formulation loses about ``1e-16 / (k0 h_min)^2`` of the modes, which
    their backward error does not show (measured on a line loaded by an
    inductor: frequency off by 2e-5 at ``k0 h_min = 2e-6`` and by 0.2 at
    2e-8, with a reported error of 1e-16).

    Args:
        model: The model, its solve domain and its mesh.
        target: ``Eigenmode.target`` (Hz).
        lowest: The lowest ``Re f`` of the modes found (Hz).
        stacklevel: Passed on to `warnings.warn`, counted from the caller.
        comm: The resolved communicator: the warning is the job's, once.
    """
    floor = LowFrequencyFloor.of(model)
    f = min(target, lowest)
    k0h = floor.k0h(f)
    if k0h >= LOW_FREQUENCY_FLOOR:
        return
    what = "target" if target <= lowest else "lowest mode found"
    _mpi.warn(
        comm,
        f"Eigenmode: the {what}, {hertz(f, 4)}, is below {floor.describe(k0h)}. "
        "Below it the eigenfrequencies, the mode fields "
        "and the participations carry rounding errors up to about "
        "1e-16/(k0*h_min)^2, which EigenmodeResult.error does not show. Raise "
        "the resonance or coarsen the smallest elements.",
        LowFrequencyWarning,
        stacklevel=stacklevel + 1,
    )


@contextlib.contextmanager
def _agreed(*errors: type[BaseException]) -> Generator[None]:
    """``errors`` raised inside are raised by every rank together (ADR 0058).

    Around code whose failures read agreed or replicated values only: the band
    checks, and the mode search, whose every branch reads agreed values.
    """
    try:
        yield
    except errors as error:
        _mpi.agreed(error)
        raise


def _require_rank_slepc(comm: Any) -> None:
    """SLEPc on the MPI PETSc stack, on every rank, or raise on every rank.

    Raises:
        gfem.SolverUnavailableError: The MPI stack is missing, or a rank cannot
            use slepc4py.
    """
    if not _mpi.agree_all(comm, flag=_petsc.has_mpi()):
        raise SolverUnavailableError(
            "An eigenmode solve above one rank runs SLEPc on the MPI PETSc stack, "
            "and this environment does not have it. Build it with "
            f"tools/petsc_wheelhouse.sh --mpi as {_mpi.RECIPE} describes, or pass "
            "comm=MPI.COMM_SELF for independent one-process copies."
        )
    error = None
    try:
        _petsc.require_slepc("eigen_solver")
    except SolverUnavailableError as err:
        error = err
    if not _mpi.agree_all(comm, flag=error is None):
        raise _mpi.agreed(
            error
            or SolverUnavailableError(
                "eigen_solver='slepc' cannot run: another rank cannot use slepc4py "
                "(that rank's error names why)."
            )
        )


def _layout(op: DrivenOperator) -> DofLayout:
    """The driven operator's layout: `ONE_PROCESS` on one process."""
    return op.layout or ONE_PROCESS


def _sheet_energy(op: DrivenOperator) -> Callable[[np.ndarray], float] | None:
    """``x -> Σ 1/2 ∫ w C_s |E_t|^2 dS`` of the capacitive sheets (J), or ``None``.

    A capacitive `Impedance`'s C piece is ``∫ w C_s / (eps0 scale) u_t .
    v_t``, as the mass piece is ``∫ eps_r u . v``: its quadratic form with
    ``(conj(x), x)`` times ``eps0 scale / 2`` is the sheet's electric
    energy, as `element_energy` gives the volume's (ADR 0103). ``None``
    without a capacitive sheet, so such models add nothing.
    """
    sheets = sheet_capacitors(op)
    if not sheets:
        return None
    factor = 0.5 * const.epsilon_0 * op.built.scale
    pair = _layout(op).pair

    def energy(x: np.ndarray) -> float:
        # the load A x paired with the state conj(x)
        return sum(
            factor * float(np.real(pair(op.apply(k, x), np.conj(x)))) for k in sheets
        )

    return energy


def _is_real(op: EigenOperator) -> bool:
    """Whether ``K`` and ``M`` are real: a lossless, undamped problem."""
    probe = np.where(op.free, np.random.default_rng(1).standard_normal(len(op.free)), 0)
    kx, _, mx = op.parts(probe.astype(complex))
    # a real matrix gives a real load on any rank's DOFs: every rank must agree
    real = not kx.imag.any() and not mx.imag.any()
    return _mpi.agree_all(op.layout.comm, flag=real)


def _is_hermitian(op: EigenOperator) -> bool:
    """Whether ``K`` and ``M`` are Hermitian: a lossless Bloch problem.

    A Bloch-phased ``K`` is complex even when lossless (`_is_real` never
    holds): two random complex probes ``u``, ``v`` test ``|v^H A u -
    conj(u^H A v)| <= _HERMITIAN |A| |u| |v|`` for ``K`` and ``M``.
    """
    rng = np.random.default_rng(1)
    n = len(op.free)
    u, v = (
        np.where(op.free, rng.standard_normal(n) + 1j * rng.standard_normal(n), 0)
        for _ in range(2)
    )
    ku, _, mu = op.parts(u)
    kv, _, mv = op.parts(v)
    nk, _, nm, *_ = op.norms
    size = np.linalg.norm(u) * np.linalg.norm(v)
    return all(
        abs(np.vdot(v, au) - np.conj(np.vdot(u, av))) <= _HERMITIAN * norm * size
        for au, av, norm in ((ku, kv, nk), (mu, mv, nm))
    )


def _check_supported(model: Model) -> None:
    """Raise for model terms this version cannot solve as eigenmodes.

    The first Floquet or wave port in model order is reported; when a
    wave port's surface is shared, the message names the surface and all
    its wave ports.
    """
    for port in model.ports:
        if isinstance(port, FloquetPort):
            raise NotImplementedError(
                f"Eigenmode does not support Floquet port {port.name!r} in this "
                "version of gfem: its orders' propagation constants must be "
                "evaluated at the complex eigenfrequency. Find the resonances and "
                "Q from a gfem.Driven sweep, or end the cell with gfem.Absorbing "
                f"on {port.surface!r} (first order: it reflects oblique waves)."
            )
        if isinstance(port, WavePort):
            shared = [p.name for p in wave_port_surfaces(model.ports)[port.surface]]
            if len(shared) > 1:
                raise NotImplementedError(
                    f"Eigenmode does not support the shared port surface "
                    f"{port.surface!r} (wave ports "
                    f"{', '.join(repr(n) for n in shared)}) in this version of "
                    "gfem: the propagation constants of its modes must be solved "
                    "at the complex eigenfrequency. Find the resonances and Q "
                    "from a gfem.Driven sweep."
                )
            raise NotImplementedError(
                f"Eigenmode does not support wave port {port.name!r} in this "
                "version of gfem: its propagation constant must be solved at the "
                "complex eigenfrequency. Replace it with gfem.LumpedPort("
                f"{port.name!r}, ..., impedance=<line impedance>), or find the "
                "resonance and Q from a gfem.Driven sweep."
            )
    terms.check_absorbing(model)
