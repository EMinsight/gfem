"""Estimate kinds: how one solved result is estimated for one estimator.

`estimate` is the one entry: a `FieldEstimator` or a keyword
`QuantityEstimator` (capacitance, S, omega, Q, a cluster's mean omega) reads
its kernel directly; an objective is classified (`gfem._objective.classify`)
and estimated by its kind's function in the closed `_OBJECTIVE_KINDS` table.
Each kind translates its reads into kernel keys, raises the refusals its
computation on the result's values decides, asks the mesh's
`gfem._dual_supply.DualSupply` for the dual families it reads, and calls its
kernel (`gfem._engine.estimate`); `_packed` makes the `ErrorEstimate`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem._domain import physics_of
from gfem._dual_supply import DualNeed, DualSupply, _adjoint_ports, _needs_adjoint
from gfem._estimate_admission import (
    _check_lossy_mode,
    _check_lossy_q,
    _lossy_elements,
    _undefined_rows,
    _unsupported_result,
)
from gfem._pytree import readonly
from gfem.errors import DegenerateModeError, MeshRefinementUnsupportedError
from gfem.model import PEC, PMC, Impedance, LumpedPort, Model
from gfem.options import SolverOptions
from gfem.refinement import (
    ErrorEstimate,
    FieldEstimator,
    QuantityEstimator,
    _describe,
    _frequency,
    _track_mode,
)
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from gfem._engine.estimate import RecoveryCache
    from gfem._objective import Objective
    from gfem.refinement import Estimator


def estimate(
    model: Model,
    result: Any,
    estimator: Estimator,
    *,
    threads: int | None,
    mode: int | tuple[int, ...] | None = None,
    duals: DualSupply | None = None,
    cache: RecoveryCache | None = None,
    rtol: float | None = None,
    reach: str | None = None,
) -> ErrorEstimate:
    """The estimate of ``result``: dispatch on the estimator and result type.

    An objective is estimated by its kind's function (`_OBJECTIVE_KINDS`).
    ``mode`` is the eigenmode a `QuantityEstimator` follows on this result,
    or the modes of a cluster it follows.
    ``duals`` is the mesh's `gfem._dual_supply.DualSupply`: the solved
    duals that rode on the solve of ``result``, the others solved there;
    ``None`` (nothing rode) solves every one the estimate needs.
    ``cache`` is the mesh's `gfem._engine.estimate.RecoveryCache`, shared by
    every estimate of ``result``'s mesh; ``None`` makes one for this call.
    ``rtol`` is the solve's ``cluster_rtol`` (`SolverOptions`' default for
    ``None``): an objective's modes are checked for degeneracy with it.
    ``reach`` is why the solve's options leave a participation's mode-field
    dual out of reach (`mode_field_reach`), else ``None``.
    """
    from gfem import _objective
    from gfem._engine import estimate as kernels

    cache = cache or kernels.RecoveryCache()
    duals = duals or DualSupply()
    kind = _kind(estimator)
    if isinstance(estimator, QuantityEstimator) and estimator.objective is not None:
        call = _Call(
            model,
            result,
            estimator,
            threads=threads,
            duals=duals,
            cache=cache,
            rtol=SolverOptions().cluster_rtol if rtol is None else rtol,
            reach=reach,
        )
        objective = _objective.classify(estimator.objective, result, model)
        of_kind = _OBJECTIVE_KINDS.get(objective.kind)
        if of_kind is None:
            raise MeshRefinementUnsupportedError(
                objective.kind,
                f"{_describe(estimator)} is an {objective.kind}, which has no "
                "verified estimate in this version of gfem: use "
                "gfem.FieldEstimator(), or estimate the quantities it reads one "
                "by one.",
            )
        return of_kind(call, objective)
    if isinstance(estimator, QuantityEstimator):
        if estimator.capacitance is not None:
            parts = [
                kernels.capacitance(
                    model, result, *estimator.capacitance, threads, cache=cache
                )
            ]
        elif estimator.s is not None:
            ports = _adjoint_ports(estimator, model, list(result.s.excited))
            adjoints = duals.s_adjoints(model, result, ports, threads)
            fields = _dual_fields(result, adjoints, cache)
            parts = kernels.s_entry(
                model, result, *estimator.s, threads, adjoints=fields, cache=cache
            )
        else:
            parts = [
                _mode_terms(
                    model, result, estimator, mode, threads=threads, cache=cache
                )
            ]
        values = tuple(terms.value for terms in parts)
        _, indicators = kernels.combine([[terms.weights] for terms in parts])
    else:
        if isinstance(result, ElectrostaticResult):
            solves = kernels.electrostatic(model, result, threads, cache=cache)
        elif isinstance(result, DrivenResult):
            solves = kernels.driven(model, result, threads, cache=cache)
        elif isinstance(result, EigenmodeResult):
            solves = kernels.eigenmode(model, result, threads, cache=cache)
        else:  # pragma: no cover - `_check_estimable` refuses the rest first
            raise _unsupported_result(result, estimator)
        values, indicators = kernels.combine(solves)
    return ErrorEstimate(
        estimator=estimator,
        kind=kind,
        values=values,
        indicators=readonly(indicators),
    )


def mode_field_reach(options: SolverOptions) -> str | None:
    """Why a solve with ``options`` leaves mode-field duals out of reach."""
    from gfem._eigen.reach import FIELD_TOLERANCE, blocker
    from gfem._linalg import ResolvedLinearSolver

    linear = ResolvedLinearSolver.resolve(
        options, default="direct", problem="eigenmode"
    ).kind
    # refine_mesh refuses a Floquet port, a Bloch wave vector and a periodic
    # model before any estimate, so neither of those conditions can block
    # here: they have no text, and reaching one raises KeyError
    why = blocker(options, linear, wave_vector=None, periodic=False)
    if why is None:
        return None
    return {
        "device": f'device="{options.device}": it runs on the CPU only',
        "linear": f'linear_solver="{linear}": it runs on linear_solver="direct" only',
        "tolerance": (
            f"SolverOptions.eigen_tolerance = {options.eigen_tolerance:g}: it "
            f"needs eigen_tolerance <= {FIELD_TOLERANCE:g} (the default), at "
            "which the forward polishes the modes"
        ),
    }[why]


@dataclasses.dataclass(frozen=True)
class _Call:
    """One objective estimate's inputs, as `estimate` was called (defaults applied)."""

    model: Model
    result: Any
    estimator: QuantityEstimator
    threads: int | None
    duals: DualSupply
    cache: RecoveryCache
    rtol: float
    reach: str | None


def _packed(
    call: _Call,
    objective: Objective,
    terms: Any,
    *,
    kind: str | None = None,
    duals: DualNeed = DualNeed(),  # noqa: B008 - frozen value object
    modes: tuple[int, ...] = (),
    defect: float = 0.0,
) -> ErrorEstimate:
    """An objective's estimate from its kernel's terms: one value, one indicator set.

    ``kind`` is the objective's kind unless given (a lossless rewrite),
    ``duals`` the solved duals it read, ``modes`` the modes it follows and
    ``defect`` a lossless rewrite's defect term.
    """
    return ErrorEstimate(
        estimator=call.estimator,
        kind=objective.kind if kind is None else kind,
        values=(terms.value,),
        indicators=readonly(np.sqrt(terms.weights)),
        _quantity=objective.value,
        _duals=duals,
        _sensitivity=objective.sensitivity,
        _modes=modes,
        _defect=defect,
    )


def _dual_fields(
    result: DrivenResult, adjoints: Mapping[Any, np.ndarray], cache: RecoveryCache
) -> dict[tuple[float, Hashable], Any]:
    """The adjoints as fields, each on its frequency's kept field's space.

    An unexcited port has no field of its own; every kept field of a
    frequency shares one space. Each is made once per mesh (``cache``, the
    mesh's), so the estimates reading it share its recovered errors.
    """
    fields = {
        f: field for (f, _), field in zip(result.kept, result.e_fields, strict=True)
    }
    return {
        (f, key): cache.dual_field(fields[f], z) for (f, key), z in adjoints.items()
    }


def _s_objective(call: _Call, objective: Objective) -> ErrorEstimate:
    """An objective of S's estimate.

    On a model that conserves power (`_conserves_power`) it is estimated
    through its lossless rewrite, over every row of the columns it reads;
    otherwise through the C form.
    The S-functional adjoints of the rows that need one (a port not
    excited) and did not ride on the solve are
    solved by ``duals``: one more factorization per frequency
    (`gfem._dual_supply.DualSupply.s_adjoints`).
    """
    from gfem._engine import estimate as kernels

    model, result, threads = call.model, call.result, call.threads
    duals, cache = call.duals, call.cache
    s = result.s
    reads = [
        (float(s.frequency[f]), s.ports[i], s.excited[j], r.cotangent)
        for r in objective.reads
        for f, i, j in [r.index]
    ]
    rewrite = _conserves_power(model)
    rows = [to for _, to, _, _ in reads]
    if rewrite:
        rows += list(s.ports)
    excited = list(s.excited)
    ports = tuple(
        dict.fromkeys(to for to in rows if _needs_adjoint(to, model, excited))
    )
    solved = duals.s_adjoints(model, result, ports, threads)
    fields = _dual_fields(result, solved, cache)
    if rewrite:
        terms = kernels.lossless_rewrite(
            model,
            result,
            reads,
            objective.curvature,
            objective.diagonal,
            threads=threads,
            adjoints=fields,
            cache=cache,
        )
        kind = "objective of s, lossless rewrite"
        defect = float(terms.lam @ terms.defect)
    else:
        terms = kernels.objective_of_s(
            model,
            result,
            reads,
            objective.curvature,
            threads,
            adjoints=fields,
            cache=cache,
        )
        kind, defect = objective.kind, 0.0
    return _packed(
        call, objective, terms, kind=kind, duals=DualNeed(ports=ports), defect=defect
    )


def _conserves_power(model: Model) -> bool:
    """Whether the true S of ``model`` conserves power, sum_i |S_ic|^2 = 1.

    A conservative whitelist: every material lossless (real
    eps_r and mu_r, no tan_delta, conductivity or lossy law), every surface
    PEC or PMC, and every port lumped, whose resistance is the reference.
    A wave port is left out: its kept modes need not carry all the power;
    so is a lumped port with no resistance, whose row of S is undefined
    (NaN): the rewrite sums every row.
    """
    return (
        all(isinstance(p, LumpedPort) for p in model.ports)
        and not _undefined_rows(model)
        and all(
            isinstance(b, PEC | PMC) for b in model.boundaries if physics_of(b) == "em"
        )
        and not _lossy_elements(model).any()
    )


def _field_objective(call: _Call, objective: Objective) -> ErrorEstimate:
    """An objective of driven fields' estimate.

    One dual per measurement output read (`gfem._engine.field_duals`):
    those that did not ride on the solve are solved by ``duals``, one
    more factorization per frequency read
    (`gfem._dual_supply.DualSupply.field_duals`).
    """
    from gfem._engine import estimate as kernels

    model, result, threads = call.model, call.result, call.threads
    duals, cache = call.duals, call.cache
    reads = []
    for r in objective.reads:
        f, excitation = result.kept[r.at]
        if r.source == "power_flow":
            key = ("power", float(f), excitation, r.regions)
            reads.append((key, complex(np.real(r.cotangent))))
        else:
            key = ("mean", float(f), excitation, r.regions, r.index[0])
            reads.append((key, complex(r.cotangent)))
    keys = list(dict.fromkeys(key for key, _ in reads))
    solved = duals.field_duals(model, result, keys, threads)
    terms = kernels.objective_of_fields(
        model,
        result,
        reads,
        objective.curvature,
        threads,
        duals=solved,
        cache=cache,
    )
    return _packed(call, objective, terms, duals=DualNeed(keys=tuple(keys)))


def _potential_objective(call: _Call, objective: Objective) -> ErrorEstimate:
    """An electrostatic objective's estimate.

    J(C) reads the potentials' free duals. An energy or potential mean reads
    one dual per read energy and per averaged region set: those that did not
    ride on the solve are solved by ``duals``, one more factorization
    (`gfem._dual_supply.DualSupply.potential_duals`).
    """
    from gfem._engine import estimate as kernels

    model, result, threads = call.model, call.result, call.threads
    duals, cache = call.duals, call.cache
    terminals = list(result.terminals)
    reads, keys = [], []
    for r in objective.reads:
        ct = float(np.real(r.cotangent))
        if r.source == "capacitance":
            i, j = r.index
            reads.append((("capacitance", terminals[j], terminals[i]), ct))
        elif r.source == "energy":
            key = ("energy", r.at, terminals[r.index[0]])
            reads.append((key, ct))
            keys.append(key)
        else:
            key = ("mean", r.regions)
            reads.append((("mean", key[1], r.at), ct))
            keys.append(key)
    keys = list(dict.fromkeys(keys))
    solved = duals.potential_duals(model, result, keys, threads)
    terms = kernels.objective_of_potentials(
        model,
        result,
        reads,
        objective.curvature,
        threads,
        duals=solved,
        cache=cache,
    )
    return _packed(call, objective, terms, duals=DualNeed(keys=tuple(keys)))


def _omega_objective(call: _Call, objective: Objective) -> ErrorEstimate:
    """An objective of the modes' omega: free duals.

    Each mode read is followed as a mode of its own (``_modes``), and must
    be clearly apart from the others (`gfem.refinement._track_mode`). The
    terms lifted for the omega estimate alone raise (`_check_lossy_mode`).
    """
    from gfem._engine import estimate as kernels

    model, result, threads = call.model, call.result, call.threads
    rtol, cache = call.rtol, call.cache
    _check_lossy_mode(model, "objective")
    reads = [(r.index[0], r.cotangent) for r in objective.reads]
    for mode, _ in reads:
        _track_mode(result, mode, None, rtol, None)
    terms = kernels.objective_of_omega(
        model, result, reads, objective.curvature, threads, cache=cache
    )
    return _packed(call, objective, terms, modes=tuple(mode for mode, _ in reads))


def _participation_objective(call: _Call, objective: Objective) -> ErrorEstimate:
    """An objective of bulk participations, and omega.

    Each participation read needs its mode-field dual: ridden on the
    forward when it was solved for the same mode, else solved by ``duals``
    on an operator prepared at the mode
    (`gfem._dual_supply.DualSupply.mode_duals`). Raises for a mode out of
    mode-field reach, for the mode Q's refusals, and for a mode energy
    holding a surface term, which the equilibrated dual flux has no volume
    load for.
    """
    from gfem._engine import estimate as kernels

    model, result, threads = call.model, call.result, call.threads
    rtol, duals, cache = call.rtol, call.duals, call.cache
    _check_lossy_mode(model, "objective")
    _check_mode_field_reach(model, call.reach)
    reads, regions = [], {}
    omega = np.asarray(result.omega)
    for r in objective.reads:
        mode = r.index[0]
        if r.source == "omega":
            reads.append(("omega", mode, r.cotangent))
            continue
        region = r.at
        reads.append(("bulk", mode, region, r.cotangent.real))
        regions.setdefault(mode, []).append(region)
    modes = sorted({read[1] for read in reads})
    for mode in modes:
        _track_mode(result, mode, None, rtol, None)
    solved: dict[tuple[int, str], Any] = {}
    for mode, names in regions.items():
        _check_isolated(omega, mode, rtol)
        ridden = duals.mode_duals(model, result, mode, names, threads)
        solved |= {(mode, name): dual for name, dual in ridden.items()}
    terms = kernels.objective_of_participation(
        model, result, reads, objective.curvature, threads, duals=solved, cache=cache
    )
    return _packed(
        call,
        objective,
        terms,
        modes=tuple(modes),
        duals=DualNeed(modes=tuple(solved)),
    )


#: Each objective kind's estimate (`gfem._objective.classify`'s kinds).
_OBJECTIVE_KINDS: dict[str, Callable[[_Call, Objective], ErrorEstimate]] = {
    "objective of s": _s_objective,
    "objective of omega": _omega_objective,
    "objective of participation": _participation_objective,
    "objective of capacitance": _potential_objective,
    "objective of electrostatic energy": _potential_objective,
    "objective of potential means": _potential_objective,
    "objective of absorbed power": _field_objective,
    "objective of field means": _field_objective,
}


def _check_mode_field_reach(model: Model, reach: str | None) -> None:
    """Raise for a participation dual out of reach, or a surface energy term."""
    if reach is not None:
        raise MeshRefinementUnsupportedError(
            "mode-field dual",
            "An objective of a participation is estimated with its mode-field "
            f"dual, which is out of reach with {reach}. Solve with "
            "those options, or use gfem.FieldEstimator().",
        )
    surface = [
        f"lumped port {p.name!r}"
        for p in model.ports
        if isinstance(p, LumpedPort) and p.rlc[2] is not None
    ] + [
        f"capacitive Impedance {b.surface!r}"
        for b in model.boundaries
        if isinstance(b, Impedance) and b.capacitance is not None
    ]
    if surface:
        raise MeshRefinementUnsupportedError(
            "mode-field dual",
            "An objective of a participation is estimated with an equilibrated "
            "mode-field dual, whose flux takes volume loads only; the mode "
            f"energy here holds the surface energy of the {surface[0]}, which "
            "no verified estimate covers. Use gfem.FieldEstimator().",
        )


def _check_isolated(omega: np.ndarray, mode: int, rtol: float) -> None:
    """Raise unless ``mode`` is apart from every other mode."""
    from gfem._eigen.reach import threshold as isolation_threshold

    others = [i for i in range(len(omega)) if i != mode]
    if not others:
        return
    gaps = np.abs(omega[others] - omega[mode]) / abs(omega[mode])
    near = others[int(np.argmin(gaps))]
    threshold = isolation_threshold(rtol)
    if gaps.min() < threshold:
        raise DegenerateModeError(
            f"An objective of the participation of mode {mode} at "
            f"{_frequency(omega, mode)} needs the mode isolated, but mode "
            f"{near} at {_frequency(omega, near)} lies within a relative gap "
            f"of {gaps.min():.3g}, below {threshold:g}: near another "
            "mode the field is a mix of the two. Separate the modes, or use "
            "gfem.FieldEstimator()."
        )


def _mode_terms(
    model: Model,
    result: Any,
    estimator: QuantityEstimator,
    mode: int | tuple[int, ...] | None,
    *,
    threads: int | None,
    cache: RecoveryCache,
) -> Any:
    """The kernel terms of a mode's ``q`` or ``omega``, or a cluster's mean omega."""
    from gfem._engine import estimate as kernels

    if isinstance(mode, tuple):
        for each in mode:
            _check_mode_eps(model, result, each)
        return kernels.omega_cluster(model, result, mode, threads, cache=cache)
    if estimator.q is not None:
        _check_lossy_q(result, int(mode or 0))
        return kernels.q(model, result, int(mode or 0), threads, cache=cache)
    _check_mode_eps(model, result, int(mode or 0))
    return kernels.omega(model, result, int(mode or 0), threads, cache=cache)


def _kind(estimator: Estimator) -> str:
    """The `ESTIMATE_KINDS` entry an estimator's estimate is of."""
    if isinstance(estimator, FieldEstimator):
        return "field"
    if isinstance(estimator.omega, tuple):
        return "cluster mean"
    for kind in ("capacitance", "s", "omega", "q", "objective"):
        if getattr(estimator, kind) is not None:
            return kind
    raise AssertionError(estimator)  # pragma: no cover - checked on construction


def _check_mode_eps(model: Model, result: EigenmodeResult, mode: int) -> None:
    """Raise for a mode bound to a law region at Re eps < 0.

    The omega estimate of such a mode (a plasmon) measured below the error,
    0.81 to 1.02, falling as Re eps approaches -1. Each law region's eps is
    read at the mode's complex omega, the value the estimate weighs with
    (`gfem._engine.dispersion.eps_at`).
    """
    from gfem._engine import dispersion

    omega = complex(np.asarray(result.omega)[mode])
    for lr in dispersion.law_regions(model):
        eps = complex(lr.law.eps_inf) + dispersion.susceptibility(lr.law, omega)
        if eps.real < 0:
            raise MeshRefinementUnsupportedError(
                "dispersive eigenmode",
                f"QuantityEstimator(omega={mode}) has no verified estimate for a "
                f"mode at {omega.real / (2 * np.pi) / 1e9:.6g} GHz, where the "
                f"permittivity law of region {lr.region!r} gives Re eps = "
                f"{eps.real:.3g} < 0 (a mode bound to a metal surface): use "
                "gfem.FieldEstimator().",
            )
