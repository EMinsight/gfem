"""Mesh refinement: error estimates of a solved result, and the refinement loop.

`refine_mesh` solves, estimates the error, marks the tetrahedra holding most
of it (Dörfler marking), bisects them and solves again, until the error
estimate meets a tolerance or a cap stops it (ADR 0085). `error_estimate`
gives one estimate of a result without refining (a measurement, ADR 0004).
Estimators are value objects named by what they estimate (ADR 0086): the
`FieldEstimator` estimates the field's error in the energy norm by flux
recovery (ADR 0087); the `QuantityEstimator` names one quantity of interest.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Any

import numpy as np

from gfem import _mpi
from gfem._checks import is_staging, is_traced
from gfem._pytree import readonly
from gfem._refine import Refinement, refine_model, transfer
from gfem.errors import (
    DistributedUnsupportedError,
    EigenmodeConvergenceError,
    LinearSolverConvergenceError,
    MeshRefinementNotConvergedWarning,
    MeshRefinementUnsupportedError,
    NonDifferentiableError,
)
from gfem.mesh import Mesh
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult
from gfem.solve import solve

__all__ = [
    "ErrorEstimate",
    "FieldEstimator",
    "MeshRefinement",
    "QuantityEstimator",
    "RefinementCycle",
    "error_estimate",
    "refine_mesh",
]

#: Why `refine_mesh` stopped.
STOPS = ("tolerance", "max_cycles", "max_tetrahedra", "solve failed")


@dataclasses.dataclass(frozen=True)
class FieldEstimator:
    """Estimates the error of a result's fields in the energy norm.

    The estimate is relative to the field's energy norm: 1e-2 means the
    field is about 1 % off its value on the same geometry with an infinitely
    fine mesh. Each tetrahedron's indicator is the distance between the
    solved electric flux and a smoother flux recovered from it (flux
    recovery), so no extra solve is needed. It is an estimate, not a bound;
    it does not include the faceting of curved surfaces or the error left by
    the linear solve.
    """


@dataclasses.dataclass(frozen=True)
class QuantityEstimator:
    """Estimates the error of one quantity of a solved result.

    Give exactly one keyword, named after the result attribute that holds
    the quantity and indexed the same way.

    Not available in this version of gfem: `refine_mesh` and
    `error_estimate` raise `gfem.MeshRefinementUnsupportedError` for it.

    Attributes:
        capacitance: Two terminal names, the entry
            ``ElectrostaticResult.capacitance[i, j]``.
        s: ``(to, from)`` port names, the entry ``result.s[to, from]``.
        omega: An index into ``EigenmodeResult.omega``.

    Raises:
        TypeError: Not exactly one keyword is given, or a value has the
            wrong type.
        ValueError: ``omega`` is negative.
    """

    capacitance: tuple[str, str] | None = None
    s: tuple[str, str] | None = None
    omega: int | None = None

    def __post_init__(self) -> None:
        """Check that exactly one quantity is named, with valid keys."""
        given = [
            name
            for name in ("capacitance", "s", "omega")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise TypeError(
                "QuantityEstimator takes exactly one of capacitance=(t1, t2), "
                f"s=(to, from) or omega=index; got {given or 'none'}."
            )
        for name in ("capacitance", "s"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, str)
                or len(tuple(value)) != 2
                or not all(isinstance(k, str) for k in value)
            ):
                raise TypeError(
                    f"QuantityEstimator({name}=...) takes two names, got {value!r}."
                )
            object.__setattr__(self, name, tuple(value))
        if self.omega is not None:
            if isinstance(self.omega, bool) or not isinstance(
                self.omega, (int, np.integer)
            ):
                raise TypeError(
                    "QuantityEstimator(omega=...) takes a mode index, got "
                    f"{self.omega!r}."
                )
            if self.omega < 0:
                raise ValueError(
                    "QuantityEstimator(omega=...) takes an index >= 0, got "
                    f"{self.omega}."
                )
            object.__setattr__(self, "omega", int(self.omega))


Estimator = FieldEstimator | QuantityEstimator


@dataclasses.dataclass(frozen=True, eq=False)
class ErrorEstimate:
    """An error estimator's estimate for one solved result.

    Attributes:
        estimator: The estimator that made it.
        values: Estimated size of the error per frequency (a driven result),
            per mode (an eigenmode result), or one entry, in the units of
            ``tolerance``: for a `FieldEstimator`, relative to the field's
            energy norm. Never negative. An estimate, not a bound.
        indicators: Each tetrahedron's share of the estimated error, shape
            ``(n_tetrahedra,)``, in the order of the model's
            ``mesh.tetrahedra``, combined over frequencies, terminals, ports
            and modes; zero outside the solve domain. For a single solve
            ``sqrt(sum(indicators**2))`` is ``value``.
    """

    estimator: Estimator
    values: tuple[float, ...]
    indicators: np.ndarray

    @property
    def value(self) -> float:
        """The largest of `values`: what `refine_mesh` compares to ``tolerance``."""
        return max(self.values)


@dataclasses.dataclass(frozen=True)
class RefinementCycle:
    """The numbers of one solved mesh of a `MeshRefinement`.

    Attributes:
        tetrahedra: The mesh's tetrahedron count.
        dofs: The field's coefficient count (one solve's unknowns).
        estimate: The error estimate's ``value`` on this mesh.
        value: The quantity of a `QuantityEstimator` on this mesh; ``None``
            for a `FieldEstimator`.
        marked: How many tetrahedra Dörfler marking picked on this mesh; 0
            when the loop stopped on it before marking.
    """

    tetrahedra: int
    dofs: int
    estimate: float
    value: float | complex | None
    marked: int


@dataclasses.dataclass(frozen=True, eq=False)
class MeshRefinement:
    """What `refine_mesh` returns: the last solved model, its result and history.

    Not a JAX pytree and not differentiable: differentiate `gfem.solve` on
    ``model`` instead.

    Attributes:
        model: The last solved model (its mesh refined, its design carried).
        result: That model's solve result.
        estimate: That result's `ErrorEstimate`.
        cycles: One `RefinementCycle` per solved mesh; ``cycles[0]`` is the
            input mesh.
        stop: Why the loop stopped: ``"tolerance"``, ``"max_cycles"``,
            ``"max_tetrahedra"`` or ``"solve failed"``.
        converged: Whether ``estimate.value`` is at or below the tolerance.
        tetrahedron_parents: For each tetrahedron of ``model.mesh``, the row
            of the input mesh's ``tetrahedra`` it lies in.
        triangle_parents: For each triangle of ``model.mesh``, the row of the
            input mesh's ``triangles`` it lies in.
    """

    model: Model
    result: Any
    estimate: ErrorEstimate
    cycles: tuple[RefinementCycle, ...]
    stop: str
    converged: bool
    tetrahedron_parents: np.ndarray
    triangle_parents: np.ndarray
    _input_mesh: Mesh = dataclasses.field(repr=False)

    def transfer(self, values: Any, region: str) -> Any:
        """Carry per-element values of a region from the input mesh to ``model``.

        Each element of the refined mesh takes the value of the input element
        it lies in. Plain indexing: a JAX array stays one, and `jax.grad`
        passes through it.

        Args:
            values: Shape ``(n,)`` or ``(n, k)``, one value per element of
                ``region`` on the input mesh, in ``mesh.elements(region)``
                order.
            region: A volume (tetrahedra) or surface (triangles) region name.

        Returns:
            One value per element of ``region`` on ``model.mesh``, in
            ``model.mesh.elements(region)`` order.

        Raises:
            RegionNotFoundError: ``region`` is not a region of the mesh.
            ValueError: ``values`` does not hold one value per element.
        """
        return transfer(
            values,
            region,
            coarse=self._input_mesh,
            fine=self.model.mesh,
            refinement=Refinement(self.tetrahedron_parents, self.triangle_parents),
        )


def error_estimate(result: Any, estimator: Estimator, *, model: Model) -> ErrorEstimate:
    """Estimate the error of a solved result, without refining.

    Example:
        The field error of a capacitance solve::

            result = gfem.solve(model, gfem.Electrostatic(terminals=["t1"]))
            est = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
            est.value, est.indicators  # relative energy error, per tetrahedron

    Args:
        result: A result of `gfem.solve` on ``model``: electrostatic,
            eigenmode (every mode is estimated), or a uniform driven sweep
            solved with ``keep_fields=True`` (every kept field).
        estimator: `FieldEstimator()`.
        model: The model the result was solved on (its materials weight the
            fields).

    Returns:
        The estimate, with NumPy values.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: The result was not solved on ``model``'s mesh, or is a
            driven result without the fields of every frequency (solve with
            ``keep_fields=True``).
        MeshRefinementUnsupportedError: The estimator or the result has no
            estimate in this version of gfem (``feature`` names what): an
            adaptive sweep, a periodic model, another result type.
        NonDifferentiableError: A model or result leaf is a JAX tracer, or
            the call is inside `jax.jit`.
    """
    if not isinstance(model, Model):
        raise TypeError(f"model must be a gfem.Model, got {type(model).__name__}.")
    _check_estimator(estimator)
    _check_untraced("error_estimate", (model, result))
    _check_estimable(result, estimator, model)
    return _estimate(model, result, estimator, threads=None)


def refine_mesh(
    model: Model,
    problem: Electrostatic | Driven | Eigenmode,
    *,
    estimator: Estimator = FieldEstimator(),  # noqa: B008 - frozen value object
    tolerance: float = 1e-2,
    error_fraction: float = 0.5,
    max_cycles: int = 10,
    max_tetrahedra: int | None = None,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> MeshRefinement:
    """Refine the mesh where the error is, until the error estimate meets a tolerance.

    Each refinement cycle solves the problem, estimates the error, stops if
    the estimate is at or below ``tolerance``, marks the fewest tetrahedra
    holding ``error_fraction`` of the squared error (Dörfler marking),
    bisects them and their neighbours until the mesh conforms (new nodes at
    edge midpoints, physical groups and imported facets kept), and carries
    the model's per-element design (`MaterialField` values, sheet fills) to
    the children. Cycle 0 solves the input mesh.

    A driven problem is solved with ``keep_fields=True`` whatever it asks:
    the estimate reads every frequency's field of every excited port, and
    ``result`` keeps them. Its low-frequency floor is checked once, on the
    returned mesh, not on every cycle.

    Example:
        Refine a capacitance solve to a 1 % field error::

            ref = gfem.refine_mesh(model, gfem.Electrostatic(terminals=["t1"]))
            ref.model, ref.result, ref.estimate.value, ref.stop

    Args:
        model: The device, with concrete values.
        problem: An `Electrostatic` problem, a `Driven` problem on a uniform
            sweep (lumped or wave ports), or an `Eigenmode` problem. The
            estimate of a driven run is the worst frequency's, of an
            eigenmode run the worst mode's.
        estimator: `FieldEstimator()`.
        tolerance: The error estimate to reach: relative to the field's energy
            norm for a `FieldEstimator`. At least 10 times
            ``options.tolerance`` (``options.eigen_tolerance`` for an
            `Eigenmode`), since the estimate does not include the linear
            solve's error.
        error_fraction: Dörfler's fraction, in (0, 1]: the share of the summed
            squared error indicators the marked tetrahedra hold.
        max_cycles: How many times the mesh may be refined; 0 solves and
            estimates only.
        max_tetrahedra: A cap on the refined mesh's tetrahedra: a bisected
            mesh above it is discarded and the loop stops on the last solved
            one. ``None`` sets no cap; each cycle grows the mesh, by about
            1.6 times on the first cycle and 1.25 times on later ones.
        options: The numerics of every solve; ``None`` is ``SolverOptions()``.
        comm: As for `gfem.solve`; mesh refinement runs on one process.

    Returns:
        The last solved model, its result and estimate, and one
        `RefinementCycle` per solved mesh.

    Warns:
        MeshRefinementNotConvergedWarning: The loop stopped on a cap or a
            failed solve with the estimate above ``tolerance``.
        LowFrequencyWarning: A frequency of a driven problem is below the
            returned mesh's low-frequency floor.

    Raises:
        TypeError: An argument has the wrong type.
        ValueError: A parameter is out of range, or ``tolerance`` is below 10
            times the solve tolerance.
        MeshRefinementUnsupportedError: The problem or the estimator has no
            mesh refinement (``feature`` names what): an adaptive sweep, a
            port mode problem, a periodic model or a Bloch wave vector.
        DistributedUnsupportedError: Launched on more than one rank.
        NonDifferentiableError: A model or problem leaf is a JAX tracer, or
            the call is inside `jax.jit`. Under `jax.grad` or `jax.vmap`
            with concrete inputs it runs; refine outside the transformed
            function, then differentiate `gfem.solve` on the refined model.
        LinearSolverConvergenceError, EigenmodeConvergenceError: The solve
            on the input mesh failed. On a later cycle the loop stops instead
            (``stop="solve failed"``).
    """
    if not isinstance(model, Model):
        raise TypeError(f"refine_mesh needs a gfem.Model, got {type(model).__name__}.")
    if options is None:
        options = SolverOptions()
    elif not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    _check_estimator(estimator)
    _check_problem(model, problem)
    if _mpi.resolve_comm(comm) is not None:
        raise DistributedUnsupportedError(
            "mesh refinement",
            "Mesh refinement has no distributed path: refine on one process, or "
            "run independent one-process copies with comm=MPI.COMM_SELF, then "
            "solve the refined model on ranks.",
        )
    _check_parameters(problem, options, tolerance, error_fraction, max_cycles)
    if max_tetrahedra is not None:
        _check_int(max_tetrahedra, "max_tetrahedra", 1)
    _check_untraced("refine_mesh", (model, problem))
    _check_supported(estimator)
    return _loop(
        model,
        problem,
        estimator=estimator,
        tolerance=float(tolerance),
        error_fraction=float(error_fraction),
        max_cycles=int(max_cycles),
        max_tetrahedra=max_tetrahedra,
        options=options,
        comm=comm,
    )


def _loop(
    model: Model,
    problem: Any,
    *,
    estimator: Estimator,
    tolerance: float,
    error_fraction: float,
    max_cycles: int,
    max_tetrahedra: int | None,
    options: SolverOptions,
    comm: Any,
) -> MeshRefinement:
    """Solve, estimate, stop, mark, bisect, cap, transfer; see `refine_mesh`."""
    from gfem._engine import driven

    if isinstance(problem, Driven):
        problem = dataclasses.replace(problem, keep_fields=True)
    tet_parents = np.arange(len(model.mesh.tetrahedra))
    tri_parents = np.arange(len(model.mesh.triangles))
    with driven.low_frequency_deferred():
        result = solve(model, problem, options=options, comm=comm)
    current = model
    cycles: list[RefinementCycle] = []
    failure: Exception | None = None
    while True:
        estimate = _estimate(current, result, estimator, threads=options.threads)
        cycle = RefinementCycle(
            tetrahedra=len(current.mesh.tetrahedra),
            dofs=_dofs(result),
            estimate=estimate.value,
            value=None,
            marked=0,
        )
        cycles.append(cycle)
        if estimate.value <= tolerance:
            stop = "tolerance"
            break
        if len(cycles) > max_cycles:
            stop = "max_cycles"
            break
        marked = dorfler(estimate.indicators, error_fraction)
        cycles[-1] = dataclasses.replace(cycle, marked=int(marked.sum()))
        refined, refinement = refine_model(current, marked)
        if max_tetrahedra is not None and len(refined.mesh.tetrahedra) > max_tetrahedra:
            stop = "max_tetrahedra"
            break
        try:
            with driven.low_frequency_deferred():
                refined_result = solve(refined, problem, options=options, comm=comm)
        except (LinearSolverConvergenceError, EigenmodeConvergenceError) as err:
            stop, failure = "solve failed", err
            break
        current, result = refined, refined_result
        tet_parents = tet_parents[refinement.tetrahedron_parents]
        tri_parents = tri_parents[refinement.triangle_parents]
    if isinstance(problem, Driven):
        driven.check_low_frequency(current, problem.frequency, stacklevel=3)
    converged = estimate.value <= tolerance
    if not converged:
        reason = {
            "max_cycles": f"after max_cycles={max_cycles} refinements",
            "max_tetrahedra": (
                f"because the next mesh would exceed max_tetrahedra={max_tetrahedra}"
            ),
            "solve failed": f"because the solve on the next mesh failed: {failure}",
        }[stop]
        warnings.warn(
            f"Mesh refinement stopped {reason}, with the error estimate "
            f"{estimate.value:.3g} above tolerance={tolerance:g}. The record "
            "holds the last solved mesh. Raise the cap, loosen tolerance, or "
            "start from a finer mesh.",
            MeshRefinementNotConvergedWarning,
            stacklevel=3,
        )
    return MeshRefinement(
        model=current,
        result=result,
        estimate=estimate,
        cycles=tuple(cycles),
        stop=stop,
        converged=converged,
        tetrahedron_parents=readonly(tet_parents),
        triangle_parents=readonly(tri_parents),
        _input_mesh=model.mesh,
    )


def dorfler(indicators: np.ndarray, error_fraction: float) -> np.ndarray:
    """Dörfler marking: the fewest tetrahedra holding a share of the squared error.

    Marks the tetrahedra with the largest indicators until their squared
    indicators add up to at least ``error_fraction`` of the total.

    Returns:
        A boolean mask over the tetrahedra; at least one is marked.
    """
    squared = np.asarray(indicators, float) ** 2
    order = np.argsort(-squared, kind="stable")
    cumulative = np.cumsum(squared[order])
    count = int(np.searchsorted(cumulative, error_fraction * cumulative[-1])) + 1
    marked = np.zeros(len(squared), bool)
    marked[order[: min(count, len(squared))]] = True
    return marked


def _estimate(
    model: Model, result: Any, estimator: Estimator, *, threads: int | None
) -> ErrorEstimate:
    """The estimate of ``result``: dispatch on the result type."""
    from gfem._engine import estimate as kernels

    if isinstance(result, ElectrostaticResult):
        solves = kernels.electrostatic(model, result, threads)
    elif isinstance(result, DrivenResult):
        solves = kernels.driven(model, result, threads)
    elif isinstance(result, EigenmodeResult):
        solves = kernels.eigenmode(model, result, threads)
    else:  # pragma: no cover - `_check_estimable` refuses the rest first
        raise _unsupported_result(result)
    values, indicators = kernels.combine(solves)
    return ErrorEstimate(
        estimator=estimator, values=values, indicators=readonly(indicators)
    )


def _dofs(result: Any) -> int:
    """A solve's unknowns: one field's coefficient count."""
    return int(_fields(result)[0].coefficients.shape[0])


def _fields(result: Any) -> list:
    """The fields of a result an estimate reads."""
    if isinstance(result, ElectrostaticResult):
        return list(result.potential.values())
    return list(result.e_fields)


def _check_estimator(estimator: Any) -> None:
    if not isinstance(estimator, (FieldEstimator, QuantityEstimator)):
        raise TypeError(
            "estimator must be gfem.FieldEstimator() or gfem.QuantityEstimator(...), "
            f"got {type(estimator).__name__}."
        )


def _check_problem(model: Model, problem: Any) -> None:
    """Raise for a problem mesh refinement has no path for, before any solve."""
    if isinstance(problem, PortMode):
        raise MeshRefinementUnsupportedError(
            "port mode",
            "Mesh refinement has no path for a gfem.PortMode problem: refine "
            "the device with a gfem.Driven problem on its wave ports, whose "
            "mode solve runs on the refined port faces.",
        )
    if not isinstance(problem, (Electrostatic, Driven, Eigenmode)):
        raise TypeError(
            "refine_mesh needs a gfem.Electrostatic, gfem.Driven or "
            f"gfem.Eigenmode problem, got {type(problem).__name__}."
        )
    if isinstance(problem, Driven) and problem.sweep == "adaptive":
        raise MeshRefinementUnsupportedError(
            "adaptive sweep",
            "Mesh refinement has no path for an adaptive sweep: refine on a "
            "uniform sweep at a few frequencies of the band, then sweep "
            "adaptively on refinement.model.",
        )
    if getattr(problem, "wave_vector", None) is not None:
        raise MeshRefinementUnsupportedError(
            "bloch",
            "Mesh refinement has no path for a Bloch wave vector: refining one "
            "side of a periodic pair would break its node pairing. Refine the "
            "mesh in your mesher.",
        )
    if model._periodic:  # noqa: SLF001
        raise MeshRefinementUnsupportedError(
            "periodic",
            "Mesh refinement has no path for a periodic model: refining one "
            "side of a periodic pair would break its node pairing. Refine the "
            "mesh in your mesher.",
        )


def _check_supported(estimator: Estimator) -> None:
    """Raise for an estimator this version cannot estimate with."""
    if isinstance(estimator, QuantityEstimator):
        raise _unsupported_quantity()


def _check_estimable(result: Any, estimator: Estimator, model: Model) -> None:
    if isinstance(estimator, QuantityEstimator):
        raise _unsupported_quantity()
    if not isinstance(result, (ElectrostaticResult, DrivenResult, EigenmodeResult)):
        raise _unsupported_result(result)
    if model._periodic:  # noqa: SLF001
        raise MeshRefinementUnsupportedError(
            "periodic",
            "The field estimator has no path for a periodic model: its "
            "recovery does not pair the periodic surfaces.",
        )
    if isinstance(result, DrivenResult):
        _check_kept(result)
    mesh = _fields(result)[0].mesh
    if mesh._topology is not model.mesh._topology or not np.array_equal(  # noqa: SLF001
        np.asarray(mesh.nodes), np.asarray(model.mesh.nodes)
    ):
        raise ValueError(
            "error_estimate needs the model the result was solved on: the "
            "result's fields live on another mesh, or on other node positions."
        )


def _check_kept(result: DrivenResult) -> None:
    """Raise unless a driven result is a uniform sweep with every field kept."""
    if result.sweep is not None and result.sweep.kind == "adaptive":
        raise MeshRefinementUnsupportedError(
            "adaptive sweep",
            "The field estimator has no path for an adaptive sweep: estimate a "
            "uniform sweep at a few frequencies of the band.",
        )
    kept = {f for f, _ in result.kept}
    missing = [f for f in result.s.frequency if f not in kept]
    if missing:
        raise ValueError(
            "error_estimate reads the electric fields of every frequency, and "
            f"this result kept none at {', '.join(f'{f:g}' for f in missing)} Hz: "
            "solve with gfem.Driven(..., keep_fields=True)."
        )


def _unsupported_quantity() -> MeshRefinementUnsupportedError:
    return MeshRefinementUnsupportedError(
        "quantity estimator",
        "gfem.QuantityEstimator has no estimate in this version of gfem: use "
        "gfem.FieldEstimator().",
    )


def _unsupported_result(result: Any) -> Exception:
    kind = type(result).__name__
    if not kind.endswith("Result"):
        return TypeError(f"error_estimate needs a result of gfem.solve, got {kind}.")
    return MeshRefinementUnsupportedError(
        kind.removesuffix("Result").lower(),
        f"The field estimator has no path for a gfem.{kind}: it estimates "
        "electrostatic, driven and eigenmode results.",
    )


def _check_untraced(name: str, values: Any) -> None:
    """Raise `NonDifferentiableError` on a traced leaf or a staging trace."""
    if is_traced(values) or is_staging():
        raise NonDifferentiableError(
            f"gfem.{name} cannot run inside jax.jit, or on values computed "
            "inside jax.grad, jax.jit or jax.vmap: mesh refinement builds a new "
            "mesh, which has no derivative. Refine with concrete values outside "
            "the transformed function, then differentiate gfem.solve on the "
            "refined model."
        )


def _check_parameters(
    problem: Any,
    options: SolverOptions,
    tolerance: Any,
    error_fraction: Any,
    max_cycles: Any,
) -> None:
    for value, name in ((tolerance, "tolerance"), (error_fraction, "error_fraction")):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a number, got {value!r}.")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be a positive number, got {tolerance!r}.")
    if not 0 < error_fraction <= 1:
        raise ValueError(f"error_fraction must be in (0, 1], got {error_fraction!r}.")
    _check_int(max_cycles, "max_cycles", 0)
    if isinstance(problem, Eigenmode):
        floor, floor_name = options.eigen_tolerance, "eigen_tolerance"
    else:
        floor, floor_name = options.tolerance, "tolerance"
    if tolerance < 10 * floor:
        raise ValueError(
            f"refine_mesh tolerance={tolerance:g} must be at least 10 x "
            f"SolverOptions.{floor_name}={floor:g}: the error estimate does not "
            "include the linear solve's error. Lower "
            f"SolverOptions.{floor_name}, or raise tolerance."
        )


def _check_int(value: Any, name: str, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {value!r}.")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}.")
