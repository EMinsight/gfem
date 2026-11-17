"""`gfem.solve`: the one verb that turns a model and a problem into a result."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any, overload

from gfem import _mpi
from gfem._adjoint import solve_vjp
from gfem._checks import is_traced
from gfem._materials import MaterialField
from gfem._periodic import check_nodes
from gfem.errors import LinearSolverConvergenceError
from gfem.model import _PHYSICS_NAMES, Model
from gfem.options import SolverOptions
from gfem.shape import _check_quality

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gfem._eigen.rule import DualRequest
    from gfem._engine.driven import AdjointLoads
    from gfem.problems import (
        DCConduction,
        Driven,
        Eigenmode,
        Electrostatic,
        PortMode,
        SteadyThermal,
    )
    from gfem.results import (
        DCConductionResult,
        DrivenResult,
        EigenmodeResult,
        ElectrostaticResult,
        PortModeResult,
        SteadyThermalResult,
    )

__all__ = ["release_memory", "solve"]


@overload
def solve(
    model: Model,
    problem: Electrostatic,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> ElectrostaticResult: ...
@overload
def solve(
    model: Model,
    problem: DCConduction,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> DCConductionResult: ...
@overload
def solve(
    model: Model,
    problem: SteadyThermal,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> SteadyThermalResult: ...
@overload
def solve(
    model: Model,
    problem: Driven,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> DrivenResult: ...
@overload
def solve(
    model: Model,
    problem: Eigenmode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> EigenmodeResult: ...
@overload
def solve(
    model: Model,
    problem: PortMode,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> PortModeResult: ...
def solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
) -> Any:
    """Solve one problem type on a model.

    A pure function of its inputs: the model and problem are not changed,
    and the same inputs give the same result.

    It is also a JAX function of the model's arrays (the mesh node
    coordinates and every `MaterialField` array): `jax.grad`, `jax.jit`
    and `jax.vmap` (sequential solves) apply. Each input then
    has a gradient (for an electrostatic capacitance: the `MaterialField`
    ``eps_r`` values), an exact zero gradient if the problem never reads it
    (``conductivity`` in electrostatics), or raises
    `gfem.NonDifferentiableError` when `jax.grad`, `jax.jit` or `jax.vmap`
    traces the function, if the problem reads it but gfem cannot
    differentiate it (for example a problem value such as
    a frequency, or a field-derived output). `jax.lax.stop_gradient` on an
    input or an output is the opt-out. Only reverse mode is supported.

    Example:
        S-parameters of a model over a band::

            result = gfem.solve(model, gfem.Driven(frequency=[4e9, 5e9, 6e9]))
            s21 = result.s["p2", "p1"]  # S21 over frequency

    Args:
        model: The device.
        problem: What to solve, e.g. `gfem.Electrostatic`.
        options: The numerics; ``None`` is ``SolverOptions()``.
        comm: An mpi4py communicator. ``None`` lets the launcher decide: one
            process without one, the world communicator under ``mpiexec`` or
            ``srun``, where every rank makes the same call. A communicator of
            size 1, such as ``MPI.COMM_SELF``, runs one independent copy.

    Returns:
        The problem type's result, e.g. `gfem.ElectrostaticResult` for an
        `Electrostatic` problem.

    Raises:
        TypeError: An argument has the wrong type.
        DistributedUnsupportedError: On several ranks, this solve has no
            distributed path; ``feature`` names what, the message the remedy.
        SolverUnavailableError: Launched on several ranks without a usable
            mpi4py and Open MPI.
        RuntimeError: On several ranks, the ranks hold different inputs;
            the message names one.
        RegionNotFoundError: The problem names a region the mesh lacks.
        ValueError: The problem does not fit the model; the message gives
            the remedy.
        NonDifferentiableError: Under `jax.grad`, an input or output gfem
            does not differentiate is perturbed or used.
        LinearSolverConvergenceError: An iterative linear solve
            (``linear_solver="bddc_gmres"``, or ``"petsc"`` with a Krylov
            ``ksp_type``) did not converge; the message gives the remedy. A
            uniform `gfem.Driven` sweep stops there, and the error's
            ``result`` holds the frequencies it solved, ``unsolved`` the
            rest.
    """
    return _solve(model, problem, options=options, comm=comm)[0]


def _solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions | None = None,
    comm: Any = None,
    adjoint_loads: AdjointLoads | Sequence[DualRequest] | None = None,
) -> tuple[Any, dict[Any, Any] | None]:
    """`solve`, and the adjoints of the per-frequency ``adjoint_loads`` with it.

    ``gfem.refine_mesh`` asks for an estimate's solved duals (ADR 0089
    amendment, ADR 0132): extra right-hand sides on the forward's
    factorization, formed per frequency from that frequency's solution
    (`gfem._engine.driven.AdjointLoads`); for an eigenmode problem, the
    participation duals of its `gfem._eigen.rule.DualRequest`s, on the
    forward's live factors (ADR 0142). Returns
    the result, and those adjoints per (frequency, key), or ``None``
    without ``adjoint_loads``. Warnings point at the caller of `solve`,
    one frame above this one.
    """
    if not isinstance(model, Model):
        raise TypeError(f"solve needs a gfem.Model, got {type(model).__name__}.")
    if options is None:
        options = SolverOptions()
    elif not isinstance(options, SolverOptions):
        raise TypeError(
            f"options must be a gfem.SolverOptions, got {type(options).__name__}."
        )
    resolved = _mpi.resolve_comm(comm)
    if resolved is not None:
        _mpi.refuse(model, problem, options)
    rule_class = _rule_class(problem)
    _check_domain(model, problem)
    _check_blends(model, problem)
    if resolved is None:
        rule = rule_class(model, problem, options)
    else:
        _mpi.entry_check(resolved, "solve", model, problem, options)
        try:
            rule = rule_class(model, problem, options, comm=resolved)
        except (TypeError, ValueError) as err:
            # the rule reads inputs the entry check has just found equal
            _mpi.agreed(err)
            raise
    if adjoint_loads is not None:
        rule.adjoint_loads = adjoint_loads
    if is_traced((model, problem)):
        return rule.result(solve_vjp(rule, model, problem)), None
    # A moved mesh must keep its periodic pairs and stay valid on the solve
    # domain.
    check_nodes(model.mesh.nodes, model._periodic)  # noqa: SLF001
    physics = getattr(type(problem), "_physics", "em")
    _check_quality(
        model.mesh,
        model._domain_of(physics).active,  # noqa: SLF001
        stacklevel=4,
        comm=resolved,
    )
    # a uniform sweep's partial result, built here: never in a JAX callback
    partial_result = getattr(rule, "partial_result", None)
    if partial_result is not None:
        rule.keeps_partial = True
    try:
        with _mpi.section(resolved, agreed=_mpi.FORWARD_ERRORS):
            outputs, _ = rule.forward(model, keep=False)
    except LinearSolverConvergenceError as err:
        if partial_result is not None:
            err.result = partial_result(err)
        raise
    adjoints = rule.adjoints if adjoint_loads is not None else None
    return rule.result(outputs), adjoints


def _check_domain(model: Model, problem: Any) -> None:
    """Raise when the problem's physics has nothing left to solve on.

    A model needs some volume in some physics' domain (ADR 0167); an empty
    domain for one physics raises when a problem of that physics is solved.
    A problem names its physics as ``_physics`` (electromagnetic by default).

    Raises:
        ValueError: Every volume is excluded from the problem's physics.
    """
    physics = getattr(type(problem), "_physics", "em")
    if model._domain_of(physics).active.any():  # noqa: SLF001
        return
    name = _PHYSICS_NAMES[physics]
    raise ValueError(
        f"Every volume is excluded from {name} problems, so the "
        f"{type(problem).__name__} problem has nothing to solve on. Exclude "
        f"fewer volumes for {name} problems (excluded=)."
    )


def _check_blends(model: Model, problem: Any) -> None:
    """Raise for a density blend with a law end off the driven uniform sweep.

    Such a blend (ADR 0113) is solved by `gfem.Driven` on the uniform sweep
    only (`gfem._engine.dispersion.refuse_blends`).

    Raises:
        ValueError: An electrostatic problem and a blend with a `Drude` end.
        NotImplementedError: Any other problem than a uniform driven sweep.
    """
    if not any(
        isinstance(m, MaterialField) and m.blend_ends is not None
        for m in model.materials.values()
    ):
        return
    kind = type(problem).__name__
    if kind == "Driven" and problem.sweep == "uniform":
        return
    # imported here: importing gfem does not import the engine
    from gfem._engine.dispersion import refuse_blends

    where = (
        'the adaptive sweep (solve sweep="uniform")'
        if kind == "Driven"
        else f"the {kind} problem"
    )
    refuse_blends(model, where, static=kind == "Electrostatic")


def _rule_class(problem: Any) -> type:
    """The engine rule of a problem type, imported on first use.

    Each problem class names its rule as ``_rule = "module:Class"``, so a new
    problem type needs no change here, and importing gfem does not import
    the engine.
    """
    path = getattr(type(problem), "_rule", None)
    if not isinstance(path, str):
        raise TypeError(
            "solve needs a problem type such as gfem.Electrostatic(...), got "
            f"{type(problem).__name__}."
        )
    module, name = path.split(":")
    return getattr(importlib.import_module(module), name)


def release_memory() -> None:
    """Free everything gfem keeps between solves in this process.

    That is what ``SolverOptions.retain_memory`` bounds, and the meshes and
    spaces it does not: what differentiated solves keep for their gradients,
    the solver plans, factorizations, assembled parts and wave port couplings
    kept for later designs, and the last discretizations of each mesh.

    Results, fields, meshes and models you hold are not touched, and no later
    result changes. A gradient whose solve was released is recomputed from
    the design when it runs, and the next solve builds its mesh, spaces and
    solver plan again: this costs time, never accuracy. Call it between two
    studies in one process, for example before a check at a higher order
    after an optimization loop.

    ``retain_memory=0`` keeps nothing from then on; this call drops what is
    kept now. JAX's own compilation caches are not touched: see
    ``jax.clear_caches()``. Under ``mpiexec`` or ``srun`` every rank must
    call it at the same point.

    Example:
        After an optimization loop, before another study::

            gfem.release_memory()
    """
    # nothing is kept before the engine's cache was first loaded
    cache = sys.modules.get("gfem._engine.cache")
    if cache is not None:
        cache.release_memory()
