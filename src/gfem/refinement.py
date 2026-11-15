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
import itertools
import math
import types
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem import _mpi
from gfem._checks import is_staging, is_traced
from gfem._pytree import readonly
from gfem._refine import Refinement, bisect, transfer, transfer_design
from gfem._text import hertz
from gfem.errors import (
    DegenerateModeError,
    EigenmodeConvergenceError,
    LinearSolverConvergenceError,
    MeshQualityWarning,
    MeshRefinementNotConvergedWarning,
    MeshRefinementUnsupportedError,
    NonDifferentiableError,
)
from gfem.mesh import Mesh
from gfem.model import Absorbing, Model, WavePort
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult
from gfem.shape import _check_quality
from gfem.solve import _solve

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from gfem._engine.driven import AdjointLoads
    from gfem._engine.estimate import RecoveryCache

__all__ = [
    "ErrorEstimate",
    "FieldEstimator",
    "MeshRefinement",
    "QuantityEstimator",
    "RefinementCycle",
    "error_estimate",
    "refine_mesh",
]


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

    With loss, the flux is the total current ``eps_eff E``, ``eps_eff =
    eps_r (1 - i tan_delta) - i sigma / (omega eps_0)``, and the norm is
    ``||curl E||^2_{mu^-1} + |k|^2 int |eps_eff| |E|^2``: the conduction
    current counts, as it does in the field. Weighting the flux error by
    ``Re(eps_eff)^-1`` instead overestimates; recovering ``eps_r E``, as
    without loss, underestimates without bound under conduction loss.

    A dispersive material (a `gfem.PermittivityLaw`) enters at its
    permittivity at each solved frequency, lossless or not, Re eps < 0
    included. Where it is exactly zero (a lossless Drude term at its plasma
    frequency, without conductivity) the flux has no weight, and
    `refine_mesh` and `error_estimate` raise `ValueError`.

    `refine_mesh` and `error_estimate` raise
    `gfem.MeshRefinementUnsupportedError` for a driven or eigenmode field
    with loss inside an anisotropic material, and at order 1.
    """


@dataclasses.dataclass(frozen=True)
class QuantityEstimator:
    """Estimates the error of one quantity of a solved result.

    Estimates how far one quantity of a solved result (a capacitance entry,
    an S entry whose source port is excited, one mode's ``omega`` or its
    Q) is from its value on the same geometry with an infinitely fine mesh.
    The estimate uses the fields already solved, with no extra solve, except
    for the adjoint duals below. It does not include
    the faceting of curved surfaces or the error left by the linear solve.

    The estimate is a magnitude in the units of ``tolerance``: relative to
    the capacitance entry, to ``omega`` or to Q, absolute for an S entry. It has
    no sign or phase, and gfem never adds it to the quantity. It is an
    estimate, not a bound. A diagonal capacitance entry never lies below
    its value on the meshed geometry: ``C_h >= C``; an off-diagonal entry
    may lie on either side.

    Give exactly one keyword, named after the result attribute that holds
    the quantity and indexed the same way.

    Example:
        Refine for S21, then read its estimated error::

            est = gfem.QuantityEstimator(s=("p2", "p1"))
            ref = gfem.refine_mesh(model, gfem.Driven([5e9]), estimator=est)
            ref.result.s["p2", "p1"], ref.estimate.values

    Attributes:
        capacitance: Two terminal names ``(i, j)``, the entry
            ``ElectrostaticResult.capacitance[i, j]``; the same name twice
            for a diagonal entry.
        s: ``(to, from)`` port names, the entry ``result.s[to, from]``; the
            source port ``from`` must be excited (`Driven.excite`), ``to``
            may be any port.
        omega: An index into ``EigenmodeResult.omega`` on the first mesh;
            on each refined mesh the estimate follows the mode whose omega
            is nearest the previous mesh's, and `refine_mesh` raises
            `gfem.DegenerateModeError` when that mode is not clearly the
            same one.
        q: An index into ``EigenmodeResult.q``, followed as ``omega``; the
            estimate is relative to Q. Q's relative error is cot(arg omega)
            times the imaginary part of omega's relative error, estimated on
            the same terms, with no extra solve. A lossless mode (infinite Q)
            raises `ValueError`.

    Raises:
        TypeError: Not exactly one keyword is given, or a value has the
            wrong type (a tuple for ``q``: the Q of a cluster is not
            estimated).
        ValueError: ``omega`` or ``q`` is negative.

    `refine_mesh` and `error_estimate` raise
    `gfem.MeshRefinementUnsupportedError` for entries with no verified
    estimate: a mode with a second-order absorbing boundary, a complex
    ``mu_r`` or a permittivity law, and an S entry or a mode with loss
    inside an anisotropic material (as `FieldEstimator`). Both estimators
    also raise it at order 1. For a lossy mode the ``omega`` estimate covers
    the complex ``omega``; ``q`` estimates Q, and raises as ``omega`` does
    above, and for a mode with loss inside an anisotropic material
    (``feature`` ``"lossy eigenmode"``). An S entry ``(to, from)``
    whose port ``to`` is not excited, or is a wave port with a lossy mode not
    parallel to its excitation, needs that port's adjoint dual: `refine_mesh`
    adds it to each of its solves, one more right-hand side with its
    options; `error_estimate` on a finished result factorizes once more per
    frequency, with the direct solver at the result's order, so estimate a
    large model through `refine_mesh` (``max_cycles=0``).
    """

    capacitance: tuple[str, str] | None = None
    s: tuple[str, str] | None = None
    omega: int | None = None
    q: int | None = None

    def __post_init__(self) -> None:
        """Check that exactly one quantity is named, with valid keys."""
        given = [
            name
            for name in ("capacitance", "s", "omega", "q")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise TypeError(
                "QuantityEstimator takes exactly one of capacitance=(t1, t2), "
                f"s=(to, from), omega=index or q=index; got {given or 'none'}."
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
        for name in ("omega", "q"):
            value = getattr(self, name)
            if value is None:
                continue
            if name == "q" and isinstance(value, (tuple, list)):
                raise TypeError(
                    f"QuantityEstimator(q=...) takes one mode index, got {value!r}: "
                    "the Q of a cluster is not estimated."
                )
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(
                    f"QuantityEstimator({name}=...) takes a mode index, got {value!r}."
                )
            if value < 0:
                raise ValueError(
                    f"QuantityEstimator({name}=...) takes an index >= 0, got {value}."
                )
            object.__setattr__(self, name, int(value))

    @property
    def _mode_keyword(self) -> str | None:
        """``"omega"`` or ``"q"`` when the quantity is of one mode, else ``None``."""
        if self.omega is not None:
            return "omega"
        return "q" if self.q is not None else None


Estimator = FieldEstimator | QuantityEstimator

#: What an `ErrorEstimate` can be of (its ``kind``): a closed set (ADR 0143).
#: ``"field"`` is the `FieldEstimator`'s; ``"capacitance"``, ``"s"`` and
#: ``"omega"`` a `QuantityEstimator`'s entries; ``"q"`` and ``"cluster mean"``
#: a mode's Q and a degenerate cluster's mean omega; every ``"objective of
#: ..."`` an objective's, by what it reads. Each kind has its own verified
#: effectivity range (``EFFECTIVITY_BY_KIND`` in the estimate tests), which
#: the docs generate per kind. Private: not in ``gfem.__all__``.
ESTIMATE_KINDS = (
    "field",
    "capacitance",
    "s",
    "omega",
    "q",
    "cluster mean",
    "objective of s",
    "objective of s, lossless rewrite",
    "objective of omega",
    "objective of capacitance",
    "objective of electrostatic energy",
    "objective of potential means",
    "objective of absorbed power",
    "objective of field means",
    "objective of participation",
)

#: The kinds of `ESTIMATE_KINDS` the estimators emit today, in that order. A
#: kind joins when its estimate lands; the docs build requires verification
#: cases for exactly these. Equal to `ESTIMATE_KINDS` once every kind has.
EMITTED_KINDS = ("field", "capacitance", "s", "omega", "q")


@dataclasses.dataclass(frozen=True, eq=False)
class ErrorEstimate:
    """An error estimator's estimate for one solved result.

    Attributes:
        estimator: The estimator that made it.
        kind: What was estimated: ``"field"`` for a `FieldEstimator`;
            ``"capacitance"``, ``"s"``, ``"omega"`` or ``"q"`` for a
            `QuantityEstimator`, after its keyword. Each kind has its own
            measured effectivity range (verification page).
        values: Estimated size of the error per frequency (a driven result),
            per mode (an eigenmode result under a `FieldEstimator`), or one
            entry, in the units of ``tolerance``. For a `FieldEstimator`
            this is relative to the field's energy norm. For a
            `QuantityEstimator` this is the quantity's error: relative to
            the capacitance entry, to ``omega`` or to Q, absolute for an S
            entry.
            It is never negative and carries no sign or phase. It is an
            estimate, not a bound.
        indicators: Each tetrahedron's share of the estimated error, shape
            ``(n_tetrahedra,)``, in the order of the model's
            ``mesh.tetrahedra``, combined over frequencies, terminals, ports
            and modes; zero outside the solve domain. For a single solve
            ``sqrt(sum(indicators**2))`` is ``value`` for a
            `FieldEstimator`. For a `QuantityEstimator`, ``indicators**2``
            is each tetrahedron's term of the quantity's error, in its
            units; ``sum(indicators**2)`` is ``value`` for a capacitance,
            and may lie above or below it for an S entry or ``omega``,
            whose ``value`` keeps the curl and mass terms from cancelling.
    """

    estimator: Estimator
    kind: str
    values: tuple[float, ...]
    indicators: np.ndarray

    @property
    def value(self) -> float:
        """The largest of `values`: what `refine_mesh` compares to ``tolerance``."""
        return max(self.values)


@dataclasses.dataclass(frozen=True)
class RefinementCycle:
    """The numbers of one solved mesh of a `MeshRefinement`.

    With a tuple of estimators, ``estimate`` and ``value`` are tuples, one
    entry per estimator in the given order.

    Attributes:
        tetrahedra: The mesh's tetrahedron count.
        dofs: The field's coefficient count (one solve's unknowns).
        estimate: The error estimate's ``value`` on this mesh.
        value: The quantity of a `QuantityEstimator` on this mesh: the
            capacitance entry in farads, a tuple of the S entry per
            frequency, the followed mode's complex ``omega`` in rad/s, or its
            Q; ``None`` for a `FieldEstimator`.
        marked: How many tetrahedra Dörfler marking picked on this mesh; 0
            when the loop stopped on it before marking.
        modes: For an eigenmode quantity, each mode index the estimators
            name on the first mesh, mapped to the index of the mode followed
            to this mesh (read-only); empty otherwise.
    """

    tetrahedra: int
    dofs: int
    estimate: float | tuple[float, ...]
    value: Any
    marked: int
    modes: Mapping[int, int] = dataclasses.field(
        default_factory=lambda: types.MappingProxyType({}), hash=False
    )


@dataclasses.dataclass(frozen=True, eq=False)
class MeshRefinement:
    """What `refine_mesh` returns: the last solved model, its result and history.

    Not a JAX pytree and not differentiable: differentiate `gfem.solve` on
    ``model`` instead.

    Attributes:
        model: The last solved model (its mesh refined, its design carried).
        result: That model's solve result.
        estimate: That result's `ErrorEstimate`; with a tuple of estimators,
            a tuple of them in the estimators' order.
        cycles: One `RefinementCycle` per solved mesh; ``cycles[0]`` is the
            input mesh.
        stop: Why the loop stopped: ``"tolerance"``, ``"max_cycles"``,
            ``"max_tetrahedra"`` or ``"solve failed"``.
        converged: Whether ``estimate.value`` is at or below the tolerance;
            with a tuple of estimators, whether every one is at or below its
            own.
        tetrahedron_parents: For each tetrahedron of ``model.mesh``, the row
            of the input mesh's ``tetrahedra`` it lies in.
        triangle_parents: For each triangle of ``model.mesh``, the row of the
            input mesh's ``triangles`` it lies in.
    """

    model: Model
    result: Any
    estimate: ErrorEstimate | tuple[ErrorEstimate, ...]
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


#: The lowest ratio of an estimate to the error over the verified cases; mode
#: following bounds a mode's shift between two meshes with it.
_ESTIMATE_FLOOR = 0.9


def error_estimate(result: Any, estimator: Estimator, *, model: Model) -> ErrorEstimate:
    """Estimate the error of a solved result, without refining.

    No solve is needed, except for an S entry whose port ``to`` is not
    excited, or is a wave port with a lossy mode not parallel to its
    excitation: its adjoint dual factorizes the system once more per
    frequency, with the direct solver at the result's order (the result does
    not record its solve options). On a large model,
    estimate through `refine_mesh` with ``max_cycles=0``, whose adjoint rides
    on its own solve with your options.

    Example:
        The field error of a capacitance solve::

            result = gfem.solve(model, gfem.Electrostatic(terminals=["t1"]))
            est = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
            est.value, est.indicators  # relative energy error, per tetrahedron

    Args:
        result: A result of `gfem.solve` on ``model``: electrostatic,
            eigenmode (every mode is estimated), or a uniform driven sweep
            solved with ``keep_fields=True`` (every kept field).
        estimator: `FieldEstimator()`, or a `QuantityEstimator` naming an
            entry of ``result`` (its ``omega`` index is a mode of
            ``result``).
        model: The model the result was solved on (its materials weight the
            fields).

    Returns:
        The estimate, with NumPy values.

    Raises:
        TypeError: An argument has the wrong type (a tuple of estimators
            too: call it once per quantity), or a `QuantityEstimator` names a
            quantity of another result type.
        KeyError: A `QuantityEstimator` names no terminal or port of the
            result.
        ValueError: The result was not solved on ``model``'s mesh, or is a
            driven result without the fields of every frequency (solve with
            ``keep_fields=True``); a `QuantityEstimator` names a source port
            that was not excited, a mode the result does not hold, or the Q of
            a lossless mode; a permittivity law is zero at a driven frequency.
        MeshRefinementUnsupportedError: The estimator or the result has no
            estimate in this version of gfem (``feature`` names what): an
            adaptive sweep, a periodic model, another result type, a result
            solved at order 1, a driven or eigenmode field with loss inside
            an anisotropic material (see `FieldEstimator`), or a quantity
            entry with no verified estimate (see `QuantityEstimator`).
        DegenerateModeError: A `QuantityEstimator` names a mode of a
            degenerate cluster, or one within the default
            `SolverOptions.cluster_rtol` of another mode.
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        DistributedUnsupportedError: The result was solved across ranks.
        NonDifferentiableError: A model or result leaf is a JAX tracer, or
            the call is inside `jax.jit`.
    """
    if not isinstance(model, Model):
        raise TypeError(f"model must be a gfem.Model, got {type(model).__name__}.")
    if isinstance(estimator, (tuple, list)):
        raise TypeError(
            "error_estimate takes one estimator, got a "
            f"{type(estimator).__name__}: call it once per quantity."
        )
    _check_estimator(estimator)
    _refuse_blends(model, "error_estimate")
    _check_untraced("error_estimate", (model, result))
    _check_estimable(result, estimator, model)
    mode = _estimator_mode(estimator)
    if mode is not None:
        # `refine_mesh`'s rule, at the default tolerance: the result does not
        # record the options it was solved with
        _track_mode(result, mode, None, SolverOptions().cluster_rtol, None)
    return _estimate(model, result, estimator, threads=None, mode=mode)


def refine_mesh(
    model: Model,
    problem: Electrostatic | Driven | Eigenmode,
    *,
    estimator: Estimator | tuple[QuantityEstimator, ...] = FieldEstimator(),  # noqa: B008 - frozen value object
    tolerance: float | tuple[float, ...] | None = None,
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

    With a tuple of estimators, several quantities refine one mesh: every
    one is estimated on every mesh, the marking sums each quantity's
    ``indicators**2`` divided by its tolerance over the quantities still
    above it, and the loop stops when every one meets its tolerance. Modes
    named by several estimators are followed once each.

    A driven problem is solved with ``keep_fields=True`` whatever it asks:
    the estimate reads every frequency's field of every excited port, and
    ``result`` keeps them. Its low-frequency floor is checked once, on the
    returned mesh, not on every cycle.

    Example:
        Refine a capacitance solve to a 1 % field error::

            ref = gfem.refine_mesh(model, gfem.Electrostatic(terminals=["t1"]))
            ref.model, ref.result, ref.estimate.value, ref.stop

        Refine for S21 and S11 at once, each to its own tolerance::

            estimators = (
                gfem.QuantityEstimator(s=("p2", "p1")),
                gfem.QuantityEstimator(s=("p1", "p1")),
            )
            ref = gfem.refine_mesh(
                model, gfem.Driven([5e9]), estimator=estimators, tolerance=(1e-4, 1e-3)
            )
            s21_estimate, s11_estimate = ref.estimate

    Args:
        model: The device, with concrete values.
        problem: An `Electrostatic` problem, a `Driven` problem on a uniform
            sweep (lumped or wave ports), or an `Eigenmode` problem. The
            estimate of a driven run is the worst frequency's, of an
            eigenmode run the worst mode's.
        estimator: `FieldEstimator()`, a `QuantityEstimator` naming a
            quantity of ``problem``, or a tuple of distinct
            `QuantityEstimator`s; the record then holds tuples, one entry per
            estimator in this order.
        tolerance: The error estimate to reach, one float for every
            estimator or a tuple with one per estimator; ``None`` is 1e-2.
            Relative to the field's energy
            norm for a `FieldEstimator`; for a `QuantityEstimator` relative
            to the capacitance entry, to ``omega`` or to Q, absolute for an S
            entry. At least 10 times
            ``options.tolerance`` (``options.eigen_tolerance`` for an
            `Eigenmode`; for Q times cot(arg omega), about 2Q, checked on
            the first solve), since the estimate does not include the linear
            solve's error. Each tolerance is checked on its own.
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
        The last solved model, its result and estimate (a tuple with a tuple
        of estimators), and one `RefinementCycle` per solved mesh.

    Warns:
        MeshRefinementNotConvergedWarning: The loop stopped on a cap or a
            failed solve with an estimate above its tolerance; it names each
            such quantity.
        LowFrequencyWarning: A frequency of a driven problem is below the
            returned mesh's low-frequency floor.
        MeshQualityWarning: The input mesh's nodes have moved and an element
            fell far below its imported quality.

    Raises:
        TypeError: An argument has the wrong type, a `QuantityEstimator`
            names a quantity of another problem type, or a tuple of
            estimators holds a `FieldEstimator`.
        KeyError: A `QuantityEstimator` names no terminal or port of the
            problem.
        ValueError: A parameter is out of range; a tuple of estimators is
            empty or names a quantity twice, or the tolerances do not match
            the estimators one to one; a tolerance is below 10
            times the solve tolerance (for Q, times cot(arg omega)); a
            `QuantityEstimator` names a source port that is not excited, a
            mode index at or above ``n_modes``, or the Q of a lossless mode; a
            permittivity law is zero at a driven frequency.
        MeshRefinementUnsupportedError: The problem or the estimator has no
            mesh refinement (``feature`` names what): an adaptive sweep, a
            port mode problem, a periodic model or a Bloch wave vector,
            ``options.order=1`` (the ``"fast"`` preset), a driven or
            eigenmode field with loss inside an anisotropic material (see
            `FieldEstimator`), a quantity entry with no verified estimate
            (see `QuantityEstimator`), or ``frequency_workers`` above 1 for
            an S estimate that solves its adjoint dual (an unexcited ``to``
            or a lossy wave port), which rides on the forward solve.
        DegenerateModeError: The mode a `QuantityEstimator` follows is not
            alone: another mode lies within ``options.cluster_rtol`` of it.
            Or, on a refined mesh, the nearest mode is not clearly the
            followed one: a second mode is less than twice as far from the
            previous frequency, or the nearest moved further than the
            previous estimate allows (the mode probably left the solved
            modes: ask for more modes or start from a finer mesh). Or two
            modes named on the first mesh reached one mode.
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        DistributedUnsupportedError: Launched on more than one rank.
        MeshQualityError: The input mesh's nodes have moved and an element is
            inverted.
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
    estimators, tolerances = _quantities(estimator, tolerance)
    _check_problem(model, problem)
    _refuse_blends(model, "refine_mesh")
    for each in estimators:
        if isinstance(each, QuantityEstimator):
            _check_quantity_keys(each, problem, model)
            _check_quantity_model(each, model)
        if isinstance(problem, (Driven, Eigenmode)):
            _check_field_model(each, model)
    if isinstance(problem, Driven):
        from gfem._engine import dispersion

        dispersion.check_estimator_frequencies(
            model, [float(f) for f in problem.frequency]
        )
    for each in estimators:
        _check_order(options.order, each)
    if _mpi.resolve_comm(comm) is not None:
        _mpi.refuse_feature("mesh_refinement")
    _check_adjoint_workers(estimators, model, problem, options)
    for each in tolerances:
        _check_parameters(problem, options, each, error_fraction, max_cycles)
    if max_tetrahedra is not None:
        _check_int(max_tetrahedra, "max_tetrahedra", 1)
    _check_untraced("refine_mesh", (model, problem))
    # a moved input mesh is checked here, so a warning names the caller's line;
    # the loop's solves and bisections skip the check
    _check_quality(model.mesh, model._domain.active, stacklevel=2)  # noqa: SLF001
    _free_consumed(options)
    return _loop(
        model,
        problem,
        estimators=estimators,
        tolerances=tuple(float(t) for t in tolerances),
        joint=isinstance(estimator, tuple),
        error_fraction=float(error_fraction),
        max_cycles=int(max_cycles),
        max_tetrahedra=max_tetrahedra,
        options=options,
        comm=comm,
    )


#: `refine_mesh`'s tolerance when none is given.
_DEFAULT_TOLERANCE = 1e-2


def _quantities(
    estimator: Any, tolerance: Any
) -> tuple[tuple[Estimator, ...], tuple[Any, ...]]:
    """The estimators of `refine_mesh` and one tolerance each (ADR 0130).

    A tuple of estimators holds `QuantityEstimator`s, at least one, none
    twice; a tuple of tolerances needs a tuple of estimators of its length.
    The tolerances' values are checked by `_check_parameters`.
    """
    if isinstance(estimator, tuple):
        if not estimator:
            raise ValueError(
                "refine_mesh needs at least one estimator in a tuple of estimators."
            )
        for each in estimator:
            if isinstance(each, FieldEstimator):
                raise TypeError(
                    "A tuple of estimators takes gfem.QuantityEstimator(...)s only: "
                    "gfem.FieldEstimator() already estimates every field of the "
                    "result, in units no quantity has. Refine for it on its own."
                )
            _check_estimator(each)
        for first, second in itertools.combinations(estimator, 2):
            if first == second:
                raise ValueError(
                    f"The tuple of estimators names {_describe(first)} twice: "
                    "give each quantity once."
                )
        estimators: tuple[Estimator, ...] = estimator
    else:
        _check_estimator(estimator)
        estimators = (estimator,)
    if tolerance is None:
        return estimators, (_DEFAULT_TOLERANCE,) * len(estimators)
    if not isinstance(tolerance, tuple):
        return estimators, (tolerance,) * len(estimators)
    if not isinstance(estimator, tuple):
        raise TypeError(
            "A tuple of tolerances needs a tuple of estimators, one tolerance "
            "each: give tolerance as a float."
        )
    if len(tolerance) != len(estimators):
        raise ValueError(
            f"tolerance gives {len(tolerance)} values for {len(estimators)} "
            "estimators: give one per estimator, in their order, or one float "
            "for all."
        )
    return estimators, tolerance


def _describe(estimator: Estimator) -> str:
    """``QuantityEstimator(s=('p2', 'p1'))``: an estimator by its one keyword."""
    if isinstance(estimator, FieldEstimator):
        return "FieldEstimator()"
    kind = _kind(estimator)
    return f"QuantityEstimator({kind}={getattr(estimator, kind)!r})"


def _loop(
    model: Model,
    problem: Any,
    *,
    estimators: tuple[Estimator, ...],
    tolerances: tuple[float, ...],
    joint: bool,
    error_fraction: float,
    max_cycles: int,
    max_tetrahedra: int | None,
    options: SolverOptions,
    comm: Any,
) -> MeshRefinement:
    """Solve, estimate, stop, mark, bisect, cap, transfer; see `refine_mesh`.

    Every estimator is estimated on every mesh; marking sums the
    tolerance-normalized weights of those still above their tolerance
    (`_marking`), and the loop stops when none is (ADR 0130). ``joint``
    (a tuple was given) makes the records tuples. Each solved mesh gets one
    `RecoveryCache`, shared by its estimates and dropped with the mesh.
    """
    from gfem._engine import driven

    if isinstance(problem, Driven):
        problem = dataclasses.replace(problem, keep_fields=True)
    tet_parents = np.arange(len(model.mesh.tetrahedra))
    tri_parents = np.arange(len(model.mesh.triangles))
    adjoint_loads = _adjoint_loads(estimators, model, problem)
    with driven.low_frequency_deferred(), warnings.catch_warnings():
        warnings.simplefilter("ignore", MeshQualityWarning)  # `refine_mesh` warned
        result, adjoints = _solve(
            model, problem, options=options, comm=comm, adjoint_loads=adjoint_loads
        )
    current = model
    cycles: list[RefinementCycle] = []
    failure: Exception | None = None
    tracks = _Tracks(estimators)
    while True:
        cache = _recovery_cache()  # this mesh's, dropped before the next solve
        modes = tracks.follow(result, options.cluster_rtol)
        if not cycles:
            _check_q_floors(result, estimators, tolerances, modes, options)
        estimates = tuple(
            _estimate(
                current,
                result,
                each,
                threads=options.threads,
                mode=_followed(modes, each),
                adjoints=adjoints,
                cache=cache,
            )
            for each in estimators
        )
        above = [
            i
            for i, (estimate, tolerance) in enumerate(
                zip(estimates, tolerances, strict=True)
            )
            if estimate.value > tolerance
        ]
        cycle = RefinementCycle(
            tetrahedra=len(current.mesh.tetrahedra),
            dofs=_dofs(result),
            estimate=_shaped([e.value for e in estimates], joint),
            value=_shaped(
                [_quantity(result, e, _followed(modes, e)) for e in estimators],
                joint,
            ),
            marked=0,
            modes=types.MappingProxyType(modes),
        )
        cycles.append(cycle)
        if not above:
            stop = "tolerance"
            break
        if len(cycles) > max_cycles:
            stop = "max_cycles"
            break
        marked = _dorfler(_marking(estimates, tolerances, above), error_fraction)
        cycles[-1] = dataclasses.replace(cycle, marked=int(marked.sum()))
        fine, refinement = bisect(current.mesh, marked)
        if max_tetrahedra is not None and len(fine.tetrahedra) > max_tetrahedra:
            stop = "max_tetrahedra"
            break
        tracks.bound(current, result, estimates, threads=options.threads, cache=cache)
        refined = transfer_design(current, fine, refinement)
        try:
            with driven.low_frequency_deferred():
                refined_result, refined_adjoints = _solve(
                    refined,
                    problem,
                    options=options,
                    comm=comm,
                    adjoint_loads=adjoint_loads,
                )
        except (LinearSolverConvergenceError, EigenmodeConvergenceError) as err:
            stop, failure = "solve failed", err
            break
        current, result, adjoints = refined, refined_result, refined_adjoints
        tet_parents = tet_parents[refinement.tetrahedron_parents]
        tri_parents = tri_parents[refinement.triangle_parents]
    if isinstance(problem, Driven):
        driven.check_low_frequency(current, problem.frequency, stacklevel=3)
    if above:
        _warn_not_converged(
            stop,
            failure,
            [(estimators[i], estimates[i].value, tolerances[i]) for i in above],
            joint=joint,
            max_cycles=max_cycles,
            max_tetrahedra=max_tetrahedra,
        )
    return MeshRefinement(
        model=current,
        result=result,
        estimate=_shaped(list(estimates), joint),
        cycles=tuple(cycles),
        stop=stop,
        converged=not above,
        tetrahedron_parents=readonly(tet_parents),
        triangle_parents=readonly(tri_parents),
        _input_mesh=model.mesh,
    )


def _check_q_floors(
    result: Any,
    estimators: tuple[Estimator, ...],
    tolerances: tuple[float, ...],
    modes: dict[int, int],
    options: SolverOptions,
) -> None:
    """`_check_q_floor` of each ``q=`` estimator, on the first result."""
    for each, tolerance in zip(estimators, tolerances, strict=True):
        if isinstance(each, QuantityEstimator) and each.q is not None:
            _check_q_floor(result, modes[each.q], tolerance, options)


def _followed(modes: dict[int, int], estimator: Estimator) -> int | None:
    """The index on this mesh of the mode ``estimator`` names, else ``None``."""
    mode = _estimator_mode(estimator)
    return None if mode is None else modes[mode]


def _shaped(items: list, joint: bool) -> Any:
    """``items`` as a tuple for a tuple of estimators, else its one item."""
    return tuple(items) if joint else items[0]


def _marking(
    estimates: tuple[ErrorEstimate, ...],
    tolerances: tuple[float, ...],
    above: list[int],
) -> np.ndarray:
    """The indicators Dörfler marks on: those of the quantities above tolerance.

    Each quantity's ``indicators**2`` is in its own units; divided by its
    tolerance they add up across quantities (ADR 0130). A lone quantity's
    indicators are marked on as they are.
    """
    if len(above) == 1:
        return estimates[above[0]].indicators
    squared = sum(
        np.asarray(estimates[i].indicators, float) ** 2 / tolerances[i] for i in above
    )
    return np.sqrt(squared)


class _Tracks:
    """The modes the estimators follow: one track per distinct starting index.

    Estimators naming the same mode on the first mesh share its track
    (ADR 0130). On each mesh every track follows the mode nearest its
    previous omega (`_track_mode`), its shift bounded by the omega estimate
    of the mode on its own sums on the previous mesh (`bound`); two tracks
    landing on one mode raise `DegenerateModeError`.
    """

    def __init__(self, estimators: tuple[Estimator, ...]) -> None:
        starts = sorted(
            {m for e in estimators if (m := _estimator_mode(e)) is not None}
        )
        #: Per starting index: the followed index, its omega and its omega
        #: estimate on the last mesh (``None`` before the first).
        self._state: dict[int, tuple[int, complex | None, float | None]] = {
            start: (start, None, None) for start in starts
        }
        self._omega = {
            e.omega: i
            for i, e in enumerate(estimators)
            if isinstance(e, QuantityEstimator) and e.omega is not None
        }

    def follow(self, result: Any, rtol: float) -> dict[int, int]:
        """Each track's mode on ``result``: starting index -> its index there."""
        if not isinstance(result, EigenmodeResult) or not self._state:
            return {}
        modes = {
            start: _track_mode(result, index, previous, rtol, bound)
            for start, (index, previous, bound) in self._state.items()
        }
        for (a, ma), (b, mb) in itertools.combinations(modes.items(), 2):
            if ma == mb:
                omega = complex(np.asarray(result.omega)[ma])
                raise DegenerateModeError(
                    f"Two followed modes reached one mode: the modes named {a} "
                    f"and {b} on the first mesh both followed mode {ma} at "
                    f"{hertz(omega.real / (2 * np.pi))} on a refined mesh. Start "
                    "from a finer mesh, on which the modes are apart, or ask "
                    "for more modes (gfem.Eigenmode(n_modes=...))."
                )
        self._modes = modes
        self._omegas = np.asarray(result.omega)
        return modes

    def bound(
        self,
        model: Model,
        result: Any,
        estimates: tuple[ErrorEstimate, ...],
        *,
        threads: int | None,
        cache: RecoveryCache,
    ) -> None:
        """Keep each track's omega and omega estimate on ``result`` for the next mesh.

        An ``omega=`` estimate of the track's mode is its bound; any other
        track computes the omega kernel of its mode on the mesh's cache.
        """
        if not self._state:
            return
        from gfem._engine import estimate as kernels

        for start, index in self._modes.items():
            if start in self._omega:
                bound = estimates[self._omega[start]].value
            else:
                bound = kernels.omega(model, result, index, threads, cache=cache).value
            self._state[start] = (index, complex(self._omegas[index]), bound)


def _warn_not_converged(
    stop: str,
    failure: Exception | None,
    above: list[tuple[Estimator, float, float]],
    *,
    joint: bool,
    max_cycles: int,
    max_tetrahedra: int | None,
) -> None:
    """Warn that `refine_mesh` stopped on ``stop`` with quantities above tolerance.

    ``above`` holds each such estimator, its estimate and its tolerance; with
    a tuple of estimators (``joint``) the warning names each. The warning
    points at the caller of `refine_mesh`, three frames up.
    """
    reason = {
        "max_cycles": f"after max_cycles={max_cycles} refinements",
        "max_tetrahedra": (
            f"because the next mesh would exceed max_tetrahedra={max_tetrahedra}"
        ),
        "solve failed": f"because the solve on the next mesh failed: {failure}",
    }[stop]
    if joint:
        what = "; ".join(
            f"{_describe(e)} at {value:.3g} above its tolerance={tolerance:g}"
            for e, value, tolerance in above
        )
        estimates = f"with the error estimate of {what}"
    else:
        ((_, value, tolerance),) = above
        estimates = f"with the error estimate {value:.3g} above tolerance={tolerance:g}"
    warnings.warn(
        f"Mesh refinement stopped {reason}, {estimates}. The record "
        "holds the last solved mesh. Raise the cap, loosen tolerance, or "
        "start from a finer mesh.",
        MeshRefinementNotConvergedWarning,
        stacklevel=4,
    )


def _free_consumed(options: SolverOptions) -> None:
    """Free the solve entries a backward has read, before the first solve.

    As a differentiated forward does: their base handles return to the plan
    pool, for the solve of the input mesh to take, within
    ``options.retain_memory``. `refine_mesh` runs on one process.
    """
    from gfem._engine.cache import ENTRIES, retain_budget

    ENTRIES.free_consumed(None, retain_budget(options.retain_memory))


def _dorfler(indicators: np.ndarray, error_fraction: float) -> np.ndarray:
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
    model: Model,
    result: Any,
    estimator: Estimator,
    *,
    threads: int | None,
    mode: int | None = None,
    adjoints: dict[tuple[float, Hashable], np.ndarray] | None = None,
    cache: RecoveryCache | None = None,
) -> ErrorEstimate:
    """The estimate of ``result``: dispatch on the estimator and result type.

    ``mode`` is the eigenmode a `QuantityEstimator` follows on this result.
    ``adjoints`` are an S entry's adjoints solved with ``result``
    (`gfem.solve._solve`);
    ``None`` solves them here if the entry needs them (`_adjoint_ports`).
    ``cache`` is the mesh's `gfem._engine.estimate.RecoveryCache`, shared by
    every estimate of ``result``'s mesh; ``None`` makes one for this call.
    """
    from gfem._engine import estimate as kernels

    cache = cache or kernels.RecoveryCache()
    kind = _kind(estimator)
    if isinstance(estimator, QuantityEstimator):
        if estimator.capacitance is not None:
            parts = [
                kernels.capacitance(
                    model, result, *estimator.capacitance, threads, cache=cache
                )
            ]
        elif estimator.s is not None:
            if adjoints is None:
                ports = _adjoint_ports(estimator, model, list(result.s.excited))
                adjoints = _adjoints(model, result, ports, threads) if ports else {}
            fields = dict(zip(result.kept, result.e_fields, strict=True))
            source = estimator.s[1]
            # a dual lives on the source field's space (an unexcited port has
            # no field of its own)
            duals = {
                (f, key): dataclasses.replace(fields[f, source], coefficients=z)
                for (f, key), z in adjoints.items()
            }
            parts = kernels.s_entry(
                model, result, *estimator.s, threads, adjoints=duals, cache=cache
            )
        elif estimator.q is not None:
            _check_lossy_q(result, int(mode or 0))
            parts = [kernels.q(model, result, int(mode or 0), threads, cache=cache)]
        else:
            parts = [kernels.omega(model, result, int(mode or 0), threads, cache=cache)]
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


def _kind(estimator: Estimator) -> str:
    """The `ESTIMATE_KINDS` entry an estimator's estimate is of."""
    if isinstance(estimator, FieldEstimator):
        return "field"
    for kind in ("capacitance", "s", "omega", "q"):
        if getattr(estimator, kind) is not None:
            return kind
    raise AssertionError(estimator)  # pragma: no cover - checked on construction


def _recovery_cache() -> RecoveryCache:
    """A new `gfem._engine.estimate.RecoveryCache`: one per solved mesh."""
    from gfem._engine import estimate as kernels

    return kernels.RecoveryCache()


def _estimator_mode(estimator: Estimator) -> int | None:
    """The mode index of a `QuantityEstimator` (``omega=``, ``q=``), else ``None``."""
    if not isinstance(estimator, QuantityEstimator):
        return None
    return estimator.omega if estimator.omega is not None else estimator.q


def _check_lossy_q(result: EigenmodeResult, mode: int) -> None:
    """Raise for the Q of a lossless mode: infinite, with no error to estimate."""
    if complex(np.asarray(result.omega)[mode]).imag == 0:
        raise ValueError(
            f"QuantityEstimator(q={mode}): mode {mode} is lossless, so its Q is "
            "infinite and has no error to estimate; estimate "
            f"QuantityEstimator(omega={mode})."
        )


def _check_q_floor(
    result: EigenmodeResult, mode: int, tolerance: float, options: SolverOptions
) -> None:
    """Raise for a Q tolerance below the eigen solve's error carried into Q.

    Q's relative error is cot(arg omega) times omega's relative error (ADR
    0131), so the solve's ``eigen_tolerance`` reaches Q multiplied by it.
    """
    _check_lossy_q(result, mode)
    omega = complex(np.asarray(result.omega)[mode])
    cot = abs(omega.real / omega.imag)
    floor = 10 * options.eigen_tolerance * cot
    if tolerance < floor:
        raise ValueError(
            f"refine_mesh tolerance={tolerance:g} for QuantityEstimator(q={mode}) "
            f"must be at least 10 x SolverOptions.eigen_tolerance="
            f"{options.eigen_tolerance:g} x cot(arg omega)={cot:.4g}, that is "
            f"{floor:.3g}: Q's relative error is cot(arg omega) times omega's, "
            "and the error estimate does not include the eigen solve's error. "
            "Lower SolverOptions.eigen_tolerance to at most "
            f"{tolerance / (10 * cot):.2g}, or raise tolerance."
        )


def _quantity(
    result: Any, estimator: Estimator, mode: int | None
) -> float | complex | tuple[complex, ...] | None:
    """The quantity a `QuantityEstimator` names, read off ``result``."""
    if not isinstance(estimator, QuantityEstimator):
        return None
    if estimator.capacitance is not None:
        i, j = (result.terminals.index(t) for t in estimator.capacitance)
        return float(np.asarray(result.capacitance)[i, j])
    if estimator.s is not None:
        return tuple(complex(v) for v in np.asarray(result.s[estimator.s]))
    if estimator.q is not None:
        return float(np.asarray(result.q)[mode])
    return complex(np.asarray(result.omega)[mode])


def _track_mode(
    result: EigenmodeResult,
    mode: int,
    previous: complex | None,
    rtol: float,
    previous_estimate: float | None,
) -> int:
    """The mode of ``result`` a quantity follows: nearest ``previous`` omega.

    On the first result (``previous`` is ``None``) it is ``mode`` itself.
    ``previous_estimate`` is the omega estimate on the previous mesh, relative
    to ``previous``.

    Raises:
        DegenerateModeError: Another mode lies within ``rtol`` of the
            followed one, or it belongs to a degenerate cluster; the nearest
            mode is further from ``previous`` than the previous estimate
            allows, or the second nearest is less than twice as far.
    """
    omegas = np.asarray(result.omega)
    if previous is not None:
        distances = np.abs(omegas - previous)
        mode = int(np.argmin(distances))
        _check_followed(omegas, mode, distances, previous, previous_estimate)
        reference = previous
    else:
        reference = omegas[mode]
    close = np.flatnonzero(np.abs(omegas - reference) <= rtol * abs(reference))
    cluster = np.asarray(result.cluster)
    if len(close) > 1 or np.count_nonzero(cluster == cluster[mode]) > 1:
        raise DegenerateModeError(
            f"QuantityEstimator(omega=...) follows one mode by its frequency, "
            f"and mode {mode} at {hertz(omegas[mode].real / (2 * np.pi))} is "
            "not alone: another mode lies within SolverOptions.cluster_rtol="
            f"{rtol:g} of it. A degenerate cluster has no one mode to follow: "
            "use gfem.FieldEstimator(), which estimates every mode, or a "
            "design that breaks the symmetry."
        )
    return mode


def _check_followed(
    omegas: np.ndarray,
    mode: int,
    distances: np.ndarray,
    previous: complex,
    previous_estimate: float | None,
) -> None:
    """Raise unless ``mode`` is clearly the mode followed from ``previous``.

    The followed mode's errors on both meshes are at most the previous one,
    and the estimate is at least `_ESTIMATE_FLOOR` of the error on every
    verified case, so the mode moves at most twice the previous estimate over
    that floor. Further away, the nearest mode is another one.
    """
    nearest = float(distances[mode])
    frequency = hertz(omegas[mode].real / (2 * np.pi))
    if previous_estimate is not None:
        allowed = 2 * previous_estimate * abs(previous) / _ESTIMATE_FLOOR
        if nearest > allowed:
            raise DegenerateModeError(
                "QuantityEstimator(omega=...) follows one mode by its frequency, "
                f"and the nearest mode on the refined mesh, mode {mode} at "
                f"{frequency}, moved {nearest / abs(previous):.3g} of the "
                "followed frequency, further than the previous error estimate "
                f"{previous_estimate:.3g} allows. The followed mode probably "
                "left the solved modes: ask for more modes "
                "(gfem.Eigenmode(n_modes=...)) or start from a finer mesh."
            )
    others = np.delete(distances, mode)
    if len(others) and float(others.min()) < 2 * nearest:
        raise DegenerateModeError(
            "QuantityEstimator(omega=...) follows one mode by its frequency, "
            f"and mode {mode} at {frequency} is not clearly nearest the "
            "followed frequency on the refined mesh: another mode is less than "
            "twice as far. Start from a finer mesh, on which the two modes "
            "are apart, or use gfem.FieldEstimator(), which estimates every "
            "mode."
        )


def _dofs(result: Any) -> int:
    """A solve's unknowns: one field's coefficient count."""
    return int(_fields(result)[0].coefficients.shape[0])


def _fields(result: Any) -> list:
    """The fields of a result an estimate reads."""
    if isinstance(result, ElectrostaticResult):
        return list(result.potential.values())
    return list(result.e_fields)


def _refuse_blends(model: Model, where: str) -> None:
    """Raise for a law blend: the estimators do not read one (ADR 0113)."""
    from gfem._engine import dispersion

    dispersion.refuse_blends(model, where)


def _check_estimator(estimator: Any) -> None:
    if not isinstance(estimator, (FieldEstimator, QuantityEstimator)):
        raise TypeError(
            "estimator must be gfem.FieldEstimator(), gfem.QuantityEstimator(...) "
            "or a tuple of gfem.QuantityEstimator(...)s, got "
            f"{type(estimator).__name__}."
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


def _check_quantity_keys(
    estimator: QuantityEstimator, problem: Any, model: Model
) -> None:
    """Raise unless the quantity names an entry ``problem`` solves for."""
    if estimator.capacitance is not None:
        _check_kind("capacitance", problem, Electrostatic)
        _check_terminals(estimator.capacitance, [name for name, _ in problem.terminals])
    elif estimator.s is not None:
        _check_kind("s", problem, Driven)
        ports = [p.name for p in model.ports]
        excited = ports if problem.excite is None else list(problem.excite)
        _check_ports(estimator.s, ports, excited)
    else:
        keyword = estimator._mode_keyword or "omega"  # noqa: SLF001
        _check_kind(keyword, problem, Eigenmode)
        _check_mode(_estimator_mode(estimator), problem.n_modes, keyword)


def _check_quantity_result(estimator: QuantityEstimator, result: Any) -> None:
    """Raise unless the quantity names an entry of ``result``."""
    if estimator.capacitance is not None:
        _check_kind("capacitance", result, ElectrostaticResult)
        _check_terminals(estimator.capacitance, list(result.terminals))
    elif estimator.s is not None:
        _check_kind("s", result, DrivenResult)
        _check_ports(estimator.s, list(result.s.ports), list(result.s.excited))
    else:
        keyword = estimator._mode_keyword or "omega"  # noqa: SLF001
        _check_kind(keyword, result, EigenmodeResult)
        _check_mode(_estimator_mode(estimator), len(np.asarray(result.omega)), keyword)


def _check_kind(keyword: str, given: Any, expected: type) -> None:
    if not isinstance(given, expected):
        raise TypeError(
            f"QuantityEstimator({keyword}=...) needs a gfem.{expected.__name__}, "
            f"got {type(given).__name__}: name a quantity of this problem "
            "(capacitance= for Electrostatic, s= for Driven, omega= or q= for "
            "Eigenmode)."
        )


def _check_terminals(names: tuple[str, str], terminals: list[str]) -> None:
    for name in names:
        if name not in terminals:
            raise KeyError(
                f"No terminal {name!r}; the terminals are "
                f"{', '.join(map(repr, terminals))}."
            )


def _check_ports(names: tuple[str, str], ports: list[str], excited: list[str]) -> None:
    """Raise unless both ports exist and the source port (the column) is excited.

    The read port ``to`` may be any port: when it is not excited, its
    S-functional adjoint is solved (`_adjoint_ports`).
    """
    for name in names:
        if name not in ports:
            raise KeyError(
                f"No port {name!r}; the ports are {', '.join(map(repr, ports))}."
            )
    source = names[1]
    if source not in excited:
        raise ValueError(
            f"QuantityEstimator(s=...) reads the field of its source port, and "
            f"port {source!r} is not excited: add it to Driven.excite."
        )


def _check_mode(mode: int | None, n_modes: int, keyword: str = "omega") -> None:
    if mode is not None and mode >= n_modes:
        raise ValueError(
            f"QuantityEstimator({keyword}={mode}) names a mode index at or above "
            f"n_modes={n_modes}: ask for more modes, or name one of 0 to "
            f"{n_modes - 1}."
        )


def _check_quantity_model(estimator: QuantityEstimator, model: Model) -> None:
    """Raise for a quantity entry the verification tier has no case for (ADR 0089)."""
    if estimator.omega is not None or estimator.s is not None:
        _check_lossy_anisotropic(estimator, model)
    if estimator.omega is not None:
        _check_lossy_mode(model)
    if estimator.q is not None:
        _check_lossy_mode(model, "q")
        region = _lossy_anisotropic_region(model)
        if region is not None:
            raise MeshRefinementUnsupportedError(
                "lossy eigenmode",
                "QuantityEstimator(q=...) has no verified estimate for a mode "
                f"with loss inside an anisotropic material (region {region!r}): "
                "use gfem.FieldEstimator().",
            )


def _adjoint_ports(
    estimator: Estimator, model: Model, excited: list[str]
) -> tuple[str, ...]:
    """The port whose S-functional adjoint an S estimate may need (ADR 0089).

    ``to`` of ``QuantityEstimator(s=(to, source))`` when it is not among
    ``excited`` (it has no field of its own), or when it is a wave port
    touching a lossy element: its mode may then not be parallel to its
    excitation (the solve checks), and its adjoint is not its own field.
    """
    if not isinstance(estimator, QuantityEstimator) or estimator.s is None:
        return ()
    to = estimator.s[0]
    if to not in excited:
        return (to,)
    port = next((p for p in model.ports if p.name == to), None)
    if not isinstance(port, WavePort):
        return ()
    return (to,) if _lossy_elements(model)[_touching(model, port.surface)].any() else ()


def _excited(problem: Driven, model: Model) -> list[str]:
    """The ports ``problem`` excites on ``model``."""
    if problem.excite is None:
        return [p.name for p in model.ports]
    return list(problem.excite)


def _adjoint_loads(
    estimators: tuple[Estimator, ...], model: Model, problem: Any
) -> AdjointLoads | None:
    """The per-frequency adjoint loads the estimators' duals solve, or ``None``.

    The union over the estimators (ADR 0130), each port once: two S entries
    reading one port share its adjoint. They ride on each of `refine_mesh`'s
    forward solves (ADR 0089 amendment); ``None`` for free duals only.
    """
    from gfem._engine.driven import port_loads

    if not isinstance(problem, Driven):
        return None
    excited = _excited(problem, model)
    ports = tuple(
        dict.fromkeys(
            port for each in estimators for port in _adjoint_ports(each, model, excited)
        )
    )
    return port_loads(ports) if ports else None


def _check_adjoint_workers(
    estimators: tuple[Estimator, ...],
    model: Model,
    problem: Any,
    options: SolverOptions,
) -> None:
    """Refuse frequency workers for an estimate that solves a dual.

    Its adjoint loads (`_adjoint_loads`) ride on the forward's
    factorization (ADR 0089), which a frequency worker does not hand back:
    frequency workers take wave ports (ADR 0099) but spread no adjoint
    solve.
    """
    if not isinstance(problem, Driven):
        return
    workers = min(options.frequency_workers, len(problem.frequency))
    excited = _excited(problem, model)
    solving = [e for e in estimators if _adjoint_ports(e, model, excited)]
    if workers > 1 and solving:
        raise MeshRefinementUnsupportedError(
            "frequency workers",
            f"{_describe(solving[0])} solves its "
            "adjoint dual with the forward, which frequency workers do not do "
            "in this version of gfem: it needs frequency_workers=1, got "
            f"{options.frequency_workers}.",
        )


def _adjoints(
    model: Model, result: DrivenResult, ports: tuple[str, ...], threads: int | None
) -> dict[tuple[float, Hashable], np.ndarray]:
    """The adjoints of ``ports`` on a finished result: one more factorization.

    Only at the frequencies where one is not its port's own field scaled.
    The result does not record its options: the direct solver, which the
    estimate's few digits need no iterative tolerance for.
    """
    from gfem._engine import driven

    problem = Driven(list(result.s.frequency), excite=list(result.s.excited))
    order = _fields(result)[0].order
    options = SolverOptions(order=order, linear_solver="direct", threads=threads)
    with driven.low_frequency_deferred(), warnings.catch_warnings():
        for category in driven.FORWARD_WARNINGS:  # the solve of ``result`` warned
            warnings.simplefilter("ignore", category)
        rule = driven.DrivenRule(model, problem, options)
        rule.adjoint_loads = driven.port_loads(ports)
        return rule.solve_adjoints(model)


def _lossy_elements(model: Model) -> np.ndarray:
    """Whether each tetrahedron is lossy: tan_delta, conductivity, complex eps, mu.

    A law region is lossy when a term of its law carries loss (ADR 0111).
    """
    from gfem._engine import dispersion
    from gfem._engine.coefficients import element_tensors, lossy_eps

    laws = dispersion.law_regions(model)
    out = np.zeros(len(model.mesh.tetrahedra), bool)
    for name in ("tan_delta", "conductivity", "eps_r", "mu_r"):
        if name == "eps_r" and laws:
            # eps_r (1 - i tan_delta) is complex exactly where eps_r is or
            # tan_delta (counted above) is nonzero; a law region holds a real
            # placeholder (no traced value read), its loss is the law's
            values = np.asarray(lossy_eps(model, law=1.0))
        else:
            values = np.asarray(element_tensors(model, name))
        lossy = values.imag if name in ("eps_r", "mu_r") else values
        lossy = np.abs(lossy) > 0
        out |= lossy.reshape(len(out), -1).any(axis=1)
    for lr in laws:
        if dispersion.has_loss(lr.law):
            out |= lr.mask
    return out


def _check_lossy_mode(model: Model, keyword: str = "omega") -> None:
    """Raise for a mode whose loss no verification case covers (ADR 0089).

    Covered: tan_delta, conductivity and complex eps_r of an isotropic
    material, impedance, first-order absorbing and conductor surfaces, and
    resistive lumped ports. A permittivity law is refused until a case
    covers it (ADR 0111). (A wave port on an eigenmode is refused by the
    eigen solver; loss inside an anisotropic material by
    `_check_lossy_anisotropic`.)
    """
    from gfem._engine import dispersion
    from gfem._engine.coefficients import element_tensors

    laws = dispersion.law_regions(model)
    if laws:
        raise MeshRefinementUnsupportedError(
            "dispersive eigenmode",
            f"QuantityEstimator({keyword}=...) has no verified estimate for a mode "
            f"of a model with a permittivity law (region {laws[0].region!r}): "
            "use gfem.FieldEstimator().",
        )
    what = None
    if any(isinstance(b, Absorbing) and b.order == 2 for b in model.boundaries):
        what = "a second-order absorbing boundary"
    elif np.any(np.asarray(element_tensors(model, "mu_r")).imag):
        what = "a complex mu_r"
    if what is not None:
        raise MeshRefinementUnsupportedError(
            "lossy eigenmode",
            f"QuantityEstimator({keyword}=...) has no verified estimate for a mode "
            f"with {what}: use gfem.FieldEstimator().",
        )


def _lossy_anisotropic_region(model: Model) -> str | None:
    """The first anisotropic material region holding loss, if any."""
    from gfem._engine.coefficients import anisotropic_regions

    lossy = _lossy_elements(model)
    for region in anisotropic_regions(model, tuple(model.materials)):
        if lossy[model.mesh.elements(region)].any():
            return region
    return None


def _check_field_model(estimator: Estimator, model: Model) -> None:
    """Raise for a wave field whose loss no field case covers (ADR 0087)."""
    if isinstance(estimator, FieldEstimator):
        _check_lossy_anisotropic(estimator, model)


def _check_lossy_anisotropic(estimator: Estimator, model: Model) -> None:
    """Raise for loss inside an anisotropic material (ADRs 0087, 0089).

    One refusal for the field, S and omega estimates: the complex tensor
    eps_eff, which weights the recovered flux error of each, has no
    verification case.
    """
    region = _lossy_anisotropic_region(model)
    if region is None:
        return
    if isinstance(estimator, FieldEstimator):
        name = "FieldEstimator()"
    else:
        name = (
            "QuantityEstimator(s=...)"
            if estimator.s
            else "QuantityEstimator(omega=...)"
        )
    raise MeshRefinementUnsupportedError(
        "lossy anisotropic material",
        f"{name} has no verified estimate with loss in the anisotropic material "
        f"of {region!r}, and no estimator has one. Refine a copy of the model "
        "with that material lossless, then solve the lossy model on the "
        "refined mesh.",
    )


def _touching(model: Model, surface: str) -> np.ndarray:
    """Whether each tetrahedron has a node on ``surface``."""
    mesh = model.mesh
    nodes = np.zeros(len(mesh.nodes), bool)
    nodes[np.asarray(mesh.triangles)[mesh.elements(surface)].ravel()] = True
    return nodes[np.asarray(mesh.tetrahedra)].any(axis=1)


def _check_estimable(result: Any, estimator: Estimator, model: Model) -> None:
    if isinstance(estimator, QuantityEstimator):
        _check_quantity_result(estimator, result)
        _check_quantity_model(estimator, model)
    if not isinstance(result, (ElectrostaticResult, DrivenResult, EigenmodeResult)):
        raise _unsupported_result(result, estimator)
    if any(field._comm is not None for field in _fields(result)):  # noqa: SLF001
        _mpi.refuse_feature("error_estimate")
    if not isinstance(result, ElectrostaticResult):
        _check_field_model(estimator, model)
    if model._periodic:  # noqa: SLF001
        raise MeshRefinementUnsupportedError(
            "periodic",
            f"gfem.{type(estimator).__name__} has no path for a periodic model: "
            "its flux recovery does not pair the periodic surfaces.",
        )
    if isinstance(result, DrivenResult):
        _check_kept(result, estimator)
        from gfem._engine import dispersion

        dispersion.check_estimator_frequencies(
            model, [float(f) for f in result.s.frequency]
        )
    _check_order(_fields(result)[0].order, estimator)
    mesh = _fields(result)[0].mesh
    if mesh._topology is not model.mesh._topology or not np.array_equal(  # noqa: SLF001
        np.asarray(mesh.nodes), np.asarray(model.mesh.nodes)
    ):
        raise ValueError(
            "error_estimate needs the model the result was solved on: the "
            "result's fields live on another mesh, or on other node positions."
        )


def _check_order(order: int, estimator: Estimator) -> None:
    """Raise at order 1, where no estimate is verified."""
    if order == 1:
        raise MeshRefinementUnsupportedError(
            "order 1",
            f"gfem.{type(estimator).__name__} has no verified estimate at order 1: "
            "there, the frequency of a thin layer meshed with one cell across "
            "its thickness moved away from the exact one as the mesh was "
            "refined, and the estimate missed it. Solve at order 2 or 3 "
            "(gfem.SolverOptions(order=2)).",
        )


def _check_kept(result: DrivenResult, estimator: Estimator) -> None:
    """Raise unless a driven result is a uniform sweep with every field kept."""
    if result.sweep is not None and result.sweep.kind == "adaptive":
        raise MeshRefinementUnsupportedError(
            "adaptive sweep",
            f"gfem.{type(estimator).__name__} has no path for an adaptive sweep: "
            "estimate a uniform sweep at a few frequencies of the band.",
        )
    kept = {f for f, _ in result.kept}
    missing = [f for f in result.s.frequency if f not in kept]
    if missing:
        raise ValueError(
            "error_estimate reads the electric fields of every frequency, and "
            f"this result kept none at {', '.join(f'{f:g}' for f in missing)} Hz: "
            "solve with gfem.Driven(..., keep_fields=True)."
        )


def _unsupported_result(result: Any, estimator: Estimator) -> Exception:
    kind = type(result).__name__
    if not kind.endswith("Result"):
        return TypeError(f"error_estimate needs a result of gfem.solve, got {kind}.")
    return MeshRefinementUnsupportedError(
        kind.removesuffix("Result").lower(),
        f"gfem.{type(estimator).__name__} has no path for a gfem.{kind}: it "
        "estimates electrostatic, driven and eigenmode results.",
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
