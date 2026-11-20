"""Mesh refinement: error estimates of a solved result, and the refinement loop.

`refine_mesh` solves, estimates the error, marks the tetrahedra holding most
of it (Dörfler marking), bisects them and solves again, until the error
estimate meets a tolerance or a cap stops it. `error_estimate`
gives one estimate of a result without refining (a measurement).
Estimators are value objects named by what they estimate: the
`FieldEstimator` estimates the field's error in the energy norm by flux
recovery; the `QuantityEstimator` names one quantity of interest.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import types
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem import _mpi
from gfem._checks import is_staging, is_traced
from gfem._dim2 import refuse_entry
from gfem._domain import physics_of
from gfem._floquet import floquet_ports
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
from gfem.mesh import Mesh, _is_model_mesh
from gfem.model import (
    PEC,
    PMC,
    Absorbing,
    Impedance,
    LumpedPort,
    Model,
    WavePort,
    surfaces_of,
)
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic, PortMode
from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult
from gfem.shape import _check_quality
from gfem.solve import _solve

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from gfem._engine.estimate import RecoveryCache
    from gfem._engine.gauge import GaugeChoice

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
    r"""Estimates the error of a result's fields in the energy norm.

    The estimate is relative to the field's energy norm: 1e-2 means the
    field is about 1 % off its value on the same geometry with an infinitely
    fine mesh. Each tetrahedron's indicator is the distance between the
    solved electric flux and a smoother flux recovered from it (flux
    recovery), so no extra solve is needed. It is an estimate, not a bound;
    it does not include the faceting of curved surfaces or the error left by
    the linear solve.

    With loss, the flux is the total current
    :math:`\varepsilon_\mathrm{eff} E`, :math:`\varepsilon_\mathrm{eff} =
    \varepsilon_r (1 - i \tan\delta) - i \sigma / (\omega \varepsilon_0)`,
    and the norm is :math:`\|\nabla \times E\|^2_{\mu^{-1}} + |k|^2 \int
    |\varepsilon_\mathrm{eff}| |E|^2`: the conduction current counts, as it
    does in the field. Weighting the flux error by
    :math:`\mathrm{Re}(\varepsilon_\mathrm{eff})^{-1}` instead overestimates;
    recovering :math:`\varepsilon_r E`, as
    without loss, underestimates without bound under conduction loss.

    A dispersive material (a `gfem.PermittivityLaw`) enters at its permittivity
    at each solved frequency, lossless or not,
    :math:`\mathrm{Re}\,\varepsilon < 0` included. Where it is exactly zero (a
    lossless Drude term at its plasma frequency, without conductivity) the flux
    has no weight, and `refine_mesh` and `error_estimate` raise `ValueError`.

    `refine_mesh` and `error_estimate` raise
    `gfem.MeshRefinementUnsupportedError` at order 1.
    """


@dataclasses.dataclass(frozen=True)
class QuantityEstimator:
    r"""Estimates the error of one quantity of a solved result.

    Estimates how far one quantity of a solved result (a capacitance entry,
    an S entry whose source port is excited, one mode's ``omega`` or its
    Q, the mean ``omega`` of a degenerate cluster, or an objective of the
    result) is from its value on the same geometry with an infinitely fine
    mesh. `ErrorEstimate.kind` names which, and each kind has its own
    measured effectivity range (verification page).
    The estimate uses the fields already solved, with no extra solve, except
    for the adjoint duals below. It does not include
    the faceting of curved surfaces or the error left by the linear solve.

    The estimate is a magnitude in the units of ``tolerance``: relative to
    the capacitance entry, to ``omega`` or to Q, absolute for an S entry. It has
    no sign or phase, and gfem never adds it to the quantity. It is an
    estimate, not a bound. A diagonal capacitance entry never lies below
    its value on the meshed geometry: :math:`C_h \ge C`; an off-diagonal entry
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
            same one. Or a tuple of indices naming whole degenerate clusters
            (``result.cluster``), stored sorted, whose mean omega is the
            quantity, estimated relative to its modulus: on each refined
            mesh the estimate follows as many modes, those nearest the
            previous mean, and raises `gfem.DegenerateModeError` when the
            mean moved further than the previous estimate allows, or another
            mode is less than twice as far from it as the farthest of them.
            Ask for at least one mode more than the tuple names.
        q: An index into ``EigenmodeResult.q``, followed as ``omega``; the
            estimate is relative to Q. Q's relative error is :math:`\cot(\arg\omega)`
            times the imaginary part of omega's relative error, estimated on
            the same terms, with no extra solve. A lossless mode (infinite Q)
            raises `ValueError`.
        objective: ``objective(result, model)``, a real 0-d JAX value: the
            same function you differentiate. gfem traces it to find what it
            reads (``ErrorEstimate.kind``) and raises
            `gfem.MeshRefinementUnsupportedError` for anything it does not
            estimate (``feature`` names it): a field read other than through
            `gfem.power_flow` or `gfem.region_mean`, a point value, a port
            mode output, a mixture of kinds, a branch on a value it reads
            (max, min, sort, where) or a custom derivative of its own. In
            this version an objective of S is estimated (S and its networks:
            ``db``, ``y()``, ``z()``, ``renormalize``, ``with_offsets``),
            and of an electrostatic result: of its capacitance (``mutual``
            and ``inverse`` too), of its region energies, or of potential
            means (`gfem.region_mean` of ``result.potential``), and of a
            driven result's fields: the power absorbed in a side without
            ports and with only volume losses of isotropic materials without
            a permittivity law (``-gfem.power_flow``), or the means of
            `gfem.region_mean` over named regions; and an objective of an
            eigenmode result's ``omega`` (``frequency``, ``q``) and
            ``bulk_participation``, each mode it reads followed as a mode of
            its own; ``q_port``, ``participation``, ``sheet_participation``,
            ``principal_participation`` and the mode fields raise. A
            participation needs its mode-field dual: an isolated mode, the
            direct solver on the CPU and ``eigen_tolerance`` at most 1e-10
            (in `refine_mesh` it rides on the eigen solves from the second
            mesh on; `error_estimate` factorizes once more per mode). The estimate
            is absolute, in the objective's units, and `refine_mesh` needs
            its ``tolerance`` given. A read row whose port is not excited,
            or is a lossy wave port, uses that port's adjoint dual, as for
            ``s``, and a read energy, potential mean, absorbed power or
            field mean its own dual; in
            `refine_mesh` each is solved once more on the first mesh and
            rides on the solves after it.

    Raises:
        TypeError: Not exactly one keyword is given, or a value has the
            wrong type (a tuple for ``q``: the Q of a cluster is not
            estimated; an ``objective`` that is not callable).
        ValueError: ``omega`` or ``q`` is negative, or a tuple ``omega``
            names fewer than two modes or a mode twice.

    `refine_mesh` and `error_estimate` raise
    `gfem.MeshRefinementUnsupportedError` for entries with no verified
    estimate. An S entry and every objective of a driven result raise it
    before any solve on a model with an `gfem.Impedance` or `gfem.Absorbing`
    surface (``feature`` ``"impedance or absorbing surface"``), with a
    lumped port whose face borders materials of different ``mu_r``
    (``"lumped port spanning a permeability contrast"``), or with a
    permittivity law at :math:`-20 < \mathrm{Re}\,\varepsilon < 0` at a solve
    frequency (``"plasmonic permittivity"``); refine those with
    `FieldEstimator`. The ``omega`` of a mode bound to a permittivity law
    region where :math:`\mathrm{Re}\,\varepsilon < 0` at the mode raises it
    when it is estimated. Both estimators also raise it at order 1. For a lossy
    mode the ``omega`` estimate covers the complex ``omega``; ``q`` estimates
    Q, and raises on a
    mode of a model with a permittivity law (``feature`` ``"dispersive
    eigenmode"``), a second-order absorbing boundary, a complex ``mu_r`` or
    loss inside an anisotropic material (``"lossy eigenmode"``). An S entry
    ``(to, from)`` whose port ``to`` is not excited, or is a wave port with a
    lossy mode not parallel to its excitation, needs that port's adjoint
    dual: `refine_mesh` adds it to each of its solves, one more right-hand
    side with its options; `error_estimate` on a finished result factorizes
    once more per frequency, with the direct solver at the result's order,
    so estimate a large model through `refine_mesh` (``max_cycles=0``).
    """

    capacitance: tuple[str, str] | None = None
    s: tuple[str, str] | None = None
    omega: int | tuple[int, ...] | None = None
    q: int | None = None
    objective: Callable[[Any, Model], Any] | None = None

    def __post_init__(self) -> None:
        """Check that exactly one quantity is named, with valid keys."""
        given = [
            name
            for name in ("capacitance", "s", "omega", "q", "objective")
            if getattr(self, name) is not None
        ]
        if len(given) != 1:
            raise TypeError(
                "QuantityEstimator takes exactly one of capacitance=(t1, t2), "
                "s=(to, from), omega=index, q=index or objective=J; got "
                f"{given or 'none'}."
            )
        if self.objective is not None and not callable(self.objective):
            raise TypeError(
                "QuantityEstimator(objective=...) takes a function "
                f"objective(result, model), got {self.objective!r}."
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
            if isinstance(value, (tuple, list)):
                object.__setattr__(self, name, _cluster_members(value))
                continue
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


def _cluster_members(value: tuple | list) -> tuple[int, ...]:
    """The sorted modes of ``omega=(i, j, ...)``: two or more, each once."""
    for each in value:
        if isinstance(each, bool) or not isinstance(each, (int, np.integer)):
            raise TypeError(
                "QuantityEstimator(omega=(...)) takes mode indices, got "
                f"{each!r} in {value!r}."
            )
        if each < 0:
            raise ValueError(
                f"QuantityEstimator(omega=(...)) takes indices >= 0, got {each} "
                f"in {value!r}."
            )
    members = tuple(sorted(int(each) for each in value))
    if len(set(members)) != len(members):
        raise ValueError(
            f"QuantityEstimator(omega={tuple(value)!r}) names a mode twice: name "
            "each mode of the cluster once."
        )
    if len(members) < 2:
        raise ValueError(
            f"QuantityEstimator(omega={tuple(value)!r}) names one mode: a cluster "
            "mean needs at least two modes; for one mode, give omega=i."
        )
    return members


Estimator = FieldEstimator | QuantityEstimator

#: What an `ErrorEstimate` can be of (its ``kind``): a closed set.
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

#: The kinds the estimators emit, in `ESTIMATE_KINDS`'s order. A kind joined
#: when its estimate landed; every kind has now, and the docs build requires
#: the two to be equal, each kind with its verification cases.
EMITTED_KINDS = ESTIMATE_KINDS


@dataclasses.dataclass(frozen=True, eq=False)
class ErrorEstimate:
    """An error estimator's estimate for one solved result.

    Attributes:
        estimator: The estimator that made it.
        kind: What was estimated: ``"field"`` for a `FieldEstimator`;
            ``"capacitance"``, ``"s"``, ``"omega"`` or ``"q"`` for a
            `QuantityEstimator`, after its keyword, ``"cluster mean"``
            for a tuple ``omega``; for an objective, what it reads
            (``"objective of s"``), or ``"objective of s, lossless
            rewrite"`` on a lossless model with lumped ports only, whose
            estimate also uses the power each port's column must carry
            (tighter). Each kind has its own measured
            effectivity range (verification page).
        values: Estimated size of the error per frequency (a driven result),
            per mode (an eigenmode result under a `FieldEstimator`), or one
            entry, in the units of ``tolerance``. For a `FieldEstimator`
            this is relative to the field's energy norm. For a
            `QuantityEstimator` this is the quantity's error: relative to
            the capacitance entry, to ``omega``, to a cluster's mean
            ``omega`` or to Q, absolute for an S entry.
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
    #: An objective's value on the result it estimated (`RefinementCycle.value`).
    _quantity: float | None = dataclasses.field(default=None, repr=False)
    #: The ports whose S-functional adjoints an objective's dual reads.
    _dual_ports: tuple[str, ...] = dataclasses.field(default=(), repr=False)
    #: An objective's first-order sensitivity, sum_a |ct_a| s_a (its floor).
    _sensitivity: float = dataclasses.field(default=0.0, repr=False)
    #: The modes an objective reads, by index on its result: each is followed.
    _modes: tuple[int, ...] = dataclasses.field(default=(), repr=False)
    #: The (mode, region) participations whose mode-field duals it reads.
    _dual_modes: tuple[tuple[int, str], ...] = dataclasses.field(default=(), repr=False)
    #: A lossless rewrite's computed defect term lam . D, signed:
    #: ``value - abs(_defect)`` is the rewritten objective's own estimate.
    _defect: float = dataclasses.field(default=0.0, repr=False)
    #: The electrostatic dual loads an objective's estimate reads, by key
    #: (`gfem._engine.electrostatic.objective_loads`).
    _dual_keys: tuple[tuple[Any, ...], ...] = dataclasses.field(default=(), repr=False)

    @property
    def value(self) -> float:
        """The largest of `values`: what `refine_mesh` compares to ``tolerance``.

        NaN when any of `values` is NaN.
        """
        values = np.asarray(self.values, float)
        return float(np.nan) if np.isnan(values).any() else float(values.max())


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
            frequency, the followed mode's complex ``omega`` in rad/s, its
            Q, the followed cluster's mean complex ``omega``, or an
            objective's value; ``None`` for a `FieldEstimator`.
        marked: How many tetrahedra Dörfler marking picked on this mesh; 0
            when the loop stopped on it before marking.
        modes: For an eigenmode quantity, each mode index the estimators
            name on the first mesh, mapped to the index of the mode followed
            to this mesh (read-only); empty otherwise. A cluster's members,
            in order, map to the modes followed as the cluster, in order.
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
            ``"max_tetrahedra"``, ``"solve failed"`` or ``"non-finite
            estimate"`` (an estimate is NaN or infinite: it counts as above
            its tolerance, and the loop warns).
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
    r"""Estimate the error of a solved result, without refining.

    No solve is needed, except for an S entry, or an objective reading a
    row of S, whose port ``to`` is not excited, or is a wave port with a
    lossy mode not parallel to its excitation: its adjoint dual factorizes
    the system once more per frequency, with the direct solver at the
    result's order (the result does not record its solve options). An
    objective reading an electrostatic energy or a potential mean
    factorizes once more the same way, for all its duals at once, and one
    reading a driven field through `gfem.power_flow` or `gfem.region_mean`
    once more per frequency it reads. On a
    large model, estimate through `refine_mesh` with ``max_cycles=0``,
    whose adjoint rides on its own solve with your options (from its second
    mesh on, for an objective).

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
            that was not excited, a mode the result does not hold, the Q of a
            lossless mode, or a cluster with no other mode in the result; a
            permittivity law is zero at a driven frequency.
        MeshRefinementUnsupportedError: The estimator or the result has no
            estimate in this version of gfem (``feature`` names what): an
            adaptive sweep, a periodic model, another result type, a result
            solved at order 1, a quantity entry with no verified estimate (an S
            entry read at a lumped port with no resistance, whose row is
            undefined; an S entry or a driven objective beside an impedance or
            absorbing surface, a lumped port across a permeability contrast or
            a permittivity law at :math:`-20 < \mathrm{Re}\,\varepsilon < 0`; a
            plasmon-bound mode; see `QuantityEstimator`), or an objective
            reading what it may not.
        DegenerateModeError: A `QuantityEstimator` names a mode of a
            degenerate cluster, or one within the default
            `SolverOptions.cluster_rtol` of another mode (the message names
            the cluster); a cluster ``omega=(i, ...)`` holds part of a
            degenerate cluster, or another mode lies less than twice as far
            from its mean as its farthest member.
        NotImplementedError: ``model`` holds a metal-dielectric blend
            (`MaterialField.interpolate` with a permittivity law), which is
            supported on the driven uniform sweep only.
        DistributedUnsupportedError: The result was solved across ranks.
        NonDifferentiableError: A model or result leaf is a JAX tracer, or
            the call is inside `jax.jit`.
    """
    if not isinstance(model, Model):
        raise TypeError(f"model must be a gfem.Model, got {type(model).__name__}.")
    refuse_entry(model.mesh, "error_estimate")
    if isinstance(estimator, (tuple, list)):
        raise TypeError(
            "error_estimate takes one estimator, got a "
            f"{type(estimator).__name__}: call it once per quantity."
        )
    _check_estimator(estimator)
    _refuse_blends(model, "error_estimate")
    _check_untraced("error_estimate", (model, result))
    _check_estimable(result, estimator, model)
    mode: int | tuple[int, ...] | None = _estimator_mode(estimator)
    if mode is not None:
        # `refine_mesh`'s rule, at the default tolerance: the result does not
        # record the options it was solved with
        _track_mode(result, mode, None, SolverOptions().cluster_rtol, None)
    elif (start := _track_start(estimator)) is not None:
        mode = _track_cluster(result, start, None, None)
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
    r"""Refine the mesh where the error is, until the error estimate meets a tolerance.

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
            `QuantityEstimator` objects; the record then holds tuples, one
            entry per estimator in this order.
        tolerance: The error estimate to reach, one float for every
            estimator or a tuple with one per estimator; ``None`` is 1e-2,
            except for an objective, whose tolerance is in its own units and
            must be given (at least 10 x the solve tolerance times the
            objective's first-order sensitivity, checked on the first
            estimate).
            Relative to the field's energy
            norm for a `FieldEstimator`; for a `QuantityEstimator` relative
            to the capacitance entry, to ``omega``, to a cluster's mean
            ``omega`` or to Q, absolute for an S entry. At least 10 times
            ``options.tolerance`` (``options.eigen_tolerance`` for an
            `Eigenmode`; for Q times :math:`\cot(\arg\omega)`, about 2Q, checked on
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
            the estimators one to one, or a cluster shares a mode with
            another estimator; a tolerance is below 10
            times the solve tolerance (for Q, times :math:`\cot(\arg\omega)`); a
            `QuantityEstimator` names a source port that is not excited, a
            mode index at or above ``n_modes``, a cluster of at least
            ``n_modes`` modes (one more is needed to tell it apart), or the Q
            of a lossless mode; a permittivity law is zero at a driven
            frequency.
        MeshRefinementUnsupportedError: The problem or the estimator has no
            mesh refinement (``feature`` names what): an adaptive sweep, a port
            mode problem, a periodic model or a Bloch wave vector,
            ``options.order=1`` (the ``"fast"`` preset), a quantity entry with
            no verified estimate (an S entry read at a lumped port with no
            resistance, whose row is undefined; an S entry or a driven
            objective beside an impedance or absorbing surface, a lumped port
            across a permeability contrast or a permittivity law at
            :math:`-20 < \mathrm{Re}\,\varepsilon < 0`, all before any solve; a
            plasmon-bound mode on the first estimate, after the first solve;
            see `QuantityEstimator`), an objective reading what it may not (on
            the first estimate), or ``frequency_workers`` above 1 for an
            estimate that solves an adjoint dual (an unexcited ``to``, a
            lossy wave port, or an objective's dual), which rides on the
            forward solve.
        DegenerateModeError: The mode a `QuantityEstimator` follows is not
            alone: another mode lies within ``options.cluster_rtol`` of it,
            or shares its ``result.cluster`` label (the message names the
            cluster, to estimate its mean with ``omega=(i, j, ...)``).
            Or, on a refined mesh, the nearest mode is not clearly the
            followed one: a second mode is less than twice as far from the
            previous frequency, or the nearest moved further than the
            previous estimate allows (the mode probably left the solved
            modes: ask for more modes or start from a finer mesh). A
            cluster ``omega=(i, ...)`` raises it when it holds part of a
            labelled cluster, when another mode lies less than twice as far
            from its mean (on the first mesh) or from its previous mean as
            its farthest member, or when its mean moved further than the
            previous estimate allows. Or two modes named on the first mesh
            reached one mode.
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
    refuse_entry(model.mesh, "refine_mesh")
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
        if isinstance(each, QuantityEstimator) and each.objective is None:
            _check_quantity_keys(each, problem, model)
            _check_quantity_model(each, model)
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
    for each, limit in zip(estimators, tolerances, strict=True):
        _check_parameters(
            problem,
            options,
            limit,
            error_fraction,
            max_cycles,
            floor=not _is_objective(each),
        )
    if max_tetrahedra is not None:
        _check_int(max_tetrahedra, "max_tetrahedra", 1)
    _check_untraced("refine_mesh", (model, problem))
    if isinstance(problem, Driven):
        frequencies = [float(f) for f in problem.frequency]
        _check_driven_estimable(estimators, model, frequencies)
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
    """The estimators of `refine_mesh` and one tolerance each.

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
            _check_disjoint_tracks(first, second)
        estimators: tuple[Estimator, ...] = estimator
    else:
        _check_estimator(estimator)
        estimators = (estimator,)
    if tolerance is None:
        if any(_is_objective(e) for e in estimators):
            raise TypeError(
                "An objective's estimate is in the objective's own units, so it "
                "has no default tolerance: pass refine_mesh(tolerance=...) in "
                "those units."
            )
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


def _check_disjoint_tracks(first: Estimator, second: Estimator) -> None:
    """Raise when a cluster shares a mode with another estimator's mode or cluster.

    A cluster is followed as a whole, and one mode of it has no
    track of its own; two estimators of one mode share its track.
    """
    a, b = _track_start(first), _track_start(second)
    if a is None or b is None or a == b or not set(a) & set(b):
        return
    raise ValueError(
        f"{_describe(first)} and {_describe(second)} share mode "
        f"{min(set(a) & set(b))}: a cluster is followed as a whole, so a mode "
        "of it can be named in one cluster only, and not alone. Estimate the "
        "cluster, or its modes one by one where they are apart."
    )


def _describe(estimator: Estimator) -> str:
    """``QuantityEstimator(s=('p2', 'p1'))``: an estimator by its one keyword."""
    if isinstance(estimator, FieldEstimator):
        return "FieldEstimator()"
    if estimator.objective is not None:
        name = getattr(
            estimator.objective, "__name__", type(estimator.objective).__name__
        )
        return f"QuantityEstimator(objective={name})"
    keyword = next(
        name
        for name in ("capacitance", "s", "omega", "q")
        if getattr(estimator, name) is not None
    )
    return f"QuantityEstimator({keyword}={getattr(estimator, keyword)!r})"


def _deferred_solve(
    model: Model,
    problem: Any,
    *,
    options: SolverOptions,
    comm: Any,
    adjoint_loads: Any,
) -> tuple[Any, dict[Any, Any] | None, GaugeChoice | None]:
    """`_solve` with the driven low-frequency check deferred.

    Also returns the gauge choice of the solve's driven rule (``None`` for
    another problem): each pass resolves it afresh, and the deferred check
    on the returned mesh reads the last pass's.
    """
    from gfem._engine import driven

    with driven.low_frequency_deferred() as deferred:
        result, adjoints = _solve(
            model, problem, options=options, comm=comm, adjoint_loads=adjoint_loads
        )
    return result, adjoints, deferred.gauge


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
    (`_marking`), and the loop stops when none is. ``joint``
    (a tuple was given) makes the records tuples. Each solved mesh gets one
    `RecoveryCache`, shared by its estimates and dropped with the mesh.
    """
    from gfem._engine import driven

    if isinstance(problem, Driven):
        problem = dataclasses.replace(problem, keep_fields=True)
    tet_parents = np.arange(len(model.mesh.tetrahedra))
    tri_parents = np.arange(len(model.mesh.triangles))
    duals = _Duals(_adjoint_port_union(estimators, model, problem), problem)
    tracks = _Tracks(estimators)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MeshQualityWarning)  # `refine_mesh` warned
        result, adjoints, gauge = _deferred_solve(
            model,
            problem,
            options=options,
            comm=comm,
            adjoint_loads=duals.loads(tracks),
        )
    current = model
    cycles: list[RefinementCycle] = []
    failure: Exception | None = None
    while True:
        cache = _recovery_cache()  # this mesh's, dropped before the next solve
        estimates, modes = _estimates(
            current,
            result,
            estimators,
            tracks,
            options,
            adjoints=adjoints,
            ridden=duals.ridden,
            cache=cache,
            first=not cycles,
        )
        _check_first_estimates(
            problem,
            result,
            (estimators, estimates, tolerances),
            modes,
            options,
            first=not cycles,
        )
        duals.add(estimates)
        above = [
            i
            for i, (estimate, tolerance) in enumerate(
                zip(estimates, tolerances, strict=True)
            )
            if not estimate.value <= tolerance  # a NaN is above
        ]
        cycle = RefinementCycle(
            tetrahedra=len(current.mesh.tetrahedra),
            dofs=_dofs(result),
            estimate=_shaped([e.value for e in estimates], joint),
            value=_shaped(
                [
                    _quantity(result, e, _followed(modes, e), estimate)
                    for e, estimate in zip(estimators, estimates, strict=True)
                ],
                joint,
            ),
            marked=0,
            modes=types.MappingProxyType(modes),
        )
        cycles.append(cycle)
        stop = _settled(estimates, above)
        if stop is not None:
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
            refined_result, refined_adjoints, refined_gauge = _deferred_solve(
                refined,
                problem,
                options=options,
                comm=comm,
                adjoint_loads=duals.loads(tracks),
            )
        except (LinearSolverConvergenceError, EigenmodeConvergenceError) as err:
            stop, failure = "solve failed", err
            break
        current, result, adjoints, gauge = (
            refined,
            refined_result,
            refined_adjoints,
            refined_gauge,
        )
        tet_parents = tet_parents[refinement.tetrahedron_parents]
        tri_parents = tri_parents[refinement.triangle_parents]
    if isinstance(problem, Driven):
        driven.check_low_frequency(current, problem.frequency, gauge, stacklevel=3)
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


def _settled(estimates: tuple[ErrorEstimate, ...], above: list[int]) -> str | None:
    """Why the loop stops before marking: none above, or one not finite."""
    if not above:
        return "tolerance"
    if not all(np.isfinite(estimates[i].value) for i in above):
        return "non-finite estimate"
    return None


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


def _followed(
    modes: dict[int, int], estimator: Estimator
) -> int | tuple[int, ...] | None:
    """The index on this mesh of the mode ``estimator`` names, else ``None``.

    For a cluster, the indices of the modes followed as the cluster.
    """
    start = _track_start(estimator)
    if start is None:
        return None
    if len(start) > 1:
        return tuple(modes[each] for each in start)
    return modes[start[0]]


def _estimates(
    model: Model,
    result: Any,
    estimators: tuple[Estimator, ...],
    tracks: _Tracks,
    options: SolverOptions,
    *,
    adjoints: Any,
    ridden: tuple[Any, ...],
    cache: RecoveryCache,
    first: bool,
) -> tuple[tuple[ErrorEstimate, ...], dict[int, int]]:
    """Every estimator's estimate of one mesh's ``result``, and the modes followed.

    The modes are ``tracks``' (`_Tracks.follow`), plus, on the ``first``
    mesh, those its objectives read (`_Tracks.add`). An objective sees an
    eigenmode result's followed modes in the first mesh's order
    (`_aligned`).
    """
    modes = tracks.follow(result, options.cluster_rtol)
    view = _aligned(result, modes)
    estimates = tuple(
        _estimate(
            model,
            view if _is_objective(each) else result,
            each,
            threads=options.threads,
            mode=_followed(modes, each),
            adjoints=adjoints,
            ridden=ridden,
            cache=cache,
            rtol=options.cluster_rtol,
            reach=_mode_field_reach(options),
        )
        for each in estimators
    )
    if first:
        modes = tracks.add(estimates, modes, result)
    return estimates, modes


def _aligned(result: Any, modes: dict[int, int]) -> Any:
    """An eigenmode ``result`` with the followed modes in the first mesh's order."""
    if not isinstance(result, EigenmodeResult) or not modes:
        return result
    from gfem import _objective

    return _objective.aligned(result, modes)


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
    tolerance they add up across quantities. A lone quantity's
    indicators are marked on as they are.
    """
    if len(above) == 1:
        return estimates[above[0]].indicators
    squared = sum(
        np.asarray(estimates[i].indicators, float) ** 2 / tolerances[i] for i in above
    )
    return np.sqrt(squared)


class _Tracks:
    """The modes the estimators follow: one track per distinct start.

    A start is one mode index, or a cluster's members (``omega=(i, ...)``).
    Estimators naming the same mode on the first mesh share its track.
    On each mesh a mode's track follows the mode nearest its
    previous omega (`_track_mode`), a cluster's the as many modes nearest its
    previous mean (`_track_cluster`), each shift bounded by the omega
    estimate on the previous mesh (`bound`); two tracks landing on one mode
    raise `DegenerateModeError`.
    """

    def __init__(self, estimators: tuple[Estimator, ...]) -> None:
        starts = sorted({m for e in estimators if (m := _track_start(e)) is not None})
        #: Per start: the followed indices, their mean omega and its omega
        #: estimate on the last mesh (``None`` before the first).
        self._state: dict[
            tuple[int, ...], tuple[tuple[int, ...], complex | None, float | None]
        ] = {start: (start, None, None) for start in starts}
        self._omega = {
            _track_start(e): i
            for i, e in enumerate(estimators)
            if isinstance(e, QuantityEstimator) and e.omega is not None
        }
        self._reached: dict[tuple[int, ...], tuple[int, ...]] = {}
        self._omegas = np.zeros(0, complex)

    def add(
        self, estimates: tuple[ErrorEstimate, ...], modes: dict[int, int], result: Any
    ) -> dict[int, int]:
        """Follow the modes the first mesh's objectives read, one track each.

        On the first mesh each such mode is itself:
        ``modes`` with them added.
        """
        read = sorted({m for e in estimates for m in e._modes})  # noqa: SLF001
        new = [m for m in read if (m,) not in self._state]
        if not new:
            return modes
        self._omegas = np.asarray(result.omega)
        for m in new:
            self._state[(m,)] = ((m,), None, None)
            self._reached[(m,)] = (m,)
        return {**modes, **{m: m for m in new}}

    def omega(self, start: int) -> complex | None:
        """The omega of mode ``start``'s track on the last mesh, if followed."""
        state = self._state.get((start,))
        return None if state is None else state[1]

    def follow(self, result: Any, rtol: float) -> dict[int, int]:
        """Each named mode's index on ``result``: index on the first mesh -> there."""
        if not isinstance(result, EigenmodeResult) or not self._state:
            return {}
        reached: dict[tuple[int, ...], tuple[int, ...]] = {}
        for start, (indices, previous, bound) in self._state.items():
            if len(start) == 1:
                reached[start] = (
                    _track_mode(result, indices[0], previous, rtol, bound),
                )
            else:
                reached[start] = _track_cluster(result, indices, previous, bound)
        for (a, ma), (b, mb) in itertools.combinations(reached.items(), 2):
            if set(ma) & set(mb):
                mode = min(set(ma) & set(mb))
                omega = complex(np.asarray(result.omega)[mode])
                raise DegenerateModeError(
                    f"Two followed modes reached one mode: the modes named "
                    f"{_named(a)} and {_named(b)} on the first mesh both "
                    f"followed mode {mode} at "
                    f"{hertz(omega.real / (2 * np.pi))} on a refined mesh. Start "
                    "from a finer mesh, on which the modes are apart, or ask "
                    "for more modes (gfem.Eigenmode(n_modes=...))."
                )
        self._reached = reached
        self._omegas = np.asarray(result.omega)
        return {
            named: index
            for start, indices in reached.items()
            for named, index in zip(start, indices, strict=True)
        }

    def bound(
        self,
        model: Model,
        result: Any,
        estimates: tuple[ErrorEstimate, ...],
        *,
        threads: int | None,
        cache: RecoveryCache,
    ) -> None:
        """Keep each track's mean omega and estimate on ``result`` for the next mesh.

        An ``omega=`` estimate of the track's mode or cluster is its bound;
        any other track computes the omega kernel of its mode on the mesh's
        cache.
        """
        if not self._state:
            return
        from gfem._engine import estimate as kernels

        for start, indices in self._reached.items():
            if start in self._omega:
                bound = estimates[self._omega[start]].value
            else:
                bound = kernels.omega(
                    model, result, indices[0], threads, cache=cache
                ).value
            mean = complex(self._omegas[list(indices)].mean())
            self._state[start] = (indices, mean, bound)


def _named(start: tuple[int, ...]) -> str:
    """A track's start in words: ``"2"`` or ``"(0, 1)"``."""
    return str(start[0]) if len(start) == 1 else str(start)


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
        "non-finite estimate": (
            "because an error estimate is not finite, so it has nothing to mark on"
        ),
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


#: Squared indicators within this share of the largest are one tie group
#: in Dörfler marking (`_dorfler`).
_TIE_RTOL = 1e-9


def _dorfler(indicators: np.ndarray, error_fraction: float) -> np.ndarray:
    """Dörfler marking: the fewest tetrahedra holding a share of the squared error.

    Marks the tetrahedra with the largest indicators until their squared
    indicators add up to at least ``error_fraction`` of the total. Squared
    indicators equal to `_TIE_RTOL` of the largest are ties: they are sorted
    by tetrahedron index, and the tie group at the cut is marked whole, so
    round-off never picks among the equal indicators of a symmetric mesh,
    which refines symmetrically.

    Returns:
        A boolean mask over the tetrahedra; at least one is marked.
    """
    squared = np.asarray(indicators, float) ** 2
    largest = float(squared.max(initial=0.0))
    keys = (
        np.rint(squared / (_TIE_RTOL * largest)).astype(np.int64)
        if largest > 0
        else np.zeros(len(squared), np.int64)
    )
    order = np.argsort(-keys, kind="stable")
    cumulative = np.cumsum(squared[order])
    count = int(np.searchsorted(cumulative, error_fraction * cumulative[-1])) + 1
    count = min(count, len(squared))
    if largest > 0:
        sorted_keys = keys[order]
        # the whole tie group of the last marked tetrahedron
        count = int(np.searchsorted(-sorted_keys, -sorted_keys[count - 1], "right"))
    marked = np.zeros(len(squared), bool)
    marked[order[:count]] = True
    return marked


def _estimate(
    model: Model,
    result: Any,
    estimator: Estimator,
    *,
    threads: int | None,
    mode: int | tuple[int, ...] | None = None,
    adjoints: Mapping[Any, np.ndarray] | None = None,
    ridden: tuple[Any, ...] = (),
    cache: RecoveryCache | None = None,
    rtol: float | None = None,
    reach: str | None = None,
) -> ErrorEstimate:
    """The estimate of ``result``: dispatch on the estimator and result type.

    ``mode`` is the eigenmode a `QuantityEstimator` follows on this result,
    or the modes of a cluster it follows.
    ``adjoints`` are the S-functional adjoints solved with ``result``
    (`gfem.solve._solve`) for the ports ``ridden``;
    ``None`` solves them here if the entry needs them (`_adjoint_ports`). An
    objective solves here those of its dual's ports not ``ridden``.
    ``cache`` is the mesh's `gfem._engine.estimate.RecoveryCache`, shared by
    every estimate of ``result``'s mesh; ``None`` makes one for this call.
    ``rtol`` is the solve's ``cluster_rtol`` (`SolverOptions`' default for
    ``None``): an objective's modes are checked for degeneracy with it.
    ``reach`` is why the solve's options leave a participation's mode-field
    dual out of reach (`_mode_field_reach`), else ``None``.
    """
    from gfem._engine import estimate as kernels

    cache = cache or kernels.RecoveryCache()
    kind = _kind(estimator)
    if isinstance(estimator, QuantityEstimator) and estimator.objective is not None:
        return _estimate_objective(
            model,
            result,
            estimator,
            threads=threads,
            adjoints=adjoints or {},
            ridden=ridden if adjoints is not None else (),
            cache=cache,
            rtol=SolverOptions().cluster_rtol if rtol is None else rtol,
            reach=reach,
        )
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
            duals = _dual_fields(result, adjoints, cache)
            parts = kernels.s_entry(
                model, result, *estimator.s, threads, adjoints=duals, cache=cache
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


def _estimate_objective(
    model: Model,
    result: Any,
    estimator: QuantityEstimator,
    *,
    threads: int | None,
    adjoints: Mapping[Any, np.ndarray],
    ridden: tuple[Any, ...],
    cache: RecoveryCache,
    rtol: float,
    reach: str | None = None,
) -> ErrorEstimate:
    """An objective's estimate: classify it, then its kind's kernel.

    An objective of S on a model that conserves power (`_conserves_power`)
    is estimated through its lossless rewrite, over every row of
    the columns it reads; any other through the C form.
    The S-functional adjoints of the rows that need one (a port not
    excited, or a lossy wave port) and did not ride on the solve
    (``ridden``) are solved here: one more factorization per frequency
    (`_adjoints`).
    """
    from gfem import _objective
    from gfem._engine import estimate as kernels

    objective = _objective.classify(estimator.objective, result, model)  # ty: ignore[invalid-argument-type]
    if objective.kind == "objective of omega":
        return _estimate_objective_of_omega(
            model, result, estimator, objective, threads=threads, rtol=rtol, cache=cache
        )
    if objective.kind == "objective of participation":
        return _estimate_objective_of_participation(
            model,
            result,
            estimator,
            objective,
            threads=threads,
            rtol=rtol,
            reach=reach,
            duals=adjoints,
            cache=cache,
        )
    if objective.kind in _POTENTIAL_KINDS:
        return _estimate_potential_objective(
            model,
            result,
            estimator,
            objective,
            threads=threads,
            adjoints=adjoints,
            ridden=ridden,
            cache=cache,
        )
    if objective.kind in _FIELD_KINDS:
        return _estimate_field_objective(
            model,
            result,
            estimator,
            objective,
            threads=threads,
            adjoints=adjoints,
            ridden=ridden,
            cache=cache,
        )
    if objective.kind != "objective of s":
        raise MeshRefinementUnsupportedError(
            objective.kind,
            f"{_describe(estimator)} is an {objective.kind}, which has no "
            "verified estimate in this version of gfem: use "
            "gfem.FieldEstimator(), or estimate the quantities it reads one "
            "by one.",
        )
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
    missing = tuple(p for p in ports if p not in ridden)
    solved: dict[Any, np.ndarray] = dict(adjoints)
    if missing:
        solved.update(_adjoints(model, result, missing, threads))
    duals = _dual_fields(result, solved, cache)
    if rewrite:
        terms = kernels.lossless_rewrite(
            model,
            result,
            reads,
            objective.curvature,
            objective.diagonal,
            threads=threads,
            adjoints=duals,
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
            adjoints=duals,
            cache=cache,
        )
        kind, defect = objective.kind, 0.0
    return ErrorEstimate(
        estimator=estimator,
        kind=kind,
        values=(terms.value,),
        indicators=readonly(np.sqrt(terms.weights)),
        _quantity=objective.value,
        _dual_ports=ports,
        _sensitivity=objective.sensitivity,
        _defect=defect,
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


#: The estimate kinds of an objective of driven fields.
_FIELD_KINDS = ("objective of absorbed power", "objective of field means")


def _estimate_field_objective(
    model: Model,
    result: DrivenResult,
    estimator: QuantityEstimator,
    objective: Any,
    *,
    threads: int | None,
    adjoints: Mapping[Any, np.ndarray],
    ridden: tuple[Any, ...],
    cache: RecoveryCache,
) -> ErrorEstimate:
    """An objective of driven fields' estimate.

    One dual per measurement output read (`gfem._engine.field_duals`):
    those that did not ride on the solve (``ridden``) are solved here, one
    more factorization per frequency read (`_field_duals`).
    """
    from gfem._engine import estimate as kernels

    reads = []
    for r in objective.reads:
        spec = objective.measurements[r.leaf]
        index = int(spec["label"].removeprefix(".e_fields[").removesuffix("]"))
        f, excitation = result.kept[index]
        if spec["measurement"] == "power_flow":
            key = ("power", float(f), excitation, tuple(spec["side"]))
            reads.append((key, complex(np.real(r.cotangent))))
        else:
            key = ("mean", float(f), excitation, tuple(spec["regions"]), r.index[0])
            reads.append((key, complex(r.cotangent)))
    keys = list(dict.fromkeys(key for key, _ in reads))
    solved: dict[Any, np.ndarray] = {
        (k[1], k): adjoints[k[1], k]
        for k in keys
        if k in ridden and (k[1], k) in adjoints
    }
    missing = [k for k in keys if (k[1], k) not in solved]
    if missing:
        solved.update(_field_duals(model, result, missing, threads))
    terms = kernels.objective_of_fields(
        model,
        result,
        reads,
        objective.curvature,
        threads,
        duals=solved,
        cache=cache,
    )
    return ErrorEstimate(
        estimator=estimator,
        kind=objective.kind,
        values=(terms.value,),
        indicators=readonly(np.sqrt(terms.weights)),
        _quantity=objective.value,
        _dual_keys=tuple(keys),
        _sensitivity=objective.sensitivity,
    )


def _field_duals(
    model: Model, result: DrivenResult, keys: list[Any], threads: int | None
) -> dict[tuple[float, Hashable], np.ndarray]:
    """The duals of field measurement outputs on a finished result.

    One more factorization per frequency read, with the direct solver at the
    result's order, the loads formed from the result's fields there.
    """
    from gfem._engine import driven, field_duals

    frequencies = sorted({k[1] for k in keys})
    excite = list(result.s.excited)
    fields = dict(zip(result.kept, result.e_fields, strict=True))
    solutions = {
        f: np.stack([np.asarray(fields[f, p].coefficients) for p in excite], axis=1)
        for f in frequencies
    }
    problem = Driven(frequencies, excite=excite)
    order = _fields(result)[0].order
    options = SolverOptions(order=order, linear_solver="direct", threads=threads)
    with driven.low_frequency_deferred(), warnings.catch_warnings():
        for category in driven.FORWARD_WARNINGS:  # the solve of ``result`` warned
            warnings.simplefilter("ignore", category)
        rule = driven.DrivenRule(model, problem, options)
        rule.adjoint_loads = field_duals.field_loads(keys)
        return rule.solve_adjoints(model, solutions)


#: The estimate kinds of an objective of an electrostatic result.
_POTENTIAL_KINDS = (
    "objective of capacitance",
    "objective of electrostatic energy",
    "objective of potential means",
)


def _estimate_potential_objective(
    model: Model,
    result: ElectrostaticResult,
    estimator: QuantityEstimator,
    objective: Any,
    *,
    threads: int | None,
    adjoints: Mapping[Any, np.ndarray],
    ridden: tuple[Any, ...],
    cache: RecoveryCache,
) -> ErrorEstimate:
    """An electrostatic objective's estimate.

    J(C) reads the potentials' free duals. An energy or potential mean reads
    one dual per read energy and per averaged region set: those that did not
    ride on the solve (``ridden``) are solved here, one more factorization
    (`gfem._engine.electrostatic.solve_duals`).
    """
    from gfem._engine import electrostatic
    from gfem._engine import estimate as kernels

    terminals = list(result.terminals)
    reads, keys = [], []
    for r in objective.reads:
        ct = float(np.real(r.cotangent))
        if r.leaf == ".capacitance":
            i, j = r.index
            reads.append((("capacitance", terminals[j], terminals[i]), ct))
        elif r.leaf.startswith(".energy["):
            region = r.leaf.removeprefix(".energy[").removesuffix("]").strip("'\"")
            key = ("energy", region, terminals[r.index[0]])
            reads.append((key, ct))
            keys.append(key)
        else:
            spec = objective.measurements[r.leaf]
            terminal = spec["label"].removeprefix(".potential[").removesuffix("]")
            key = ("mean", tuple(spec["regions"]))
            reads.append((("mean", key[1], terminal.strip("'\"")), ct))
            keys.append(key)
    keys = list(dict.fromkeys(keys))
    solved: dict[Any, np.ndarray] = {
        k: adjoints[k] for k in keys if k in ridden and k in adjoints
    }
    missing = [k for k in keys if k not in solved]
    if missing:
        loads = electrostatic.objective_loads(missing)
        solved.update(electrostatic.solve_duals(model, result, loads, threads))
    terms = kernels.objective_of_potentials(
        model,
        result,
        reads,
        objective.curvature,
        threads,
        duals=solved,
        cache=cache,
    )
    return ErrorEstimate(
        estimator=estimator,
        kind=objective.kind,
        values=(terms.value,),
        indicators=readonly(np.sqrt(terms.weights)),
        _quantity=objective.value,
        _dual_keys=tuple(keys),
        _sensitivity=objective.sensitivity,
    )


def _estimate_objective_of_omega(
    model: Model,
    result: EigenmodeResult,
    estimator: QuantityEstimator,
    objective: Any,
    *,
    threads: int | None,
    rtol: float,
    cache: RecoveryCache,
) -> ErrorEstimate:
    """An objective of the modes' omega: free duals.

    Each mode read is followed as a mode of its own (``_modes``), and must
    be clearly apart from the others (`_track_mode`). The terms
    lifted for the omega estimate alone raise (`_check_lossy_mode`).
    """
    from gfem._engine import estimate as kernels

    _check_lossy_mode(model, "objective")
    reads = [(r.index[0], r.cotangent) for r in objective.reads]
    for mode, _ in reads:
        _track_mode(result, mode, None, rtol, None)
    terms = kernels.objective_of_omega(
        model, result, reads, objective.curvature, threads, cache=cache
    )
    return ErrorEstimate(
        estimator=estimator,
        kind=objective.kind,
        values=(terms.value,),
        indicators=readonly(np.sqrt(terms.weights)),
        _quantity=objective.value,
        _sensitivity=objective.sensitivity,
        _modes=tuple(mode for mode, _ in reads),
    )


def _estimate_objective_of_participation(
    model: Model,
    result: EigenmodeResult,
    estimator: QuantityEstimator,
    objective: Any,
    *,
    threads: int | None,
    rtol: float,
    reach: str | None,
    duals: Mapping[Any, Any],
    cache: RecoveryCache,
) -> ErrorEstimate:
    """An objective of bulk participations, and omega.

    Each participation read needs its mode-field dual: ridden on the
    forward (``duals``, by mode and region, when it was solved for the same
    mode), else solved here on an operator prepared at the mode
    (`_participation_duals`). Raises for a mode out of mode-field reach,
    for the mode Q's refusals, and for a mode energy holding a surface
    term, which the equilibrated dual flux has no volume load for.
    """
    from gfem._engine import estimate as kernels

    _check_lossy_mode(model, "objective")
    _check_mode_field_reach(model, reach)
    reads, regions = [], {}
    omega = np.asarray(result.omega)
    for r in objective.reads:
        mode = r.index[0]
        if r.leaf == ".omega":
            reads.append(("omega", mode, r.cotangent))
            continue
        region = r.leaf.removeprefix(".bulk_participation[").rstrip("]").strip("'\"")
        reads.append(("bulk", mode, region, r.cotangent.real))
        regions.setdefault(mode, []).append(region)
    modes = sorted({read[1] for read in reads})
    for mode in modes:
        _track_mode(result, mode, None, rtol, None)
    solved: dict[tuple[int, str], Any] = {}
    for mode, names in regions.items():
        _check_isolated(omega, mode, rtol)
        ridden = {
            name: duals.get((mode, name))
            for name in names
            if _rides(duals.get((mode, name)), omega[mode])
        }
        missing = [name for name in names if name not in ridden]
        if missing:
            ridden |= _participation_duals(model, result, mode, missing, threads)
        solved |= {(mode, name): dual for name, dual in ridden.items()}
    terms = kernels.objective_of_participation(
        model, result, reads, objective.curvature, threads, duals=solved, cache=cache
    )
    return ErrorEstimate(
        estimator=estimator,
        kind=objective.kind,
        values=(terms.value,),
        indicators=readonly(np.sqrt(terms.weights)),
        _quantity=objective.value,
        _sensitivity=objective.sensitivity,
        _modes=tuple(modes),
        _dual_modes=tuple(solved),
    )


def _rides(dual: Any, omega: complex) -> bool:
    """Whether a ridden ``dual`` was solved for the mode at ``omega``."""
    return dual is not None and abs(dual.omega - omega) <= 1e-12 * abs(omega)


def _mode_field_reach(options: SolverOptions) -> str | None:
    """Why a solve with ``options`` leaves mode-field duals out of reach."""
    from gfem._eigen.rule import _FIELD_TOLERANCE
    from gfem._linalg import ResolvedLinearSolver

    linear = ResolvedLinearSolver.resolve(
        options, default="direct", problem="eigenmode"
    ).kind
    if options.device != "cpu":
        return f'device="{options.device}": it runs on the CPU only'
    if linear != "direct":
        return f'linear_solver="{linear}": it runs on linear_solver="direct" only'
    if options.eigen_tolerance > _FIELD_TOLERANCE:
        return (
            f"SolverOptions.eigen_tolerance = {options.eigen_tolerance:g}: it "
            f"needs eigen_tolerance <= {_FIELD_TOLERANCE:g} (the default), at "
            "which the forward polishes the modes"
        )
    return None


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
    from gfem._eigen.rule import _ISOLATED

    others = [i for i in range(len(omega)) if i != mode]
    if not others:
        return
    gaps = np.abs(omega[others] - omega[mode]) / abs(omega[mode])
    near = others[int(np.argmin(gaps))]
    threshold = max(_ISOLATED, rtol)
    if gaps.min() < threshold:
        raise DegenerateModeError(
            f"An objective of the participation of mode {mode} at "
            f"{_frequency(omega, mode)} needs the mode isolated, but mode "
            f"{near} at {_frequency(omega, near)} lies within a relative gap "
            f"of {gaps.min():.3g}, below {threshold:g}: near another "
            "mode the field is a mix of the two. Separate the modes, or use "
            "gfem.FieldEstimator()."
        )


def _participation_duals(
    model: Model,
    result: EigenmodeResult,
    mode: int,
    regions: list[str],
    threads: int | None,
) -> dict[str, Any]:
    """The participation duals of ``mode`` on a finished result: one factorization.

    The operator is prepared at k (1 + d) next to the mode, as a backward
    without its forward's operator does (`EigenmodeRule._field_pairs`),
    with the direct solver: the result does not record its options.
    """
    import scipy.constants

    from gfem._eigen.adjoint import reshift_offset
    from gfem._eigen.rule import EigenmodeRule

    omega = np.asarray(result.omega)
    field = result.e_fields[mode]
    problem = Eigenmode(target=omega[mode].real / (2 * np.pi), n_modes=len(omega))
    options = SolverOptions(order=field.order, linear_solver="direct", threads=threads)
    rule = EigenmodeRule(model, problem, options)
    op = rule.driven(model)
    k = complex(omega[mode]) * op.built.scale / scipy.constants.c
    others = np.delete(omega, mode)
    gap = (
        float(np.min(np.abs(others - omega[mode])) / abs(omega[mode]))
        if len(others)
        else math.inf
    )
    eop = rule.operator(op, k * (1 + reshift_offset(gap)))
    try:
        return rule.participation_duals(
            model,
            op,
            eop,
            u=np.asarray(field.coefficients, complex),
            k=k,
            gap=gap,
            mode=mode,
            regions=tuple(regions),
        )
    finally:
        eop.free_handle()


def _check_first_estimates(
    problem: Any,
    result: Any,
    quantities: tuple[
        tuple[Estimator, ...], tuple[ErrorEstimate, ...], tuple[float, ...]
    ],
    modes: dict[int, int],
    options: SolverOptions,
    *,
    first: bool,
) -> None:
    """On the first mesh, raise for a Q or objective floor, or for workers.

    ``quantities`` holds the estimators, their estimates and tolerances.
    Q's floor is `_check_q_floors`. An objective's is the solve tolerance
    carried through the objective's first-order sensitivity, 10 x
    tolerance x sum_a |ct_a| s_a,
    since the estimate does not include the solve's error. A dual that
    rides on the forward solve needs one worker.
    """
    if not first:
        return
    estimators, estimates, tolerances = quantities
    _check_q_floors(result, estimators, tolerances, modes, options)
    solve_tolerance, name = _solve_tolerance(problem, options)
    workers = 1
    if isinstance(problem, Driven):
        workers = min(options.frequency_workers, len(problem.frequency))
    for estimator, estimate, tolerance in zip(
        estimators, estimates, tolerances, strict=True
    ):
        if not _is_objective(estimator):
            continue
        sensitivity = estimate._sensitivity  # noqa: SLF001
        floor = 10 * solve_tolerance * sensitivity
        if tolerance < floor:
            raise ValueError(
                f"refine_mesh tolerance={tolerance:g} for {_describe(estimator)} "
                f"must be at least {floor:.3g}: 10 x SolverOptions.{name}="
                f"{solve_tolerance:g} x the objective's first-order sensitivity "
                f"{sensitivity:.3g} (the sum of |dJ/dy| |y| over what it reads, "
                "|y| = 1 for S). The error estimate does not include the "
                f"solve's error. Lower SolverOptions.{name} to at most "
                f"{tolerance / (10 * sensitivity):.2g}, or raise tolerance."
            )
        if workers > 1 and (estimate._dual_ports or estimate._dual_keys):  # noqa: SLF001
            raise MeshRefinementUnsupportedError(
                "frequency workers",
                f"{_describe(estimator)} solves its adjoint duals with the "
                "forward, which frequency workers do not do in this version of "
                "gfem: it needs frequency_workers=1, got "
                f"{options.frequency_workers}.",
            )


def _is_objective(estimator: Any) -> bool:
    return isinstance(estimator, QuantityEstimator) and estimator.objective is not None


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


def _recovery_cache() -> RecoveryCache:
    """A new `gfem._engine.estimate.RecoveryCache`: one per solved mesh."""
    from gfem._engine import estimate as kernels

    return kernels.RecoveryCache()


def _estimator_mode(estimator: Estimator) -> int | None:
    """The mode index of a `QuantityEstimator` (``omega=i``, ``q=``), else ``None``."""
    if not isinstance(estimator, QuantityEstimator) or isinstance(
        estimator.omega, tuple
    ):
        return None
    return estimator.omega if estimator.omega is not None else estimator.q


def _track_start(estimator: Estimator) -> tuple[int, ...] | None:
    """The modes an estimator follows on the first mesh: ``(i,)`` or a cluster."""
    if isinstance(estimator, QuantityEstimator) and isinstance(estimator.omega, tuple):
        return estimator.omega
    mode = _estimator_mode(estimator)
    return None if mode is None else (mode,)


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

    Q's relative error is cot(arg omega) times omega's relative error,
    so the solve's ``eigen_tolerance`` reaches Q multiplied by it.
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
    result: Any,
    estimator: Estimator,
    mode: int | tuple[int, ...] | None,
    estimate: ErrorEstimate | None = None,
) -> float | complex | tuple[complex, ...] | None:
    """The quantity a `QuantityEstimator` names, read off ``result``.

    An objective's is its value, which its ``estimate`` holds.
    """
    if not isinstance(estimator, QuantityEstimator) or estimator.objective is not None:
        # None for a FieldEstimator, whose estimate holds no quantity
        return getattr(estimate, "_quantity", None)
    if estimator.capacitance is not None:
        i, j = (result.terminals.index(t) for t in estimator.capacitance)
        return float(np.asarray(result.capacitance)[i, j])
    if estimator.s is not None:
        return tuple(complex(v) for v in np.asarray(result.s[estimator.s]))
    if estimator.q is not None:
        return float(np.asarray(result.q)[mode])
    if isinstance(mode, tuple):
        return complex(np.asarray(result.omega)[list(mode)].mean())
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
            allows, or the second nearest is less than twice as far. The
            message names the cluster's modes and points to the cluster
            keyword.
    """
    omegas = np.asarray(result.omega)
    if previous is not None:
        distances = np.abs(omegas - previous)
        mode = int(np.argmin(distances))
        _check_followed(omegas, mode, distances, previous, previous_estimate)
        reference = previous
    else:
        reference = omegas[mode]
    close = np.abs(omegas - reference) <= rtol * abs(reference)
    cluster = np.asarray(result.cluster)
    together = np.flatnonzero(close | (cluster == cluster[mode]))
    if len(together) > 1:
        others = [int(i) for i in together if i != mode]
        named = (
            f", QuantityEstimator(omega={tuple(int(i) for i in together)})"
            if previous is None
            else " on the first mesh, QuantityEstimator(omega=(i, j, ...))"
        )
        raise DegenerateModeError(
            f"QuantityEstimator follows one mode by its frequency, and mode "
            f"{mode} at {_frequency(omegas, mode)} is in a degenerate cluster "
            f"with {_modes_text(omegas, others)} (labelled together in "
            "result.cluster, or within SolverOptions.cluster_rtol="
            f"{rtol:g}): name all of them{named}, or use "
            "gfem.FieldEstimator(), which estimates every mode."
        )
    return mode


def _frequency(omegas: np.ndarray, mode: int) -> str:
    """Mode ``mode``'s frequency in words, ``"9.12 GHz"``."""
    return hertz(complex(omegas[mode]).real / (2 * np.pi))


def _modes_text(omegas: np.ndarray, modes: list[int]) -> str:
    """``"modes 0 (9.12 GHz) and 2 (9.12 GHz)"``: modes with their frequencies."""
    items = [f"{i} ({_frequency(omegas, i)})" for i in modes]
    if len(items) == 1:
        return f"mode {items[0]}"
    return f"modes {', '.join(items[:-1])} and {items[-1]}"


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
    frequency = _frequency(omegas, mode)
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
    near = [int(i) for i in np.flatnonzero(distances < 2 * nearest) if int(i) != mode]
    if near:
        raise DegenerateModeError(
            "QuantityEstimator(omega=...) follows one mode by its frequency, "
            f"and mode {mode} at {frequency} is not clearly nearest the "
            f"followed frequency on the refined mesh: "
            f"{_modes_text(omegas, near)} {'is' if len(near) == 1 else 'are'} "
            "less than twice as far, a degenerate cluster with it. Name all of "
            "them on the first mesh, QuantityEstimator(omega=(i, j, ...)), "
            "start from a finer mesh, on which the modes are apart, or use "
            "gfem.FieldEstimator(), which estimates every mode."
        )


def _track_cluster(
    result: EigenmodeResult,
    members: tuple[int, ...],
    previous: complex | None,
    previous_estimate: float | None,
) -> tuple[int, ...]:
    """The modes of ``result`` a cluster mean follows.

    On the first result (``previous`` is ``None``) they are ``members``, which
    must be whole ``result.cluster`` groups. On a refined mesh they are the
    as many modes nearest ``previous``, the previous mean, whose mean moved
    at most twice the previous estimate over `_ESTIMATE_FLOOR` (the
    single-mode bound, `_check_followed`). On every mesh the nearest other
    mode must lie at least twice as far from the mean (the first mesh) or
    from ``previous`` as the farthest member (the gap test).

    Raises:
        DegenerateModeError: ``members`` holds part of a labelled cluster;
            the mean moved too far; another mode is too near.
    """
    omegas = np.asarray(result.omega)
    count = len(members)
    keyword = f"QuantityEstimator(omega={members})"
    if previous is None:
        cluster = np.asarray(result.cluster)
        for mode in members:
            group = [int(i) for i in np.flatnonzero(cluster == cluster[mode])]
            missing = [i for i in group if i not in members]
            if missing:
                others = [i for i in group if i != mode]
                raise DegenerateModeError(
                    f"{keyword}: mode {mode} at {_frequency(omegas, mode)} is in "
                    f"a degenerate cluster with {_modes_text(omegas, others)} "
                    "(result.cluster): name all of them, QuantityEstimator("
                    f"omega={tuple(sorted({*members, *group}))})."
                )
        reference = complex(omegas[list(members)].mean())
        chosen = members
        where = "on the first mesh"
    else:
        distances = np.abs(omegas - previous)
        order = np.argsort(distances, kind="stable")
        chosen = tuple(sorted(int(i) for i in order[:count]))
        mean = complex(omegas[list(chosen)].mean())
        reference = previous
        if previous_estimate is not None:
            moved = abs(mean - previous)
            allowed = 2 * previous_estimate * abs(previous) / _ESTIMATE_FLOOR
            if moved > allowed:
                raise DegenerateModeError(
                    f"{keyword} follows a cluster by its mean frequency, and "
                    f"the {count} modes nearest it on the refined mesh, "
                    f"{_modes_text(omegas, list(chosen))}, moved their mean "
                    f"{moved / abs(previous):.3g} of the followed frequency, "
                    "further than the previous error estimate "
                    f"{previous_estimate:.3g} allows. The cluster probably left "
                    "the solved modes: ask for more modes "
                    "(gfem.Eigenmode(n_modes=...)) or start from a finer mesh."
                )
        where = "on the refined mesh"
    distances = np.abs(omegas - reference)
    farthest = float(distances[list(chosen)].max())
    others = [int(i) for i in np.argsort(distances, kind="stable") if i not in chosen]
    nearest = others[0]
    if distances[nearest] < 2 * farthest or distances[nearest] == 0:
        raise DegenerateModeError(
            f"{keyword} follows a cluster by its mean frequency, and the "
            f"cluster, {_modes_text(omegas, list(chosen))} {where}, is not "
            f"clearly apart from mode {nearest} at {_frequency(omegas, nearest)}: "
            "it is less than twice as far from the mean as the farthest "
            "member. Start from a finer mesh, on which the modes are apart, "
            "name it in the cluster too, or use gfem.FieldEstimator(), which "
            "estimates every mode."
        )
    return chosen


def _dofs(result: Any) -> int:
    """A solve's unknowns: one field's coefficient count."""
    return int(_fields(result)[0].coefficients.shape[0])


def _fields(result: Any) -> list:
    """The fields of a result an estimate reads."""
    if isinstance(result, ElectrostaticResult):
        return list(result.potential.values())
    return list(result.e_fields)


def _refuse_blends(model: Model, where: str) -> None:
    """Raise for a law blend: the estimators do not read one."""
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
    # "thermal" or "dc_conduction": the estimators are electromagnetic
    feature = physics_of(problem)
    if feature != "em":
        raise MeshRefinementUnsupportedError(
            feature,
            f"Mesh refinement has no path for a gfem.{type(problem).__name__} "
            "problem: its error estimators are electromagnetic. Refine the mesh "
            "in your mesher.",
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
    if floquet_ports(model.ports):
        raise MeshRefinementUnsupportedError(
            "floquet port",
            "Mesh refinement has no path for a Floquet port: refining one side "
            "of a periodic pair would break its node pairing. Refine the mesh in "
            "your mesher.",
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
        _check_modes(estimator, problem.n_modes, keyword)


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
        _check_modes(estimator, len(np.asarray(result.omega)), keyword)


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

    The read port ``to`` may be any port with a resistance
    (`_check_defined_row`, with the model): when it is not excited, its
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


def _check_modes(estimator: QuantityEstimator, n_modes: int, keyword: str) -> None:
    """Raise unless the estimator's modes, and a cluster's guard mode, are solved.

    A cluster's gap test (`_track_cluster`) needs one mode more than the
    cluster holds.
    """
    members = _track_start(estimator) or ()
    if len(members) < 2:
        _check_mode(_estimator_mode(estimator), n_modes, keyword)
        return
    _check_mode(max(members), n_modes, keyword)
    if n_modes <= len(members):
        raise ValueError(
            f"QuantityEstimator(omega={members}) follows a cluster of "
            f"{len(members)} modes, told apart from the nearest other mode, "
            f"and n_modes={n_modes} leaves none: ask for one more mode: "
            f"gfem.Eigenmode(n_modes={len(members) + 1})."
        )


def _check_mode(mode: int | None, n_modes: int, keyword: str = "omega") -> None:
    if mode is not None and mode >= n_modes:
        raise ValueError(
            f"QuantityEstimator({keyword}={mode}) names a mode index at or above "
            f"n_modes={n_modes}: ask for more modes, or name one of 0 to "
            f"{n_modes - 1}."
        )


def _check_quantity_model(estimator: QuantityEstimator, model: Model) -> None:
    """Raise for a quantity entry the verification tier has no case for.

    ``q`` keeps the mode refusals lifted for ``omega``
    (`_check_lossy_mode`); the omega estimate checks its mode's Re eps when
    it estimates (`_check_mode_eps`).
    """
    if estimator.q is not None:
        _check_lossy_mode(model, "q")
    if estimator.s is not None:
        _check_defined_row(estimator.s[0], model)


def _undefined_rows(model: Model) -> tuple[str, ...]:
    """The lumped ports with no resistance: their rows of S are undefined (NaN)."""
    from gfem._engine.port_network import resistance

    return tuple(
        p.name
        for p in model.ports
        if isinstance(p, LumpedPort) and math.isnan(resistance(p))
    )


def _check_defined_row(to: str, model: Model) -> None:
    """Raise when the read port ``to`` has no resistance (an undefined S row)."""
    if to in _undefined_rows(model):
        raise MeshRefinementUnsupportedError(
            "undefined S row",
            f"QuantityEstimator(s=({to!r}, ...)) reads the row of port {to!r}, "
            "which has no resistance: its row of S is undefined (NaN), so "
            "there is no error to estimate. Give the port a resistance, or "
            "read another row.",
        )


def _adjoint_ports(
    estimator: Estimator, model: Model, excited: list[str]
) -> tuple[str, ...]:
    """The port whose S-functional adjoint an S estimate may need.

    ``to`` of ``QuantityEstimator(s=(to, source))`` when it is not among
    ``excited`` (it has no field of its own), or when it is a wave port
    touching a lossy element: its mode may then not be parallel to its
    excitation (the solve checks), and its adjoint is not its own field.
    """
    if not isinstance(estimator, QuantityEstimator) or estimator.s is None:
        return ()
    to = estimator.s[0]
    return (to,) if _needs_adjoint(to, model, excited) else ()


def _needs_adjoint(to: str, model: Model, excited: list[str]) -> bool:
    """Whether the row of port ``to`` may need its solved S-functional adjoint.

    When it is not among ``excited`` (it has no field of its own), or is a
    wave port touching a lossy element: its mode may then not be parallel
    to its excitation (the solve checks), and its adjoint is not its own
    field.
    """
    if to not in excited:
        return True
    port = next((p for p in model.ports if p.name == to), None)
    if not isinstance(port, WavePort):
        return False
    return bool(_lossy_elements(model)[_touching(model, port.surface)].any())


def _excited(problem: Driven, model: Model) -> list[str]:
    """The ports ``problem`` excites on ``model``."""
    if problem.excite is None:
        return [p.name for p in model.ports]
    return list(problem.excite)


def _adjoint_port_union(
    estimators: tuple[Estimator, ...], model: Model, problem: Any
) -> tuple[str, ...]:
    """The ports whose S-functional adjoints the S entries' duals solve.

    The union over the estimators, each port once: two S entries
    reading one port share its adjoint. An objective's ports are known once
    it has been classified, on the first result (`ErrorEstimate`).
    """
    if not isinstance(problem, Driven):
        return ()
    excited = _excited(problem, model)
    return tuple(
        dict.fromkeys(
            port for each in estimators for port in _adjoint_ports(each, model, excited)
        )
    )


class _Duals:
    """The duals that ride on `refine_mesh`'s solves.

    On a driven problem, the ports whose S-functional adjoints ride: an S
    entry's from the first solve; an objective's once its estimate has read
    them (`add`), from the next solve on. On
    an electrostatic problem, an objective's energy and potential-mean
    duals, the same way.
    """

    def __init__(self, ports: tuple[str, ...], problem: Any = None) -> None:
        self.ports = ports
        #: The participations whose mode-field duals ride on the eigen
        #: forward: (mode on the first mesh, region).
        self.modes: tuple[tuple[int, str], ...] = ()
        self.keys: tuple[tuple[Any, ...], ...] = ()
        self._electrostatic = isinstance(problem, Electrostatic)

    def loads(self, tracks: _Tracks) -> Any:
        """The solves' adjoint loads; ``None`` for free duals only.

        On an eigenmode problem, the participations' dual requests: each
        asks for the mode nearest its track's omega on the last mesh
        (`_Tracks.omega`).
        """
        if self.modes:
            from gfem._eigen.rule import DualRequest

            regions: dict[int, list[str]] = {}
            for start, region in self.modes:
                regions.setdefault(start, []).append(region)
            return [
                DualRequest(start, omega, tuple(names))
                for start, names in regions.items()
                if (omega := tracks.omega(start)) is not None
            ]
        if self._electrostatic:
            from gfem._engine.electrostatic import objective_loads

            return objective_loads(self.keys) if self.keys else None
        from gfem._engine.driven import port_loads
        from gfem._engine.field_duals import field_loads

        parts = []
        if self.ports:
            parts.append(port_loads(self.ports))
        if self.keys:
            parts.append(field_loads(self.keys))
        if len(parts) < 2:
            return parts[0] if parts else None

        def loads(context: Any) -> dict[Hashable, np.ndarray]:
            return {k: v for part in parts for k, v in part(context).items()}

        return loads

    @property
    def ridden(self) -> tuple[Any, ...]:
        """What rides: the ports and field-dual keys, or the electrostatic keys."""
        return self.keys if self._electrostatic else (*self.ports, *self.keys)

    def add(self, estimates: tuple[ErrorEstimate, ...]) -> None:
        """Add the duals the estimates' objectives read."""
        read = (p for e in estimates for p in e._dual_ports)  # noqa: SLF001
        self.ports = tuple(dict.fromkeys([*self.ports, *read]))
        keys = (k for e in estimates for k in e._dual_keys)  # noqa: SLF001
        self.keys = tuple(dict.fromkeys([*self.keys, *keys]))
        modes = (m for e in estimates for m in e._dual_modes)  # noqa: SLF001
        self.modes = tuple(dict.fromkeys([*self.modes, *modes]))


def _check_adjoint_workers(
    estimators: tuple[Estimator, ...],
    model: Model,
    problem: Any,
    options: SolverOptions,
) -> None:
    """Refuse frequency workers for an estimate that solves a dual.

    Its adjoint loads (`_loads`) ride on the forward's
    factorization, which a frequency worker does not hand back:
    frequency workers take wave ports but spread no adjoint
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

    A law region is lossy when a term of its law carries loss.
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


def _check_lossy_mode(model: Model, keyword: str) -> None:
    """Raise for a mode term an estimate built on the mode has no case for.

    The terms lifted for omega alone: a permittivity law
    ("dispersive eigenmode"), and a second-order absorbing boundary, a
    complex mu_r or loss inside an anisotropic material ("lossy
    eigenmode"). Q and the objectives of a mode keep refusing them;
    ``keyword`` names the refused estimator's keyword.
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
    elif (region := _lossy_anisotropic_region(model)) is not None:
        what = f"loss inside the anisotropic material of {region!r}"
    if what is not None:
        raise MeshRefinementUnsupportedError(
            "lossy eigenmode",
            f"QuantityEstimator({keyword}=...) has no verified estimate for a mode "
            f"with {what}: use gfem.FieldEstimator().",
        )


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


def _lossy_anisotropic_region(model: Model) -> str | None:
    """The first anisotropic material region holding loss, if any."""
    from gfem._engine.coefficients import anisotropic_regions

    lossy = _lossy_elements(model)
    for region in anisotropic_regions(model, tuple(model.materials)):
        if lossy[model.mesh.elements(region)].any():
            return region
    return None


#: The plasmonic window of a law region's Re eps, open at both ends.
_PLASMONIC_WINDOW = (-20.0, 0.0)


def _check_driven_estimable(
    estimators: tuple[Estimator, ...], model: Model, frequencies: list[float]
) -> None:
    """Raise for a driven quantity estimate the identity has no term for.

    The S estimate's and every objective's on a driven result (objectives
    of S and driven field objectives alike).
    """
    for estimator in estimators:
        if isinstance(estimator, QuantityEstimator) and estimator.s is not None:
            _check_driven_quantity(model, frequencies, "QuantityEstimator(s=...)")
        elif _is_objective(estimator):
            _check_driven_quantity(model, frequencies, _describe(estimator))


def _check_driven_quantity(model: Model, frequencies: list[float], what: str) -> None:
    """Raise where a driven quantity estimate was measured underestimating.

    One check for the S estimate, objectives of S and driven field
    objectives, before any solve. It runs after
    `dispersion.check_estimator_frequencies` has gone over every frequency,
    whose zero-weight `ValueError` comes first. The field estimate is not
    refused here: it is unclaimed (recipe Limits).

    Args:
        model: The driven model.
        frequencies: The solve frequencies, Hz.
        what: The estimate, as the message names it (for example
            ``"QuantityEstimator(s=...)"``).

    Raises:
        MeshRefinementUnsupportedError: ``feature`` is ``"impedance or
            absorbing surface"`` (any `Impedance` or `Absorbing` surface),
            ``"lumped port spanning a permeability contrast"`` (the
            tetrahedra beside a `LumpedPort`'s surface differ in mu_r,
            tensors compared whole) or ``"plasmonic permittivity"`` (a law
            region with Re eps in `_PLASMONIC_WINDOW` at a frequency).
    """
    remedy = (
        "Refine with gfem.FieldEstimator() and check the quantity by a finer solve."
    )
    surface = next(
        (b.surface for b in model.boundaries if isinstance(b, (Impedance, Absorbing))),
        None,
    )
    if surface is not None:
        raise MeshRefinementUnsupportedError(
            "impedance or absorbing surface",
            f"{what} has no verified estimate on a driven model with an "
            f"impedance or absorbing surface ({surface!r}): the current on such "
            "a surface carries a share of the error that the estimate has no "
            f"term for. {remedy}",
        )
    port = _lumped_port_on_mu_contrast(model)
    if port is not None:
        raise MeshRefinementUnsupportedError(
            "lumped port spanning a permeability contrast",
            f"{what} has no verified estimate where the face of lumped port "
            f"{port!r} borders materials of different mu_r: the estimate "
            "misses the error at the edge where the permeability interface "
            f"meets the port's face. {remedy}",
        )
    plasmonic = _plasmonic_law(model, frequencies)
    if plasmonic is not None:
        region, f, eps = plasmonic
        low, high = _PLASMONIC_WINDOW
        raise MeshRefinementUnsupportedError(
            "plasmonic permittivity",
            f"{what} has no verified estimate where a permittivity law has Re "
            f"eps between {low:g} and {high:g}: region {region!r} has Re eps = "
            f"{eps:.3g} at {hertz(f)}. There, metal edges and lumped port edges "
            f"on the metal make the solution itself converge poorly. {remedy}",
        )


def _lumped_port_on_mu_contrast(model: Model) -> str | None:
    """The first lumped port whose surface borders tetrahedra of different mu_r."""
    from gfem._engine.coefficients import element_tensors

    lumped = [p for p in model.ports if isinstance(p, LumpedPort)]
    if not lumped:
        return None
    mesh = model.mesh
    mu = np.asarray(element_tensors(model, "mu_r"))
    mu = mu.reshape(len(mu), -1)
    active = model._domain.active  # noqa: SLF001
    neighbours = np.asarray(mesh.triangle_neighbours)
    for port in lumped:
        for surface in surfaces_of(port):
            beside = neighbours[mesh.elements(surface)].ravel()
            beside = beside[beside >= 0]
            values = mu[beside[active[beside]]]
            if len(values) and np.any(values != values[0]):
                return port.name
    return None


def _plasmonic_law(
    model: Model, frequencies: list[float]
) -> tuple[str, float, float] | None:
    """The first (region, frequency, Re eps) of a law in `_PLASMONIC_WINDOW`.

    Re eps is the kernel's (`dispersion.eps_at`): eps_inf plus the law's
    susceptibility; a conductivity adds to Im eps only.
    """
    from gfem._engine import dispersion

    low, high = _PLASMONIC_WINDOW
    regions = [lr for lr in dispersion.law_regions(model) if not is_traced(lr.law)]
    for f in frequencies:
        for lr in regions:
            eps = complex(lr.law.eps_inf) + dispersion.susceptibility(
                lr.law, 2 * math.pi * f
            )
            if low < eps.real < high:
                return lr.region, f, eps.real
    return None


def _touching(model: Model, surface: str) -> np.ndarray:
    """Whether each tetrahedron has a node on ``surface``."""
    mesh = model.mesh
    nodes = np.zeros(len(mesh.nodes), bool)
    nodes[np.asarray(mesh.triangles)[mesh.elements(surface)].ravel()] = True
    return nodes[np.asarray(mesh.tetrahedra)].any(axis=1)


def _check_estimable(result: Any, estimator: Estimator, model: Model) -> None:
    if isinstance(estimator, QuantityEstimator) and estimator.objective is None:
        _check_quantity_result(estimator, result)
        _check_quantity_model(estimator, model)
    if not isinstance(result, (ElectrostaticResult, DrivenResult, EigenmodeResult)):
        raise _unsupported_result(result, estimator)
    if any(field._comm is not None for field in _fields(result)):  # noqa: SLF001
        _mpi.refuse_feature("error_estimate")
    if floquet_ports(model.ports):
        raise MeshRefinementUnsupportedError(
            "floquet port",
            f"gfem.{type(estimator).__name__} has no path for a Floquet port: its "
            "flux recovery does not pair the periodic surfaces.",
        )
    if model._periodic:  # noqa: SLF001
        raise MeshRefinementUnsupportedError(
            "periodic",
            f"gfem.{type(estimator).__name__} has no path for a periodic model: "
            "its flux recovery does not pair the periodic surfaces.",
        )
    if isinstance(result, DrivenResult):
        _check_kept(result, estimator)
        from gfem._engine import dispersion

        frequencies = [float(f) for f in result.s.frequency]
        dispersion.check_estimator_frequencies(model, frequencies)
        _check_driven_estimable((estimator,), model, frequencies)
    _check_order(_fields(result)[0].order, estimator)
    mesh = _fields(result)[0].mesh
    if not _is_model_mesh(mesh, model.mesh):
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
    *,
    floor: bool = True,
) -> None:
    """Check the loop's parameters; ``floor``: the solve tolerance's floor too.

    An objective's floor is its sensitivity's (`_check_objective_floors`),
    checked on the first estimate.
    """
    for value, name in ((tolerance, "tolerance"), (error_fraction, "error_fraction")):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a number, got {value!r}.")
    if not (math.isfinite(tolerance) and tolerance > 0):
        raise ValueError(f"tolerance must be a positive number, got {tolerance!r}.")
    if not 0 < error_fraction <= 1:
        raise ValueError(f"error_fraction must be in (0, 1], got {error_fraction!r}.")
    _check_int(max_cycles, "max_cycles", 0)
    if not floor:
        return
    solve_tolerance, floor_name = _solve_tolerance(problem, options)
    if tolerance < 10 * solve_tolerance:
        raise ValueError(
            f"refine_mesh tolerance={tolerance:g} must be at least 10 x "
            f"SolverOptions.{floor_name}={solve_tolerance:g}: the error estimate "
            "does not include the linear solve's error. Lower "
            f"SolverOptions.{floor_name}, or raise tolerance."
        )


def _solve_tolerance(problem: Any, options: SolverOptions) -> tuple[float, str]:
    """The solve's tolerance the estimate does not include, and its option name."""
    if isinstance(problem, Eigenmode):
        return options.eigen_tolerance, "eigen_tolerance"
    return options.tolerance, "tolerance"


def _check_int(value: Any, name: str, low: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {value!r}.")
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}.")
