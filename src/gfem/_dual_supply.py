"""Dual supply: an estimate's solved duals, ridden on the solve or solved here.

A quantity estimate reads the solved duals of what it measures: the
S-functional adjoints of the ports whose row needs one, the duals of field
measurement outputs, of electrostatic energies and potential means, and the
mode-field duals of participations. `refine_mesh` asks each solve to ride
those its estimates read (`RiddenDuals`, the run's ledger); each mesh's
estimates read them from one frozen `DualSupply`, which solves here, on the
finished result, those that did not ride. `error_estimate` uses the empty
`DualSupply()`: nothing rode, every solved dual is solved here.
"""

from __future__ import annotations

import dataclasses
import math
import types
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from gfem.model import WavePort
from gfem.options import SolverOptions
from gfem.problems import Driven, Eigenmode, Electrostatic

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from gfem.model import Model
    from gfem.refinement import ErrorEstimate, Estimator, _Tracks
    from gfem.results import DrivenResult, EigenmodeResult, ElectrostaticResult


@dataclasses.dataclass(frozen=True)
class DualNeed:
    """The solved duals one estimate reads (free duals excluded)."""

    #: The ports whose S-functional adjoints it reads.
    ports: tuple[str, ...] = ()
    #: The keys of its field-output duals, or of its electrostatic duals
    #: (`gfem._engine.electrostatic.objective_loads`).
    keys: tuple[tuple[Any, ...], ...] = ()
    #: The (mode, region) participations whose mode-field duals it reads,
    #: by mode on the first mesh.
    modes: tuple[tuple[int, str], ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class DualSupply:
    """One mesh's solved duals: those that rode on its solve, the rest solved here.

    ``asked`` is what the solve was asked to ride, ``reply`` what came back
    (`gfem.solve._solve`'s adjoints). Each family has its own test of what
    rode, and solves the rest on the finished result: one more factorization
    each (`solved` records these calls).
    """

    asked: DualNeed = DualNeed()
    reply: Mapping[Any, Any] = types.MappingProxyType({})
    _solved: dict[str, list] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def solved(self) -> Mapping[str, tuple]:
        """The duals solved here, after the fact, in call order.

        By family (``"s"``, ``"field"``, ``"potential"``, ``"mode"``): the
        ports, the keys, the keys, or the mode of each post-hoc solve.
        """
        return types.MappingProxyType({k: tuple(v) for k, v in self._solved.items()})

    def _record(self, family: str, missing: Any) -> None:
        self._solved.setdefault(family, []).append(missing)

    def s_adjoints(
        self,
        model: Model,
        result: DrivenResult,
        ports: tuple[str, ...],
        threads: int | None,
    ) -> dict[tuple[float, Hashable], np.ndarray]:
        """The S-functional adjoints of ``ports``, by ``(frequency, port)``.

        Those of the ports asked of the solve rode (a port asked but not in
        the reply at a frequency needs none there); the others are solved
        here (`_adjoints`).
        """
        missing = tuple(p for p in ports if p not in self.asked.ports)
        solved: dict[Any, np.ndarray] = {
            (f, p): z for (f, p), z in self.reply.items() if p in ports
        }
        if missing:
            self._record("s", missing)
            solved.update(_adjoints(model, result, missing, threads))
        return solved

    def field_duals(
        self,
        model: Model,
        result: DrivenResult,
        keys: list[Any],
        threads: int | None,
    ) -> dict[Any, np.ndarray]:
        """The duals of field measurement outputs ``keys``, by ``(frequency, key)``.

        Those asked of the solve and in its reply rode; the others are
        solved here, one more factorization per frequency read
        (`_field_duals`).
        """
        solved: dict[Any, np.ndarray] = {
            (k[1], k): self.reply[k[1], k]
            for k in keys
            if k in self.asked.keys and (k[1], k) in self.reply
        }
        missing = [k for k in keys if (k[1], k) not in solved]
        if missing:
            self._record("field", missing)
            solved.update(_field_duals(model, result, missing, threads))
        return solved

    def potential_duals(
        self,
        model: Model,
        result: ElectrostaticResult,
        keys: list[Any],
        threads: int | None,
    ) -> dict[Any, np.ndarray]:
        """The electrostatic duals of ``keys``, by key.

        Those asked of the solve and in its reply rode; the others are
        solved here, one more factorization
        (`gfem._engine.electrostatic.solve_duals`).
        """
        from gfem._engine import electrostatic

        solved: dict[Any, np.ndarray] = {
            k: self.reply[k] for k in keys if k in self.asked.keys and k in self.reply
        }
        missing = [k for k in keys if k not in solved]
        if missing:
            self._record("potential", missing)
            loads = electrostatic.objective_loads(missing)
            solved.update(electrostatic.solve_duals(model, result, loads, threads))
        return solved

    def mode_duals(
        self,
        model: Model,
        result: EigenmodeResult,
        mode: int,
        regions: list[str],
        threads: int | None,
    ) -> dict[str, Any]:
        """The participation duals of ``mode`` in ``regions``, by region.

        Those in the reply solved for the same mode (`_rides`) rode; the
        others are solved here on an operator prepared at the mode
        (`_participation_duals`).
        """
        omega = np.asarray(result.omega)
        ridden = {
            name: self.reply.get((mode, name))
            for name in regions
            if _rides(self.reply.get((mode, name)), omega[mode])
        }
        missing = [name for name in regions if name not in ridden]
        if missing:
            self._record("mode", mode)
            ridden |= _participation_duals(model, result, mode, missing, threads)
        return ridden


class RiddenDuals:
    """The duals that ride on `refine_mesh`'s solves.

    On a driven problem, the ports whose S-functional adjoints ride: an S
    entry's from the first solve; an objective's once its estimate has read
    them (`add`), from the next solve on. On
    an electrostatic problem, an objective's energy and potential-mean
    duals, the same way.
    """

    def __init__(
        self, estimators: tuple[Estimator, ...], model: Model, problem: Any
    ) -> None:
        self.ports = _adjoint_port_union(estimators, model, problem)
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
    def need(self) -> DualNeed:
        """What rides: the ports, the field or electrostatic keys, the modes."""
        return DualNeed(self.ports, self.keys, self.modes)

    def supply(self, adjoints: Mapping[Any, Any] | None) -> DualSupply:
        """The supply of the mesh whose solve replied ``adjoints``.

        ``None`` (the solve was asked nothing) leaves nothing asked.
        """
        return DualSupply(
            asked=self.need if adjoints is not None else DualNeed(),
            reply=types.MappingProxyType(adjoints or {}),
        )

    def add(self, estimates: tuple[ErrorEstimate, ...]) -> None:
        """Add the duals the estimates' objectives read."""
        read = (p for e in estimates for p in e._duals.ports)  # noqa: SLF001
        self.ports = tuple(dict.fromkeys([*self.ports, *read]))
        keys = (k for e in estimates for k in e._duals.keys)  # noqa: SLF001
        self.keys = tuple(dict.fromkeys([*self.keys, *keys]))
        modes = (m for e in estimates for m in e._duals.modes)  # noqa: SLF001
        self.modes = tuple(dict.fromkeys([*self.modes, *modes]))


def _adjoint_ports(
    estimator: Estimator, model: Model, excited: list[str]
) -> tuple[str, ...]:
    """The port whose S-functional adjoint an S estimate may need.

    ``to`` of ``QuantityEstimator(s=(to, source))`` when it is not among
    ``excited`` (it has no field of its own), or when it is a wave port
    touching a lossy element: its mode may then not be parallel to its
    excitation (the solve checks), and its adjoint is not its own field.
    """
    from gfem.refinement import QuantityEstimator

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
    from gfem._estimate_admission import _lossy_elements, _touching

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


def _adjoints(
    model: Model, result: DrivenResult, ports: tuple[str, ...], threads: int | None
) -> dict[tuple[float, Hashable], np.ndarray]:
    """The adjoints of ``ports`` on a finished result: one more factorization.

    Only at the frequencies where one is not its port's own field scaled.
    The result does not record its options: the direct solver, which the
    estimate's few digits need no iterative tolerance for.
    """
    from gfem._engine import driven

    frequencies = list(result.s.frequency)
    loads = driven.port_loads(ports)
    return _finished_driven(
        model, result, loads, frequencies, solutions=None, threads=threads
    )


def _field_duals(
    model: Model, result: DrivenResult, keys: list[Any], threads: int | None
) -> dict[tuple[float, Hashable], np.ndarray]:
    """The duals of field measurement outputs on a finished result.

    One more factorization per frequency read, with the direct solver at the
    result's order, the loads formed from the result's fields there.
    """
    from gfem._engine import field_duals

    frequencies = sorted({k[1] for k in keys})
    excite = list(result.s.excited)
    fields = dict(zip(result.kept, result.e_fields, strict=True))
    solutions = {
        f: np.stack([np.asarray(fields[f, p].coefficients) for p in excite], axis=1)
        for f in frequencies
    }
    loads = field_duals.field_loads(keys)
    return _finished_driven(
        model, result, loads, frequencies, solutions=solutions, threads=threads
    )


def _finished_driven(
    model: Model,
    result: DrivenResult,
    loads: Any,
    frequencies: list[Any],
    *,
    solutions: Mapping[float, np.ndarray] | None,
    threads: int | None,
) -> dict[tuple[float, Hashable], np.ndarray]:
    """The adjoints of ``loads`` at ``frequencies`` on a finished driven result.

    With the direct solver at the result's order, its warnings silenced: the
    solve of ``result`` warned.
    """
    from gfem._engine import driven
    from gfem.refinement import _fields

    problem = Driven(frequencies, excite=list(result.s.excited))
    order = _fields(result)[0].order
    options = SolverOptions(order=order, linear_solver="direct", threads=threads)
    with driven.low_frequency_deferred(), warnings.catch_warnings():
        for category in driven.FORWARD_WARNINGS:  # the solve of ``result`` warned
            warnings.simplefilter("ignore", category)
        rule = driven.DrivenRule(model, problem, options)
        return rule.solve_adjoints(model, loads, solutions)


def _rides(dual: Any, omega: complex) -> bool:
    """Whether a ridden ``dual`` was solved for the mode at ``omega``."""
    return dual is not None and abs(dual.omega - omega) <= 1e-12 * abs(omega)


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
