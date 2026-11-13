"""The loads of the mode-field adjoint for the eigenmode ratios (ADR 0102).

The ratios on an `EigenmodeResult` (``q_port``, ``participation``,
``bulk_participation``, ``principal_participation``,
``sheet_participation``) are scale- and phase-invariant functions of a
mode ``(k, u)``: quadratic forms of ``u`` over the electric energy

    E = 1/2 eps0 int Re(eps_r) |u|^2 + sum_ports C_p |V_p|^2 / 2
        + sum_sheets 1/2 eps0 scale u^H A_C u

(``V_p = l_p^T u`` the port voltage, ``A_C`` a capacitive sheet's C piece,
ADR 0103), and ``Re k``. A sheet's participation is its L piece's form
``u^H A_L u / (2 mu0 scale Re(omega)^2 E)``. A real objective
``G = sum ct g`` of them gives the load of quantum-metrics ticket 03: its
Wirtinger partial ``w = dG/du`` (``dG = 2 Re(w^T du + kappa dk) + ...``),
``kappa = dG/dk``, and the explicit terms, the derivative of ``G`` by each
energy part at fixed ``u``. Scale and phase invariance give ``u^T w = 0``, so
the normalization multipliers vanish and the adjoint's right-hand side is
``w`` itself.

The explicit part is contracted by the sensitivity kernel through one
`gfem._engine.sensitivity.energy_term` per region (``Re(eps_r)`` there) and
per principal axis of an anisotropic region, weighted by ``1/2 eps0 scale``
times ``dG`` by that part's energy: a pair ``(conj(u), u)``. Every element
lies in one region, so the whole energy's derivative is spread over the
region terms. The sheets' L and C pieces are explicit too, through the
same pair weighted by piece name: the kernel's sheet pass gives their fill
derivative and its node pass their transport, as for the operator.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping

import numpy as np
import scipy.constants as const

from gfem._engine import terms
from gfem._engine.driven import DrivenOperator
from gfem._engine.eigen import energy_parts
from gfem._engine.sensitivity import Term, energy_term
from gfem.errors import NonDifferentiableError
from gfem.model import Model

__all__ = ["Load", "Ratios", "energy_name", "sheet_capacitors", "sheet_inductors"]

#: The prefix of the explicit energy terms' names (no piece has it).
_ENERGY = "mode energy"


def energy_name(region: str, axis: int | None = None) -> str:
    """The name of a region's (or one of its principal axes') energy term."""
    if axis is None:
        return f"{_ENERGY} {region}"
    return f"{_ENERGY} {region} axis {axis}"


def sheet_capacitors(op: DrivenOperator) -> list[int]:
    """The index of each capacitive `Impedance`'s C piece in ``op``."""
    return [
        k
        for k, piece in enumerate(op.pieces)
        if piece.lumped is not None
        and piece.lumped.kind[0] == "boundary"
        and piece.lumped.kind[2] == "capacitance"
    ]


def sheet_inductors(op: DrivenOperator, surfaces: tuple[str, ...]) -> list[int]:
    """The index of each sheet's L piece (``"impedance <surface> L"``) in ``op``."""
    index = {piece.name: k for k, piece in enumerate(op.pieces)}
    return [index[f"impedance {surface} L"] for surface in surfaces]


@dataclasses.dataclass(frozen=True)
class Load:
    """What one mode's cotangents ask of the adjoint.

    Attributes:
        w: ``dG/du`` (Wirtinger, unconjugated pairing) on the solve space.
        kappa: ``dG/dk``.
        energy: The explicit terms: the weight of each energy term
            (`energy_name`) and sheet piece (by name) on the pair
            ``(conj(u), u)``.
    """

    w: np.ndarray
    kappa: complex
    energy: dict[str, float]


class Ratios:
    """The ratio outputs' forms on one driven operator, and their loads."""

    def __init__(
        self,
        model: Model,
        op: DrivenOperator,
        *,
        regions: tuple[str, ...],
        anisotropic: tuple[str, ...],
        resistive: tuple[str, ...],
        inductive: tuple[str, ...],
        kinetic: tuple[str, ...] = (),
    ) -> None:
        built = op.built
        self.regions = regions
        self.anisotropic = anisotropic
        self.resistive = resistive
        self.inductive = inductive
        self.kinetic = kinetic
        #: J per file-unit integral of ``Re(eps_r) |u|^2``.
        self.unit = 0.5 * const.epsilon_0 * built.scale
        #: rad/s per unit of ``k``.
        self.per_k = const.c / built.scale
        self.parts, self.axis_parts = energy_parts(model, op, regions, anisotropic)
        self._op = op
        self.rlc = {data.port.name: data.port.rlc for data in op.ports}
        #: The capacitive sheets' C pieces and the ``kinetic`` sheets' L
        #: pieces: indices in ``op.pieces``.
        self.capacitors = sheet_capacitors(op)
        self.inductors = sheet_inductors(op, kinetic)
        #: The L pieces' form per unit of ``1 / (2 mu0 scale)``.
        self.per_form = 1 / (2 * const.mu_0 * built.scale)

    def apply(self, piece: int, u: np.ndarray) -> np.ndarray:
        """``A u`` of ``op``'s piece ``piece``."""
        return self._op.apply(piece, u)

    def piece_name(self, piece: int) -> str:
        """The name of ``op``'s piece ``piece`` (its key in the pairs' weights)."""
        return self._op.pieces[piece].name

    @functools.cached_property
    def voltage(self) -> dict[str, np.ndarray]:
        """Each lumped port's voltage functional ``l`` (built on first use)."""
        op = self._op
        return {
            data.port.name: terms.port_vectors(op.space, op.built, op.order, data)[1]
            for data in op.ports
        }

    def terms(self) -> list[Term]:
        """The explicit energy terms, one per region and per principal axis."""
        out = [
            energy_term(energy_name(region), part.coefficient, part.elements)
            for region, part in zip(self.regions, self.parts, strict=True)
        ]
        for region, parts in zip(self.anisotropic, self.axis_parts, strict=True):
            out += [
                energy_term(
                    energy_name(region, part.axis),
                    part.coefficient,
                    part.elements,
                    axis=part.axis,
                )
                for part in parts
            ]
        return out

    def voltages(self, u: np.ndarray) -> dict[str, complex]:
        """Each lumped port's voltage ``l^T u``."""
        return {name: complex(v @ u) for name, v in self.voltage.items()}

    def load(
        self, u: np.ndarray, k: complex, ct: Mapping[str, np.ndarray], *, mode: int
    ) -> Load:
        """The load of the cotangents ``ct`` on the mode ``(k, u)``.

        Args:
            u: The mode on the solve space (any scale: the ratios are
                invariant).
            k: Its eigenvalue.
            ct: This mode's cotangents by output: ``"q_port"`` ``(n_resistive,)``,
                ``"participation"`` ``(n_inductive,)``, ``"bulk"``
                ``(n_regions,)``, ``"principal"`` ``(n_anisotropic, 3)``,
                ``"sheet"`` ``(n_kinetic,)``; a missing output carries none.
            mode: The mode's index, for the error message.

        Raises:
            NonDifferentiableError: A nonzero cotangent on an infinite
                ``q_port`` (no voltage on the port).
        """
        mode_state = _Mode(self, u, k)
        partials = _Partials(self)
        bulk = ct.get("bulk")
        if bulk is not None:
            for r, c in enumerate(bulk):
                partials.share(c, mode_state, r, mode_state.region_energy[r])
        principal = ct.get("principal")
        if principal is not None and np.any(principal):
            mode_state.apply_axes(self)
            for (r, a), share in np.ndenumerate(mode_state.axis_energy):
                partials.share(principal[r, a], mode_state, (r, a), share)
        for c, name in zip(ct.get("q_port", ()), self.resistive, strict=False):
            if c:
                partials.q_port(c, mode_state, name, index=mode)
        for c, name in zip(ct.get("participation", ()), self.inductive, strict=False):
            if c:
                partials.participation(c, mode_state, name)
        for s, c in enumerate(ct.get("sheet", ())):
            if c:
                partials.sheet(c, mode_state, s)
        return partials.load(mode_state)


class _Mode:
    """One mode's quadratic forms: port voltages, region and axis energies."""

    def __init__(self, ratios: Ratios, u: np.ndarray, k: complex) -> None:
        self.u = u
        self.k = k
        self.volts = ratios.voltages(u)
        self.region_applied = [part.apply(u) for part in ratios.parts]
        self.region_energy = np.array(
            [ratios.unit * float(np.real(np.conj(u) @ a)) for a in self.region_applied]
        )
        self.capacitors = {
            name: rlc[2] for name, rlc in ratios.rlc.items() if rlc[2] is not None
        }
        #: The capacitive sheets' C pieces on the mode, and their energies.
        self.sheet_applied = [ratios.apply(piece, u) for piece in ratios.capacitors]
        self.sheet_energy = [
            ratios.unit * float(np.real(np.conj(u) @ a)) for a in self.sheet_applied
        ]
        #: The kinetic sheets' L pieces on the mode, and their forms.
        self.inductor_applied = [ratios.apply(piece, u) for piece in ratios.inductors]
        self.inductor_form = [
            float(np.real(np.conj(u) @ a)) for a in self.inductor_applied
        ]
        #: The mode energy, J.
        self.energy = (
            self.region_energy.sum()
            + sum(
                0.5 * c * abs(self.volts[name]) ** 2
                for name, c in self.capacitors.items()
            )
            + sum(self.sheet_energy)
        )
        self.omega = k.real * ratios.per_k
        self.axis_applied: list[list[np.ndarray]] = [[] for _ in ratios.anisotropic]
        self.axis_energy = np.zeros((len(ratios.anisotropic), 3))

    def apply_axes(self, ratios: Ratios) -> None:
        """The principal axes' parts on the mode, and their energies."""
        for r, parts in enumerate(ratios.axis_parts):
            self.axis_applied[r] = [part.apply(self.u) for part in parts]
            self.axis_energy[r] = [
                ratios.unit * float(np.real(np.conj(self.u) @ a))
                for a in self.axis_applied[r]
            ]


class _Partials:
    """``dG`` by the energy, each region's and axis's part, each ``|V|^2``, ``k``."""

    def __init__(self, ratios: Ratios) -> None:
        self.ratios = ratios
        self.energy = 0.0
        self.region = np.zeros(len(ratios.regions))
        self.axis = np.zeros((len(ratios.anisotropic), 3))
        self.volts2 = dict.fromkeys(ratios.voltage, 0.0)
        self.inductor = np.zeros(len(ratios.kinetic))
        self.kappa = 0.0

    def share(self, c: float, mode: _Mode, where: int | tuple, value: float) -> None:
        """``c`` times a share ``value / E`` of a region (or axis ``where``)."""
        if not c:
            return
        if isinstance(where, tuple):
            self.axis[where] += c / mode.energy
        else:
            self.region[where] += c / mode.energy
        self.energy -= c * value / mode.energy**2

    def q_port(self, c: float, mode: _Mode, name: str, *, index: int) -> None:
        """``c`` times ``q = omega 2 R E / |V|^2`` (mode ``index``, for messages)."""
        v2 = abs(mode.volts[name]) ** 2
        if v2 == 0:
            raise NonDifferentiableError(
                f"NonDifferentiableError: q_port[{name!r}] of mode {index} is "
                "infinite: the mode puts no voltage on the port, and an infinite "
                f"value has no derivative. Differentiate 1 / q_port[{name!r}] "
                "instead, whose value there is an exact 0 (and the sum of 1 / Q "
                "terms is what a loss objective wants anyway)."
            )
        q = mode.omega * 2 * self.ratios.rlc[name][0] * mode.energy / v2
        self.energy += c * q / mode.energy
        self.volts2[name] -= c * q / v2
        self.kappa += c * q / (2 * mode.k.real)

    def participation(self, c: float, mode: _Mode, name: str) -> None:
        """``c`` times ``p = sign(Re V) |V|^2 / (2 omega^2 L E)``."""
        sign = math.copysign(1.0, mode.volts[name].real)
        inductance = self.ratios.rlc[name][1]
        factor = sign / (2 * mode.omega**2 * inductance * mode.energy)
        g = factor * abs(mode.volts[name]) ** 2
        self.energy -= c * g / mode.energy
        self.volts2[name] += c * factor
        self.kappa -= c * g / mode.k.real

    def sheet(self, c: float, mode: _Mode, s: int) -> None:
        """``c`` times sheet ``s``'s ``p = u^H A_L u / (2 mu0 scale omega^2 E)``."""
        factor = self.ratios.per_form / (mode.omega**2 * mode.energy)
        g = factor * mode.inductor_form[s]
        self.energy -= c * g / mode.energy
        self.inductor[s] += c * factor
        self.kappa -= c * g / mode.k.real

    def load(self, mode: _Mode) -> Load:
        """The load: ``w``, ``kappa`` and the energy terms' weights.

        ``w = dG/du``, with ``d(u^H A u)/du = conj(A u)`` (``A`` real
        symmetric) and ``d|V|^2/du = conj(V) l``.
        """
        ratios = self.ratios
        w = np.zeros_like(mode.u, dtype=complex)
        weights = {}
        whole = self.region + self.energy
        for region, applied, c in zip(
            ratios.regions, mode.region_applied, whole, strict=True
        ):
            if c:
                w += c * ratios.unit * np.conj(applied)
                weights[energy_name(region)] = c * ratios.unit
        for (r, a), c in np.ndenumerate(self.axis):
            if c:
                w += c * ratios.unit * np.conj(mode.axis_applied[r][a])
                weights[energy_name(ratios.anisotropic[r], a)] = c * ratios.unit
        # the energy's sheet terms, and the sheets' L forms
        sheets = [
            (piece, applied, self.energy * ratios.unit)
            for piece, applied in zip(
                ratios.capacitors, mode.sheet_applied, strict=True
            )
        ] + list(
            zip(ratios.inductors, mode.inductor_applied, self.inductor, strict=True)
        )
        for piece, applied, c in sheets:
            if c:
                w += c * np.conj(applied)
                weights[ratios.piece_name(piece)] = c
        volts2 = dict(self.volts2)
        for name, c in mode.capacitors.items():
            volts2[name] += self.energy * 0.5 * c
        for name, c in volts2.items():
            if c:
                w += c * np.conj(mode.volts[name]) * ratios.voltage[name]
        return Load(w=w, kappa=complex(self.kappa), energy=weights)
