r"""Cell outputs of a periodic cell: ports, efficiencies and the power balance.

The public functions are re-exported by `gfem.diffraction`. The private
helpers (`cell_of`, `incidence_of`, `check_complete`, `stack`, ...) are shared
with the other cell outputs: they read only the static labels of a result and
`jax.numpy` of ``result.s.values``, so every output differentiates through S.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import check_fixed, is_traced
from gfem._pytree import register_fields
from gfem._text import hertz, hertz_band
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gfem._boundaries import FloquetPort
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = [
    "Efficiencies",
    "PowerBalance",
    "efficiencies",
    "floquet_ports",
    "power_balance",
]

_NAMED = ("TE", "TM", "RCP", "LCP", "unpolarized")
_SQRT_HALF = math.sqrt(0.5)


# --- floquet_ports ---------------------------------------------------------------


def floquet_ports(
    model: Model,
    surface: str,
    frequency: ArrayLike,
    *,
    wave_vector: ArrayLike | None = None,
    wave_vector_frequency: float | None = None,
    offset: float = 0.0,
) -> list[FloquetPort]:
    r"""A `gfem.FloquetPort` for every order that propagates behind a face.

    One port per diffraction order that propagates in the medium behind
    ``surface`` at some of the frequencies (`gfem.Model.propagating_orders`),
    in TE then TM, sorted as `gfem.Model.propagating_orders` sorts the orders,
    all at ``offset``. Declaring every propagating order is what
    `power_balance` needs: the power of an order without a port is missing
    from S. The names read ``"top (0, 0) TE"``. The model needs the periodic
    pairs only, not Floquet ports yet.

    With ``wave_vector_frequency`` the wave vector is scaled with the
    frequency, :math:`k_t(f) = k_t f / f_0`, as `gfem.Driven` does (a
    fixed-angle sweep); an order that opens inside the band is declared. A
    sequence of wave vectors (shape ``(n, 3)``, e.g. one per angle of an angle
    sweep) declares the union of the orders over all of them.

    Args:
        model: A periodic cell (`gfem.Model`).
        surface: An open face of the cell.
        frequency: One frequency or several, Hz.
        wave_vector: One 3-vector or a sequence of them, shape ``(n, 3)``,
            rad/m (`gfem.Model.bloch_wave_vector`); ``None`` for normal
            incidence.
        wave_vector_frequency: The frequency in Hz that ``wave_vector`` is
            given at, to scale it with the frequency; ``None`` keeps it
            fixed.
        offset: The ports' de-embedding distance, m.

    Returns:
        The ports, for ``model.replace(ports=[...])``.

    Raises:
        ValueError: ``wave_vector_frequency`` without ``wave_vector``, or not
            a positive frequency; a wave vector of another shape; and what
            `gfem.Model.propagating_orders` raises.
        NonDifferentiableError: An argument is traced by JAX.
    """
    from gfem._boundaries import FloquetPort
    from gfem._floquet import sort_orders

    who = "floquet_ports"
    if wave_vector is None and wave_vector_frequency is not None:
        raise ValueError(
            "floquet_ports: wave_vector_frequency scales wave_vector, but "
            "wave_vector is None."
        )
    check_fixed(wave_vector, f"{who}: wave_vector must be fixed numbers")
    check_fixed(
        wave_vector_frequency, f"{who}: wave_vector_frequency must be a fixed number"
    )
    check_fixed(frequency, f"{who}: frequency must be fixed numbers")
    f = np.atleast_1d(np.asarray(frequency, float)).ravel()
    found: set[tuple[int, int]] = set()
    if wave_vector is None:
        found |= set(model.propagating_orders(surface, f))
    else:
        k = _wave_vectors(wave_vector)
        if wave_vector_frequency is None:
            for w in k:
                found |= set(model.propagating_orders(surface, f, wave_vector=w))
        else:
            f0 = wave_vector_frequency
            ok = isinstance(f0, (int, float, np.integer, np.floating)) and not (
                isinstance(f0, (bool, np.bool_))
            )
            if not ok or not math.isfinite(f0) or f0 <= 0:
                raise ValueError(
                    f"{who}: wave_vector_frequency must be a positive finite "
                    "frequency in Hz, or None for the same wave vector at every "
                    f"frequency; got {f0!r}."
                )
            for w in k:
                for fi in f:
                    found |= set(
                        model.propagating_orders(
                            surface, [fi], wave_vector=w * (fi / f0)
                        )
                    )
    return [
        FloquetPort(
            f"{surface} ({m}, {n}) {pol}",
            surface,
            order=(m, n),
            polarization=pol,
            offset=offset,
        )
        for m, n in sort_orders(found)
        for pol in ("TE", "TM")
    ]


def _wave_vectors(wave_vector: ArrayLike) -> np.ndarray:
    """``wave_vector`` as an ``(n, 3)`` float array of wave vectors."""
    try:
        k = np.asarray(wave_vector, float)
    except (TypeError, ValueError):  # ragged: no float array, report its outer shape
        k = np.empty(np.shape(np.asarray(wave_vector, dtype=object)), dtype=object)
    shape = k.shape
    if shape == (3,):
        k = k[None]
    if k.dtype == object or k.ndim != 2 or k.shape[1] != 3 or not len(k):
        raise ValueError(
            "floquet_ports: wave_vector must be one 3-vector or a sequence of "
            f"them, shape (n, 3), in rad/m, got shape {shape}."
        )
    return k


# --- result types ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Efficiencies:
    r"""The efficiency of each diffraction order for one incidence.

    The efficiency of a row is :math:`|S|^2` of the incident column (or
    combination), the fraction of the incident power, per cell, that the
    order carries away. Non-propagating rows read 0.

    Attributes:
        frequency: ``(n_freq,)`` Hz.
        values: ``(..., n_freq, n_rows)`` real, differentiable like S; a
            leading axis for an angle sweep.
        rows: ``(surface, order, polarization)`` of each last-axis entry:
            ``"TE"``/``"TM"`` in the linear basis, ``"RCP"``/``"LCP"`` in the
            circular one.
        incidence: ``(surface, order, polarization)`` of the incident wave,
            the polarization as given (a Jones pair as a normalized complex
            pair).

    ``eff[surface, order, pol]`` is one row, ``(..., n_freq)``;
    ``eff[surface, order]`` the sum of the order's two polarization rows.
    """

    frequency: np.ndarray
    values: jax.Array
    rows: tuple[tuple[str, tuple[int, int], str], ...]
    incidence: tuple[str, tuple[int, int], Any]

    def __getitem__(self, key: Any) -> jax.Array:
        """One efficiency row, or an order's total over its two polarizations.

        Raises:
            KeyError: No such row, or the order's other polarization has no
                row.
        """
        rows = self.rows
        if isinstance(key, tuple) and len(key) == 3:
            surface, order, pol = key
            try:
                order = tuple(order)
            except TypeError:
                order = key[1]
            if (surface, order, pol) in rows:
                return self.values[..., rows.index((surface, order, pol))]
        elif isinstance(key, tuple) and len(key) == 2:
            surface, order = key
            try:
                order = tuple(order)
            except TypeError:
                order = key[1]
            found = {r[2]: i for i, r in enumerate(rows) if r[:2] == (surface, order)}
            if found:
                pair = ("RCP", "LCP") if "RCP" in found else ("TE", "TM")
                for pol, other in (pair, pair[::-1]):
                    if pol not in found:
                        raise KeyError(
                            f"Order {order} on {surface!r} has no {pol} row, so its "
                            "total over both polarizations is not known; read "
                            f"eff[{surface!r}, {order}, {other!r}] or add the "
                            "missing Floquet port."
                        )
                return (
                    self.values[..., found[pair[0]]] + self.values[..., found[pair[1]]]
                )
        raise KeyError(
            f"No efficiency row {key!r}; the rows are {', '.join(map(repr, rows))}."
        )

    def __repr__(self) -> str:
        return (
            f"Efficiencies(incidence {self.incidence}, {len(self.frequency)} "
            f"frequencies, {len(self.rows)} rows)"
        )


register_fields(Efficiencies, ("values",))


@dataclasses.dataclass(frozen=True, eq=False)
class PowerBalance:
    r"""The reflected, transmitted, coupled and absorbed power for one incidence.

    Fractions of the incident power. ``absorptance`` is
    :math:`1 - R - T - \text{coupled}` exactly.

    Attributes:
        frequency: ``(n_freq,)`` Hz.
        reflectance: ``(..., n_freq)``: the sum of every Floquet row on the
            incident surface.
        transmittance: ``(..., n_freq)``: the sum of every Floquet row on
            any other Floquet surface (0 on a one-face cell).
        coupled: ``(..., n_freq)``: the power into the lumped and wave ports.
        absorptance: ``(..., n_freq)``.
        incidence: ``(surface, order, polarization)`` of the incident wave.
    """

    frequency: np.ndarray
    reflectance: jax.Array
    transmittance: jax.Array
    coupled: jax.Array
    absorptance: jax.Array
    incidence: tuple[str, tuple[int, int], Any]

    def __repr__(self) -> str:
        return (
            f"PowerBalance(incidence {self.incidence}, {len(self.frequency)} "
            "frequencies)"
        )


register_fields(
    PowerBalance, ("reflectance", "transmittance", "coupled", "absorptance")
)


# --- shared pieces ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class Cell:
    """What a cell output reads of its results and model (static labels)."""

    who: str
    results: tuple[Any, ...]
    sequence: bool
    model: Any
    #: The result's Floquet port names, in S's row order.
    floquet: tuple[str, ...]
    #: The result's other port names (lumped, wave), in S's row order.
    others: tuple[str, ...]
    info: dict[str, Any]
    offsets: dict[str, float]
    _directions: dict[str, int] = dataclasses.field(default_factory=dict)

    @property
    def s(self) -> Any:
        return self.results[0].s

    @property
    def frequency(self) -> np.ndarray:
        return np.asarray(self.s.frequency, float)

    @property
    def surfaces(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.info[n].surface for n in self.floquet))

    def face(self, surface: str) -> Any:
        return self.model._face(surface, self.who)  # noqa: SLF001

    def h_out(self, surface: str) -> int:
        """The direction sign of beams leaving through ``surface``, ±1 along ê3."""
        if surface not in self._directions:
            face = self.face(surface)
            self._directions[surface] = int(np.sign(-face.normal @ face.frame[2]))
        return self._directions[surface]

    def port(self, surface: str, order: tuple[int, int], pol: str) -> str | None:
        """The result's Floquet port reading (surface, order, pol), else ``None``."""
        for name in self.floquet:
            d = self.info[name]
            if (d.surface, d.order, d.polarization) == (surface, order, pol):
                return name
        return None


def unitary(h: int) -> np.ndarray:
    r""":math:`U_h = \tfrac1{\sqrt2}\begin{pmatrix}1&1\\-ih&ih\end{pmatrix}`.

    Columns RCP and LCP, rows TM and TE.
    """
    return np.asarray([[1.0, 1.0], [-1j * h, 1j * h]]) * _SQRT_HALF


def _member_key(r: Any) -> tuple[Any, ...]:
    return (
        r.s.ports,
        r.s.excited,
        tuple(r.s.frequency),
        tuple(
            (n, d.surface, d.order, d.polarization)
            for n, d in r.diffraction_orders.items()
        )
        + tuple(sorted(dict(r.offsets).items())),
    )


def cell_of(result: Any, model: Any, who: str) -> Cell:
    """Validate ``result`` and ``model`` and read the Floquet labels (M1-M4, M13).

    Raises:
        TypeError: ``result`` is not a `DrivenResult` or a sequence of them;
            ``model`` is not a `Model`.
        ValueError: The result has no Floquet port, the sequence is empty or
            its members differ, or ``model`` is not the solved model.
    """
    from gfem._boundaries import FloquetPort
    from gfem.model import Model
    from gfem.results import DrivenResult

    sequence = isinstance(result, Sequence) and not isinstance(result, str)
    members = tuple(result) if sequence else (result,)
    if sequence and not members:
        raise ValueError(f"{who}: the sequence of results is empty.")
    for r in members:
        if not isinstance(r, DrivenResult):
            raise TypeError(
                f"{who} needs a gfem.DrivenResult or a sequence of them (an angle "
                f"sweep), got {type(result if not sequence else r).__name__}."
            )
    if not isinstance(model, Model):
        raise TypeError(f"{who} needs model= a gfem.Model, got {type(model).__name__}.")
    first = members[0]
    if not first.diffraction_orders:
        raise ValueError(
            f"{who}: the result has no Floquet port; {who} reads the "
            "gfem.FloquetPort rows of S of a periodic cell."
        )
    base = _member_key(first)
    labels = ("ports", "excited ports", "frequencies", "Floquet ports")
    for i, r in enumerate(members[1:], start=1):
        key = _member_key(r)
        for what, a, b in zip(labels, base, key, strict=True):
            if a != b:
                raise ValueError(
                    f"{who}: the results of a sequence must share their ports, "
                    "excited ports, frequencies and Floquet ports (one angle "
                    f"sweep); result {i} differs from result 0 in its {what}."
                )
    info = dict(first.diffraction_orders)
    in_model = {p.name: p for p in model.ports if isinstance(p, FloquetPort)}
    for name in first.s.ports:
        if name not in info:
            continue
        d = info[name]
        p = in_model.get(name)
        if p is None or (p.surface, p.order, p.polarization) != (
            d.surface,
            d.order,
            d.polarization,
        ):
            where = (
                "the model has no such Floquet port"
                if p is None
                else f"order {p.order} in {p.polarization} on {p.surface!r} "
                "in the model"
            )
            raise ValueError(
                f"{who}: model= is not the model the result was solved with: "
                f"Floquet port {name!r} reads order {d.order} in {d.polarization} "
                f"on {d.surface!r} in the result, but {where}. Pass the model "
                "given to gfem.solve."
            )
    return Cell(
        who=who,
        results=members,
        sequence=sequence,
        model=model,
        floquet=tuple(n for n in first.s.ports if n in info),
        others=tuple(n for n in first.s.ports if n not in info),
        info=info,
        offsets=dict(first.offsets),
    )


def check_order(order: Any, who: str, arg: str = "order") -> tuple[int, int]:
    """``order`` as two ints (M11)."""
    ok = (
        isinstance(order, (list, tuple))
        and len(order) == 2
        and all(
            isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
            for i in order
        )
    )
    if not ok:
        raise TypeError(f"{who}: {arg} must be two integers (m, n), got {order!r}.")
    return (int(order[0]), int(order[1]))


@dataclasses.dataclass(frozen=True)
class Incident:
    """An incidence: weighted coherent combinations of the TM and TE columns."""

    surface: str
    order: tuple[int, int]
    label: Any
    #: ``(weight, (a_TM, a_TE))`` of each coherent combination.
    combos: tuple[tuple[float, tuple[complex, complex]], ...]
    #: The S column index of each polarization of the incident order.
    columns: dict[str, int]

    @property
    def incidence(self) -> tuple[str, tuple[int, int], Any]:
        return (self.surface, self.order, self.label)

    def columns_of(self, s_values: jax.Array) -> list[tuple[float, jax.Array]]:
        """``(weight, column)`` of each combination, columns ``(n_freq, n_ports)``."""
        out = []
        for w, (a_tm, a_te) in self.combos:
            col: Any = 0.0
            if a_tm != 0:
                col = col + a_tm * s_values[:, :, self.columns["TM"]]
            if a_te != 0:
                col = col + a_te * s_values[:, :, self.columns["TE"]]
            out.append((w, col))
        return out


def _pair(polarization: Any, who: str) -> tuple[complex, complex]:
    bad = ValueError(
        f"{who}: polarization must be 'TE', 'TM', 'RCP', 'LCP', 'unpolarized' or "
        f"a Jones pair (a_TM, a_TE), got {polarization!r}."
    )
    if isinstance(polarization, str):
        raise bad
    try:
        arr = np.asarray(polarization, dtype=complex)
    except (TypeError, ValueError):
        raise bad from None
    if arr.shape != (2,):
        raise bad
    if not np.isfinite(arr).all() or not arr.any():
        raise ValueError(
            f"{who}: the Jones pair {polarization!r} must be two finite numbers, "
            "not both zero."
        )
    arr = arr / np.linalg.norm(arr)
    return complex(arr[0]), complex(arr[1])


def _combos(
    label: Any, h_in: int
) -> tuple[tuple[tuple[float, tuple[complex, complex]], ...], tuple[str, ...], bool]:
    """The weighted Jones pairs of an incidence, its polarizations, coherence."""
    one = 1 + 0j
    if label == "TE":
        return ((1.0, (0j, one)),), ("TE",), False
    if label == "TM":
        return ((1.0, (one, 0j)),), ("TM",), False
    if label == "unpolarized":
        return ((0.5, (0j, one)), (0.5, (one, 0j))), ("TM", "TE"), False
    if label in ("RCP", "LCP"):
        u = unitary(h_in)
        c = 0 if label == "RCP" else 1
        return ((1.0, (complex(u[0, c]), complex(u[1, c]))),), ("TM", "TE"), True
    return ((1.0, label),), ("TM", "TE"), True


def incidence_of(
    cell: Cell,
    surface: str,
    polarization: Any,
    order: tuple[int, int],
) -> Incident:
    """Resolve the incident (surface, order, polarization) (M5-M10).

    Raises:
        NonDifferentiableError: ``polarization`` is traced.
        ValueError: An invalid polarization, a surface or port the result
            lacks, a port that was not excited, a pair at two offsets.
    """
    who = cell.who
    if is_traced(polarization):
        raise NonDifferentiableError(
            f"{who}: polarization must be fixed numbers: the incident "
            "polarization is a constant of the measurement. Build the pair "
            "outside jax.grad, jax.jit or jax.vmap; to optimize the incident "
            "polarization, combine the columns of a Jones matrix "
            "(gfem.diffraction.jones(...).values) with your own JAX vector."
        )
    if isinstance(polarization, str):
        if polarization not in _NAMED:
            raise ValueError(
                f"{who}: polarization must be 'TE', 'TM', 'RCP', 'LCP', "
                f"'unpolarized' or a Jones pair (a_TM, a_TE), got {polarization!r}."
            )
        label: Any = polarization
    else:
        pair = _pair(polarization, who)
        label = pair
    surfaces = cell.surfaces
    if surface not in surfaces:
        raise ValueError(
            f"{who}: {surface!r} carries no Floquet port of the result; its "
            f"Floquet surfaces are {', '.join(map(repr, surfaces))}."
        )
    combos, need, coherent = _combos(label, -cell.h_out(surface))
    columns: dict[str, int] = {}
    for pol in need:
        name = cell.port(surface, order, pol)
        if name is None:
            why = (
                ""
                if len(need) == 1
                else f" (polarization={polarization!r} combines the order's TM "
                "and TE columns)"
            )
            raise ValueError(
                f"{who}: no Floquet port reads order {order} in {pol} on "
                f"{surface!r}{why}. Add gfem.FloquetPort(name, {surface!r}, "
                f"order={order}, polarization={pol!r}) to the model."
            )
        if name not in cell.s.excited:
            d = cell.info[name]
            raise ValueError(
                f"{who}: Floquet port {name!r} (order {d.order}, {d.polarization} "
                f"on {d.surface!r}) was not excited, so S has no column for it. "
                f"Solve with Driven(..., excite=None) or add {name!r} to "
                "Driven.excite."
            )
        columns[pol] = cell.s.excited.index(name)
    if coherent:
        check_offsets(cell, surface, order)
    return Incident(surface, order, label, combos, columns)


def check_offsets(cell: Cell, surface: str, order: tuple[int, int]) -> None:
    """Raise unless an order's TM and TE ports share one offset (M8)."""
    a, b = (cell.port(surface, order, p) for p in ("TM", "TE"))
    if a is None or b is None:
        return
    da, db = cell.offsets.get(a, 0.0), cell.offsets.get(b, 0.0)
    if da != db:
        raise ValueError(
            f"{cell.who}: Floquet ports {a!r} and {b!r} read order {order} on "
            f"{surface!r} at different offsets ({da:g} m and {db:g} m), so their "
            "relative phase mixes two reference planes. Give both the same "
            "FloquetPort(offset=...), or align them with result.with_offsets(...)."
        )


def stack(cell: Cell, per_result: list[Any]) -> Any:
    """The per-result outputs, stacked on a leading axis for a sequence."""
    if cell.sequence:
        return jnp.stack(per_result)
    return per_result[0]


# --- completeness ----------------------------------------------------------------


def _runs(idx: list[int], freqs: np.ndarray) -> str:
    out, run = [], [idx[0]]
    for i in idx[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            out.append(run)
            run = [i]
    out.append(run)
    return ", ".join(
        hertz(freqs[r[0]]) if len(r) == 1 else hertz_band(freqs[r[0]], freqs[r[-1]])
        for r in out
    )


def _listing(
    surfaces: tuple[str, ...],
    missing: dict[tuple[str, tuple[int, int], str], set[int]],
    freqs: np.ndarray,
) -> list[str]:
    """``order (m, n) TE and TM on 's' at 200 THz`` for each missing group."""
    from gfem._floquet import sort_orders

    entries = []
    for surface in surfaces:
        for o in sort_orders(o for s, o, _ in missing if s == surface):
            te = missing.get((surface, o, "TE"), set())
            tm = missing.get((surface, o, "TM"), set())
            if te and te == tm:
                groups = [("TE and TM", te)]
            else:
                groups = [(p, s) for p, s in (("TE", te), ("TM", tm)) if s]
            entries += [
                f"order {o} {pols} on {surface!r} at {_runs(sorted(idx), freqs)}"
                for pols, idx in groups
            ]
    return entries


def check_complete(cell: Cell) -> None:
    r"""Raise unless every propagating (order, polarization) has a port.

    Per frequency, at the Bloch wave vector recovered from the declared
    orders: :math:`k_t = \kappa_{(m,n)} - m b_1 - n b_2`.

    Raises:
        NonDifferentiableError: The result's diffraction orders are traced.
        ValueError: A propagating (order, polarization) has no port.
    """
    who = cell.who
    freqs = cell.frequency
    missing: dict[tuple[str, tuple[int, int], str], set[int]] = {}
    for r in cell.results:
        for surface in cell.surfaces:
            ports = [d for d in r.diffraction_orders.values() if d.surface == surface]
            d0 = ports[0]
            if is_traced(d0.wave_vector):
                raise NonDifferentiableError(
                    f"{who}: the result's diffraction orders are traced by JAX "
                    "(the result entered jax.jit or jax.vmap as an argument), so "
                    "the orders that propagate cannot be checked; they are "
                    f"constants of the solve. Call {who} on the result inside the "
                    "function that calls gfem.solve, or outside the transformation."
                )
            b1, b2 = cell.face(surface).reciprocal
            kt = np.asarray(d0.wave_vector, float) - d0.order[0] * b1 - d0.order[1] * b2
            declared = {(d.order, d.polarization) for d in ports}
            for i, f in enumerate(freqs):
                for o in cell.model.propagating_orders(surface, f, wave_vector=kt[i]):
                    for pol in ("TE", "TM"):
                        if (o, pol) not in declared:
                            missing.setdefault((surface, o, pol), set()).add(i)
    if not missing:
        return
    entries = _listing(cell.surfaces, missing, freqs)
    one = len(missing) == 1
    raise ValueError(
        f"{who}: Diffraction {'; '.join(entries)} "
        f"{'propagates' if one else 'propagate'} without a Floquet port: the "
        f"surface absorbs {'it' if one else 'them'}, so "
        f"{'its' if one else 'their'} power is missing from S and 1 - R - T would "
        "count it as absorbed. Add a gfem.FloquetPort for each "
        "(gfem.diffraction.floquet_ports declares every propagating order in TE "
        "and TM), or read the declared orders with gfem.diffraction.efficiencies."
    )


# --- efficiencies and power balance ----------------------------------------------


def efficiencies(
    result: DrivenResult | Sequence[DrivenResult],
    surface: str,
    *,
    polarization: str | tuple[complex, complex],
    model: Model,
    order: tuple[int, int] = (0, 0),
    basis: Literal["linear", "circular"] = "linear",
) -> Efficiencies:
    r"""The efficiency of every declared diffraction order for one incidence.

    Light of ``polarization`` enters the cell through ``surface`` in
    ``order`` (the specular order by default). Each row is :math:`|S|^2` of
    the incident column, or of the combination of the incident TM and TE
    columns with the Jones pair
    :math:`(a_\mathrm{TM}, a_\mathrm{TE})` (in the incident order's Floquet
    references; ``"RCP"`` and ``"LCP"`` resolve through
    :math:`U_{h_\mathrm{in}}`, so on the top face of a cell ``"RCP"`` is
    :math:`(1, +i)/\sqrt2`). ``"unpolarized"`` is the mean of the TE and TM
    columns' efficiencies. In the circular basis each declared order has an
    ``"RCP"`` and an ``"LCP"`` row,
    :math:`b = U_{h_\mathrm{out}}^\dagger (b_\mathrm{TM}, b_\mathrm{TE})`.

    No completeness is needed: a propagating order without a port is not a
    row. A sequence of results (an angle sweep) is stacked on a leading axis.
    Differentiable through S; the polarization, orders and basis are
    constants, and a traced Jones pair raises.

    Args:
        result: A `gfem.DrivenResult` of a model with Floquet ports, or a
            sequence of them.
        surface: The Floquet surface the light enters through.
        polarization: ``"TE"``, ``"TM"``, ``"RCP"``, ``"LCP"``,
            ``"unpolarized"`` or a Jones pair :math:`(a_\mathrm{TM},
            a_\mathrm{TE})`.
        model: The model the result was solved with.
        order: The incident order.
        basis: ``"linear"`` or ``"circular"``.

    Returns:
        An `Efficiencies`.

    Raises:
        TypeError: ``result`` or ``model`` has the wrong type, ``order`` is
            not two integers.
        ValueError: An invalid ``polarization`` or ``basis``, a missing or
            unexcited Floquet port, ports at different offsets where they
            combine, a ``model`` that is not the solved one, an empty or
            inconsistent sequence.
        NonDifferentiableError: ``polarization`` is traced.
    """
    who = "efficiencies"
    cell = cell_of(result, model, who)
    order = check_order(order, who)
    if basis not in ("linear", "circular"):
        raise ValueError(f"{who}: basis must be 'linear' or 'circular', got {basis!r}.")
    inc = incidence_of(cell, surface, polarization, order)
    f_idx = [cell.s.ports.index(n) for n in cell.floquet]
    if basis == "linear":
        rows: list[tuple[str, tuple[int, int], str]] = [
            (cell.info[n].surface, cell.info[n].order, cell.info[n].polarization)
            for n in cell.floquet
        ]
    else:
        groups = dict.fromkeys(
            (cell.info[n].surface, cell.info[n].order) for n in cell.floquet
        )
        for s, o in groups:
            for pol in ("TM", "TE"):
                if cell.port(s, o, pol) is None:
                    raise ValueError(
                        f"{who}: no Floquet port reads order {o} in {pol} on "
                        f"{s!r} (basis='circular' needs both polarizations of "
                        f"every order). Add gfem.FloquetPort(name, {s!r}, "
                        f"order={o}, polarization={pol!r}) to the model."
                    )
            check_offsets(cell, s, o)
        rows = [(s, o, p) for s, o in groups for p in ("RCP", "LCP")]
    per = []
    for r in cell.results:
        total = 0.0
        for w, col in inc.columns_of(r.s.values):
            if basis == "linear":
                part = jnp.abs(col[:, f_idx]) ** 2
            else:
                parts = []
                for s, o in groups:
                    b = jnp.stack(
                        [
                            col[:, cell.s.ports.index(cell.port(s, o, "TM"))],
                            col[:, cell.s.ports.index(cell.port(s, o, "TE"))],
                        ],
                        axis=-1,
                    )
                    circ = b @ jnp.conj(unitary(cell.h_out(s)))
                    parts.append(jnp.abs(circ) ** 2)
                part = jnp.concatenate(parts, axis=-1)
            total = total + w * part
        per.append(total)
    return Efficiencies(
        frequency=cell.frequency,
        values=stack(cell, per),
        rows=tuple(rows),
        incidence=inc.incidence,
    )


def power_balance(
    result: DrivenResult | Sequence[DrivenResult],
    surface: str,
    *,
    polarization: str | tuple[complex, complex],
    model: Model,
    order: tuple[int, int] = (0, 0),
) -> PowerBalance:
    r"""The reflectance, transmittance, coupled power and absorptance of a cell.

    For the incidence of `efficiencies`: ``reflectance`` is the sum of every
    Floquet row on the incident surface, ``transmittance`` the sum over every
    other Floquet surface (0 on a one-face cell), ``coupled``
    :math:`\sum |S|^2` over the lumped and wave port rows (a lumped port
    without resistance carries no power), and

    .. math:: A = 1 - R - T - \mathrm{coupled}

    exactly. ``"unpolarized"`` averages the totals of TE and TM.

    The balance is complete only if every propagating order has ports: the
    power of an order without a port is missing from S and would count as
    absorbed. So at every frequency, at the Bloch wave vector recovered from
    the declared orders, every propagating (order, polarization) of every
    Floquet surface of the result must have a port; grazing orders are not
    required. `gfem.diffraction.floquet_ports` declares them all.

    Args:
        result: A `gfem.DrivenResult` of a model with Floquet ports, or a
            sequence of them (an angle sweep, stacked on a leading axis).
        surface: The Floquet surface the light enters through.
        polarization: As for `efficiencies`.
        model: The model the result was solved with.
        order: The incident order.

    Returns:
        A `PowerBalance`.

    Raises:
        TypeError: As for `efficiencies`.
        ValueError: As for `efficiencies`, and a propagating (order,
            polarization) without a Floquet port.
        NonDifferentiableError: ``polarization`` is traced, or the result's
            diffraction orders are traced (the result entered `jax.jit` or
            `jax.vmap` as an argument).
    """
    who = "power_balance"
    cell = cell_of(result, model, who)
    order = check_order(order, who)
    inc = incidence_of(cell, surface, polarization, order)
    check_complete(cell)
    from gfem._boundaries import LumpedPort

    # a lumped port without resistance has an undefined (NaN) S row: no power
    no_power = {
        p.name for p in model.ports if isinstance(p, LumpedPort) and p.rlc[0] is None
    }
    ports = cell.s.ports
    on = [
        i
        for i, n in enumerate(ports)
        if n in cell.info and cell.info[n].surface == surface
    ]
    off = [
        i
        for i, n in enumerate(ports)
        if n in cell.info and cell.info[n].surface != surface
    ]
    coupled_idx = [
        i for i, n in enumerate(ports) if n not in cell.info and n not in no_power
    ]
    per = []
    for r in cell.results:
        refl = trans = coup = 0.0
        for w, col in inc.columns_of(r.s.values):
            p = jnp.abs(col) ** 2
            refl = refl + w * p[:, on].sum(axis=1)
            trans = trans + w * p[:, off].sum(axis=1)
            coup = coup + w * p[:, coupled_idx].sum(axis=1)
        per.append((refl, trans, coup, 1 - refl - trans - coup))
    refl, trans, coup, absn = (stack(cell, [p[k] for p in per]) for k in range(4))
    return PowerBalance(
        frequency=cell.frequency,
        reflectance=refl,
        transmittance=trans,
        coupled=coup,
        absorptance=absn,
        incidence=inc.incidence,
    )
