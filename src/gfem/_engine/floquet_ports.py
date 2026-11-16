"""Floquet ports in the driven problem: plane-wave orders coupled to the cell.

A Floquet surface (`gfem._floquet`) enters the driven operator as (ADR 0148,
amended by ADR 0153), in the units of `gfem._engine.terms` (lengths in mesh
file units):

- one Robin term ``i k0 ∫ (n_f / mu_f) E_t · v_t`` on the surface, the
  first-order absorbing admittance of the half-space behind it
  (`floquet_piece`);
- one rank-one correction ``(λ_o - i k0 n_f / mu_f) / A ṽ_o ṽ_oᴴ`` per
  modelled (order, polarization), ``λ_o`` the order's Dirichlet-to-Neumann
  value (`gfem._floquet.dtn`) and ``A`` the face area, applied by Woodbury
  and never assembled: a `LowRankTerm` ``U C Vᵀ`` with ``U`` the vectors
  ``ṽ = Pᴴ v``, ``V = conj(U)`` and ``C`` diagonal, row-scaled, with an
  infinite TM shift at grazing as an exact constraint row.

The vectors are Fourier projections ``v_o = ∫ N · ψ_o dS``, ``ψ_o = ê_o
exp(-i κ_o · r)``, assembled as complex linear forms on the unpaired H(curl)
space and carried to the solve space by ``Pᴴ`` (the expansion of the
periodic space, `gfem._engine.driven.expansion_matrix`); no port mode is
solved.

Each Floquet port reads its order's coefficient ``c_o = vᴴ (P x) / A`` and
is fed by the load ``2 λ_q Pᴴ v_q``, scaled to power waves (ADR 0154): an
order carries ``|c|² A / (2 Z_o)`` through one cell, so the load is
multiplied by ``sqrt(2 Z_q / A)`` (1 W incident) and the row by
``sqrt(A / (2 Z_p))``. A row reads exactly 0 where its order does not
propagate, its ``z_ref`` is the order's wave impedance (NaN there) and its
de-embedding ``exp(gamma offset)`` (1 there).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem._boundaries import FloquetPort
from gfem._engine import quadrature, terms
from gfem._engine.domain import boundary_region
from gfem._engine.mesh import EngineMesh
from gfem._engine.system_operator import LowRankTerm
from gfem._engine.wave_ports import deembedding
from gfem._floquet import POLARIZATIONS, SurfaceOrders
from gfem._linalg.base import free_mask
from gfem._text import hertz
from gfem.model import Model

__all__ = [
    "FloquetPortCoupling",
    "FloquetSurfaceCoupling",
    "check_excitable",
    "floquet_couplings",
    "floquet_piece",
]


def _at(omegas: np.ndarray, omega: float) -> int:
    """The index of ``omega`` on the output grid ``omegas``."""
    k = int(np.argmin(np.abs(omegas - omega)))
    if abs(omegas[k] - omega) > 1e-12 * abs(omega):  # pragma: no cover
        raise ValueError(f"No Floquet order at {omega / (2 * math.pi):g} Hz.")
    return k


@dataclasses.dataclass(eq=False)
class FloquetPortCoupling:
    """One Floquet port's terms in the driven problem, over the output grid.

    Attributes:
        index: The port's position in the model's ports.
        port: The Floquet port.
        column: Its (order, polarization) column in its surface's vectors.
        omegas: ``(n_freq,)`` the output grid, rad/s.
        lam: ``(n_freq,)`` λ of its order, per mesh file unit; 0 where it is
            infinite (TM at grazing, never excited).
        impedance: ``(n_freq,)`` Z_o, ohm; NaN where not propagating.
        gamma: ``(n_freq,)`` gamma of its order, 1/m.
        propagating: ``(n_freq,)`` whether its order propagates.
        area: The face area, file units².
    """

    index: int
    port: FloquetPort
    column: int
    omegas: np.ndarray
    lam: np.ndarray
    impedance: np.ndarray
    gamma: np.ndarray
    propagating: np.ndarray
    area: float

    def excitation(self, omega: float) -> complex:
        """The load's factor on its column: ``2 λ sqrt(2 Z / A)`` (1 W incident)."""
        k = _at(self.omegas, omega)
        return complex(2 * self.lam[k] * math.sqrt(2 * self.impedance[k] / self.area))

    def functional(self, omega: float) -> float:
        """The row's factor on its conjugated column: ``sqrt(A / (2 Z)) / A``.

        0 where the order does not propagate.
        """
        k = _at(self.omegas, omega)
        if not self.propagating[k]:
            return 0.0
        return math.sqrt(self.area / (2 * self.impedance[k])) / self.area

    @property
    def z_ref(self) -> np.ndarray:
        """``(n_freq,)`` the order's wave impedance, NaN where not propagating."""
        return self.impedance

    def deembedding(self, omega: float) -> complex:
        """``exp(gamma offset)`` where the order propagates, 1 elsewhere."""
        k = _at(self.omegas, omega)
        if not self.propagating[k]:
            return 1.0 + 0j
        return complex(deembedding(self.gamma[k], self.port.offset))


@dataclasses.dataclass(eq=False)
class FloquetSurfaceCoupling:
    """The terms of one Floquet surface, over the output grid.

    A surface coupling of `gfem._engine.system_operator.SystemOperator`.

    Attributes:
        surface: The Floquet surface.
        piece: Its Robin term (`floquet_piece`).
        vectors: ``(ndof, 2 n_o)``: ``Pᴴ v`` per (order, polarization), the
            columns order-major in `gfem._floquet.POLARIZATIONS` order; zero
            on the solve space's fixed DOFs.
        shift: ``(n_freq, 2 n_o)``: the diagonal of ``C``, ``(λ - i k0 n /
            mu) / A`` on the modelled columns, 0 elsewhere and where the
            shift is infinite.
        infinite: ``(n_freq, 2 n_o)``: where the shift is infinite (TM at
            grazing).
        omegas: ``(n_freq,)`` the output grid, rad/s.
        ports: The surface's Floquet ports, in model port order.
        transpose: Whether this is the transposed system's coupling
            (``conj(U) Cᵀ Uᵀ``).
    """

    surface: str
    piece: terms.Piece
    vectors: np.ndarray
    shift: np.ndarray
    infinite: np.ndarray
    omegas: np.ndarray
    ports: tuple[FloquetPortCoupling, ...]
    transpose: bool = False

    def term(self, omega: float) -> LowRankTerm:
        """``W(ω) = U C Uᴴ`` (``V = conj(U)``), or its transpose."""
        k = _at(self.omegas, omega)
        term = LowRankTerm(
            self.vectors,
            np.diag(self.shift[k]),
            right=np.conj(self.vectors),
            infinite=self.infinite[k] if self.infinite[k].any() else None,
            row_scaled=True,
        )
        return term.transposed() if self.transpose else term

    def transposed(self) -> FloquetSurfaceCoupling:
        """The coupling of the transposed system: ``conj(U) Cᵀ Uᵀ``."""
        return dataclasses.replace(self, transpose=not self.transpose)


def floquet_piece(
    built: EngineMesh, order: int, surface_orders: SurfaceOrders
) -> terms.Piece:
    """The Robin term of a Floquet surface: ``i k0 ∫ (n / mu) E_t · v_t``."""
    face = surface_orders.face
    region = boundary_region(built, (face.surface,))
    weight = math.sqrt(face.eps_r / face.mu_r)
    scale = built.scale

    def integrand(u: Any, v: Any) -> Any:
        ds = quadrature.ds(order, definedon=region)
        return weight * u.Trace() * v.Trace() * ds

    return terms.Piece(
        f"floquet {face.surface}",
        lambda w: 1j * w * scale / const.c,  # i k0, per file unit
        integrand,
        kind="port",
    )


def _projections(
    built: EngineMesh,
    unpaired: ngs.FESpace,
    order: int,
    surface_orders: SurfaceOrders,
    row: int = 0,
) -> np.ndarray:
    """``(unpaired ndof, 2 n_o)``: ``v = ∫ N · ψ dS`` per (order, polarization).

    κ and the reference vectors at the table's row ``row``, the output
    grid's first frequency: one wave vector holds at every frequency of a
    solve (a fixed-angle sweep solves each frequency on its own, ADR 0151).
    Zero on the unpaired space's fixed DOFs.
    """
    region = boundary_region(built, (surface_orders.face.surface,))
    # ψ oscillates across the face: integrate it above the pinned rule
    rule = ngs.IntegrationRule(ngs.ET.TRIG, 2 * order + 6)
    ds = ngs.ds(definedon=region, intrules={ngs.ET.TRIG: rule})
    v = unpaired.TestFunction()
    free = free_mask(unpaired.FreeDofs(), unpaired.ndof)
    kappa = surface_orders.rows.wave_vector[row] * built.scale  # rad per file unit
    references = (surface_orders.rows.te[row], surface_orders.rows.tm[row])
    out = np.zeros((unpaired.ndof, 2 * len(surface_orders.orders)), complex)
    for o, k in enumerate(kappa):
        phase = ngs.exp(-1j * (k[0] * ngs.x + k[1] * ngs.y + k[2] * ngs.z))
        for p, reference in enumerate(references):
            form = ngs.LinearForm(unpaired)
            psi = phase * ngs.CF(tuple(float(c) for c in reference[o]))
            form += psi * v.Trace() * ds
            form.Assemble()
            out[:, 2 * o + p] = form.vec.FV().NumPy() * free
    return out


def floquet_couplings(
    model: Model,
    op: Any,
    table: dict[str, SurfaceOrders],
    frequency: Sequence[float],
    *,
    grid: Sequence[float] | None = None,
) -> list[FloquetSurfaceCoupling]:
    """Each Floquet surface's coupling on the driven operator ``op``.

    Args:
        model: The model.
        op: The driven operator (`gfem._engine.driven.DrivenOperator`) of a
            periodic model: its space, expansion and mesh.
        table: The order table over ``grid`` (`gfem._floquet.order_table`):
            its modelled orders are the whole sweep's.
        frequency: The output grid, Hz: a subset of ``grid`` (a frequency
            worker's block).
        grid: The table's grid, Hz; ``frequency`` by default.

    Returns:
        One coupling per Floquet surface, in the order of the table.
    """
    if not table:
        return []
    if op.expansion is None:  # pragma: no cover - a Floquet model is periodic
        raise ValueError("A Floquet port needs the periodic space's expansion.")
    built, scale = op.built, op.built.scale
    grid = list(frequency if grid is None else grid)
    at = [grid.index(f) for f in frequency]  # the table's rows of the output grid
    omegas = 2 * np.pi * np.asarray(frequency, float)
    k0 = omegas * scale / const.c
    index = {p.name: i for i, p in enumerate(model.ports)}
    expansion = op.expansion.matrix
    free = free_mask(op.space.FreeDofs(), op.space.ndof)
    out = []
    for surface, surface_orders in table.items():
        face = surface_orders.face
        area = face.area / scale**2
        v = _projections(built, op.expansion.space, op.order, surface_orders, at[0])
        vectors = (expansion.conj().T @ v) * free[:, None]
        n_o = len(surface_orders.orders)
        infinite = surface_orders.infinite[at].reshape(len(omegas), 2 * n_o)
        # an infinite λ stays a flag (scaling inf+0j would make it inf+nanj)
        lam = np.where(
            infinite, 0, surface_orders.lam[at].reshape(len(omegas), 2 * n_o)
        )
        lam = lam * scale
        modelled = np.repeat(surface_orders.modelled, 2)[None, :] & ~infinite
        robin = 1j * k0 * math.sqrt(face.eps_r / face.mu_r)
        safe = np.where(modelled, lam, 0.0)
        shift = np.where(modelled, (safe - robin[:, None]) / area, 0.0)
        infinite = infinite & np.repeat(surface_orders.modelled, 2)[None, :]
        z = surface_orders.impedance[at].reshape(len(omegas), 2 * n_o)
        ports = []
        for port in surface_orders.ports:
            o = surface_orders.orders.index(port.order)
            p = POLARIZATIONS.index(port.polarization)
            column = 2 * o + p
            ports.append(
                FloquetPortCoupling(
                    index=index[port.name],
                    port=port,
                    column=column,
                    omegas=omegas,
                    lam=lam[:, column],
                    impedance=z[:, column],
                    gamma=surface_orders.rows.gamma[at, o],
                    propagating=surface_orders.rows.propagating[at, o],
                    area=area,
                )
            )
        out.append(
            FloquetSurfaceCoupling(
                surface=surface,
                piece=floquet_piece(built, op.order, surface_orders),
                vectors=vectors,
                shift=shift,
                infinite=infinite,
                omegas=omegas,
                ports=tuple(ports),
            )
        )
    return out


def check_excitable(
    table: dict[str, SurfaceOrders], excite: Sequence[str], frequency: Sequence[float]
) -> None:
    """Raise for an excited Floquet port whose order does not propagate somewhere.

    Raises:
        ValueError: An excited Floquet port's order has gamma² >= 0 at a
            frequency of the grid: it carries no incident power there (ADR
            0153). Raised for the whole sweep, before any factorization.
    """
    propagating = {}
    for surface_orders in table.values():
        for port in surface_orders.ports:
            o = surface_orders.orders.index(port.order)
            propagating[port.name] = (
                port,
                surface_orders.face.surface,
                surface_orders.rows.propagating[:, o],
            )
    always = [n for n, (_, _, ok) in propagating.items() if ok.all()]
    for name in excite:
        if name not in propagating:
            continue
        port, surface, ok = propagating[name]
        if ok.all():
            continue
        freqs = ", ".join(
            hertz(f) for f, good in zip(frequency, ok, strict=True) if not good
        )
        m, n = port.order
        listed = ", ".join(repr(p) for p in always)
        raise ValueError(
            f"FloquetPort {name!r} (order ({m}, {n}), {port.polarization}) cannot "
            f"be excited at {freqs}: the order is evanescent or grazing in the "
            f"medium behind {surface!r} there and carries no incident power. "
            "Excite only orders that propagate over the whole sweep, e.g. "
            f"Driven(..., excite=[{listed}]), or narrow the frequency grid."
        )
