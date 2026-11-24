"""Wave ports in the driven problem: the port mode sweep coupled to the 3D field.

A wave port surface enters the driven problem through four terms (the 3D
field ``E`` and test function ``v`` in the units of
`gfem._engine.terms`, each mode ``(E_t, E_n)`` propagating into the domain
as ``exp(-gamma s)``, ``gamma = alpha + i beta``). One surface carries one
wave port, or several, one mode each (a shared port surface):

- one Robin term ``gamma_0 ∫ (A E_t) · v_t`` per surface
  (`gfem._engine.terms`), with ``gamma_0`` the propagation constant of the
  surface's smallest-``mode`` port (keeping its attenuation: dropping
  it would leave ``-alpha_0 M e`` on a lossy mode) and
  ``A`` the port's tangential inverse-permeability tensor (``mu_r^-1`` for
  an isotropic material; the port mode problem's ``nu_tt``);
- one modal correction of rank ``2 n`` per surface of ``n`` wave ports,
  ``W = H R_sym⁻¹ Hᵀ - gamma_0 M E G⁻¹ Eᵀ M``, never assembled, with ``H``
  the modes' n x H as 3D vectors (``h = ∫ A (grad E_n + gamma E_t) ·
  v_t``), ``E`` their tangential traces, ``M`` the Robin mass, ``R = Hᵀ
  E`` (symmetrized) and ``G = Eᵀ M E``. For one port it is the full
  minus scalar port operator ``W_full - W_scalar``,
  ``h hᵀ / (hᵀ e) - h_s h_sᵀ / (h_sᵀ e)`` with
  ``h_s`` the n x H without ``grad E_n``. On every requested mode the
  Robin term and ``W`` together impose the mode's full n x H, so a hybrid
  mode (``E_n != 0``) leaves the port without reflection; for a lone TEM
  mode ``h = h_s`` and ``W = 0``. ``W`` is complex symmetric and does not
  depend on the modes' scales;
- each port's incident wave's excitation ``2 h``;
- each port's S functional, the biorthogonal projection of the field on
  the surface's modes: the modal amplitudes ``c = R⁻¹ Hᵀ E`` with ``R =
  Hᵀ E_modes`` the unconjugated reactions (by reciprocity ``h_lᵀ e_k = 0``
  for ``l != k``, lossy or not), times the port's pseudo-wave factor
  ``kappa``, minus 1 at the excited port. The incident load
  ``2 h / kappa`` feeds a pseudo-wave of amplitude 1. Projecting on
  the conjugated mode instead reflects a lossy line's own mode at
  order ``alpha / beta``.

The 3D trace on the port face and the 2D Nédélec space of the port mode
problem share the face's triangles and order, and NGSolve orders both bases
by vertex number, which runs with gfem's node number on either mesh, also on
a rank's part. So the trace of a 3D field is the 2D field with the
same coefficients on matching edges and faces (`trace_dofs`). With
the modes written in the surface's port mode sweep's fixed basis,
``mode(omega) = W c(omega)``, the excitations and the functionals are fixed
vectors (one per basis column, assembled on the 2D port once) times
tabulated scalars, and ``W(omega) = vectors C(omega) vectorsᵀ`` with a
tabulated ``(2 m, 2 m)`` coefficient of rank ``2 n``, which is what the
adaptive sweep needs.

The wave ports of one surface share its Robin term, its vectors and its
modal correction (`PortSurfaceCoupling`); the excitation and the S
functional are each port's own (`WavePortCoupling`).

Under ranks the sweep and the arrays read off its basis run once on rank 0 and
are broadcast, so every coupling below holds replicated values. Only ``vectors``
is a rank's own: a load column, each value shared by the ranks that write it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence
from typing import Any

import ngsolve as ngs
import numpy as np

from gfem._boundaries import (
    WavePort,
    lead_port,
    wave_port_label,
    wave_port_surfaces,
)
from gfem._engine import quadrature, terms
from gfem._engine.coefficients import surface_p0
from gfem._engine.domain import boundary_region
from gfem._engine.line_mode import LineModeSpace, ModeSpace, mode_space
from gfem._engine.mesh import EngineMesh, inverse_order
from gfem._linalg.base import free_mask
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem._line_section import LineSection
from gfem._pml import Stretch
from gfem._port_section import PortSection, port_section
from gfem._sweep.port_mode import PortModeSweep, rows_of, sweeps_on, swept_once
from gfem._text import hertz
from gfem.errors import PortModeError
from gfem.model import Model
from gfem.problems import Polarization

__all__ = [
    "FaceTrace",
    "PortSurfaceCoupling",
    "SweptSurface",
    "WavePortCoupling",
    "deembedding",
    "face_trace",
    "face_vectors",
    "line_robin_pieces",
    "port_mode_sweeps",
    "trace_dofs",
    "wave_port_couplings",
]


def deembedding(gamma: Any, offset: float) -> Any:
    """``exp(gamma offset)``: S moves a wave port's reference plane into the domain.

    ``gamma`` the port mode's propagation constant (1/m), per frequency or
    one; ``offset`` the port's de-embedding distance (m).
    """
    return np.exp(gamma * offset)


def _at(omegas: np.ndarray, omega: float) -> int:
    """The index of ``omega`` on the output grid ``omegas``."""
    k = int(np.argmin(np.abs(omegas - omega)))
    if abs(omegas[k] - omega) > 1e-12 * abs(omega):  # pragma: no cover
        raise ValueError(f"No port mode at {omega / (2 * math.pi):g} Hz.")
    return k


@dataclasses.dataclass(eq=False)
class WavePortCoupling:
    """One wave port's terms in the driven problem, over the output grid.

    Its surface's `PortSurfaceCoupling` holds the terms the wave ports of
    one surface share.

    Attributes:
        index: The port's position in the model's ports.
        port: The wave port.
        sweep: Its port mode sweep over the output frequencies.
        phi: ``(n_freq, 2 m)``: the mode's n x H on the columns of its
            surface's ``vectors``, ``gamma c_t`` and ``c_n`` (gamma per file
            unit): ``h = vectors @ phi``.
        dual: ``(n_freq, 2 m)``: the port's S functional on the columns of
            ``vectors``: ``kappa`` times row ``k`` of ``R⁻¹ Hᵀ``, so that
            ``dual @ (vectorsᵀ x)`` is the port's outgoing pseudo-wave.
        kappa: ``(n_freq,)`` the wave of the sweep's mode at unit modal
            amplitude (`pseudo_wave_factor`): its pseudo-wave with a voltage
            path, its reciprocity-normalized modal wave without one; of
            modulus 1 on a lossless line.
        omegas: ``(n_freq,)`` the output grid, rad/s.
        scale: Metres per mesh file unit.
        field_factor: The solved field per SI field: ``scale`` for a 3D field
            in volts per file unit, 1 for a 2D model's field in SI units.
            ``None`` reads ``scale``.
    """

    index: int
    port: WavePort
    sweep: PortModeSweep
    phi: np.ndarray
    dual: np.ndarray
    kappa: np.ndarray
    omegas: np.ndarray
    scale: float
    field_factor: float | None = None

    def excitation(self, omega: float) -> np.ndarray:
        """``(2 m,)``: the incident wave's right-hand side is ``vectors @ this``."""
        k = _at(self.omegas, omega)
        factor = self.scale if self.field_factor is None else self.field_factor
        return 2.0 * factor * self.phi[k] / self.kappa[k]

    def functional(self, omega: float) -> np.ndarray:
        """``(2 m,)``: the port's S row is ``this @ (vectors.T @ x)``."""
        return self.dual[_at(self.omegas, omega)]

    @property
    def z_ref(self) -> np.ndarray:
        """``(n_freq,)`` complex reference impedance ``Z_PV``; NaN without a path.

        An imaginary part within rounding of zero (a lossless line) is dropped,
        so a lossless port's ``z_ref`` stays exactly real.
        """
        z = self.sweep.z_pv
        return np.where(np.abs(z.imag) <= 1e-12 * np.abs(z), z.real + 0j, z)

    def deembedding(self, omega: float) -> complex:
        """``exp(gamma offset)``: S moves its reference plane into the domain."""
        k = _at(self.omegas, omega)
        return complex(deembedding(self.sweep.gamma[k], self.port.offset))


@dataclasses.dataclass(frozen=True, eq=False)
class FaceTrace:
    """How the port mode's 2D DOFs sit in the 3D space: the face trace maps.

    The 3D field's tangential trace on the port has, on 2D DOF ``idx2[k]``,
    the coefficient of 3D DOF ``idx3[k]`` (`trace_dofs`, kept to this rank's
    free DOFs).

    Attributes:
        idx3: The 3D free DOFs on the port face.
        idx2: The 2D DOFs matching ``idx3``.
        writers: ``(ndof,)`` how many ranks write each 3D DOF (1 on a DOF
            of the face on one rank, more on an interface DOF).
    """

    idx3: np.ndarray
    idx2: np.ndarray
    writers: np.ndarray

    def lift(self, r: np.ndarray) -> np.ndarray:
        """The 3D vectors ``(ndof, 2 m)`` of 2D columns ``r`` ``(n_free_2d, 2 m)``."""
        vectors = np.zeros((len(self.writers), r.shape[1]), r.dtype)
        vectors[self.idx3] = r[self.idx2] / self.writers[self.idx3, None]
        return vectors


@dataclasses.dataclass(eq=False)
class PortSurfaceCoupling:
    """The terms the wave ports of one surface share, over the output grid.

    Attributes:
        piece: The surface's Robin term (θ tabulated on the grid), with
            ``beta`` of the wave port with the smallest ``mode``.
        vectors: ``(ndof, 2 m)``: per mode basis column, the 3D vector of
            ``∫ A (column) · v_t`` (tangential columns) or ``∫ A
            grad(column) · v_t`` (normal columns), ``A`` the port's
            tangential inverse-permeability tensor.
            Zero on the 3D space's fixed DOFs.
        reaction: ``(2 m, m)``: the port-face products of the columns of
            ``vectors`` with the mode basis's tangential columns, so ``hᵀ e
            = phi @ reaction @ c_t``.
        ports: The surface's wave ports, in model port order; the first
            with the smallest ``mode`` is the lead.
        extra: Further Robin pieces of a line port whose weight ``w``
            depends on the frequency (`line_robin_pieces`); empty on a 3D
            port.
        trace: The face trace maps that built ``vectors`` from the 2D
            columns, kept so a pass over the face reads the same maps.
    """

    piece: terms.Piece
    vectors: np.ndarray
    reaction: np.ndarray
    ports: tuple[WavePortCoupling, ...]
    extra: tuple[terms.Piece, ...] = ()
    trace: FaceTrace | None = None

    @property
    def pieces(self) -> tuple[terms.Piece, ...]:
        """The surface's Robin pieces: `piece`, then `extra`."""
        return (self.piece, *self.extra)

    def __post_init__(self) -> None:
        for k in range(len(self.lead.omegas)):
            _, r_sym, g = self._blocks(k)
            for name, block in (("reaction", r_sym), ("Gram", g)):
                if _singular(block):
                    raise self._singular_error(name, k)

    @property
    def lead(self) -> WavePortCoupling:
        """The wave port with the smallest ``mode``: its gamma is ``gamma_0``."""
        lead = lead_port([c.port for c in self.ports])
        return next(c for c in self.ports if c.port is lead)

    def _blocks(self, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``P``, ``R_sym`` and ``gamma_0 G`` at output frequency ``k``."""
        lead = self.lead
        m, n = self.reaction.shape[1], len(self.ports)
        gamma0 = lead.sweep.gamma[k] * lead.scale
        full = np.stack([c.phi[k] for c in self.ports], axis=1)
        tangential = np.stack([c.sweep.coords[k, :m] for c in self.ports], axis=1)
        scalar = lead.sweep.space.scalar(tangential, gamma0, lead.omegas[k])
        p = np.concatenate([full, scalar], axis=1)
        reactions = p.T @ (self.reaction @ tangential)
        r = reactions[:n]
        # R_sym, not R: it keeps D, hence C, complex symmetric
        return p, (r + r.T) / 2, reactions[n:]

    def _singular_error(self, name: str, k: int) -> PortModeError:
        label = wave_port_label([c.port for c in self.ports], lone_surface=True)
        f = self.lead.omegas[k] / (2 * math.pi)
        return PortModeError(
            f"{label}: the modal correction's {name} matrix is singular at "
            f"{hertz(f)} (the port mode's products with its own trace on "
            "the port face vanish), so the port cannot impose its mode. Check the "
            "port's cross-section and mesh, or use a LumpedPort."
        )

    def correction(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        """The modal correction at ω as ``W = vectors P D Pᵀ vectorsᵀ``.

        With ``n`` wave ports, the modes' n x H ``Φ`` and tangential
        coordinates ``C_t``, ``R = Φᵀ reaction C_t`` and ``G = C_tᵀ
        reaction[:m] C_t``.

        Returns:
            ``P`` ``(2 m, 2 n)``, ``[Φ, gamma_0 [C_t; 0]]`` (for one port
            the n x H of its mode with and without ``grad E_n``), and ``D``
            ``(2 n, 2 n)``, ``blockdiag(R_sym⁻¹, -(gamma_0 G)⁻¹)`` with
            ``R_sym = (R + Rᵀ) / 2`` (for one port ``diag(1 / R, -1 /
            R_s)``, the unconjugated reactions). Both blocks are regular:
            the coupling raises `PortModeError` when built otherwise.
        """
        p, r_sym, g = self._blocks(_at(self.lead.omegas, omega))
        n = len(self.ports)
        d = np.zeros((2 * n, 2 * n), complex)
        d[:n, :n] = np.linalg.inv(r_sym)
        d[n:, n:] = -np.linalg.inv(g)
        return p, d

    def correction_coefficients(self, omega: float) -> np.ndarray:
        """``C(ω)`` ``(2 m, 2 m)`` with ``W = vectors C vectorsᵀ`` (rank ``2 n``).

        ``C`` is complex symmetric: the adjoint reuses the forward operator
        untransposed.
        """
        p, d = self.correction(omega)
        return (p @ d) @ p.T


def _singular(a: np.ndarray) -> bool:
    """Non-finite, ``sigma_max = 0`` or ``sigma_min <= 1e-12 sigma_max``."""
    if not np.all(np.isfinite(a)):
        return True
    sigma = np.linalg.svd(a, compute_uv=False)
    return bool(sigma[0] == 0 or sigma[-1] <= 1e-12 * sigma[0])


def _phi(sweep: PortModeSweep, scale: float) -> np.ndarray:
    """`WavePortCoupling.phi` of a mode's sweep (the space's n x H columns)."""
    omegas = 2 * np.pi * sweep.frequency
    return sweep.space.phi(sweep.coords, sweep.gamma * scale, omegas)


@dataclasses.dataclass(eq=False)
class SweptSurface:
    """One wave port surface's port mode sweep over a grid: 2D only.

    What `wave_port_couplings` couples to the 3D field, and what the parent
    of frequency workers runs once over the whole grid, keeps for the
    result and hands over by rows.

    Attributes:
        ports: The surface's wave ports, in model port order.
        modes: The surface's discrete port mode problem.
        sweeps: Each port's mode over the grid.
        arrays: `gfem._sweep.port_mode.swept_once`'s arrays, with the
            columns read off the basis (``r``, ``reaction``).
    """

    ports: tuple[WavePort, ...]
    modes: ModeSpace
    sweeps: list[PortModeSweep]
    arrays: dict[str, np.ndarray]

    def rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """The arrays at the grid indices ``rows``: plain data that pickles."""
        return rows_of(self.arrays, rows)


def port_mode_sweeps(
    model: Model,
    *,
    order: int,
    frequency: tuple[float, ...],
    tolerance: float,
    comm: Any = None,
    handed: Sequence[dict[str, np.ndarray]] | None = None,
    polarization: Polarization | None = None,
    stretch: Stretch | None = None,
) -> list[SweptSurface]:
    """Each wave port surface's port mode sweep over ``frequency``.

    Needs no 3D space. With ``handed`` nothing is solved: the sweeps stand
    on those arrays, as a rank's do on the arrays rank 0 sent.

    Args:
        model: The model.
        order: The element order of the port mode problem.
        frequency: The grid, Hz.
        tolerance: ``SolverOptions.port_mode_tolerance``.
        comm: The resolved communicator (the sweep runs once, on rank 0).
        handed: Per surface, a whole-grid sweep's arrays at ``frequency``
            (`SweptSurface.rows`).
        polarization: A 2D model's, for its line ports' modes.
        stretch: The perfectly matched layer's stretch of the driven solve,
            which a section cutting the layer takes (`port_section`).

    Returns:
        One per surface with wave ports, in the order of each surface's
        first wave port.

    Raises:
        PortModeError: As `gfem._sweep.port_mode.port_mode_sweep`.
    """
    groups = list(wave_port_surfaces(model.ports).values())
    if handed is not None and len(handed) != len(groups):  # pragma: no cover
        raise ValueError(
            f"{len(handed)} handed port mode sweeps for {len(groups)} wave port "
            "surfaces."
        )
    out = []
    for k, group in enumerate(groups):
        section = port_section(model, lead_port(group), stretch)
        modes = mode_space(section, order, polarization)
        if handed is not None:
            arrays = handed[k]
            sweeps = sweeps_on(modes, frequency, arrays)
        else:
            sweeps, arrays = swept_once(
                comm,
                modes,
                frequency,
                [p.mode for p in group],
                tolerance=tolerance,
                ports=list(group),
                derived=modes.port_columns,
            )
        out.append(SweptSurface(tuple(group), modes, sweeps, arrays))
    return out


def wave_port_couplings(
    model: Model,
    built: EngineMesh,
    space: ngs.FESpace,
    *,
    order: int,
    frequency: tuple[float, ...],
    tolerance: float,
    layout: DofLayout = ONE_PROCESS,
    swept: Sequence[dict[str, np.ndarray]] | None = None,
    polarization: Polarization | None = None,
    stretch: Stretch | None = None,
) -> list[PortSurfaceCoupling]:
    """Run each wave port surface's port mode sweep and build its driven terms.

    On a 2D mesh each surface is a line port, coupled to the 2D field of
    ``polarization`` through its Robin pieces (`line_robin_pieces`).

    Args:
        model: The model.
        built: Its engine mesh.
        space: The 3D driven space, or a 2D model's H1 space.
        order: The element order (the port mode problem uses the same).
        frequency: The output grid, Hz.
        tolerance: ``SolverOptions.port_mode_tolerance``.
        layout: How ``space`` lies across ranks.
        swept: Per surface, a port mode sweep's arrays at ``frequency``,
            handed over in place of the sweep (a frequency worker's rows of
            its parent's whole-grid sweep, `port_mode_sweeps`).
        polarization: A 2D model's, ``"Ez"`` or ``"Hz"``.
        stretch: The perfectly matched layer's stretch of the solve: the
            port sections and the Robin terms take it.

    Returns:
        One coupling per surface with wave ports, in the order of each
        surface's first wave port.
    """
    omegas = 2 * np.pi * np.asarray(frequency, float)
    index = {p.name: i for i, p in enumerate(model.ports)}
    out = []
    surfaces = port_mode_sweeps(
        model,
        order=order,
        frequency=frequency,
        tolerance=tolerance,
        comm=layout.comm,
        handed=swept,
        polarization=polarization,
        stretch=stretch,
    )
    for surface in surfaces:
        group, modes, sweeps, columns = (
            surface.ports,
            surface.modes,
            surface.sweeps,
            surface.arrays,
        )
        lead = lead_port(group)
        section = modes.section
        r, reaction = columns["r"], columns["reaction"]
        trace = face_trace(model, modes, built, space, layout)
        vectors = trace.lift(r)
        scale = section.scale
        phis = [_phi(sweep, scale) for sweep in sweeps]
        dual, r_kk = _dual(reaction, [s.coords for s in sweeps], phis)
        # each mode's reciprocity pairing (1/2) int (e x h) . n, unconjugated
        pairing = modes.pairing_factor(omegas) * r_kk
        # a 2D model's H1 field is in SI units, a 3D one in volts per file unit
        factor = 1.0 if isinstance(modes, LineModeSpace) else scale
        ports = []
        for k, (port, sweep) in enumerate(zip(group, sweeps, strict=True)):
            kappa = pseudo_wave_factor(sweep, pairing[:, k])
            ports.append(
                WavePortCoupling(
                    index=index[port.name],
                    port=port,
                    sweep=sweep,
                    phi=phis[k],
                    # the load 2 factor h feeds modal amplitude factor
                    dual=kappa[:, None] * dual[:, k] / factor,
                    kappa=kappa,
                    omegas=omegas,
                    scale=scale,
                    field_factor=None if factor == scale else factor,
                )
            )
        # gamma_0, not i beta_0: with W's -gamma_0 M E G⁻¹ Eᵀ M the two impose
        # exactly h on the modes, lossy or not
        gamma0 = next(c for c in ports if c.port is lead).sweep.gamma * scale
        if isinstance(modes, LineModeSpace):
            piece, *extra = line_robin_pieces(
                built, order, modes, port=lead.name, gamma0=gamma0, omegas=omegas
            )
            out.append(
                PortSurfaceCoupling(
                    piece,
                    vectors,
                    reaction,
                    tuple(ports),
                    extra=tuple(extra),
                    trace=trace,
                )
            )
            continue
        piece = terms.wave_port_piece(
            model,
            built,
            order,
            port=lead.name,
            surface=lead.surface,
            theta=terms.Tabulated(omegas, gamma0),
            stretch=section.stretch,
        )
        out.append(
            PortSurfaceCoupling(piece, vectors, reaction, tuple(ports), trace=trace)
        )
    return out


def face_trace(
    model: Model,
    modes: ModeSpace,
    built: EngineMesh,
    space: ngs.FESpace,
    layout: DofLayout = ONE_PROCESS,
) -> FaceTrace:
    """The face trace maps of a surface's 2D DOFs in the 3D space.

    Collective under ranks (once, here); the maps are local.

    Args:
        model: The model.
        modes: The surface's discrete port mode problem.
        built: The model's engine mesh.
        space: The 3D driven space.
        layout: How ``space`` lies across ranks.
    """
    idx3, idx2 = trace_dofs(model, modes.section, modes, built, space)
    free = free_mask(space.FreeDofs(), space.ndof)
    idx3, idx2 = idx3[free[idx3]], idx2[free[idx3]]
    # never behind a test on len(idx3): the exchange is collective
    writers = np.zeros(space.ndof)
    writers[idx3] = 1.0
    writers = np.real(layout.exchange(writers))
    return FaceTrace(idx3, idx2, writers)


def face_vectors(
    model: Model,
    modes: ModeSpace,
    built: EngineMesh,
    space: ngs.FESpace,
    layout: DofLayout = ONE_PROCESS,
) -> Callable[[np.ndarray], np.ndarray]:
    """How a surface's 2D columns become its 3D ``vectors`` (`FaceTrace.lift`).

    Collective under ranks (once, here); the returned map is local.

    Args:
        model: The model.
        modes: The surface's discrete port mode problem.
        built: The model's engine mesh.
        space: The 3D driven space.
        layout: How ``space`` lies across ranks.

    Returns:
        The map from ``r`` ``(n_free_2d, 2 m)`` (`port_columns`) to ``vectors``
        ``(ndof, 2 m)``, zero on the 3D space's fixed DOFs.
    """
    return face_trace(model, modes, built, space, layout).lift


def line_robin_pieces(
    built: EngineMesh,
    order: int,
    modes: LineModeSpace,
    *,
    port: str,
    gamma0: np.ndarray,
    omegas: np.ndarray,
) -> list[terms.Piece]:
    r"""The Robin term :math:`\gamma_0 \int w\,u\,v\,dt` of a line port.

    One piece per ``B`` slot of the line's port mode problem: :math:`w =
    \sum_j \theta_j(\omega) w_j`, each :math:`w_j` per segment of the line
    (:math:`\mu_r^{-1}` in Ez, :math:`\varepsilon_r^{-1}` off the
    frequency-dependent segments in Hz, and their indicator for each group of
    those), so piece ``j`` has :math:`\theta = \gamma_0 \theta_j(\omega)`,
    tabulated on the output grid. The first is the line's steady weight.

    Args:
        built: The 2D engine mesh.
        order: The element order.
        modes: The line's port mode problem.
        port: The lead port's name.
        gamma0: ``(n_freq,)`` its propagation constant per file unit.
        omegas: ``(n_freq,)`` the output grid, rad/s.
    """
    section = modes.section
    region = boundary_region(built, (section.port.surface,))
    ds = quadrature.ds(order, definedon=region)
    n_faces = len(built.facet_order)
    scalars = modes.weights(omegas)
    out = []
    for j, w in enumerate(modes.weight_values):
        values = np.zeros(n_faces, complex)
        values[section.faces] = w
        cf = surface_p0(built, values)
        out.append(
            terms.Piece(
                f"waveport {port}" if j == 0 else f"waveport {port} {j}",
                terms.Tabulated(omegas, gamma0 * scalars[:, j]),
                lambda u, v, cf=cf: cf * u * v * ds,
                kind="port",
            )
        )
    return out


def pseudo_wave_factor(sweep: PortModeSweep, pairing: np.ndarray) -> np.ndarray:
    """``(n_freq,)``: a mode's wave at unit modal amplitude.

    With a voltage path, Marks and Williams' pseudo-wave ``a = sqrt(Re z) /
    (2 |z|) (V + z I)`` at ``z = Z_PV``, scaled so that a lossless line's
    1 W wave has ``|a| = 1``: ``(V/|V|) sqrt(Re Z_PV / |Z_PV|)`` for the
    sweep's mode of unit power modulus. Without one, the modal wave
    normalized by reciprocity: ``sqrt(pairing)``, with ``pairing`` the
    mode's ``(1/2) ∫ (e x h) · n`` unconjugated (1 W on a lossless line).
    Either way S does not depend on the phase of the mode, and the modal
    waves make S symmetric for a reciprocal device.
    """
    v, z = sweep.voltage, sweep.z_pv
    with np.errstate(invalid="ignore"):  # NaN without a voltage path
        kappa = v / np.abs(v) * np.sqrt(z.real / np.abs(z))
    return np.where(np.isfinite(kappa), kappa, np.sqrt(pairing + 0j))


def free_dual_ratio(mode: Any, port: WavePort) -> np.ndarray:
    """``(n_freq,)``: a wave port's free dual scale over a lossless port's.

    A lone wave port's S functional is ``c`` times its excitation with
    ``c = -i kappa^2 / (4 k0 eta0 N)`` (`pseudo_wave_factor`, ``N`` the
    mode's reciprocity pairing): this is ``kappa^2 / N``. 1 without a
    voltage path (``kappa^2 = N``) and on a lossless line.

    Args:
        mode: The port's `gfem.PortModeResult`, with its field.
        port: The wave port.
    """
    from gfem._sweep.port_mode import path_voltage

    n = len(np.asarray(mode.frequency))
    field = mode.e_field
    if port.voltage_path is None or field is None:
        return np.ones(n, complex)
    space, basis, coords = field._space, field._basis, field._coords  # noqa: SLF001
    v = path_voltage(space, basis, coords, np.asarray(port.voltage_path), port.name)
    z = np.asarray(mode.Z_PV, complex)
    kappa = v / np.abs(v) * np.sqrt(z.real / np.abs(z))
    omega = 2 * np.pi * np.asarray(mode.frequency, float)
    gamma = np.asarray(mode.gamma, complex) * space.section.scale
    pairing = np.array(
        [
            space.pairing(basis @ c, g, w)
            for c, g, w in zip(coords, gamma, omega, strict=True)
        ]
    )
    return kappa**2 / pairing


def _dual(
    reaction: np.ndarray, coords: list[np.ndarray], phi: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``R⁻¹ Φᵀ``, the modal amplitude functionals, and ``R``'s diagonal.

    ``R_lk = h_lᵀ e_k = phi_l @ reaction @ c_t,k``, unconjugated: the
    biorthogonal projection, exact for a lossy port line too. Returns
    ``(n_freq, n, 2 m)`` and ``(n_freq, n)``.
    """
    m = reaction.shape[1]
    tangential = np.stack([c[:, :m] for c in coords], axis=2)  # (f, m, n)
    p = np.stack(phi, axis=1)  # (f, n, 2m)
    r = np.einsum("fli,ij,fjk->flk", p, reaction, tangential)
    return np.linalg.solve(r, p), np.einsum("fkk->fk", r)


def trace_dofs(
    model: Model,
    section: PortSection | LineSection,
    modes: ModeSpace,
    built: EngineMesh,
    space: ngs.FESpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Matching DOFs of the 3D space and the port's 2D Nédélec space.

    On a 2D mesh, of the 2D H1 space and the line port's 1D H1 space
    (`_line_trace_dofs`).

    The 3D field's tangential trace on the port has, on 2D DOF ``idx2[k]``,
    the coefficient of 3D DOF ``idx3[k]``: edges are matched through the gfem
    nodes at their ends, faces through their triangle. A rank walks its own port
    faces, so on a rank with none both come back empty.

    Returns:
        ``(idx3, idx2)``.
    """
    if isinstance(section, LineSection):
        if not isinstance(modes, LineModeSpace):  # pragma: no cover
            raise TypeError("A line section's trace needs its LineModeSpace.")
        return _line_trace_dofs(model, section, modes, built, space)
    if section.faces is None:  # pragma: no cover - sections built by gfem have them
        raise ValueError("The port section does not record its mesh triangles.")
    used = np.unique(model.mesh.triangles[section.faces])
    element_of = inverse_order(section.faces, len(model.mesh.triangles))
    nd = modes.space.components[0]
    mesh3, mesh2 = built.mesh, modes.mesh
    node = built.node_order
    pairs: dict[int, int] = {}

    def add(d3, d2) -> None:
        if len(d3) != len(d2):  # pragma: no cover - same order and element type
            raise RuntimeError("The 3D trace and the port space do not match.")
        for a, b in zip(d3, d2, strict=True):
            if a >= 0:
                pairs[int(b)] = int(a)

    elements = element_of[built.facet_order]
    for local in np.flatnonzero(elements >= 0):
        bnd = mesh3[ngs.ElementId(ngs.BND, int(local))]
        el = mesh2[ngs.ElementId(ngs.VOL, int(elements[local]))]
        edges3 = {
            tuple(sorted(int(node[v.nr]) for v in mesh3[e].vertices)): e
            for e in bnd.edges
        }
        for e in el.edges:
            key = tuple(sorted(int(used[v.nr]) for v in mesh2[e].vertices))
            add(
                space.GetDofNrs(edges3[key]),
                nd.GetDofNrs(ngs.NodeId(ngs.EDGE, e.nr)),
            )
        add(space.GetDofNrs(bnd.faces[0]), nd.GetDofNrs(el.faces[0]))
    idx2 = np.fromiter(pairs, int, len(pairs))
    idx3 = np.fromiter(pairs.values(), int, len(pairs))
    return idx3, idx2


def _line_trace_dofs(
    model: Model,
    section: LineSection,
    modes: LineModeSpace,
    built: EngineMesh,
    space: ngs.FESpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Matching DOFs of a 2D H1 space and a line port's 1D H1 space.

    Vertices match through their gfem node; a segment's interior DOFs through
    the segment. Both meshes number their vertices in increasing gfem node,
    so NGSolve orients each segment's interior basis the same way on both.

    Returns:
        ``(idx2d, idx1d)``.
    """
    surface = model.mesh.surface_elements
    used = np.unique(surface[section.faces])
    element_of = inverse_order(section.faces, len(surface))
    on_line = element_of[built.facet_order]
    line_element = inverse_order(modes.element_segment, len(section.segments))
    mesh2, mesh1, h1 = built.mesh, modes.mesh, modes.space
    node = built.node_order
    pairs: dict[int, int] = {}

    def split(fes, el, gfem_of) -> tuple[dict[int, int], list[int]]:
        vertices = {int(gfem_of[v.nr]): v.nr for v in el.vertices}
        at = {
            g: int(fes.GetDofNrs(ngs.NodeId(ngs.VERTEX, nr))[0])
            for g, nr in vertices.items()
        }
        inner = [int(d) for d in fes.GetDofNrs(el) if int(d) not in at.values()]
        return at, inner

    for local in np.flatnonzero(on_line >= 0):
        bnd = mesh2[ngs.ElementId(ngs.BND, int(local))]
        el = mesh1[ngs.ElementId(ngs.VOL, int(line_element[on_line[local]]))]
        at2, inner2 = split(space, bnd, node)
        at1, inner1 = split(h1, el, used)
        if at2.keys() != at1.keys() or len(inner2) != len(inner1):  # pragma: no cover
            raise RuntimeError("The 2D trace and the line port space do not match.")
        pairs.update({at1[g]: d2 for g, d2 in at2.items() if d2 >= 0})
        pairs.update({d1: d2 for d1, d2 in zip(inner1, inner2, strict=True) if d2 >= 0})
    idx1 = np.fromiter(pairs, int, len(pairs))
    idx2 = np.fromiter(pairs.values(), int, len(pairs))
    return idx2, idx1
