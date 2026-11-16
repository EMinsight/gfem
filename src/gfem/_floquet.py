"""Diffraction orders of Floquet surfaces: frame, order table and truncation.

A Floquet surface is an open face of a periodic cell whose rim lies on the
seams of two periodic pairs (ADR 0146). Its diffraction orders are plane
waves in the homogeneous lossless medium behind it (ADR 0147):

- Lattice frame: ê1 = a1/|a1|, ê3 = a1 x a2 normalized, ê2 = ê3 x ê1, with
  a1, a2 the translations of the face's two pairs in ``boundaries`` order.
- Order (m, n) has the in-plane wave vector κ = k_t + m b1 + n b2, with
  b_i·a_j = 2π δ_ij and k_t the problem's Bloch wave vector.
- gamma = sqrt(|κ|² - k²) with k = k0 n the medium's wavenumber: gamma = iβ (β > 0)
  for a propagating order (gamma² < 0), alpha > 0 for an evanescent one, 0 at
  grazing.
- TE reference ê3 x κ̂, TM reference κ̂, with κ̂ := ê1 at κ = 0.
- The outgoing order's Dirichlet-to-Neumann value
  n x (μ_r⁻¹ curl E) = λ E_t: λ_TE = gamma/μ_r, λ_TM = -k²/(gamma μ_r) (the
  propagating λ_TE = iβ/μ_r, λ_TM = i k²/(β μ_r), and their evanescent
  continuations). λ_TM is infinite at gamma = 0: a flag, never ``inf+nanj``.
- Modelled orders (ADR 0152): every propagating order plus the evanescent
  ones with e^{-2|gamma|d} > ``floquet_order_tolerance``, d the distance from
  the face to the nearest scatterer, never beyond the order cap
  ⌊pπ/(|b_i| h_face)⌋; one set per sweep, the union over the grid.

Everything here is NumPy on host values: the order table is a pure
function of the model, the frequencies, the wave vector(s) and the options,
so the parent and every frequency worker compute the same rows (ADR 0149).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.constants as const

from gfem._boundaries import FloquetPort, Periodic, surfaces_of
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem._periodic import Pairing, side_edges
from gfem.errors import FloquetOrderCapWarning, PortGeometryError

if TYPE_CHECKING:
    from gfem.model import Model
    from gfem.options import SolverOptions

__all__ = [
    "POLARIZATIONS",
    "FloquetFace",
    "OrderRows",
    "SurfaceOrders",
    "Truncation",
    "dtn",
    "floquet_face",
    "floquet_surfaces",
    "lattice_frame",
    "modelled_orders",
    "order_cap",
    "order_rows",
    "order_table",
    "reciprocal_lattice",
    "sort_orders",
    "warn_order_cap",
    "wave_impedance",
    "wavenumber",
]

C0 = const.c
ETA0 = const.mu_0 * const.c
#: The two polarizations, in the order of the table's last axis.
POLARIZATIONS = ("TE", "TM")
#: |κ| at or below this fraction of the reciprocal lattice's scale is κ = 0.
ZERO_KAPPA_RTOL = 1e-12
#: Largest node distance off the face's best-fit plane, relative to its size.
PLANE_RTOL = 1e-6
#: Sine below which a translation counts as lying in the face plane.
IN_PLANE_SINE = 1e-6

Order = tuple[int, int]


# --- lattice --------------------------------------------------------------------


def lattice_frame(a1: Iterable[float], a2: Iterable[float]) -> np.ndarray:
    """The lattice frame ``(ê1, ê2, ê3)`` as rows, ``(3, 3)``."""
    a1, a2 = np.asarray(tuple(a1), float), np.asarray(tuple(a2), float)
    e1 = a1 / np.linalg.norm(a1)
    e3 = np.cross(a1, a2)
    e3 /= np.linalg.norm(e3)
    return np.stack([e1, np.cross(e3, e1), e3])


def reciprocal_lattice(a1: Iterable[float], a2: Iterable[float]) -> np.ndarray:
    """``(b1, b2)`` as rows, rad/m, in the plane of a1 and a2: b_i·a_j = 2πδ_ij."""
    a = np.stack([np.asarray(tuple(a1), float), np.asarray(tuple(a2), float)])
    return 2 * np.pi * np.linalg.solve(a @ a.T, a)


def wavenumber(frequency: Any, eps_r: float, mu_r: float) -> np.ndarray:
    """K = 2πf/c0 · sqrt(eps_r mu_r), rad/m."""
    return 2 * np.pi * np.asarray(frequency, float) / C0 * math.sqrt(eps_r * mu_r)


def sort_orders(orders: Iterable[Order]) -> tuple[Order, ...]:
    """Distinct orders sorted by |m| + |n|, then m, then n."""
    return tuple(sorted(set(orders), key=lambda o: (abs(o[0]) + abs(o[1]), o)))


def _wave_vectors(wave_vector: Any, n_f: int) -> np.ndarray:
    """``None``, one 3-vector or one per frequency, as ``(n_f, 3)`` rad/m."""
    if wave_vector is None:
        return np.zeros((n_f, 3))
    k = np.asarray(wave_vector, float)
    if k.shape == (3,):
        return np.broadcast_to(k, (n_f, 3)).copy()
    if k.shape == (n_f, 3):
        return k.copy()
    raise ValueError(
        f"A wave vector is one 3-vector or one per frequency ({n_f}, 3), got "
        f"shape {k.shape}."
    )


# --- order rows -----------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class OrderRows:
    """Per frequency and diffraction order, the plane wave's values.

    Attributes:
        orders: The orders (m, n).
        frequency: ``(n_f,)`` Hz.
        wavenumber: ``(n_f,)`` k = k0 n of the medium, rad/m.
        wave_vector: ``(n_f, n_o, 3)`` κ, rad/m.
        gamma: ``(n_f, n_o)`` complex gamma, 1/m: iβ, alpha > 0 or 0.
        propagating: ``(n_f, n_o)`` gamma² < 0, strictly.
        theta: ``(n_f, n_o)`` rad, arcsin(|κ|/k); π/2 at grazing, NaN
            evanescent.
        phi: ``(n_f, n_o)`` rad from ê1 toward ê2; NaN evanescent.
        te: ``(n_f, n_o, 3)`` TE reference vector ê3 x κ̂.
        tm: ``(n_f, n_o, 3)`` TM reference vector κ̂.
    """

    orders: tuple[Order, ...]
    frequency: np.ndarray
    wavenumber: np.ndarray
    wave_vector: np.ndarray
    gamma: np.ndarray
    propagating: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    te: np.ndarray
    tm: np.ndarray

    @property
    def grazing(self) -> np.ndarray:
        """``(n_f, n_o)`` gamma = 0 exactly."""
        return self.gamma == 0


def order_rows(
    orders: Sequence[Order],
    frequency: Any,
    wave_vector: Any,
    *,
    a1: Iterable[float],
    a2: Iterable[float],
    eps_r: float,
    mu_r: float = 1.0,
) -> OrderRows:
    """The plane-wave values of ``orders`` at each frequency.

    Args:
        orders: Orders (m, n) on the reciprocal lattice of a1, a2.
        frequency: Frequencies, Hz.
        wave_vector: ``None`` (k_t = 0), one 3-vector, or one per frequency,
            rad/m, in the plane of a1 and a2.
        a1: The first pair's translation, m.
        a2: The second pair's translation, m.
        eps_r: The medium's relative permittivity (real).
        mu_r: The medium's relative permeability (real).
    """
    f = np.atleast_1d(np.asarray(frequency, float))
    k = wavenumber(f, eps_r, mu_r)
    return _rows(tuple(orders), f, k, _wave_vectors(wave_vector, len(f)), a1=a1, a2=a2)


def _rows(orders, f, k, kt, *, a1, a2) -> OrderRows:
    frame = lattice_frame(a1, a2)
    b = reciprocal_lattice(a1, a2)
    mn = np.asarray(orders, float).reshape(-1, 2)
    kappa = kt[:, None, :] + (mn @ b)[None, :, :]
    k1 = kappa @ frame[0]
    k2 = kappa @ frame[1]
    norm = np.hypot(k1, k2)
    g2 = norm**2 - k[:, None] ** 2
    gamma = np.where(
        g2 >= 0, np.sqrt(np.maximum(g2, 0)) + 0j, 1j * np.sqrt(np.maximum(-g2, 0))
    )
    propagating = g2 < 0
    evanescent = g2 > 0
    beta = np.sqrt(np.maximum(-g2, 0))
    theta = np.where(evanescent, np.nan, np.arctan2(norm, beta))
    zero = norm <= ZERO_KAPPA_RTOL * float(np.linalg.norm(b, axis=1).max())
    c = np.where(zero, 1.0, k1 / np.where(zero, 1.0, norm))
    s = np.where(zero, 0.0, k2 / np.where(zero, 1.0, norm))
    phi = np.where(evanescent, np.nan, np.arctan2(s, c))
    tm = c[..., None] * frame[0] + s[..., None] * frame[1]
    te = np.cross(frame[2], tm)
    return OrderRows(
        orders=tuple((int(m), int(n)) for m, n in orders),
        frequency=f,
        wavenumber=k,
        wave_vector=kappa,
        gamma=gamma,
        propagating=propagating,
        theta=theta,
        phi=phi,
        te=te,
        tm=tm,
    )


def dtn(
    gamma: Any, k: Any, mu_r: float, polarization: str
) -> tuple[np.ndarray, np.ndarray]:
    """λ with n x (μ_r⁻¹ curl E) = λ E_t for an outgoing order, 1/m.

    Args:
        gamma: gamma, 1/m (broadcast with ``k``).
        k: The medium's wavenumber, rad/m.
        mu_r: The medium's relative permeability.
        polarization: ``"TE"`` or ``"TM"``.

    Returns:
        ``(lam, infinite)``: λ_TE = gamma/μ_r or λ_TM = -k²/(gamma μ_r), and where
        λ is infinite (TM at gamma = 0); there ``lam`` holds ``inf + 0j``.
    """
    gamma = np.asarray(gamma, complex)
    k = np.asarray(k, float)
    if polarization == "TE":
        lam = np.broadcast_to(gamma / mu_r, np.broadcast(gamma, k).shape)
        return lam.copy(), np.zeros(lam.shape, bool)
    if polarization != "TM":
        raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}.")
    gamma, k = np.broadcast_arrays(gamma, k)
    infinite = gamma == 0
    safe = np.where(infinite, 1.0, gamma)
    lam = np.where(infinite, complex(np.inf, 0.0), -(k**2) / (safe * mu_r))
    return lam, infinite


def wave_impedance(
    gamma: Any, k: Any, eps_r: float, mu_r: float, polarization: str
) -> np.ndarray:
    """Z_o, ohm: TE η/cos θ, TM η cos θ, η = η0 sqrt(μ_r/eps_r); NaN off propagation.

    cos θ = β/k with gamma = iβ; NaN where gamma² >= 0.
    """
    gamma, k = np.broadcast_arrays(np.asarray(gamma, complex), np.asarray(k, float))
    propagating = (gamma.real == 0) & (gamma.imag > 0)
    cos = np.where(propagating, gamma.imag, 1.0) / k
    eta = ETA0 * math.sqrt(mu_r / eps_r)
    if polarization == "TE":
        z = eta / cos
    elif polarization == "TM":
        z = eta * cos
    else:
        raise ValueError(f"polarization must be 'TE' or 'TM', got {polarization!r}.")
    return np.where(propagating, z, np.nan)


# --- truncation -----------------------------------------------------------------


def order_cap(
    b1: Iterable[float], b2: Iterable[float], h_face: float, order: int
) -> tuple[int, int]:
    """The order cap ⌊pπ/(|b_i| h_face)⌋ along each lattice direction."""
    return tuple(  # ty: ignore[invalid-return-type]
        math.floor(order * math.pi / (float(np.linalg.norm(tuple(b))) * h_face))
        for b in (b1, b2)
    )


@dataclasses.dataclass(frozen=True)
class Truncation:
    """The modelled orders of a face, and what the order cap cut.

    Attributes:
        orders: The modelled orders, sorted (`sort_orders`).
        cap: The order cap (M1, M2).
        cut: Orders the truncation rule keeps beyond the cap, sorted (the
            first orders past the cap when the rule keeps every order).
        alpha_cap: The smallest alpha of the orders just past the cap over the
            grid, 1/m (0 if one propagates); ``nan`` when nothing is cut.
    """

    orders: tuple[Order, ...]
    cap: tuple[int, int]
    cut: tuple[Order, ...]
    alpha_cap: float


def modelled_orders(
    frequency: Any,
    wave_vector: Any,
    *,
    a1: Iterable[float],
    a2: Iterable[float],
    eps_r: float,
    mu_r: float = 1.0,
    distance: float,
    tolerance: float,
    cap: tuple[int, int],
) -> Truncation:
    """The orders a face models over a sweep (ADR 0152).

    Every order that propagates (or grazes) at some frequency, plus every
    evanescent one with e^{-2alpha d} > ``tolerance`` at some frequency, within
    |m| <= cap[0], |n| <= cap[1]: the union over the grid.

    Args:
        frequency: The sweep's frequencies, Hz.
        wave_vector: ``None``, one 3-vector or one per frequency, rad/m.
        a1: The first pair's translation, m.
        a2: The second pair's translation, m.
        eps_r: The face medium's relative permittivity.
        mu_r: The face medium's relative permeability.
        distance: d, m, from the face to the nearest scatterer (``inf``
            when there is none).
        tolerance: The round-trip tolerance (``floquet_order_tolerance``).
        cap: The order cap (`order_cap`).
    """
    f = np.atleast_1d(np.asarray(frequency, float))
    k = wavenumber(f, eps_r, mu_r)
    kt = _wave_vectors(wave_vector, len(f))
    a = np.stack([np.asarray(tuple(a1), float), np.asarray(tuple(a2), float)])
    # keep alpha < alpha_max, i.e. |κ|² < k² + alpha_max²
    alpha_max = math.log(1 / tolerance) / (2 * distance) if distance > 0 else math.inf
    kept: set[Order] = set()
    cut: set[Order] = set()
    if math.isfinite(alpha_max):
        for kf, ktf in zip(k, kt, strict=True):
            radius = math.sqrt(kf**2 + alpha_max**2)
            centre = a @ ktf / (2 * np.pi)
            reach = radius * np.linalg.norm(a, axis=1) / (2 * np.pi)
            ms = range(
                math.floor(-reach[0] - centre[0]), math.ceil(reach[0] - centre[0]) + 1
            )
            ns = range(
                math.floor(-reach[1] - centre[1]), math.ceil(reach[1] - centre[1]) + 1
            )
            grid = [(m, n) for m in ms for n in ns]
            rows = _rows(grid, f[:1], np.array([kf]), ktf[None], a1=a[0], a2=a[1])
            alpha = rows.gamma[0].real
            for o, al in zip(grid, alpha, strict=True):
                if al == 0 or al < alpha_max:
                    inside = abs(o[0]) <= cap[0] and abs(o[1]) <= cap[1]
                    (kept if inside else cut).add(o)
    else:
        kept = {
            (m, n)
            for m in range(-cap[0], cap[0] + 1)
            for n in range(-cap[1], cap[1] + 1)
        }
        cut = set(_ring(cap))
    alpha_cap = math.nan
    if cut:
        ring = _ring(cap)
        rows = _rows(ring, f, k, kt, a1=a[0], a2=a[1])
        alpha_cap = float(rows.gamma.real.min())
    return Truncation(sort_orders(kept), cap, sort_orders(cut), alpha_cap)


def _ring(cap: tuple[int, int]) -> list[Order]:
    """The orders just past the cap: |m| = M1 + 1 or |n| = M2 + 1."""
    m1, m2 = cap[0] + 1, cap[1] + 1
    return [
        (m, n)
        for m in range(-m1, m1 + 1)
        for n in range(-m2, m2 + 1)
        if abs(m) == m1 or abs(n) == m2
    ]


# --- the face of a Floquet surface ------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class FloquetFace:
    """A Floquet surface's geometry and medium, measured on the imported mesh.

    Attributes:
        surface: The surface name.
        region: The one volume region behind it.
        eps_r: The region's relative permittivity (real).
        mu_r: The region's relative permeability (real).
        normal: The unit normal pointing into the solve domain.
        origin: A point of the face plane (its nodes' mean), m.
        a1: The first in-plane pair's translation, m.
        a2: The second in-plane pair's translation, m.
        pairs: The two pairs, in ``boundaries`` order.
        area: The face area, m².
        h_face: The longest edge of the face's triangles, m.
        distance: d, m: the smallest normal distance from the face to a
            vertex of an element of another region or of a surface that
            carries a boundary condition or port (other than the pairs and
            Floquet surfaces); ``inf`` when there is none.
    """

    surface: str
    region: str
    eps_r: float
    mu_r: float
    normal: np.ndarray
    origin: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    pairs: tuple[Pairing, Pairing]
    area: float
    h_face: float
    distance: float

    @property
    def frame(self) -> np.ndarray:
        """The lattice frame ``(ê1, ê2, ê3)`` as rows."""
        return lattice_frame(self.a1, self.a2)

    @property
    def reciprocal(self) -> np.ndarray:
        """``(b1, b2)`` as rows, rad/m."""
        return reciprocal_lattice(self.a1, self.a2)

    @property
    def index(self) -> float:
        """N = sqrt(eps_r mu_r)."""
        return math.sqrt(self.eps_r * self.mu_r)

    def cap(self, order: int) -> tuple[int, int]:
        """The order cap at element order ``order``."""
        b1, b2 = self.reciprocal
        return order_cap(b1, b2, self.h_face, order)

    def rows(
        self, orders: Sequence[Order], frequency: Any, wave_vector: Any
    ) -> OrderRows:
        """`order_rows` of ``orders`` in this face's lattice and medium."""
        return order_rows(
            orders,
            frequency,
            wave_vector,
            a1=self.a1,
            a2=self.a2,
            eps_r=self.eps_r,
            mu_r=self.mu_r,
        )


def floquet_surfaces(ports: Iterable[Any]) -> dict[str, tuple[FloquetPort, ...]]:
    """The Floquet ports among ``ports`` by surface, in model order."""
    out: dict[str, list[FloquetPort]] = {}
    for port in ports:
        if isinstance(port, FloquetPort):
            out.setdefault(port.surface, []).append(port)
    return {s: tuple(group) for s, group in out.items()}


def _edges(triangles: np.ndarray) -> np.ndarray:
    """``(3t, 2)`` sorted node pairs of the triangles' edges."""
    return np.sort(triangles[:, [[0, 1], [1, 2], [0, 2]]].reshape(-1, 2), axis=1)


def _vec(x: Iterable[float]) -> str:
    return "(" + ", ".join(f"{float(v):.6g}" for v in x) + ")"


def floquet_face(
    model: Model, surface: str, lead: str, pairs: Sequence[Pairing] | None = None
) -> FloquetFace:
    """Check ``surface`` as a Floquet surface of ``model`` and measure it.

    The checks of ticket 08 Q6, in order (the first two after planarity
    and one-sidedness): a periodic pair exists; no pair translates out of
    the face plane; the rim lies on two pairs and nothing else; the face
    spans whole periods (rim twins on the face); one region behind it; that
    region lossless, isotropic, real and constant.

    Args:
        model: The model (its pairs may still be being built).
        surface: The surface name.
        lead: What opens every message, e.g. ``FloquetPort 'p' on surface
            'top'``.
        pairs: The model's pairings in ``boundaries`` order; read from the
            model when ``None``.

    Raises:
        ValueError: No pair, or a pair translating along the normal.
        PortGeometryError: Not planar, the domain on both sides, rim edges
            on no pair, or rim twins off the face.
        NotImplementedError: The rim on one pair only, several regions
            behind the face, or a medium that is not lossless, isotropic,
            real and constant.
    """
    pairs = list(model._periodic if pairs is None else pairs)  # noqa: SLF001
    if not pairs:
        raise ValueError(
            f"{lead}: the model has no gfem.Periodic pair, but a Floquet port's "
            "diffraction orders are those of a periodic cell. Pair the surfaces "
            f"that meet {surface!r} with gfem.Periodic, or end a closed model with "
            "gfem.WavePort or gfem.Absorbing."
        )
    tris, tets, origin, normal = _plane(model, surface, lead)
    for p in pairs:
        a = np.asarray(p.translation)
        if abs(float(a @ normal)) > IN_PLANE_SINE * float(np.linalg.norm(a)):
            raise ValueError(
                f"{lead}: {p.label} translates along the normal of {surface!r}, so "
                f"{surface!r} is not an open face of the cell. Remove that pair: a "
                "Floquet port needs the cell open along its normal."
            )
    p1, p2 = _rim_pairs(model, surface, lead, tris, pairs)
    for p in (p1, p2):
        _check_twins(model, surface, lead, tris, p)
    region, material = _medium(model, surface, lead, tets)
    x = model.mesh.imported_nodes[tris]
    area = 0.5 * float(
        np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]), axis=1).sum()
    )
    e = _edges(tris)
    nodes = model.mesh.imported_nodes
    return FloquetFace(
        surface=surface,
        region=region,
        eps_r=float(material.eps_r),  # ty: ignore[invalid-argument-type]
        mu_r=float(material.mu_r),  # ty: ignore[invalid-argument-type]
        normal=normal,
        origin=origin,
        a1=np.asarray(p1.translation, float),
        a2=np.asarray(p2.translation, float),
        pairs=(p1, p2),
        area=area,
        h_face=float(np.linalg.norm(nodes[e[:, 0]] - nodes[e[:, 1]], axis=1).max()),
        distance=_distance(model, region, origin, normal),
    )


def _plane(
    model: Model, surface: str, lead: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The face's triangles, the elements behind them, a point and the inward normal.

    Raises:
        PortGeometryError: The face is not planar, or not one-sided.
    """
    mesh = model.mesh
    x = mesh.imported_nodes
    sel = np.flatnonzero(mesh.triangle_tags == mesh.region(surface).tag)
    tris = mesh.triangles[sel]
    used = np.unique(tris)
    origin = x[used].mean(axis=0)
    *_, vt = np.linalg.svd(x[used] - origin, full_matrices=False)
    normal = vt[2]
    size = float(np.ptp(x[used], axis=0).max())
    off = float(np.abs((x[used] - origin) @ normal).max())
    if off > PLANE_RTOL * size:
        raise PortGeometryError(
            f"{lead}: the surface is not planar (its nodes are up to {off:.3g} m "
            "off their best-fit plane). A Floquet port face is one flat face "
            "across the cell."
        )
    active = model._domain.active  # noqa: SLF001
    nb = mesh.triangle_neighbours[sel]
    side_ok = (nb >= 0) & active[np.maximum(nb, 0)]
    if (side_ok.sum(axis=1) != 1).any():
        raise PortGeometryError(
            f"{lead}: the solve domain lies on both sides of {surface!r} (or on "
            "neither); a Floquet port faces the domain from one side."
        )
    tets = np.where(side_ok[:, 0], nb[:, 0], nb[:, 1])
    centroid = x[mesh.tetrahedra[tets]].mean(axis=1)
    inward = np.sign((centroid - x[tris].mean(axis=1)) @ normal)
    return tris, tets, origin, normal * inward[0]


def _rim_pairs(
    model: Model, surface: str, lead: str, tris: np.ndarray, pairs: list[Pairing]
) -> tuple[Pairing, Pairing]:
    """The two pairs the face's rim lies on, in ``boundaries`` order.

    Raises:
        NotImplementedError: The rim lies on one pair, the rest on walls.
        PortGeometryError: Rim edges lie on no pair.
    """
    mesh = model.mesh
    tag = mesh.region(surface).tag
    keys, counts = np.unique(_edges(tris), axis=0, return_counts=True)
    rim = [tuple(r) for r in keys[counts == 1].tolist()]
    names = {r.tag: r.name for r in mesh.regions if r.dim == 2}
    on: dict[tuple[int, int], set[str]] = {r: set() for r in rim}
    all_e = _edges(mesh.triangles)
    n = len(mesh.imported_nodes)
    near = np.isin(all_e[:, 0] * n + all_e[:, 1], [i * n + j for i, j in rim])
    tags = np.repeat(mesh.triangle_tags, 3)[near]
    for (i, j), t in zip(all_e[near].tolist(), tags.tolist(), strict=True):
        if t != tag:
            on[i, j].add(names.get(t, str(t)))
    pair_of = {s: k for k, p in enumerate(pairs) for s in (p.first, p.second)}
    met: set[int] = set()
    lone: list[tuple[int, int]] = []
    walls: set[str] = set()
    for key, surfaces in on.items():
        found = {pair_of[s] for s in surfaces if s in pair_of}
        if not found:
            lone.append(key)
            walls |= surfaces
        met |= found
    if lone and len(met) == 1:
        listed = ", ".join(repr(w) for w in sorted(walls)) or "no other surface"
        raise NotImplementedError(
            f"{lead}: its rim lies on one periodic pair ({pairs[min(met)].label}) "
            f"and on {listed}. In this version of gfem a Floquet port needs a cell "
            "periodic along both directions of its face: pair the two walls with "
            "gfem.Periodic as well, keep the cell one element thick across them, "
            "and give the wave vector no component along that pair. A 1D grating "
            "is modelled this way."
        )
    if lone or len(met) != 2:
        lone = lone or rim
        where = mesh.imported_nodes[list(lone[0])].mean(axis=0)
        raise PortGeometryError(
            f"{lead}: {len(lone)} edges of its rim lie on no periodic pair's "
            f"surface (one at {_vec(where)} m). A Floquet port face spans the "
            "whole cell, its rim on the surfaces of two gfem.Periodic pairs: make "
            f"{surface!r} one planar face across the cell, with no other boundary "
            "or structure on it."
        )
    i, j = sorted(met)
    return pairs[i], pairs[j]


def _check_twins(
    model: Model, surface: str, lead: str, tris: np.ndarray, p: Pairing
) -> None:
    """The face meets both surfaces of ``p`` along twin edges (whole periods).

    Raises:
        PortGeometryError: An edge on one side whose twin is not a face edge.
    """
    edges = side_edges(model.mesh, p, tris)
    for k, other, first, second in (
        (1, 2, p.first, p.second),
        (2, 1, p.second, p.first),
    ):
        missing = edges[k] - edges[other]
        if missing:
            raise PortGeometryError(
                f"{lead}: it meets {first!r} along {len(missing)} edges whose "
                f"twins on {second!r} are not edges of {surface!r}, so the face "
                f"is not one period of the cell. Mesh {surface!r} across the "
                "whole cell."
            )


def _medium(
    model: Model, surface: str, lead: str, tets: np.ndarray
) -> tuple[str, Material]:
    """The one region behind the face and its material.

    Raises:
        NotImplementedError: Several regions, or an unsupported medium.
    """
    mesh = model.mesh
    volumes = {r.tag: r.name for r in mesh.regions if r.dim == 3}
    regions = list(dict.fromkeys(volumes[int(t)] for t in mesh.tetrahedron_tags[tets]))
    if len(regions) > 1:
        raise NotImplementedError(
            f"{lead}: the elements behind {surface!r} belong to regions "
            f"{', '.join(repr(r) for r in regions)}. In this version of gfem a "
            "Floquet port needs one homogeneous region behind its face, because "
            "its orders are plane waves in that medium. Move the face off the "
            "structure, leaving a layer of one material between them."
        )
    region = regions[0]
    material = model.materials[region]
    what = _unsupported_medium(material)
    if what or not isinstance(material, Material):
        raise NotImplementedError(
            f"{lead}: region {region!r} behind it has {what}. In this version of "
            "gfem a Floquet port needs a lossless isotropic material with constant "
            "real eps_r and mu_r behind its face. Put a lossless layer between the "
            "face and that material."
        )
    return region, material


def _unsupported_medium(material: Material | MaterialField) -> str:
    """What makes a half-space medium unsupported, or ``""``."""
    if isinstance(material, MaterialField):
        return "a MaterialField"
    checks = (
        (isinstance(material.eps_r, PermittivityLaw), "a permittivity law"),
        (not material.isotropic, "anisotropic eps_r or mu_r"),
        (np.iscomplexobj(np.asarray(material.eps_r)), "a complex eps_r"),
        (material.conductivity != 0, "a conductivity"),
        (material.tan_delta != 0, "a loss tangent"),
    )
    return next((what for bad, what in checks if bad), "")


def _distance(
    model: Model, region: str, origin: np.ndarray, normal: np.ndarray
) -> float:
    """d, m: see `FloquetFace.distance`."""
    mesh = model.mesh
    x = mesh.imported_nodes
    active = model._domain.active  # noqa: SLF001
    tag = mesh.region(region).tag
    other = active & (mesh.tetrahedron_tags != tag)
    nodes = [mesh.tetrahedra[other].ravel()]
    skip = {s for b in model.boundaries if isinstance(b, Periodic) for s in b.surfaces}
    skip |= {p.surface for p in model.ports if isinstance(p, FloquetPort)}
    named = [
        s
        for item in (*model.boundaries, *model.ports)
        for s in surfaces_of(item)
        if s not in skip
    ]
    for s in dict.fromkeys([*named, *model.default_pmc]):
        if s in skip:
            continue
        nodes.append(mesh.triangles[mesh.triangle_tags == mesh.region(s).tag].ravel())
    scatter = np.unique(np.concatenate(nodes))
    if not scatter.size:
        return math.inf
    return float(np.abs((x[scatter] - origin) @ normal).min())


# --- the order table -------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class SurfaceOrders:
    """The order table of one Floquet surface over a sweep.

    Attributes:
        face: The surface's geometry and medium.
        ports: Its Floquet ports, in model order.
        truncation: The modelled orders and what the cap cut.
        rows: The plane-wave rows of `orders`.
        modelled: ``(n_o,)`` whether each of `orders` is modelled.
        lam: ``(n_f, n_o, 2)`` λ per polarization (`POLARIZATIONS`), 1/m.
        infinite: ``(n_f, n_o, 2)`` where λ is infinite (TM at gamma = 0).
        impedance: ``(n_f, n_o, 2)`` Z_o, ohm; NaN where not propagating.
    """

    face: FloquetFace
    ports: tuple[FloquetPort, ...]
    truncation: Truncation
    rows: OrderRows
    modelled: np.ndarray
    lam: np.ndarray
    infinite: np.ndarray
    impedance: np.ndarray

    @property
    def orders(self) -> tuple[Order, ...]:
        """The modelled and the declared orders, sorted (`sort_orders`)."""
        return self.rows.orders


def order_table(
    model: Model, frequency: Any, wave_vector: Any, options: SolverOptions
) -> dict[str, SurfaceOrders]:
    """The order table of every Floquet surface of ``model``, by surface.

    A pure function of its arguments: the parent and every frequency
    worker compute the same rows (ADR 0149).

    Args:
        model: The model.
        frequency: The sweep's frequencies, Hz.
        wave_vector: ``None``, one 3-vector or one per frequency, rad/m.
        options: Read for ``order`` and ``floquet_order_tolerance``.
    """
    f = np.atleast_1d(np.asarray(frequency, float))
    out: dict[str, SurfaceOrders] = {}
    for surface, ports in floquet_surfaces(model.ports).items():
        face = floquet_face(
            model, surface, f"FloquetPort {ports[0].name!r} on surface {surface!r}"
        )
        truncation = modelled_orders(
            f,
            wave_vector,
            a1=face.a1,
            a2=face.a2,
            eps_r=face.eps_r,
            mu_r=face.mu_r,
            distance=face.distance,
            tolerance=options.floquet_order_tolerance,
            cap=face.cap(options.order),
        )
        orders = sort_orders([*truncation.orders, *(p.order for p in ports)])
        rows = face.rows(orders, f, wave_vector)
        lam, infinite, z = [], [], []
        for pol in POLARIZATIONS:
            value, flag = dtn(rows.gamma, rows.wavenumber[:, None], face.mu_r, pol)
            lam.append(value)
            infinite.append(flag)
            z.append(
                wave_impedance(
                    rows.gamma, rows.wavenumber[:, None], face.eps_r, face.mu_r, pol
                )
            )
        kept = set(truncation.orders)
        out[surface] = SurfaceOrders(
            face=face,
            ports=ports,
            truncation=truncation,
            rows=rows,
            modelled=np.array([o in kept for o in orders]),
            lam=np.stack(lam, axis=-1),
            infinite=np.stack(infinite, axis=-1),
            impedance=np.stack(z, axis=-1),
        )
    return out


def warn_order_cap(table: dict[str, SurfaceOrders], tolerance: float) -> None:
    """Warn `FloquetOrderCapWarning` for each surface whose cap cut the rule.

    Called by the parent only (a worker's warnings never reach the user).
    """
    for surface, rows in table.items():
        t = rows.truncation
        if not t.cut:
            continue
        shown = ", ".join(str(o) for o in t.cut[:4]) + (
            ", ..." if len(t.cut) > 4 else ""
        )
        if t.alpha_cap > 0:
            reach = math.log(1 / tolerance) / (2 * t.alpha_cap)
            remedy = (
                f"Move {surface!r} out to {reach:.3g} m from the nearest structure, "
                f"or refine the mesh of {surface!r}."
            )
        else:
            remedy = (
                f"An order past the cap propagates: refine the mesh of {surface!r}."
            )
        warnings.warn(
            FloquetOrderCapWarning(
                f"Floquet surface {surface!r}: its face mesh resolves diffraction "
                f"orders up to |m| <= {t.cap[0]}, |n| <= {t.cap[1]} (the order cap), "
                f"but orders beyond it ({shown}) are not negligible "
                f"{rows.face.distance:.3g} m from the nearest structure (their round "
                f"trip decays by less than floquet_order_tolerance={tolerance:g}). "
                + remedy
            ),
            stacklevel=3,
        )
