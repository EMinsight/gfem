r"""The line port mode's material derivatives and the pullback of its normalization.

The line counterpart of `gfem._engine.port_mode_sensitivity` and
`gfem._engine.port_mode_normalization`. A line port's pencil is complex
symmetric, so the pencil-generic adjoint
(`gfem._engine.port_mode_adjoint`) applies unchanged; what is specific to
the line is on the 1D mesh of its segments:

- `LineKernel` contracts pairs :math:`(q, x)` with the pencil's forms per
  segment: :math:`\int q'x'` and :math:`\int qx` give the cotangents of the
  weight :math:`w` (:math:`\mu_r^{-1}` in Ez, :math:`\varepsilon_r^{-1}` in
  Hz), of the mass coefficient (:math:`\varepsilon_r` or :math:`\mu_r`) and of
  the Ez conductivity piece, and :math:`q x` at an absorbing end gives the
  cotangent of its admittance :math:`Y`.
- `line_normalize_vjp` is the reverse mode of the sweep's normalization
  (`gfem._engine.port_mode_normalization.normalize_rows`) on a line: the power
  :math:`P'` of 1 W/m, the gauge :math:`\int w u` or the overlap continuity,
  and no voltage sign. The power, the gauge and the line mass read the weight
  :math:`w`, so in Hz they carry the permittivity.
- `line_integrated` maps the segment cotangents to the owner triangles of the
  2D operator's ``stiffness``, ``mass``, ``conductivity`` and ``absorbing``
  terms (`gfem._engine.sensitivity.Integrated`).

The convention is :math:`\mathrm{d}L = \operatorname{Re}\sum \bar c\,
\mathrm{d}c`, plain (not conjugated) cotangents.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import ngsolve as ngs
import numpy as np

from gfem._engine.line_mode import LineModeSpace, LineReduction
from gfem._engine.port_mode_normalization import NormalizationCotangents, Normalized
from gfem._engine.sensitivity import Integrated

__all__ = [
    "LineCotangents",
    "LineKernel",
    "LinePair",
    "full_line_reduction",
    "line_integrated",
    "line_normalize_vjp",
]


@dataclasses.dataclass(eq=False)
class LinePair:
    r"""A line form contracted with a pair of vectors, with a weight.

    :math:`\mathrm{d}L = \operatorname{Re}\,w \int \mathrm{d}w_\text{seg}\,
    \ell\,r\,\mathrm{d}t` per segment for a bilinear form of the line's
    weight :math:`w` (``left`` the test side), and with ``left`` ``None`` the
    functional :math:`\int \mathrm{d}w_\text{seg}\,r\,\mathrm{d}t`.

    Attributes:
        left: The test-side vector, on the line's free DOFs or on all of them
            (``None`` for a functional).
        right: The trial-side vector on the free DOFs.
        weight: The complex weight, with the slot's scalar included.
    """

    left: np.ndarray | None
    right: np.ndarray
    weight: complex


@dataclasses.dataclass(eq=False)
class LineCotangents:
    r"""The cotangents of the coefficients a line port's forms read.

    Attributes:
        w: ``(n_seg,)`` of the weight :math:`w` of the line's mass and
            stiffness (:math:`\mu_r^{-1}` in Ez, :math:`\varepsilon_r^{-1}` in
            Hz).
        mass: ``(n_seg,)`` of the mass coefficient (:math:`\varepsilon_r` in
            Ez, :math:`\mu_r` in Hz).
        sigma: ``(n_seg,)`` of the Ez conductivity piece's coefficient
            :math:`\eta_0 s\sigma`; zero in Hz.
        admittance: ``(2,)`` of the admittance :math:`Y` of each end (zero off
            an absorbing end).
    """

    w: np.ndarray
    mass: np.ndarray
    sigma: np.ndarray
    admittance: np.ndarray

    @classmethod
    def zeros(cls, space: LineModeSpace) -> LineCotangents:
        """All cotangents zero, in the shapes of the line's section."""
        n = len(space.section.eps)
        return cls(*(np.zeros(k, complex) for k in (n, n, n, 2)))

    def add(self, other: LineCotangents) -> None:
        """Add ``other`` in place."""
        for f in dataclasses.fields(self):
            getattr(self, f.name).__iadd__(getattr(other, f.name))


class LineKernel:
    """The material derivatives of one `LineModeSpace` at a wavenumber.

    Args:
        space: The line's port mode problem.
        k0: The wavenumber per local unit of the pairs' frequency.
    """

    def __init__(self, space: LineModeSpace, k0: float) -> None:
        self.space = space
        self.at(k0)
        self._gf = ngs.GridFunction(space.space)
        self._gl = ngs.GridFunction(space.space)
        self._ends = _absorbing_ends(space)

    def at(self, k0: float) -> LineKernel:
        """This kernel at another wavenumber (the forms are shared)."""
        self.k0 = float(k0)
        self.theta = self.space.pencil.theta(self.k0)
        return self

    # --- integrals -------------------------------------------------------------

    def _load(self, gf: ngs.GridFunction, v: np.ndarray) -> ngs.GridFunction:
        """``gf`` holding the coefficients of ``v`` (free DOFs or all of them)."""
        space = self.space
        gf.vec.FV().NumPy()[:] = v if len(v) == space.space.ndof else space.full(v)
        return gf

    def _per_segment(self, cf: ngs.CoefficientFunction) -> np.ndarray:
        """The integral of ``cf`` over each section segment."""
        space = self.space
        per_element = ngs.Integrate(
            cf, space.mesh, ngs.VOL, order=2 * space.order, element_wise=True
        )
        out = np.zeros(len(space.section.eps), complex)
        out[space.element_segment] = np.asarray(per_element)
        return out

    def _products(
        self, left: np.ndarray | None, right: np.ndarray, *, gradient: bool = False
    ) -> np.ndarray:
        r"""Per segment, :math:`\int \ell r` (of the derivatives, or :math:`\int r`)."""
        r = self._load(self._gf, right)
        if left is None:
            return self._per_segment(r)
        left_cf = self._load(self._gl, left)
        if gradient:
            return self._per_segment(ngs.grad(left_cf) * ngs.grad(r))
        return self._per_segment(left_cf * r)

    # --- the pencil's items ------------------------------------------------------

    def cotangents(
        self, pairs: Sequence[tuple[np.ndarray, np.ndarray, complex]]
    ) -> LineCotangents:
        r"""The cotangents of :math:`\mathrm{Re}\,q^T(\mathrm{d}A-\lambda\mathrm{d}B)x`.

        Args:
            pairs: ``(q, x, lam)`` per pair: vectors on the free DOFs and the
                eigenvalue :math:`\lambda = \gamma^2` the pair belongs to
                (summed over the pairs).
        """
        space = self.space
        out = LineCotangents.zeros(space)
        theta = self.theta
        ez = space.polarization == "Ez"
        for q, x, lam in pairs:
            stiff = self._products(q, x, gradient=True)
            mass = self._products(q, x)
            out.w += theta[0] * (stiff - lam * mass)
            out.mass += theta[1] * mass
            if ez:
                out.sigma += theta[2] * mass
            for e, pos in self._ends:
                out.admittance[e] += theta[2] * q[pos] * x[pos]
        return out

    def explicit(self, pairs: Sequence[LinePair]) -> LineCotangents:
        """The weight's cotangents of the forms beside the pencil (`LinePair`)."""
        out = LineCotangents.zeros(self.space)
        for pair in pairs:
            out.w += pair.weight * self._products(pair.left, pair.right)
        return out


def _absorbing_ends(space: LineModeSpace) -> list[tuple[int, int]]:
    """``(end, free position)`` of each absorbing end off the laws."""
    s = space.section
    out = []
    for e, (node, kind, seg) in enumerate(
        zip(s.ends, s.end_kinds, s.end_segments, strict=True)
    ):
        if kind != "absorbing" or (s.law_index is not None and s.law_index[seg] >= 0):
            continue
        dof = space.space.GetDofNrs(ngs.NodeId(ngs.VERTEX, int(node)))[0]
        out.append((e, int(np.searchsorted(space.free, dof))))
    return out


def line_integrated(section, cot: LineCotangents, n_elements: int) -> Integrated:
    """The segment cotangents as per-element integrals of the 2D operator's terms.

    ``stiffness`` takes the weight's, ``mass`` the mass coefficient's (a law
    segment has none: its permittivity goes through the law), ``conductivity``
    the Ez piece's and ``absorbing`` the ends' admittances, each added to the
    area element behind its segment.
    """
    cells = section.cells
    if cells is None:  # pragma: no cover - built by line_section
        raise ValueError("The line section does not record its mesh elements.")
    mass = cot.mass
    if section.law_index is not None:
        mass = np.where(np.asarray(section.law_index) >= 0, 0.0, mass)
    values = {
        name: np.zeros(n_elements, complex)
        for name in ("stiffness", "mass", "conductivity", "absorbing")
    }
    np.add.at(values["stiffness"], cells, cot.w)
    np.add.at(values["mass"], cells, mass)
    np.add.at(values["conductivity"], cells, cot.sigma)
    np.add.at(values["absorbing"], cells[section.end_segments], cot.admittance)
    return Integrated(values=values)


# --- the normalization -------------------------------------------------------------


def full_line_reduction(space: LineModeSpace) -> LineReduction:
    """The line's power and gauge on the whole free-DOF space.

    A cotangent on the field points anywhere in the space, not only along the
    sweep's reduced basis, so the normalization is pulled back on full vectors.
    """
    return LineReduction(space, np.eye(len(space.free)))


def line_normalize_vjp(  # noqa: PLR0915, PLR0917
    fields: LineReduction,
    mass,
    norm: Normalized,
    y: np.ndarray,
    gamma: np.ndarray,
    omega: np.ndarray,
    *,
    z_bar: np.ndarray,
    polarity: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> NormalizationCotangents:
    r"""The reverse mode of `normalize_rows` at the rows it returned, on a line.

    ``fields`` is `full_line_reduction`'s; the other arguments are the
    forward's. The cotangents are those of the field coordinates :math:`z`.
    The power of a line is :math:`\frac{i\bar\gamma}{2\omega\mu_0}z^T\bar B\bar z`
    (Ez) or :math:`\frac{-i\gamma}{2\omega\varepsilon_0}z^TB\bar z` (Hz) with
    :math:`B = \sum_j\theta_j B_j` the :math:`w`-weighted mass, and the gauge
    :math:`\sum_j\theta_j\int w_j u`; the pairs returned are their cotangents
    of :math:`w`.
    """
    space = fields.space
    n = len(y)
    ez = space.polarization == "Ez"
    vacuum = space._vacuum  # noqa: SLF001
    z_bar = np.array(z_bar, dtype=complex)
    y_bar = np.zeros(y.shape, complex)
    gamma_bar = np.zeros(n, complex)
    reference_bar = None
    pairs: list[LinePair] = []
    for i in reversed(range(n)):
        row = norm.rows[i]
        gam = gamma[i]
        theta = space._weights(float(space._k0(omega[i])))  # noqa: SLF001
        b = sum(t * m for t, m in zip(theta, fields._b, strict=True))  # noqa: SLF001
        zb = z_bar[i]
        # the phase factor u = |g| / g: z2 = u z1, dphi = Im(dg / g)
        u = abs(row.value) / row.value
        s = u * (zb @ row.z1)
        g_bar = -1j * s.imag / row.value
        zb1 = u * zb
        if row.kind == "gauge":
            gauge = sum(t * g for t, g in zip(theta, fields._gauge, strict=True))  # noqa: SLF001
            zb1 = zb1 + g_bar * gauge
            pairs.append(LinePair(None, row.z1, g_bar * theta[0]))
        elif row.kind == "largest":
            zb1 = zb1.copy()
            zb1[row.entry] += g_bar
        elif row.kind == "polarity":
            zb1 = zb1 + g_bar * polarity
        else:
            previous = norm.z[i - 1] if i else reference
            assert previous is not None  # noqa: S101 - an overlap row has one
            zb1 = zb1 + g_bar * (mass.T @ previous.conj())
            lower = np.conj(g_bar * (mass @ row.z1))
            if i == 0:
                reference_bar = lower
            else:
                z_bar[i - 1] += lower
        # the power: z1 = z0 / sqrt|P|
        modulus = abs(row.power)
        direction = row.power / modulus
        zb0 = zb1 / math.sqrt(modulus)
        slope = -0.5 * modulus**-1.5 * float((zb1 @ row.z0).real)
        p_bar = slope * direction.conjugate()
        z0 = row.z0
        zc = z0.conj()
        if ez:
            c = 1j / (2 * omega[i] * vacuum)
            bc = b.conj()
            gc = gam.conjugate()
            quad = z0 @ (bc @ zc)
            zb0 = zb0 + p_bar * c * gc * (bc @ zc)
            zb0 = zb0 + np.conj(p_bar * c * gc * (bc @ z0))
            gamma_bar[i] += np.conj(p_bar * c * quad)
            pairs.append(LinePair(zc, z0, np.conj(p_bar * c * gc) * theta[0]))
        else:
            c = -1j / (2 * omega[i] * vacuum)
            quad = z0 @ (b @ zc)
            zb0 = zb0 + p_bar * c * gam * (b @ zc)
            zb0 = zb0 + np.conj(p_bar * c * gam * (b @ z0))
            gamma_bar[i] += p_bar * c * quad
            pairs.append(LinePair(z0, zc, p_bar * c * gam * theta[0]))
        # the coordinates are the pencil vector itself
        y_bar[i] += zb0
    return NormalizationCotangents(y_bar, gamma_bar, reference_bar, pairs)  # ty: ignore[invalid-argument-type]
