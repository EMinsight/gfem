r"""The loss matrices of driven fields: where their power becomes heat.

The host kernel of `gfem.dissipated_power` and of the electromagnetic heat.
For the fields :math:`E_p` of the ports driven together at one frequency,
the power dissipated in a gfem tetrahedron or on a sheet face is the
Hermitian form

.. math::

    q = \sum_{pq} \overline{a_p} a_q D[p, q], \qquad
    D[p, q] = \frac{1}{c_0} \sum_k \int \overline{E_p} \cdot
    \mathrm{Im}(\theta_k C_k) E_q,

with :math:`a_p` the incident waves (1 W per unit :math:`|a_p|^2`),
:math:`c_0 = 2 \omega \mu_0 s` (:math:`s` the mesh unit in m) and
:math:`\theta_k C_k` the lossy pieces of the driven operator
(`gfem._engine.terms`): the mass (its loss tangent and complex
permittivity), the conductivity, a permittivity law's susceptibility and
the `Conductor` and `Impedance` sheet pieces. :math:`D` is the anti-Hermitian
part of those pieces on the pairs :math:`(\overline{E_p}, E_q)`, so its
diagonal is what `gfem.power_flow` reads from them. The curl-curl
(a real permeability), absorbing, wave port and lumped port pieces are not
heat: radiation leaves the model, and a port's load takes its power. A
perfectly matched layer's tetrahedra are masked out: the field there is the
stretched one.

The backward pass of a real objective :math:`L = \mathrm{Re} \sum
\overline{G} \cdot D` (:math:`G` the cotangents of the real and imaginary
parts of D) has two terms:

- the field term, :math:`\partial L / \partial x - i \partial L /
  \partial y` of each field's coefficients, a weighted apply of the lossy
  pieces: :math:`\sum_p W[p, r] \mathrm{Im}(\theta C) \overline{e_p} / c_0`
  with :math:`W[p, q] = \overline{G[p, q]} + G[q, p]`, one P0 weight per
  element and face;
- the explicit material term: :math:`D` is linear in each piece's
  coefficient, so a leaf's cotangent is the piece's own derivative times the
  contracted integral :math:`-i \theta_k \tau_e` per element, with
  :math:`\tau_e = \mathrm{Re} \sum_{pq} \overline{G_e[p, q]}
  \int_e \overline{E_p} \cdot E_q / c_0` (its symmetric moment for a
  per-axis leaf), handed to the sensitivity kernel as
  `gfem._engine.sensitivity.Integrated`; law parameters, sheet fills and an
  `Impedance` resistance are contracted here the same way.

The fields' coefficients are on the unpaired space of the driven operator,
also on a model with `gfem.Periodic` pairs (as for `gfem.power_flow`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING, Any

import ngsolve as ngs
import numpy as np
import scipy.constants as const

from gfem import _pml
from gfem._boundaries import Conductor, Impedance
from gfem._engine import quadrature
from gfem._engine.coefficients import coefficient, element_tensors, p0, surface_p0
from gfem._engine.domain import boundary_region
from gfem._engine.driven import DrivenOperator, driven_operator, operator_terms
from gfem._engine.sensitivity import Integrated, sensitivity
from gfem._engine.terms import sheet_fill

if TYPE_CHECKING:
    from gfem.model import Model

__all__ = [
    "LOSSY_PROPERTIES",
    "Losses",
    "build",
    "check_magnetic_loss",
    "heat_elements",
    "sheet_surfaces",
]

#: The material properties the heat reads directly: its explicit term.
LOSSY_PROPERTIES = frozenset({"eps_r", "tan_delta", "conductivity"})
#: The volume pieces whose loss is heat, by name or name prefix.
_VOLUME_NAMES = ("mass", "conductivity")
_VOLUME_PREFIXES = ("susceptibility ", "blend ")
#: The sheet pieces whose loss is heat (`Conductor` and `Impedance`).
_SHEET_PREFIXES = ("conductor ", "impedance ")


def heat_elements(model: Model, support: np.ndarray) -> np.ndarray:
    """The tetrahedra whose loss is heat: the solve domain outside a layer."""
    layer = _pml.layer_tetrahedra(model.mesh, model.pml)
    return support if layer is None else support & ~layer


def sheet_surfaces(model: Model) -> tuple[str, ...]:
    """The surfaces carrying a `Conductor` or `Impedance`, in boundary order."""
    out: list[str] = []
    for b in model.boundaries:
        if isinstance(b, Conductor | Impedance) and b.surface not in out:
            out.append(b.surface)
    return tuple(out)


def check_magnetic_loss(model: Model, heat: np.ndarray) -> None:
    """Raise if a heated tetrahedron has a complex ``mu_r``.

    Raises:
        NotImplementedError: Naming the region.
    """
    mu = element_tensors(model, "mu_r")
    lossy = np.abs(np.imag(mu)).reshape(len(mu), -1).max(axis=1) > 0
    lossy &= heat
    if not lossy.any():
        return
    mesh = model.mesh
    region = next(r for r in mesh.volumes if lossy[mesh.elements(r)].any())
    raise NotImplementedError(
        f"Region {region!r} has a complex mu_r: magnetic loss is not turned "
        "into heat in this version, so its dissipated power would be missing. "
        "Model the loss as an electric one (a loss tangent or a conductivity), "
        "or give the region a real mu_r."
    )


def _lossy_volume(name: str) -> bool:
    return name in _VOLUME_NAMES or name.startswith(_VOLUME_PREFIXES)


def _lossy_sheet(name: str) -> bool:
    return name.startswith(_SHEET_PREFIXES)


@dataclasses.dataclass(eq=False)
class Losses:
    """The lossy pieces of a model at one frequency, ready to contract fields.

    Attributes:
        model: The model (concrete values).
        op: Its driven operator on the unpaired space (result fields' space).
        heat: Per gfem tetrahedron, whether its loss is heat.
        omega: The angular frequency, rad/s.
        volume: Per tetrahedron, the real symmetric ``sum_k Im(theta_k C_k)``,
            ``(n,)`` or ``(n, 3, 3)``, zero off `heat`.
        sheets: The sheet pieces with a lossy θ, with ``Im theta``.
    """

    model: Model
    op: DrivenOperator
    heat: np.ndarray
    omega: float
    volume: np.ndarray
    sheets: list[tuple[Any, float]]

    @property
    def c0(self) -> float:
        """``2 omega mu0 scale``: the forms' units to W."""
        return 2 * self.omega * const.mu_0 * self.op.built.scale

    def _fields(self, e: np.ndarray) -> tuple[list, list]:
        """The fields ``E_p`` and their conjugates as grid functions."""
        out, conj = [], []
        for column in e.T:
            for values, into in ((column, out), (np.conj(column), conj)):
                gf = ngs.GridFunction(self.op.space)
                gf.vec.FV().NumPy()[:] = values
                into.append(gf)
        return out, conj

    def _faces(self) -> int:
        return len(self.model.mesh.triangles)

    def matrices(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The loss matrices, W per unit incident waves.

        Args:
            e: The fields' coefficients, ``(ndof, P)``.

        Returns:
            ``(n_tetrahedra, P, P)`` and ``(n_triangles, P, P)`` Hermitian
            matrices, gfem order; zero off the heat tetrahedra and off the
            sheet faces.
        """
        built, order = self.op.built, self.op.order
        n = e.shape[1]
        fields, conj = self._fields(e)
        tets = np.zeros((len(self.heat), n, n), complex)
        faces = np.zeros((self._faces(), n, n), complex)
        volume = coefficient(built, self.volume) if self.volume.any() else None
        for p in range(n):
            for q in range(p, n):
                if volume is not None:
                    tets[:, p, q] = self._volume_pass(
                        (volume * fields[q]) * conj[p], order
                    )
                if self.sheets:
                    faces[:, p, q] = self._face_pass(
                        [
                            (
                                im
                                * piece.coefficient
                                * fields[q].Trace()
                                * conj[p].Trace(),
                                piece.region,
                            )
                            for piece, im in self.sheets
                        ],
                        order,
                    )
                if q > p:
                    tets[:, q, p] = np.conj(tets[:, p, q])
                    faces[:, q, p] = np.conj(faces[:, p, q])
        for p in range(n):
            # the diagonal is real: drop round-off in its imaginary part
            tets[:, p, p] = tets[:, p, p].real
            faces[:, p, p] = faces[:, p, p].real
        return tets / self.c0, faces / self.c0

    def _volume_pass(self, integrand: Any, order: int) -> np.ndarray:
        """``int_e integrand`` per gfem tetrahedron."""
        built = self.op.built
        space = ngs.L2(built.mesh, order=0, complex=True)
        form = ngs.LinearForm(space)
        form += integrand * space.TestFunction() * quadrature.dx(order)
        form.Assemble()
        out = np.zeros(len(self.heat), complex)
        out[built.cell_order] = form.vec.FV().NumPy()
        return out

    def _face_pass(self, parts: Iterable[tuple[Any, Any]], order: int) -> np.ndarray:
        """``sum int_f integrand`` per gfem triangle, each part on its region."""
        built = self.op.built
        space = ngs.SurfaceL2(built.mesh, order=0, complex=True)
        q = space.TestFunction()
        form = ngs.LinearForm(space)
        for integrand, region in parts:
            form += integrand * q * quadrature.ds(order, definedon=region)
        form.Assemble()
        out = np.zeros(self._faces(), complex)
        out[built.facet_order] = form.vec.FV().NumPy()
        return out

    def pullback(
        self, e: np.ndarray, g_tets: np.ndarray, g_faces: np.ndarray
    ) -> np.ndarray:
        """The field term: each field's coefficient cotangent, ``(ndof, P)``.

        ``g_tets`` and ``g_faces`` are the complex cotangents ``G`` of the
        matrices (real part's plus i times imaginary part's).
        """
        built, order, space = self.op.built, self.op.order, self.op.space
        n = e.shape[1]
        _, conj = self._fields(e)
        w_tets = _weights(g_tets)
        w_faces = _weights(g_faces)
        volume = coefficient(built, self.volume) if self.volume.any() else None
        v = space.TestFunction()
        out = np.zeros(e.shape, complex)
        for r in range(n):
            form = ngs.LinearForm(space)
            empty = True
            for p in range(n):
                if volume is not None and w_tets[:, p, r].any():
                    w = p0(built, w_tets[:, p, r])
                    form += (w * (volume * conj[p])) * v * quadrature.dx(order)
                    empty = False
                if self.sheets and w_faces[:, p, r].any():
                    w = surface_p0(built, w_faces[:, p, r])
                    for piece, im in self.sheets:
                        form += (
                            (im * piece.coefficient * w)
                            * conj[p].Trace()
                            * v.Trace()
                            * quadrature.ds(order, definedon=piece.region)
                        )
                    empty = False
            if not empty:
                form.Assemble()
                out[:, r] = form.vec.FV().NumPy()
        return out / self.c0

    def explicit(
        self,
        e: np.ndarray,
        g_tets: np.ndarray,
        g_faces: np.ndarray,
        kinds: set[Hashable],
    ) -> dict[Hashable, np.ndarray]:
        """The explicit term: each wanted model leaf's cotangent.

        ``kinds`` are input kinds (`gfem._engine.sensitivity`): materials,
        law parameters, sheet fills and `Impedance` resistances. The values
        are complex (``sum ct * dy/dp``): the caller takes the real part for
        a real leaf. Per material kind one value per element of the region
        (``(n, 3)`` per principal axis); per fill one per face of the
        surface; one number per law parameter or resistance.
        """
        out: dict[Hashable, np.ndarray] = {}
        materials = {k for k in kinds if k[0] == "material"}  # ty: ignore[not-subscriptable]
        laws = [k for k in kinds if k[0] == "law"]  # ty: ignore[not-subscriptable]
        boundaries = [k for k in kinds if k[0] == "boundary"]  # ty: ignore[not-subscriptable]
        if materials or laws:
            tau = self._tau(e, g_tets)
            if materials:
                out |= self._materials(e, g_tets, tau, materials)
            out |= {kind: self._law(tau, kind) for kind in laws}
        if boundaries:
            tau_f = self._tau_faces(e, g_faces)
            out |= {kind: self._boundary(tau_f, kind) for kind in boundaries}
        return out

    def _materials(
        self, e: np.ndarray, g: np.ndarray, tau: np.ndarray, kinds: set[Hashable]
    ) -> dict[Hashable, np.ndarray]:
        """The material leaves' cotangents, through the sensitivity kernel.

        Each lossy volume piece's contracted integral is ``-i theta tau``
        (its moment for a per-axis leaf), handed over as `Integrated`.
        """
        model = self.model
        axes = any(_per_axis(model, k) for k in kinds)
        moments = self._tau_moments(e, g) if axes else None
        values, given = {}, {}
        for piece in (*self.op.pieces, *self.op.latent):
            if piece.kind != "value" or not _lossy_volume(piece.name):
                continue
            theta = complex(piece.theta(self.omega))
            values[piece.name] = -1j * theta * tau
            if moments is not None:
                given[piece.name] = -1j * theta * moments
        terms = operator_terms(model, self.op, nodes=False)
        return sensitivity(
            model,
            [Integrated(values, moments=given)],
            kinds,
            form="driven",
            terms=lambda _nodes: terms,
        )

    def _law(self, tau: np.ndarray, kind: Any) -> np.ndarray:
        """A law parameter's cotangent: its pieces' dθ/dp times ``-i c tau``."""
        total = 0j
        for piece in self.op.pieces:
            if piece.law is None or piece.law[0] != kind[1]:
                continue
            slope = piece.law[1](self.omega).get(tuple(kind[2:]))
            if slope is not None and piece.values is not None:
                total += -1j * complex(slope) * np.sum(piece.values * tau)
        return np.asarray(total)

    def _boundary(self, tau_f: np.ndarray, kind: Any) -> np.ndarray:
        """A sheet fill's cotangent per face, or an `Impedance` resistance's."""
        index = kind[1]
        b = self.model.boundaries[index]
        faces = self.model.mesh.elements(b.surface)
        if tuple(kind[2:]) == ("fill", "fill"):
            per_face = np.zeros(len(faces))
            for piece in self.op.pieces:
                if piece.sheet is not None and piece.sheet[0] == index:
                    im = complex(piece.theta(self.omega)).imag
                    per_face += im * piece.sheet[1] * tau_f[faces]
            return per_face
        total = 0.0
        fill = sheet_fill(self.model, b)
        for piece in self.op.pieces:
            value = piece.lumped
            if value is None or value.kind != tuple(kind) or piece.sheet is None:
                continue
            im = complex(piece.theta(self.omega)).imag
            c = piece.sheet[1] * (1.0 if fill is None else fill[faces])
            total += im * value.slope * float(np.sum(c * tau_f[faces]))
        return np.asarray(total)

    def _tau(self, e: np.ndarray, g: np.ndarray) -> np.ndarray:
        """``Re sum conj(G[p, q]) int_e conj(E_p) . E_q / c0`` per tetrahedron."""
        fields, conj = self._fields(e)
        n = e.shape[1]
        out = np.zeros(len(self.heat))
        for p in range(n):
            for q in range(n):
                weight = np.conj(g[:, p, q])
                if not weight.any():
                    continue
                integral = self._volume_pass(conj[p] * fields[q], self.op.order)
                out += np.real(weight * integral)
        return np.where(self.heat, out, 0.0) / self.c0

    def _tau_moments(self, e: np.ndarray, g: np.ndarray) -> np.ndarray:
        """`_tau`'s symmetric moment ``int_e sym(conj(E_p) (x) E_q)``, ``(n, 3, 3)``."""
        fields, conj = self._fields(e)
        n = e.shape[1]
        out = np.zeros((len(self.heat), 3, 3))
        for p in range(n):
            for q in range(n):
                weight = np.conj(g[:, p, q])
                if not weight.any():
                    continue
                for i in range(3):
                    for j in range(i, 3):
                        part = 0.5 * (
                            conj[p][i] * fields[q][j] + conj[p][j] * fields[q][i]
                        )
                        value = np.real(weight * self._volume_pass(part, self.op.order))
                        out[:, i, j] += value
                        if j > i:
                            out[:, j, i] += value
        out[~self.heat] = 0.0
        return out / self.c0

    def _tau_faces(self, e: np.ndarray, g: np.ndarray) -> np.ndarray:
        """``Re sum conj(G[p, q]) int_f conj(E_p)_t . E_q,t / c0`` per triangle."""
        surfaces = sheet_surfaces(self.model)
        out = np.zeros(self._faces())
        if not surfaces:
            return out
        region = boundary_region(self.op.built, surfaces)
        fields, conj = self._fields(e)
        n = e.shape[1]
        for p in range(n):
            for q in range(n):
                weight = np.conj(g[:, p, q])
                if not weight.any():
                    continue
                integrand = conj[p].Trace() * fields[q].Trace()
                integral = self._face_pass([(integrand, region)], self.op.order)
                out += np.real(weight * integral)
        return out / self.c0


def _weights(g: np.ndarray) -> np.ndarray:
    """``W[p, q] = conj(G[p, q]) + G[q, p]`` per element."""
    return np.conj(g) + np.swapaxes(g, 1, 2)


def _per_axis(model: Model, kind: Any) -> bool:
    from gfem._materials import MaterialField

    material = model.materials.get(kind[1])
    return isinstance(material, MaterialField) and material.has_principal_values(
        kind[2]
    )


def build(model: Model, order: int, support: np.ndarray, omega: float) -> Losses:
    """The `Losses` of ``model`` at ``omega`` on the fields' ``support``.

    Raises:
        NotImplementedError: A heated tetrahedron has a complex ``mu_r``.
    """
    heat = heat_elements(model, support)
    check_magnetic_loss(model, heat)
    op = driven_operator(
        model,
        order,
        elements=heat,
        faces=np.ones(len(model.mesh.triangles), bool),
        paired=False,
    )
    n = len(heat)
    parts = []
    for piece in op.pieces:
        if piece.kind != "value" or not _lossy_volume(piece.name):
            continue
        if piece.values is None:  # pragma: no cover - every lossy piece has them
            msg = f"The driven piece {piece.name!r} has no element values."
            raise ValueError(msg)
        parts.append(np.imag(complex(piece.theta(omega)) * np.asarray(piece.values)))
    tensor = any(v.ndim == 3 for v in parts)
    volume = np.zeros((n, 3, 3) if tensor else n)
    for v in parts:
        volume = volume + (
            v[:, None, None] * np.eye(3) if tensor and v.ndim == 1 else v
        )
    volume[~heat] = 0.0
    sheets = []
    for piece in op.pieces:
        if piece.kind != "trace" or not _lossy_sheet(piece.name):
            continue
        im = complex(piece.theta(omega)).imag
        if im != 0:
            sheets.append((piece, im))
    return Losses(model, op, heat, omega, volume, sheets)
