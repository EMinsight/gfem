r"""Point dipoles as load columns of the driven solve.

A dipole's load is the transpose of the point probe (`gfem._engine.probe`)
at its position: with :math:`\varphi_j` the H(curl) basis in SI units and
the driven equations scaled by :math:`\mu_0 L` (`gfem._engine.terms`,
:math:`L` the metres per file unit),

.. math::

    b_j = \mu_0 L\,\omega^2\, p\cdot\varphi_j(r_0) \quad\text{(electric)},
    \qquad
    b_j = -i\mu_0 L\,\omega\, m\cdot(\nabla\times\varphi_j)(r_0)
    \quad\text{(magnetic)},

a fixed vector times a scalar of :math:`\omega` (affine, as the adaptive
sweep needs). The emitted power is the pair of that load with the solution,
:math:`P = -\operatorname{Im}(b^H x) / (2\omega\mu_0 L)`: finite for the
discrete field though the exact one is singular at :math:`r_0`.

`locate` settles a point on a face, edge or vertex in one element
deterministically (in the support, then the largest index); the load sits
on that element's DOFs. Positions where the bulk power is undefined raise
(`check_positions`).

Gradients: with :math:`\lambda = A^{-T} g` the adjoint of an output whose
cotangent on :math:`x` is :math:`g`, the node coordinates move the
electric load through its point evaluation. For a field :math:`l_h` of
fixed coefficients, the value at the fixed point :math:`r_0` moves under a
P1 node displacement :math:`W` by :math:`\delta l_h(r_0) = -[(\nabla
W)^\mathsf{T} l_h + (\nabla l_h)\,W](r_0)` (covariant transport through the
element's affine map), so :math:`\lambda^\mathsf{T}\delta b = \mu_0
L\omega^2\, p\cdot\delta\lambda_h(r_0)`: twelve numbers on the owning
element's vertices (`transport`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants as const
import scipy.sparse

from gfem import _mpi
from gfem._boundaries import (
    PEC,
    PMC,
    Absorbing,
    Conductor,
    FloquetPort,
    Impedance,
    LumpedPort,
    WavePort,
    surfaces_of,
)
from gfem._engine.probe import ON_FACE, Layout, Located, locate, probe
from gfem._linalg.base import free_mask
from gfem._linalg.layout import ONE_PROCESS, DofLayout
from gfem._sources import ElectricDipole, MagneticDipole
from gfem.mesh import Mesh

__all__ = ["PointLoads", "check_positions", "point_loads"]

#: The four faces of a tetrahedron, by the vertex each one leaves out.
_FACES = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
#: Relative imaginary part under which a law's value counts as real.
_REAL = 1e-12

_Dipole = ElectricDipole | MagneticDipole


@dataclasses.dataclass(frozen=True, eq=False)
class PointLoads:
    """The point dipoles' load columns on one operator's space.

    Attributes:
        names: The dipoles' names, in column order.
        moments: ``(n, 3)`` complex, :math:`p` (C m) or :math:`m` (A m²).
        magnetic: ``(n,)``, whether each dipole is magnetic.
        vectors: ``(N, n)`` complex ``scipy.sparse`` CSC matrix (a column holds
            the DOFs of one element), the fixed parts: dipole k's load at ω is
            ``coefficients(ω)[k] * vectors[:, k]``, zero on fixed DOFs.
        located: Each dipole's element and barycentric coordinates.
        scale: The metres per file unit.
        layout: The DOF layout of the loads (``DISTRIBUTED`` columns).
    """

    names: tuple[str, ...]
    moments: np.ndarray
    magnetic: np.ndarray
    vectors: scipy.sparse.csc_matrix
    located: Located
    scale: float
    layout: DofLayout = ONE_PROCESS

    def coefficients(self, omega: float) -> np.ndarray:
        r"""``(n,)``: :math:`\mu_0 L\omega^2` (electric), :math:`-i\mu_0 L\omega`."""
        unit = const.mu_0 * self.scale
        return unit * np.where(self.magnetic, -1j * omega, omega**2)

    def rhs(self, omega: float) -> np.ndarray:
        """The loads at ω, ``(N, n)`` dense."""
        return (self.vectors @ scipy.sparse.diags(self.coefficients(omega))).toarray()

    def __call__(self, omega: float) -> np.ndarray:
        """`rhs`: the loads of a source kind as `SourceColumns` reads them."""
        return self.rhs(omega)

    def port_waves(self, omega: float, z: np.ndarray) -> np.ndarray:
        r"""``(n_ports, n)``: :math:`c_k\, b_k^\mathsf{T} z_i` by reciprocity.

        ``z`` ``(N, n_ports)`` holds each port's own solution :math:`z_i =
        A^{-1} g_i`; the dipole's load is :math:`c_k b_k`
        (`coefficients`, `vectors`), so the product is :math:`g_i^\mathsf{T}
        A^{-1} c_k b_k` for a complex symmetric :math:`A`: the port wave
        before de-embedding, without solving the dipole.
        """
        pairs = np.asarray(self.layout.pair(self.vectors, z))
        return (pairs * self.coefficients(omega)[:, None]).T

    def reduced_port_waves(
        self, omega: float, projected: np.ndarray, coordinates: np.ndarray
    ) -> np.ndarray:
        r"""``(n_ports, n)``: `port_waves` from the ports' reduced solutions.

        ``projected`` is :math:`B^\mathsf{T} V` (`vectors` on the real basis
        :math:`V`, dense, read once for all frequencies) and ``coordinates``
        ``(n_basis, n_ports)`` the ports' solutions on :math:`V`.
        """
        pairs = projected @ coordinates
        return (pairs * self.coefficients(omega)[:, None]).T

    def port_wave_loads(self, omega: float, weights: np.ndarray) -> np.ndarray:
        r"""``(N, r)``: :math:`\sum_k w_{jk}\, c_k b_k`, one column per row ``j``.

        ``weights`` is ``(r, n)``. The backward of port waves read by
        reciprocity loads port :math:`i`'s adjoint column with its cotangent
        row, de-embedded: one column per port, whatever the number of dipoles.
        """
        scaled = self.coefficients(omega)[:, None] * np.asarray(weights).T
        return np.asarray(self.vectors @ scaled)

    def port_wave_nodes(
        self,
        mesh: Mesh,
        layout: Layout,
        omega: float,
        *,
        solutions: np.ndarray,
        weights: np.ndarray,
        comm: Any = None,
    ) -> np.ndarray:
        r"""The node term of port waves read by reciprocity at ω, ``(n_nodes, 3)``.

        `nodes` with dipole :math:`k`'s adjoint :math:`\lambda_k = \sum_i
        w_{ik} z_i`, read at the dipoles' positions from the ports' own
        solutions ``solutions`` ``(N, n_ports)`` (one jet per port with a
        nonzero row of ``weights`` ``(n_ports, n)``, the cotangent
        de-embedded), and no power term: no dipole is solved.
        """
        out = np.zeros((len(mesh.nodes), 3))
        rows = np.flatnonzero(np.any(weights != 0, axis=1))
        electric = np.flatnonzero(~self.magnetic)
        if not rows.size or not electric.size:
            return out
        probed = probe(mesh, layout, self.located, comm)
        values = np.zeros((len(self.names), 3), complex)
        gradients = np.zeros((len(self.names), 3, 3), complex)
        for i in rows:
            value, gradient = probed.jet(solutions[:, i])
            values += weights[i][:, None] * value
            gradients += weights[i][:, None, None] * gradient
        coefficients = self.coefficients(omega)
        for k in electric:
            vertices, moved = transport(
                mesh,
                self.located,
                int(k),
                values=values[k][None],
                gradients=gradients[k][None],
                weight=(coefficients[k] * self.moments[k])[None],
            )
            np.add.at(out, vertices, np.real(moved[0]))
        return out

    def power_scale(self, omega: float) -> float:
        r"""Emitted power per :math:`-\operatorname{Im}(b^H x)`, in W."""
        return 1.0 / (2 * omega * const.mu_0 * self.scale)

    def emitted_power(self, omega: float, x: np.ndarray) -> np.ndarray:
        r"""``(n,)`` W: :math:`-\operatorname{Im}(b_k^H x_k) / (2\omega\mu_0 L)`.

        ``x`` ``(N, n)`` holds the dipoles' solutions, column k dipole k's.
        """
        b = self.rhs(omega)
        pairs = np.asarray(self.layout.pair(np.conj(b), x))
        return -np.imag(np.diagonal(pairs)) * self.power_scale(omega)

    def nodes(
        self,
        mesh: Mesh,
        layout: Layout,
        omega: float,
        *,
        adjoint: np.ndarray,
        solution: np.ndarray,
        power: np.ndarray,
        comm: Any = None,
    ) -> np.ndarray:
        r"""The electric loads' node term at ω, ``(n_nodes, 3)`` real, per metre.

        :math:`\operatorname{Re}(\lambda_k^\mathsf{T}\,\delta b_k)` (the load
        moves with its element) plus the emitted power's own dependence on
        the load, :math:`-\operatorname{Im}(\delta b_k^H x_k) / (2\omega\mu_0
        L)` times its cotangent ``power[k]``. ``adjoint`` and ``solution``
        are ``(N, n)``, the dipoles' columns (this rank's piece); each rank
        adds the dipoles it holds, so a sum over ranks counts each once.
        Magnetic dipoles add nothing (their node gradient is refused).
        """
        out = np.zeros((len(mesh.nodes), 3))
        probed = probe(mesh, layout, self.located, comm)
        coefficients = self.coefficients(omega)
        scale = self.power_scale(omega)
        for k in np.flatnonzero(~self.magnetic):
            lam_value, lam_grad = probed.jet(adjoint[:, k])
            x_value, x_grad = probed.jet(solution[:, k])
            c, p = coefficients[k], self.moments[k]
            vertices, moved = transport(
                mesh,
                self.located,
                int(k),
                values=np.stack([lam_value[k], x_value[k]]),
                gradients=np.stack([lam_grad[k], x_grad[k]]),
                weight=np.stack([c * p, np.conj(c * p)]),
            )
            term = np.real(moved[0]) - power[k] * scale * np.imag(moved[1])
            np.add.at(out, vertices, term)
        return out

    def power_loads(self, omega: float, cotangent: np.ndarray) -> np.ndarray:
        r"""``(N, n)``: the cotangent on :math:`x_k` of a cotangent on `emitted_power`.

        JAX's convention: the cotangent of :math:`-\operatorname{Im}(b^H x)`
        on :math:`x` is :math:`i\bar b`.
        """
        ct = np.asarray(cotangent, float) * self.power_scale(omega)
        return 1j * np.conj(self.rhs(omega)) * ct[None, :]


def check_positions(
    model: Any,
    sources: Sequence[_Dipole],
    layout: Layout,
    frequency: Sequence[float],
    comm: Any = None,
) -> Located:
    """Locate the dipoles and raise where their bulk power is undefined.

    Raises:
        ValueError: A dipole lies outside the mesh or the solve domain (an
            excluded or conductor volume), on a material interface, in a
            perfectly matched layer, on a boundary or sheet face, or in a
            lossy, anisotropic or `MaterialField` material (a permittivity
            law is accepted where its value is real and positive).
    """
    mesh = model.mesh
    points = np.array([s.position for s in sources], float)
    try:
        located = locate(mesh, layout, points, comm)
    except ValueError as err:
        name = _first_failing(mesh, layout, points)
        raise _mpi.agreed(
            ValueError(
                f"{_label(sources[name])} cannot be placed: {err} A dipole needs a "
                "position inside the solve domain (not in an excluded or "
                "conductor volume)."
            )
        ) from None
    names = mesh._region_names()  # noqa: SLF001
    tags = mesh._topology.volume_tags  # noqa: SLF001
    pml = set(model.pml.regions) if model.pml is not None else set()
    faces = _boundary_faces(model, layout)
    for k, source in enumerate(sources):
        holders = _holders(mesh, located.tetrahedra[k], points[k])
        regions = sorted({names[int(tags[t])] for t in holders})
        layer = [r for r in regions if r in pml]
        if layer:
            raise ValueError(
                f"{_label(source)} lies in the perfectly matched layer region "
                f"{layer[0]!r}: the layer is not a physical medium, so the dipole "
                "has no emitted power there. Move it into the physical region."
            )
        face = _on_face(mesh, holders, points[k], faces)
        if face is not None:
            raise ValueError(
                f"{_label(source)} lies on the {face}; a dipole needs a position "
                "inside the volume (its bulk power is undefined on a boundary "
                "face). Move it off the face."
            )
        materials = {r: model.materials.get(r) for r in regions}
        if (
            len({id(m) for m in materials.values()}) > 1
            and len(set(map(_key, materials.values()))) > 1
        ):
            raise ValueError(
                f"{_label(source)} lies on the interface between regions "
                f"{', '.join(map(repr, regions))}, of different materials: the "
                "field normal to it and the bulk power are ambiguous there. Move "
                "the dipole off the interface."
            )
        _check_material(source, regions[0], materials[regions[0]], frequency)
    return located


def _label(source: _Dipole) -> str:
    return f"{type(source).__name__} {source.name!r}"


def _first_failing(mesh: Mesh, layout: Layout, points: np.ndarray) -> int:
    for k in range(len(points)):
        try:
            locate(mesh, layout, points[k : k + 1])
        except ValueError:
            return k
    return 0  # pragma: no cover - locate raised for some point


def _key(material: Any) -> Any:
    try:
        hash(material)
    except TypeError:
        return id(material)
    return material


def _holders(mesh: Mesh, tet: int, point: np.ndarray) -> np.ndarray:
    """The volume elements holding ``point``, to `ON_FACE`, around element ``tet``."""
    from gfem._engine.probe import _barycentric, _tets_of_nodes

    nodes = np.asarray(mesh.nodes, float)
    tets = np.asarray(mesh._topology.volume_elements)  # noqa: SLF001
    of_node, start = _tets_of_nodes(mesh)
    around = np.unique(
        np.concatenate([of_node[start[n] : start[n + 1]] for n in tets[tet]])
    )
    lowest = _barycentric(
        nodes, tets[around], np.repeat(point[None, :], len(around), axis=0)
    ).min(axis=1)
    return around[lowest >= -ON_FACE]


def _boundary_faces(model: Any, layout: Layout) -> dict[tuple[int, ...], str]:
    """The faces a dipole may not lie on, by sorted node triple, described.

    Every face of a boundary condition or a port (interior sheets
    included), and every face of the solve domain's boundary.
    """
    from gfem._engine.probe import in_support

    mesh = model.mesh
    out: dict[tuple[int, ...], str] = {}
    tets = np.asarray(mesh._topology.volume_elements)  # noqa: SLF001
    support = in_support(mesh, layout)
    faces = np.sort(tets[support][:, _FACES], axis=2).reshape(-1, 3)
    unique, counts = np.unique(faces, axis=0, return_counts=True)
    for f in unique[counts == 1]:
        out[tuple(int(v) for v in f)] = "boundary of the solve domain"
    triangles = np.sort(np.asarray(mesh.triangles), axis=1)
    for item in (*model.boundaries, *model.ports):
        kind = _kind(item)
        if kind is None:
            continue
        for surface in surfaces_of(item):
            what = f"{kind} face {surface!r}"
            if isinstance(item, (LumpedPort, WavePort, FloquetPort)):
                what = f"{kind} face {item.name!r} (surface {surface!r})"
            for f in triangles[mesh.elements(surface)]:
                out[tuple(int(v) for v in f)] = what
    return out


def _kind(item: Any) -> str | None:
    kinds = {
        PEC: "PEC",
        PMC: "PMC",
        Absorbing: "Absorbing",
        Conductor: "Conductor",
        Impedance: "Impedance",
        LumpedPort: "lumped port",
        WavePort: "wave port",
        FloquetPort: "Floquet port",
    }
    for cls, name in kinds.items():
        if isinstance(item, cls):
            return name
    return None


def _on_face(
    mesh: Mesh,
    holders: np.ndarray,
    point: np.ndarray,
    faces: dict[tuple[int, ...], str],
) -> str | None:
    """The description of a refused face holding ``point``, or ``None``."""
    from gfem._engine.probe import _barycentric

    nodes = np.asarray(mesh.nodes, float)
    tets = np.asarray(mesh._topology.volume_elements)  # noqa: SLF001
    bary = _barycentric(nodes, tets[holders], np.repeat(point[None], len(holders), 0))
    for t, b in zip(holders, bary, strict=True):
        for i in np.flatnonzero(b < ON_FACE):
            key = tuple(sorted(int(v) for v in tets[t][_FACES[i]]))
            if key in faces:
                return faces[key]
    return None


def _check_material(
    source: _Dipole, region: str, material: Any, frequency: Sequence[float]
) -> None:
    from gfem._materials import Material, PermittivityLaw

    where = f"{_label(source)} lies in region {region!r}"
    if not isinstance(material, Material):
        # a value error: the position, not the material's type, is wrong
        raise ValueError(  # noqa: TRY004
            f"{where}, whose material is a {type(material).__name__}: the bulk power "
            "is defined in a constant gfem.Material only. Give the elements around "
            "the dipole a region of their own with a constant material."
        )
    if not material.isotropic:
        raise ValueError(
            f"{where}, whose material is anisotropic: this version has no bulk "
            "power of a dipole in an anisotropic medium. Move the dipole into an "
            "isotropic region."
        )
    lossy = "a loss" if material.tan_delta or material.conductivity else None
    eps = material.eps_r
    if isinstance(eps, PermittivityLaw):
        values = [complex(np.asarray(eps(f))) for f in frequency]
        if any(abs(v.imag) > _REAL * abs(v) for v in values):
            lossy = "a lossy permittivity law"
        elif any(v.real <= 0 for v in values):
            lossy = "a permittivity law whose value is not positive"
    if lossy is not None:
        raise ValueError(
            f"{where}, which has {lossy} at the solve frequencies: the power a "
            "dipole emits into an absorbing medium grows without bound as the "
            "mesh is refined. Move the dipole into a lossless region, or remove "
            "the loss around it."
        )


def point_loads(
    model: Any,
    sources: Sequence[_Dipole],
    layout: Layout,
    space: ngs.FESpace,
    frequency: Sequence[float],
    *,
    dof_layout: DofLayout | None = None,
    comm: Any = None,
) -> PointLoads:
    """The dipoles' `PointLoads` on ``space`` (the fields' space, ``layout``).

    Raises:
        ValueError: See `check_positions`.
    """
    located = check_positions(model, sources, layout, frequency, comm)
    probed = probe(model.mesh, layout, located, comm)
    moments = np.array([s.moment for s in sources], complex)
    magnetic = np.array([isinstance(s, MagneticDipole) for s in sources])
    # one shape-function pass for all dipoles; a column holds only its element's DOFs
    vectors = probed.columns(moments, magnetic).astype(complex).tocsr()
    free = free_mask(space.FreeDofs(), space.ndof)
    vectors = (scipy.sparse.diags(free.astype(float)) @ vectors[: space.ndof]).tocsc()
    return PointLoads(
        names=tuple(s.name for s in sources),
        moments=moments,
        magnetic=magnetic,
        vectors=vectors,
        located=located,
        scale=float(probed._built.scale),  # noqa: SLF001
        layout=dof_layout or ONE_PROCESS,
    )


def transport(
    mesh: Mesh,
    located: Located,
    k: int,
    *,
    values: np.ndarray,
    gradients: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""``weight · δl(r_0)`` per vertex and axis of dipole ``k``'s element.

    ``values`` ``(m, 3)`` and ``gradients`` ``(m, 3, 3)`` (``[i, j]`` is
    :math:`\partial_j l_i`) are ``m`` fields at the point, SI; ``weight``
    ``(m, 3)``. Returns the element's four nodes and ``(m, 4, 3)``: row
    ``(a, i, c)`` is :math:`w_a\cdot\delta l_a` for a unit displacement of
    vertex ``i`` along axis ``c``, per metre.
    """
    tet = int(located.tetrahedra[k])
    vertices = np.asarray(mesh.tetrahedra[tet])
    x = np.asarray(mesh.nodes, float)[vertices]
    edges = np.stack([x[i] - x[3] for i in range(3)], axis=1)
    inverse = np.linalg.inv(edges)  # rows: gradients of barycentric 0..2
    grad_bary = np.vstack([inverse, -inverse.sum(axis=0)])  # (4, 3)
    bary = np.asarray(located.barycentric[k])  # (4,)
    # (w . grad λ_i) l_c + λ_i (w . ∂_c l)
    first = np.einsum("ad,id,ac->aic", weight, grad_bary, values)
    second = np.einsum("i,ad,adc->aic", bary, weight, gradients)
    return vertices, -(first + second)
