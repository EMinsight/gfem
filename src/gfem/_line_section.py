"""The cross-section of a line port: a straight line of a 2D mesh with materials.

Pure NumPy on the `Model` arrays, the 2D mesh's counterpart of
`gfem._port_section.PortSection`; the engine turns it into a 1D NGSolve mesh.
The port mode problem reads the material of the area element behind each
segment of the line and, at each of its two ends, the boundary condition of
the lines that meet it there.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np

from gfem._boundaries import WavePort
from gfem._domain import physics_of
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem.errors import PortGeometryError
from gfem.mesh import LENGTH_UNITS
from gfem.model import Model

#: The conditions a line port's end takes, by precedence: the first that a
#: line meeting the end carries wins, and an end that meets none is PMC.
END_KINDS = ("pec", "absorbing", "pmc")


@dataclasses.dataclass(frozen=True, eq=False)
class LineSection:
    r"""A line port's straight cross-section, ready for the port mode problem.

    Coordinates are local to the line, in the mesh file's length unit
    (``scale`` metres). The in-plane frame ``(t, n)`` has
    :math:`\hat t \times \hat n = \hat z`, and ``n`` points into the solve
    domain, the direction of propagation of an incident wave.

    Attributes:
        port: The wave port.
        scale: Metres per local coordinate unit.
        origin: A point of the line, ``(2,)`` in local units of the global frame.
        frame: Rows ``t`` and ``n`` (unit vectors, global frame), ``(2, 2)``.
        points: The coordinate along ``t`` of each node, ``(n_nodes,)``. Nodes
            are in increasing gfem node number.
        segments: ``(n_seg, 2)`` node indices of each segment.
        eps: Complex relative permittivity of the area element behind each
            segment, ``(n_seg,)``, without conductivity; a law segment's
            eps_inf.
        mu: Complex relative permeability, ``(n_seg,)``.
        sigma: Electrical conductivity, S/m, ``(n_seg,)``.
        faces: The mesh segment (a row of `gfem.Mesh.surface_elements`) of
            each of ``segments``.
        ends: The two end nodes, ``(2,)``.
        end_kinds: Each end's condition, one of `END_KINDS`.
        end_segments: The segment at each end, ``(2,)``.
        laws: ``(region, law)`` per region of the section whose eps_r is a
            `gfem.PermittivityLaw`; empty without one.
        law_index: ``(n_seg,)`` each segment's index in ``laws``, -1 off
            them; ``None`` without laws.
    """

    port: WavePort
    scale: float
    origin: np.ndarray
    frame: np.ndarray
    points: np.ndarray
    segments: np.ndarray
    eps: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    faces: np.ndarray
    ends: np.ndarray
    end_kinds: tuple[str, str]
    end_segments: np.ndarray
    laws: tuple[tuple[str, PermittivityLaw], ...] = ()
    law_index: np.ndarray | None = None

    #: A line port has no impedance edges (the sweep's slow-wave bound).
    impedance_terms = ()

    @property
    def normal(self) -> np.ndarray:
        """Unit inward normal, global frame, ``(2,)``."""
        return self.frame[1]

    @property
    def diameter(self) -> float:
        """The line's length, local units."""
        return float(np.ptp(self.points))

    @property
    def max_conductivity(self) -> float:
        """The largest conductivity over the section, S/m."""
        return float(self.sigma.max())

    def index_squared(
        self, sigma_term: float = 0.0, frequency: float | None = None
    ) -> tuple[float, float]:
        """Bounds on the squared bulk index over the section.

        Args:
            sigma_term: ``max_conductivity / (omega eps0)`` to add to ``|eps|``.
            frequency: Hz, where the laws are read (`eps_at`); without it, a
                law segment counts its eps_inf.

        Returns:
            ``max Re(eps) max Re(mu)`` (the shift) and ``(max |eps| +
            sigma_term) max |mu|`` (an upper bound on the ``|n|^2`` of any
            propagating mode).
        """
        eps = self.eps if frequency is None else self.eps_at(frequency)
        return (
            float(eps.real.max() * self.mu.real.max()),
            float((np.abs(eps).max() + sigma_term) * np.abs(self.mu).max()),
        )

    def eps_at(self, frequency: float) -> np.ndarray:
        """``eps`` with each law's permittivity at ``frequency`` (Hz)."""
        if not self.laws:
            return self.eps
        eps = self.eps.copy()
        index = np.asarray(self.law_index)
        for g, chi in enumerate(self.susceptibilities(frequency)):
            eps[index == g] += chi
        return eps

    def susceptibilities(self, frequency: float) -> list[complex]:
        """The susceptibility chi of each of ``laws`` at ``frequency`` (Hz)."""
        from gfem._engine.dispersion import susceptibility  # lazy: no engine at import

        omega = 2 * math.pi * frequency
        return [susceptibility(law, omega) for _, law in self.laws]

    def check_frequencies(self, frequencies) -> None:
        """Raise where a law of the section cannot be solved (see `PortSection`).

        Raises:
            ValueError: A frequency lies on a lossless pole of a law, or a
                law region has Re eps <= 0 there.
        """
        from gfem._port_section import check_law_frequencies

        check_law_frequencies(self.port, self.laws, self.susceptibilities, frequencies)

    def to_local(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates along ``t`` ``(n,)`` and distances along ``n`` of points (m)."""
        rel = np.asarray(points_m, float) / self.scale - self.origin
        return rel @ self.frame[0], rel @ self.frame[1]

    def to_global(self, local: np.ndarray) -> np.ndarray:
        """Global points (m) ``(n, 2)`` of coordinates along ``t``."""
        t = np.asarray(local, float)
        return (self.origin + t[:, None] * self.frame[0]) * self.scale


def line_section(model: Model, port: WavePort) -> LineSection:
    """Extract the cross-section of the line port ``port`` from a 2D ``model``.

    Raises:
        PortGeometryError: The line is not straight, not one connected line,
            or has the solve domain on both sides.
    """
    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001 - the role arrays of a 2D mesh
    scale = LENGTH_UNITS[mesh.length_unit]
    nodes = np.asarray(mesh.nodes, float) / scale
    active = model._domain.active  # noqa: SLF001
    tag = mesh.region(port.surface).tag
    nb = topology.surface_neighbours
    sel = np.flatnonzero(topology.surface_tags == tag)
    side_ok = (nb[sel] >= 0) & active[np.maximum(nb[sel], 0)]
    one = side_ok.sum(axis=1) == 1
    sel, side_ok = sel[one], side_ok[one]
    cells = np.where(side_ok[:, 0], nb[sel, 0], nb[sel, 1])
    segs = topology.surface_elements[sel]

    used, local = np.unique(segs, return_inverse=True)
    local = local.reshape(-1, 2)
    xy = nodes[used]
    origin = xy.mean(axis=0)
    *_, vt = np.linalg.svd(xy - origin, full_matrices=False)
    tangent = vt[0]
    normal = np.array([-tangent[1], tangent[0]])
    length = float(np.ptp((xy - origin) @ tangent))
    off = float(np.abs((xy - origin) @ normal).max())
    from gfem._port_section import PLANE_RTOL

    if off > PLANE_RTOL * length:
        raise PortGeometryError(
            f"WavePort {port.name!r}: line {port.surface!r} is not straight (its "
            f"nodes are up to {off * scale:.3g} m off their best-fit line). A line "
            "port must be one straight segment of the boundary; split the line "
            "in gmsh."
        )
    centroid = nodes[topology.volume_elements[cells]].mean(axis=1)
    inward = np.sign((centroid - nodes[segs].mean(axis=1)) @ normal)
    if not (inward == inward[0]).all():  # pragma: no cover - an exterior line
        raise PortGeometryError(
            f"WavePort {port.name!r}: the solve domain lies on both sides of "
            f"{port.surface!r}; a line port must face the domain from one side."
        )
    normal = normal * inward[0]
    tangent = np.array([normal[1], -normal[0]])  # t x n = +z
    frame = np.stack([tangent, normal])
    points = (xy - origin) @ tangent

    degree = np.bincount(local.ravel(), minlength=len(used))
    ends = np.flatnonzero(degree == 1)
    if len(ends) != 2 or (degree > 2).any():
        raise PortGeometryError(
            f"WavePort {port.name!r}: line {port.surface!r} is not one connected "
            "line (its segments have "
            f"{len(ends)} free ends). A line port must be one straight segment "
            "of the boundary; give each piece a port of its own."
        )
    ends = ends[np.argsort(points[ends])]
    end_segments = np.array([np.flatnonzero((local == e).any(axis=1))[0] for e in ends])
    eps, mu, sigma = _materials(model, port, cells)
    laws, law_index = _laws(model, cells)
    return LineSection(
        port=port,
        scale=scale,
        origin=origin,
        frame=frame,
        points=points,
        segments=local,
        eps=eps,
        mu=mu,
        sigma=sigma,
        faces=sel,
        ends=ends,
        end_kinds=_end_kinds(model, port, used[ends]),
        end_segments=end_segments,
        laws=laws,
        law_index=law_index if laws else None,
    )


def _region_names(model: Model) -> dict[int, str]:
    mesh = model.mesh
    return {r.tag: r.name for r in mesh.regions if r.dim == mesh.dim}


def _materials(model: Model, port: WavePort, cells: np.ndarray):
    """Per-segment ``(eps, mu, sigma)`` from the area elements behind them."""
    from gfem._port_section import _field, _permittivity

    mesh = model.mesh
    names = _region_names(model)
    region_of = mesh._topology.volume_tags[cells]  # noqa: SLF001
    n = len(cells)
    eps = np.zeros(n, complex)
    mu = np.zeros(n, complex)
    sigma = np.zeros(n)
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        rows = np.flatnonzero(region_of == rtag)
        material = model.materials[region]
        if isinstance(material, Material):
            eps[rows] = _permittivity(material)[0][0, 0]
            mu[rows] = complex(np.asarray(material.mu_r, complex))
            sigma[rows] = float(np.asarray(material.conductivity, float))
            continue
        if not isinstance(material, MaterialField):  # pragma: no cover
            raise TypeError(f"Unknown material type {type(material).__name__}.")
        if material.blend_ends is not None:
            what = (
                "a layer-grid design"
                if material.blend_ends.pixels is not None
                else "a density blend with a permittivity-law end"
            )
            raise NotImplementedError(
                f"WavePort {port.name!r}: region {region!r} has elements beside "
                f"the port's line and is {what} (MaterialField.interpolate), "
                "which the port mode problem does not read. Keep the design away "
                "from line ports (give the elements beside the port a region of "
                "their own with a constant gfem.Material)."
            )
        pos = np.searchsorted(mesh.elements(region), cells[rows])
        tan = _field(material, "tan_delta", 0.0, pos).real
        eps[rows] = _field(material, "eps_r", 1.0, pos) * (1 - 1j * tan)
        mu[rows] = _field(material, "mu_r", 1.0, pos)
        sigma[rows] = _field(material, "conductivity", 0.0, pos).real
    return eps, mu, sigma


def _laws(model: Model, cells: np.ndarray):
    """The section's laws, ``((region, law), ...)``, and each segment's index."""
    names = _region_names(model)
    laws: list[tuple[str, PermittivityLaw]] = []
    index = np.full(len(cells), -1, np.int64)
    region_of = model.mesh._topology.volume_tags[cells]  # noqa: SLF001
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        material = model.materials[region]
        if isinstance(material, Material) and isinstance(
            material.eps_r, PermittivityLaw
        ):
            index[region_of == rtag] = len(laws)
            laws.append((region, material.eps_r))
    return tuple(laws), index


def _end_kinds(model: Model, port: WavePort, nodes: np.ndarray) -> tuple[str, str]:
    """The condition of the lines that meet each of the end ``nodes`` (gfem).

    Another line meets an end when one of its segments holds the end node and
    borders the solve domain. Its kind follows the 3D port section's rule
    (another wave port and ``pec_edges`` are PEC), with the precedence of
    `END_KINDS`.
    """
    from gfem._port_section import _edge_kinds

    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001
    nb = topology.surface_neighbours
    active = model._domain.active  # noqa: SLF001
    touches = ((nb >= 0) & active[np.maximum(nb, 0)]).any(axis=1)
    other = (topology.surface_tags != mesh.region(port.surface).tag) & touches
    names = {r.tag: r.name for r in mesh.regions if r.dim == mesh.dim - 1}
    kinds = _edge_kinds(model, port)
    out = []
    for node in nodes:
        meeting = other & (topology.surface_elements == node).any(axis=1)
        found = {
            "absorbing" if kind == "absorbing2" else kind
            for kind in (
                kinds.get(names[int(t)])
                for t in np.unique(topology.surface_tags[meeting])
            )
        }
        out.append(next((k for k in END_KINDS if k in found), "pmc"))
    return out[0], out[1]


#: The material properties the port mode problem reads.
_PORT_PROPERTIES = ("eps_r", "mu_r", "tan_delta", "conductivity")


def line_section_key(model: Model, port: WavePort) -> str:
    """A digest of what `line_section` reads of ``model`` beyond its mesh.

    As `gfem._port_section.port_section_key`: the length unit, the solve
    domain, the port, every electromagnetic boundary condition and port, and
    the material values of the solve-domain area elements beside the line.
    """
    mesh = model.mesh
    topology = mesh._topology  # noqa: SLF001
    domain = model._domain  # noqa: SLF001
    digest = hashlib.blake2b(digest_size=16)

    def add(*parts: object) -> None:
        digest.update(repr(parts).encode())

    add(mesh.length_unit, domain.excluded, port, model.ports)
    add(tuple(b for b in model.boundaries if physics_of(b) == "em"))
    nb = topology.surface_neighbours[mesh.elements(port.surface)]
    cells = np.unique(nb[(nb >= 0) & domain.active[np.maximum(nb, 0)]])
    names = _region_names(model)
    region_of = topology.volume_tags[cells]
    for rtag in np.unique(region_of):
        region = names[int(rtag)]
        material = model.materials[region]
        add(region, type(material).__name__)
        if isinstance(material, Material):
            add(*(getattr(material, n) for n in _PORT_PROPERTIES))
            continue
        pos = np.searchsorted(mesh.elements(region), cells[region_of == rtag])
        for name in _PORT_PROPERTIES:
            value = getattr(material, name)
            if value is None:
                add(name, None)
                continue
            values = np.asarray(value)
            if values.ndim == 1:
                values = values[pos]
            add(name, values.dtype.str, values.shape)
            digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()
