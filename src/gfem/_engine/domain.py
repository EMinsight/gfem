"""The solve domain of each physics on the NGSolve mesh.

Excluded volume entities are left out of every space of their physics
(``definedon`` the remaining domains, then ``Compress``), so their shells
become exterior boundaries of the solve (the conductor is cracked out of
the mesh).
Each function takes the physics (``"em"``, ``"thermal"``,
``"dc_conduction"``); the electromagnetic domain is the default.
"""

from __future__ import annotations

import ngsolve as ngs
import numpy as np

from gfem._boundaries import PEC, Absorbing, Conductor, Impedance
from gfem._engine.mesh import EngineMesh
from gfem._periodic import pair_names
from gfem.model import Model


def em_region(model: Model, built: EngineMesh, physics: str = "em") -> ngs.Region:
    """The NGSolve volume region of the model's solve domain of ``physics``.

    Args:
        model: The model; ``built`` must come from ``model.mesh``.
        built: The engine mesh.
        physics: ``"em"`` (electromagnetic), ``"thermal"`` or
            ``"dc_conduction"``.
    """
    out = set(model._domain_of(physics).excluded)  # noqa: SLF001
    bits = ngs.BitArray(len(built.domains))
    bits.Clear()
    for k, key in enumerate(built.domains):
        if key not in out:
            bits.Set(k)
    return ngs.Region(built.mesh, ngs.VOL, bits)


def region_domain(
    model: Model, built: EngineMesh, name: str, physics: str = "em"
) -> ngs.Region:
    """The NGSolve volume region of region ``name`` inside the solve domain.

    Args:
        model: The model; ``built`` must come from ``model.mesh``.
        built: The engine mesh.
        name: A volume region of the mesh. Its entities excluded from
            ``physics`` are left out, so the region holds exactly its active
            tetrahedra.
        physics: As `em_region`.
    """
    tag = model.mesh.region(name).tag
    out = {e for t, e in model._domain_of(physics).excluded if t == tag}  # noqa: SLF001
    bits = ngs.BitArray(len(built.domains))
    bits.Clear()
    for k, key in enumerate(built.domains):
        if key[0] == tag and key[1] not in out:
            bits.Set(k)
    return ngs.Region(built.mesh, ngs.VOL, bits)


def hcurl_space(
    model: Model,
    built: EngineMesh,
    order: int,
    *,
    complex_valued: bool = True,
    dirichlet: tuple[str, ...] = (),
    periodic: bool = False,
    phases: tuple[complex, ...] | None = None,
) -> ngs.FESpace:
    """First-kind Nédélec space of ``order`` on the electromagnetic domain.

    DOFs of excluded volumes are compressed away, so the space's size is the
    system size. The tangential DOFs on the ``dirichlet`` surfaces (exact
    names, e.g. `PEC`) are fixed: left out of ``FreeDofs()``, which BDDC
    reads. The DOF numbering does not depend on ``dirichlet``.

    With ``periodic``, the space is periodic over the point identifications
    of ``built`` (the model's periodic pairs, `to_ngsolve`): each second
    surface's DOFs are its twins' on the first, times the pair's Bloch
    phase (``phases``, `gfem._periodic.bloch_phases`; ``None`` for plain
    periodicity), and are compressed away. Its DOF numbering is not the
    unpaired space's: map with the expansion P, never by index.
    """
    space = ngs.HCurl(
        built.mesh,
        order=order,
        type1=True,
        complex=complex_valued,
        definedon=em_region(model, built),
        dirichlet=_bc_numbers(built, dirichlet),
    )
    return _compressed(space, periodic=periodic, phases=phases)


def element_dofs(space: ngs.FESpace) -> np.ndarray:
    """The positions of ``space``'s DOFs that a local volume element uses, sorted.

    Read from the diagonal of a local volume mass form, with no exchange. On a
    paired distributed mesh, the unpaired space also holds the DOFs of partner
    triangles a rank has without their tetrahedra; these rows are the plain
    mesh's field layout, element by element.
    """
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=True, diagonal=True)
    form += u * v * ngs.dx
    form.Assemble()
    matrix = getattr(form.mat, "local_mat", form.mat)
    diagonal = np.asarray(matrix.AsVector().FV().NumPy())
    return np.flatnonzero(np.abs(diagonal) > 0)


def _compressed(
    space: ngs.FESpace, *, periodic: bool, phases: tuple[complex, ...] | None
) -> ngs.FESpace:
    """``space`` compressed, periodic first with ``periodic`` (Bloch ``phases``)."""
    if periodic:
        space = ngs.Periodic(space, phase=None if phases is None else list(phases))
    return ngs.Compress(space)


def _bc_numbers(built: EngineMesh, names: tuple[str, ...]) -> list[int]:
    """1-based NGSolve boundary numbers of these surface names (exact names)."""
    wanted = set(names)
    return [j + 1 for j, n in enumerate(built.mesh.GetBoundaries()) if n in wanted]


def pec_surfaces(model: Model) -> tuple[str, ...]:
    """The `PEC` surfaces of the model, sorted: the Dirichlet surfaces of a space."""
    return tuple(sorted({b.surface for b in model.boundaries if isinstance(b, PEC)}))


def potential_surfaces(model: Model) -> tuple[str, ...]:
    """The surfaces where the divergence-free projection fixes its potential.

    The `PEC` surfaces and every wall with a stiffness or damping term: a
    `Conductor`, an `Absorbing`, an `Impedance` with a resistance or an
    inductance. The gradient of a potential that vanishes on such a wall has
    no tangential trace there, so the wall term does not act on it and it
    is a null vector of the curl-curl operator; a gradient with a
    tangential part on the wall is not. A capacitance-only
    `Impedance` adds to the mass term alone and port surfaces carry a term
    along one tangential direction: their potential stays free. Sorted.
    """
    lossy = {
        b.surface
        for b in model.boundaries
        if isinstance(b, (Conductor, Absorbing))
        or (
            isinstance(b, Impedance)
            and (b.resistance is not None or b.inductance is not None)
        )
    }
    return tuple(sorted({*pec_surfaces(model), *lossy}))


def domain_key(model: Model, physics: str = "em") -> tuple:
    """The solve domain of ``physics``, as a cache key.

    Two models with the same mesh topology and entities excluded from
    ``physics`` share its spaces.
    """
    return (physics, model._domain_of(physics).excluded)  # noqa: SLF001


def space_key(
    model: Model,
    kind: str,
    order: int,
    *,
    periodic: bool = False,
    phases: tuple[complex, ...] | None = None,
    real: bool = False,
    fixed: tuple[str, ...] | None = None,
    physics: str = "em",
) -> tuple:
    """The cache key of a ``kind`` space (``"hcurl"``, ``"h1"``) of ``order``.

    Its ``fixed`` surfaces fixed (``None``: the `PEC` ones, `pec_surfaces`);
    with ``periodic``, periodic over the model's pairs with Bloch ``phases``
    (``None``: plain periodicity); ``real`` for a real-valued space. The one
    key builder of `gfem._engine.cache.Discretization.space` for these
    spaces, so a paired and an unpaired space never share a key, nor two
    spaces fixed on different surfaces. ``physics`` picks the solve domain
    (`em_region`).
    """
    if fixed is None:
        fixed = pec_surfaces(model)
    key = (domain_key(model, physics), kind, order, fixed)
    if periodic:
        key = (*key, pair_names(model._periodic), phases)  # noqa: SLF001
    return (*key, "real") if real else key


def h1_space(
    model: Model,
    built: EngineMesh,
    order: int,
    dirichlet: tuple[str, ...] = (),
    *,
    periodic: bool = False,
    phases: tuple[complex, ...] | None = None,
    physics: str = "em",
    complex_valued: bool = False,
) -> ngs.FESpace:
    """H1 space of ``order`` on the domain of ``physics``, compressed.

    The DOFs on the ``dirichlet`` surfaces (exact names) are fixed: they are
    left out of ``FreeDofs()``, which BDDC reads. With ``periodic``, periodic
    over the model's pairs as `hcurl_space`; Bloch ``phases`` make it
    complex, as does ``complex_valued`` (real by default). ``physics`` picks
    the solve domain (`em_region`; the electromagnetic one by default).
    """
    space = ngs.H1(
        built.mesh,
        order=order,
        complex=complex_valued or phases is not None,
        definedon=em_region(model, built, physics),
        dirichlet=_bc_numbers(built, dirichlet),
    )
    return _compressed(space, periodic=periodic, phases=phases)


def boundary_region(built: EngineMesh, surfaces: tuple[str, ...]) -> ngs.Region:
    """The NGSolve boundary region of these surface names (exact names)."""
    names = built.mesh.GetBoundaries()
    bits = ngs.BitArray(len(names))
    bits.Clear()
    wanted = set(surfaces)
    for j, name in enumerate(names):
        if name in wanted:
            bits.Set(j)
    return ngs.Region(built.mesh, ngs.BND, bits)
