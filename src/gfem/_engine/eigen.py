"""The NGSolve side of the eigenmode solve (`gfem._eigen` stays NGSolve-free).

`eigen_operator` assembles the eigenmode operator ``T(k) = K + k D + k^2 M
+ A(k)`` from the driven operator's pieces, sorted by `gfem._eigen.operator.
split`: ``D``, ``M`` and each nonpolynomial ``A_p`` become NumPy linear maps
of the H(curl) space, and ``T(k)`` is a `gfem._linalg` handle of the driven
form at ``ω = k c / scale``. `projector` builds the divergence-free
projection (`DivFreeProjector`) and `element_energy` the electric energy of
a mode per element.

Under a Bloch wave vector the operator is ``"general"`` (the driven
operator's `DrivenOperator.symmetry`): every map is assembled, because a
matrix-free form's transpose silently applies the form itself on a
Bloch-phased space, and the projection is complex and periodic with the same
phases.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ngsolve as ngs
import numpy as np
import scipy.constants as const
import scipy.sparse as sp

from gfem import _linalg, _timing
from gfem._eigen.operator import EigenOperator, LinearMap, NonPolynomial, split
from gfem._eigen.seed import MIN_SUBSPACE
from gfem._engine import dispersion, quadrature, terms
from gfem._engine.cache import discretization
from gfem._engine.coefficients import (
    coefficient,
    element_tensors,
    principal_permittivity,
)
from gfem._engine.domain import (
    h1_space,
    hcurl_space,
    pec_surfaces,
    potential_surfaces,
    space_key,
)
from gfem._engine.driven import DrivenOperator, real_space
from gfem._linalg import Handle, Operator, ResolvedLinearSolver
from gfem._linalg.base import free_mask, host_bytes
from gfem._linalg.layout import ONE_PROCESS, DofLayout, RankLayout
from gfem._linalg.petsc import FreeNumbering
from gfem._materials import PermittivityLaw
from gfem._periodic import bloch_phases
from gfem.model import Model

__all__ = [
    "DivFreeProjector",
    "assemble",
    "eigen_operator",
    "element_energy",
    "energy_parts",
    "law_scale",
    "matvec",
    "principal_energy",
    "projector",
    "real_eps",
    "scale_per_tetrahedron",
]

#: Relative weight of the mass term that makes the H1 form definite.
_REGULARIZATION = 1e-10


def eigen_operator(
    op: DrivenOperator,
    sigma: float,
    *,
    linear: ResolvedLinearSolver,
    tolerance: float,
    numbers: np.ndarray | None = None,
) -> EigenOperator:
    """The eigen operator ``T(k)`` of the driven operator, shifted at ``sigma``.

    Nothing is factorized until the handle is first used. ``linear`` at
    ``tolerance`` is the linear solver of every shift (`gfem._linalg`), of
    the driven operator's symmetry. A ``"general"`` operator's damping and
    nonpolynomial maps are assembled, so their transposes are exact.

    Under ranks (``op.layout``) every shift is prepared on the layout, the seed's
    rows are the owned free DOFs (`gfem._linalg.petsc.FreeNumbering`), and
    ``numbers`` are each local DOF's one-process number
    (`gfem._engine.fields.one_process_dofs`).
    """
    space, scale = op.space, op.built.scale
    classes = split(op.pieces, scale, sigma)
    powers = classes.powers
    # the constant θ of polynomial law pieces (lossless Drude), out of the
    # curl energy of the longitudinal-mode test
    constant = [(c, p) for c, p in powers[0] if p.law is not None]
    free = free_mask(space.FreeDofs(), space.ndof)
    symmetry, matrix_free = op.symmetry, op.symmetric
    layout = op.layout or ONE_PROCESS
    numbering = None
    if isinstance(op.layout, RankLayout):
        numbering = FreeNumbering(
            op.layout.pardofs,
            space.FreeDofs(),
            ngs.GridFunction(space).vec.CreateVector(),
            op.layout.comm,
        )

    def _linear_map(matrix) -> LinearMap:
        return _Map(matrix, layout)

    def operator(k: complex) -> Operator:
        return Operator(
            op.form(k * const.c / scale), space.FreeDofs(), layout=op.layout
        )

    def prepare_shift(t: Operator, solves: int = MIN_SUBSPACE) -> Handle:
        """Prepare ``t``: ``T(k)`` at one shift (`operator`), for ``solves``.

        The seed solve applies it `MIN_SUBSPACE` times or more; a refinement
        reshift `gfem._eigen.operator.RESHIFT_SOLVES` times.
        """
        return linear.prepare(
            t,
            symmetry=symmetry,
            width=1,
            solves=solves,
            tolerance=tolerance,
        )

    return EigenOperator(
        shift=sigma,
        operator=operator,
        prepare=prepare_shift,
        damping=(
            _linear_map(
                assemble(
                    space, powers[1], matrix_free=matrix_free, symmetric=op.symmetric
                )
            )
            if powers[1]
            else None
        ),
        mass=_linear_map(assemble(space, powers[2], symmetric=op.symmetric)),
        free=free,
        scale=scale,
        nonpolynomial=tuple(
            NonPolynomial.from_piece(
                piece,
                _linear_map(
                    assemble(
                        space,
                        [(1.0, piece)],
                        matrix_free=matrix_free,
                        symmetric=op.symmetric,
                    )
                ),
                scale,
            )
            for piece in classes.nonpolynomial
        ),
        symmetry=symmetry,
        law_constant=(
            _linear_map(assemble(space, constant, symmetric=op.symmetric))
            if constant
            else None
        ),
        layout=layout,
        numbers=numbers,
        numbering=numbering,
    )


def assemble(
    space: ngs.FESpace,
    items: list[tuple[complex, terms.Piece]],
    *,
    matrix_free: bool = False,
    symmetric: bool = True,
):
    """``Σ c_p a_p`` as an NGSolve operator.

    Assembled, or applied element by element when ``matrix_free`` (no
    matrix stored: the damping is mostly boundary terms, which an assembled
    matrix would hold in a graph of the whole space). ``symmetric``: the
    form's flag, False on a Bloch-phased space (A^T != A).
    """
    u, v = space.TnT()
    form = ngs.BilinearForm(space, symmetric=symmetric, nonassemble=matrix_free)
    for c, piece in items:
        form += c * piece.integrand(u, v)
    if not matrix_free:
        with _timing.phase("assemble"):
            form.Assemble()
    return form.mat


def matvec(
    matrix,
    x: np.ndarray,
    *,
    transpose: bool = False,
    layout: DofLayout = ONE_PROCESS,
) -> np.ndarray:
    """An NGSolve matrix (``transpose``: its transpose) applied to a NumPy vector.

    The transpose of an assembled matrix only: a matrix-free form's ``.T``
    applies the form itself on a Bloch-phased space. ``x`` is a state column of
    ``layout`` (marked so, as the driven solves mark theirs), the result a load.
    """
    a = matrix.CreateColVector()
    a.FV().NumPy()[:] = x
    layout.mark_state(a)
    b = matrix.CreateColVector()
    b.data = (matrix.T if transpose else matrix) * a
    return b.FV().NumPy().copy()


class _Map:
    """``matrix @ x`` (or ``matrix.T @ x``) on NumPy vectors, and its bytes.

    ``x`` is a state of ``layout``, the result a load (`matvec`).
    """

    def __init__(self, matrix, layout: DofLayout = ONE_PROCESS) -> None:
        self.matrix = matrix
        self.layout = layout

    def __call__(self, x: np.ndarray, /, *, transpose: bool = False) -> np.ndarray:
        """``matrix x``, or ``matrix^T x``."""
        return matvec(self.matrix, x, transpose=transpose, layout=self.layout)

    @property
    def nbytes(self) -> int:
        """The assembled matrix's host bytes; 0 for a matrix-free form."""
        return host_bytes(self.matrix)


def real_eps(model: Model) -> np.ndarray:
    """``Re(eps_r)`` per tetrahedron, ``(n,)`` or ``(n, 3, 3)``: the projector's weight.

    A permittivity law region reads its eps_inf
    (`gfem._engine.dispersion.high_frequency_eps`): the
    divergence-free projection's inner product and the phase rule's form.
    The energies integrate it too, and `law_scale` turns a law region's
    eps_inf into the weight at a frequency. A model without a law reads
    ``element_tensors`` as before (bit-identical).
    """
    if not dispersion.law_regions(model):
        return np.real(element_tensors(model, "eps_r"))
    return np.real(dispersion.high_frequency_eps(model))


def law_scale(
    model: Model, omega: float, weight: Callable[[PermittivityLaw, float], float]
) -> dict[str, float]:
    """``weight(law, omega) / eps_inf`` by law region: the energies' factor there.

    `real_eps` holds eps_inf on a law region, and an energy integrates it
    times ``|x|^2`` per element: the factor turns that into ``weight`` times
    ``|x|^2`` (`gfem._engine.dispersion.energy_eps` for the mode energy,
    `gfem._engine.dispersion.brillouin_eps` for `gfem.electric_energy`).
    Empty without a law.
    """
    return {
        lr.region: weight(lr.law, omega) / float(lr.law.eps_inf)
        for lr in dispersion.law_regions(model)
    }


def scale_per_tetrahedron(model: Model, factors: dict[str, float]) -> np.ndarray:
    """`law_scale`'s ``factors`` per tetrahedron (gfem order), 1 elsewhere."""
    mesh = model.mesh
    out = np.ones(len(mesh.tetrahedra))
    for region, factor in factors.items():
        out[mesh.elements(region)] = factor
    return out


def projector(
    model: Model,
    op: DrivenOperator,
    wave_vector: tuple[float, float, float] | None = None,
) -> DivFreeProjector:
    """The divergence-free projection on ``op``'s space (`DivFreeProjector`).

    The H(curl) space has the solve's DOF numbering and `PEC` DOFs; the H1
    space of the same order is fixed on the `PEC` surfaces and on every
    lossy wall (`gfem._engine.domain.potential_surfaces`). Both
    are real
    (`gfem._engine.driven.real_space` without pairs), and periodic over a
    model's pairs. Under a Bloch ``wave_vector`` (rad/m) the H(curl) space
    is the solve's own complex one and the H1 space complex, with the same
    Bloch phases. Under ranks (``op.layout``) both spaces are the distributed
    mesh's, and the projection maps states to states (`DivFreeProjector`).
    """
    built, order = op.built, op.order
    periodic = model._periodic  # noqa: SLF001
    disc = discretization(model.mesh, periodic, built.comm)
    pec = pec_surfaces(model)
    fixed = potential_surfaces(model)
    if not periodic:
        hcurl = real_space(model, op)
        h1_key = space_key(model, "h1", order, fixed=fixed)
        h1 = disc.space(h1_key, lambda: h1_space(model, built, order, fixed))
    else:
        phases = bloch_phases(model, wave_vector)
        h1_key = space_key(
            model, "h1", order, periodic=True, phases=phases, fixed=fixed
        )
        h1 = disc.space(
            h1_key,
            lambda: h1_space(model, built, order, fixed, periodic=True, phases=phases),
            group=phases,
        )
        if phases is None:
            hcurl = disc.space(
                space_key(model, "hcurl", order, periodic=True, real=True),
                lambda: hcurl_space(
                    model,
                    built,
                    order,
                    complex_valued=False,
                    dirichlet=pec,
                    periodic=True,
                ),
            )
        else:
            hcurl = op.space
    eps = coefficient(built, real_eps(model))
    nodes = np.asarray(model.mesh.nodes) / built.scale
    diameter = float(np.linalg.norm(np.ptp(nodes, axis=0)))
    h1_layout = None
    if op.layout is not None:
        h1_layout = disc.space((*h1_key, "layout"), lambda: RankLayout(h1, built.comm))
    return DivFreeProjector(
        hcurl,
        h1,
        eps,
        order,
        diameter,
        layout=op.layout,
        h1_layout=h1_layout,
        periodic=bool(periodic),
    )


def element_energy(
    model: Model, op: DrivenOperator
) -> Callable[[np.ndarray], np.ndarray]:
    """``x -> ∫ Re(eps_r) x . conj(x)`` per element of ``op``'s space.

    The returned function takes a coefficient vector of ``op.space`` and
    gives the integral over each tetrahedron in the engine mesh's element
    order (`gfem._engine.mesh.EngineMesh.cell_order` maps it to the
    mesh's), in the mesh's file units, at quadrature order ``2 order``.
    ``Re(eps_r)`` is `real_eps`'s: eps_inf on a law region, which the
    caller scales to its weight at a frequency (`law_scale`). Under ranks the
    integrals are those of the rank's own tetrahedra
    (`gfem._engine.mesh.EngineMesh.cell_order` is then the rank's).
    """
    built = op.built
    eps = coefficient(built, real_eps(model))
    gf = ngs.GridFunction(op.space)
    layout = op.layout or ONE_PROCESS

    def density(x: np.ndarray) -> np.ndarray:
        gf.vec.FV().NumPy()[:] = x
        layout.mark_state(gf.vec)
        values = ngs.Integrate(
            (eps * gf) * ngs.Conj(gf),
            built.mesh,
            order=2 * op.order,
            element_wise=True,
        )
        return values.NumPy()

    return density


class EnergyPart:
    """One part of the mode energy's volume term: ``int C u . v`` on some elements.

    ``C`` is ``Re(eps_r)`` on the part's tetrahedra (a region's, or one
    principal axis's ``Re(eps_k) a_k a_k^T`` of an anisotropic region's),
    zero elsewhere, as `element_energy` and `principal_energy` integrate it:
    eps_inf on a law region (`real_eps`), the phase rule's form there, which
    the mode energy scales by `law_scale` per mode.

    Attributes:
        values: ``C`` per tetrahedron (gfem order), ``(n,)`` or ``(n, 3, 3)``.
        elements: The part's tetrahedra, one boolean per tetrahedron.
        axis: The principal axis of an axis's part, else ``None``.
    """

    def __init__(
        self,
        op: DrivenOperator,
        values: np.ndarray,
        elements: np.ndarray,
        axis: int | None = None,
    ) -> None:
        self.values = values
        self.elements = elements
        self.axis = axis
        self.coefficient = coefficient(op.built, values)
        self._op = op
        self._form: ngs.BilinearForm | None = None

    def apply(self, x: np.ndarray) -> np.ndarray:
        """``A x`` of the part's form (file units), matrix-free."""
        space = self._op.space
        if self._form is None:
            u, v = space.TnT()
            form = ngs.BilinearForm(space, symmetric=True, nonassemble=True)
            form += (self.coefficient * u) * v * quadrature.dx(self._op.order)
            self._form = form
        source = ngs.GridFunction(space).vec
        target = source.CreateVector()
        source.FV().NumPy()[:] = x
        self._form.Apply(source, target)
        return target.FV().NumPy().copy()


def energy_parts(
    model: Model,
    op: DrivenOperator,
    regions: tuple[str, ...],
    anisotropic: tuple[str, ...],
) -> tuple[list[EnergyPart], list[list[EnergyPart]]]:
    """The volume energy's parts per region, and per principal axis.

    Returns one `EnergyPart` per region of ``regions`` (they add up to the
    `element_energy` form on the solve domain), and for each region of
    ``anisotropic`` its three axis parts (adding up to its region part),
    the forms of the eigenmode ratios' explicit energy terms
    (`gfem._engine.sensitivity.energy_term`).
    """
    mesh = model.mesh
    eps = real_eps(model)

    def members(region: str) -> np.ndarray:
        inside = np.zeros(len(mesh.tetrahedra), dtype=bool)
        inside[mesh.elements(region)] = True
        return inside

    def masked(values: np.ndarray, inside: np.ndarray) -> np.ndarray:
        shape = (-1,) + (1,) * (values.ndim - 1)
        return np.where(inside.reshape(shape), values, 0.0)

    parts = [
        EnergyPart(op, masked(eps, inside), inside)
        for inside in (members(r) for r in regions)
    ]
    axes_parts = []
    if anisotropic:
        values, axes = principal_permittivity(model, anisotropic)
        for region in anisotropic:
            inside = members(region)
            axes_parts.append(
                [
                    EnergyPart(
                        op,
                        masked(
                            values[:, k, None, None]
                            * axes[:, k, :, None]
                            * axes[:, k, None, :],
                            inside,
                        ),
                        inside,
                        axis=k,
                    )
                    for k in range(3)
                ]
            )
    return parts, axes_parts


def principal_energy(
    model: Model, op: DrivenOperator, regions: tuple[str, ...]
) -> Callable[[np.ndarray], np.ndarray]:
    """``x -> ∫ eps_k |a_k . x|^2`` per element and principal axis.

    `element_energy` split along the principal axes ``a_k`` of the
    anisotropic ``regions`` (`gfem._engine.coefficients.anisotropic_regions`),
    with their real principal ``eps_k``: ``(n, 3)`` in the engine mesh's
    element order, zero outside ``regions``. The three columns add up to
    `element_energy` on ``regions``.
    """
    built = op.built
    values, axes = principal_permittivity(model, regions)
    parts = [
        coefficient(
            built,
            values[:, k, None, None] * axes[:, k, :, None] * axes[:, k, None, :],
        )
        for k in range(3)
    ]
    gf = ngs.GridFunction(op.space)
    layout = op.layout or ONE_PROCESS

    def density(x: np.ndarray) -> np.ndarray:
        gf.vec.FV().NumPy()[:] = x
        layout.mark_state(gf.vec)
        return np.stack(
            [
                ngs.Integrate(
                    (part * gf) * ngs.Conj(gf),
                    built.mesh,
                    order=2 * op.order,
                    element_wise=True,
                ).NumPy()
                for part in parts
            ],
            axis=1,
        )

    return density


class DivFreeProjector:
    """``x -> x - G (G^H M0 G)^-1 G^H M0 x`` on an H(curl) space.

    The curl-curl operator vanishes on gradients, so every gradient of an
    H1 function (zero on the `PEC` surfaces and the lossy walls, where a
    gradient with a tangential part is no null vector: the wall term acts
    on it) is an eigenvector of the
    eigenmode problem at frequency 0, a huge null space next to the
    physical modes. When the target lies below about half a mode's
    frequency, shift-invert returns these first. Projecting every Arnoldi
    vector onto the complement of the gradients, orthogonal in the
    electric-energy inner product ``M0`` (``∫ Re(eps_r) u . v``), keeps
    them out, with ``G`` the exact embedding of ``grad H1(p)`` into
    first-kind Nédélec of order p and ``G^H M0 G`` the H1 form ``∫ Re(eps_r)
    grad φ . grad ψ``, factorized once. A volume of the solve domain without
    a fixed surface has constant potentials (zero gradient) in the kernel of that
    form; a mass term about 1e-10 of the stiffness removes them without
    affecting the projection.

    The spaces are real (``G`` real, ``G^H = G^T``), or complex under a
    Bloch phase: then ``G``, ``M0`` (Hermitian) and the H1 form (Hermitian
    positive definite, factored by LU, never sparse Cholesky) are complex,
    and a Bloch eigenvector satisfies ``G^H M0 x = 0``, not ``G^T M0 x =
    0``.

    Under ranks (``layout``, real spaces only) ``x`` is a state, ``M0 x`` a load,
    :math:`G^T` maps a load to a load and ``G`` a state to a state, through NGSolve's
    own ``ParallelMatrix`` ``G``, and the H1 form is factored by MUMPS on
    ``h1_layout``. Between periodic spaces ``G`` is instead the
    conversion each rank computes alone, a local CSR, as the periodic
    expansion is (`gfem._engine.driven.expansion_matrix`): NGSolve's parallel
    conversion is wrong on a paired mesh. One process keeps SciPy's
    CSR ``G``.

    Args:
        hcurl: The H(curl) space (the solve's numbering and `PEC` DOFs):
            real, or the complex Bloch-phased solve space.
        h1: The H1 space of the same order, domain, periodicity and phases,
            fixed on the `PEC` surfaces and the lossy walls (complex iff
            ``hcurl`` is).
        eps: ``Re(eps_r)`` per element, an NGSolve coefficient (scalar or
            3x3).
        order: The polynomial order (quadrature).
        diameter: Size of the domain in mesh file units.
        layout: The H(curl) space's layout across ranks, ``None`` on one process.
        h1_layout: The H1 space's, with ``layout``.
        periodic: Whether the spaces are periodic (a local ``G`` under ranks).
    """

    def __init__(
        self,
        hcurl: ngs.FESpace,
        h1: ngs.FESpace,
        eps: ngs.CoefficientFunction,
        order: int,
        diameter: float,
        *,
        layout: DofLayout | None = None,
        h1_layout: DofLayout | None = None,
        periodic: bool = False,
    ) -> None:
        """Build ``G``, ``M0`` and factorize the H1 form."""
        dx = quadrature.dx(order)
        self._layout = layout or ONE_PROCESS
        #: Whether the spaces are complex (a Bloch phase).
        self.complex = bool(hcurl.is_complex)
        # a complex form is Hermitian, not symmetric: PARDISO reads the flag
        symmetric = not self.complex
        u, v = hcurl.TnT()
        mass = ngs.BilinearForm(hcurl, symmetric=symmetric)
        mass += (eps * u) * v * dx
        with _timing.phase("assemble"):
            mass.Assemble()
        #: ``M0``, the electric-energy form on the H(curl) space.
        self.mass = mass.mat
        phi = h1.TrialFunction()
        local = layout is not None and periodic
        if local:
            g = ngs.ConvertOperator(
                h1, hcurl, trial_proxy=ngs.grad(phi), localop=True, parmat=False
            )
        else:
            g = ngs.ConvertOperator(h1, hcurl, trial_proxy=ngs.grad(phi))
        #: NGSolve's G under ranks between plain spaces, a ``ParallelMatrix``;
        #: ``None`` on one process and between periodic spaces.
        self._parallel_g: Any = g if layout is not None and not local else None
        #: Whether ``_g`` is a rank's local G (periodic spaces under ranks).
        self._local_g = local
        if self._parallel_g is None:
            rows, cols, values = g.COO()
            self._g = sp.csr_matrix(
                (np.asarray(values), (np.asarray(rows), np.asarray(cols))),
                shape=(g.height, g.width),
            )
            self._g.eliminate_zeros()
            # G^H as CSR, so a product with it runs row by row like G's
            self._gh = self._g.conj().T.tocsr()
        phi, psi = h1.TnT()
        laplace = ngs.BilinearForm(h1, symmetric=symmetric)
        laplace += (eps * ngs.grad(phi)) * ngs.grad(psi) * dx
        laplace += (_REGULARIZATION / diameter**2) * phi * psi * dx
        self._laplace = _linalg.prepare(
            Operator(laplace, h1.FreeDofs(), layout=h1_layout),
            symmetry="hermitian",
            kind="direct",
        )

    def energy(self, x: np.ndarray) -> np.ndarray:
        """``M0 x`` (on a real space, complex ``x`` part by part): a load."""
        layout = self._layout
        if self.complex:
            return matvec(self.mass, x, layout=layout)
        return matvec(self.mass, x.real, layout=layout) + 1j * matvec(
            self.mass, x.imag, layout=layout
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``x`` without its gradient part (``(n,)`` complex).

        A real ``G`` and ``G^T`` act on the real and imaginary parts
        separately: a product with a complex vector would copy ``G`` to
        complex on every call. The result is bit-identical.
        """
        energy = self.energy(x)
        if self._parallel_g is not None or self._local_g:
            return self._project_on_ranks(x, energy)
        if self.complex:
            return x - self._g @ self._laplace.solve(self._gh @ energy)
        rhs = self._gh @ energy.real + 1j * (self._gh @ energy.imag)
        phi = self._laplace.solve(rhs)
        return x - (self._g @ phi.real + 1j * (self._g @ phi.imag))

    def _project_on_ranks(self, x: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """`__call__` under ranks: :math:`G^T` of the load, MUMPS, ``G`` of the state.

        The real H1 factors solve the real and imaginary parts as two columns.
        """
        rhs = np.stack(
            [self._g_apply(e, transpose=True) for e in (energy.real, energy.imag)],
            axis=1,
        )
        phi = self._laplace.solve(rhs).real
        return x - (self._g_apply(phi[:, 0]) + 1j * self._g_apply(phi[:, 1]))

    def _g_apply(self, values: np.ndarray, *, transpose: bool = False) -> np.ndarray:
        """``G`` of a real H1 state (a state), or :math:`G^T` of a real load (a load).

        A local ``G`` maps them on each rank's own DOFs, with no exchange.
        """
        if self._local_g:
            return (self._gh if transpose else self._g) @ values
        g = self._parallel_g
        source = g.CreateColVector() if transpose else g.CreateRowVector()
        target = g.CreateRowVector() if transpose else g.CreateColVector()
        source.FV().NumPy()[:] = values
        if transpose:
            self._layout.mark_load(source)
            target.data = g.T * source
            target.Distribute()
        else:
            self._layout.mark_state(source)
            target.data = g * source
            target.Cumulate()
        return target.FV().NumPy().copy()

    def free(self) -> None:
        """Release the H1 factorization."""
        self._laplace.free()
