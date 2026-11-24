r"""Piezoelectric rule of a 2D mesh: displacement and potential in one H1 space.

For fields that do not vary along :math:`z`, the unknowns are the three
displacement components :math:`u_x, u_y, u_z` on the mechanical domain and the
potential :math:`\varphi` on the electromagnetic one, all complex H1 of one
order, in one compound space. With the stress-charge relations
:math:`T = \hat c S - e^T E`, :math:`D = e S + \varepsilon^S E` and
:math:`E = -\nabla\varphi`, the symmetric weak form (charge rows negated) is

.. math::

    a(x, y) = \int S(v)\cdot(\hat c\,S(u) + e^T\nabla\varphi)
    + \nabla\psi\cdot(e\,S(u) - \varepsilon^S\nabla\varphi)
    - \omega^2\rho\,v\cdot u,

with :math:`\hat c = c\,(1 + i(1/Q_m + \eta\,r^2))`, the damping frame's
:math:`r` (`gfem.DampingFrame`), under :math:`e^{+i\omega t}`. The operator is
complex symmetric and affine in :math:`\omega^2`: :math:`K` and :math:`M` are
assembled once and :math:`A = K - \omega^2 M` is formed per frequency, one
sparse Cholesky factorization each, shared by every terminal.

Terminal :math:`j` is the lifting :math:`w_j` (the potential 1 on its
electrode DOFs, interpolated: a hierarchical basis has zero higher-order
values there) corrected to :math:`x_j = w_j + (\text{free part})` with
:math:`(A x_j)_f = 0`. The charge per length on terminal :math:`i` is the
reaction weighted by its lifting, :math:`q'_{ij} = -w_i^T A x_j`, and
:math:`Y'_{ij} = i\omega q'_{ij}`; it equals the quadratic form
:math:`-x_i^T A x_j`. The clamped capacitance is the quadratic form of the
potential alone with :math:`\varepsilon^S` and no displacement.

Lengths are the mesh file's unit (``scale`` metres): the strain and the
potential gradient carry :math:`1/\text{scale}` and the area
:math:`\text{scale}^2`, so the stiffness and permittivity forms are the
file-unit ones and only the mass term carries :math:`\text{scale}^2`. The
displacement is in m per V.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants

from gfem._boundaries import PEC, Absorbing
from gfem._checks import is_traced
from gfem._engine import quadrature
from gfem._engine.cache import discretization
from gfem._engine.coefficients import p0, permittivity
from gfem._engine.domain import boundary_region, em_region, h1_space, region_domain
from gfem._engine.runtime import task_manager
from gfem._engine.solve_request import SolveRequest
from gfem._field import FieldMap
from gfem._linalg.base import free_mask
from gfem._linalg.direct import DirectHandle
from gfem._materials import piezoelectric_in_frame, stiffness_in_frame
from gfem.errors import MissingMaterialPropertyError, NonDifferentiableError
from gfem.model import Model
from gfem.options import SolverOptions
from gfem.problems import Piezoelectric
from gfem.results import Field, PiezoelectricResult

__all__ = ["PiezoelectricRule", "PiezoelectricSystem"]

EPS0 = scipy.constants.epsilon_0


class PiezoelectricRule:
    """Host-side forward of a `Piezoelectric` problem on a 2D `Model`.

    Construction runs every static check (it needs no array values);
    `forward` assembles the system and solves each frequency.
    """

    physics = "em"

    def __init__(
        self,
        model: Model,
        problem: Piezoelectric,
        options: SolverOptions,
        *,
        comm: Any = None,
    ) -> None:
        """Check that the problem fits the model (see `gfem.Piezoelectric`).

        Raises:
            NotImplementedError: ``linear_solver="petsc"``, or
                ``frequency_workers`` above 1.
            NonDifferentiableError: The model's leaves are traced by JAX.
            RegionNotFoundError: A name is not a region of the mesh.
            ValueError: The problem does not fit the model; the message gives
                the remedy.
            MissingMaterialPropertyError: A region of the mechanical domain
                has no stiffness or density.
        """
        if not isinstance(problem, Piezoelectric):
            raise TypeError(f"Expected gfem.Piezoelectric, got {problem!r}.")
        self.problem = problem
        self.options = options
        self.comm = comm
        if options.linear_solver == "petsc":
            raise NotImplementedError(
                "A Piezoelectric solve has no PETSc path in this version: use "
                'linear_solver="direct" or "auto".'
            )
        if options.frequency_workers > 1:
            raise NotImplementedError(
                "A Piezoelectric solve runs its frequencies in one process, one "
                "factorization each: leave frequency_workers at 1 (threads= sets "
                "the cores)."
            )
        if is_traced(model):
            raise NonDifferentiableError(
                "gradients of a Piezoelectric solve are not available yet: wrap "
                "the model's leaves in jax.lax.stop_gradient."
            )
        self.mechanical = _check_mechanical(model)
        self.electrodes = _check_electrodes(model, problem)
        _check_clamped(model, problem)
        _check_absorbing(model)
        _check_materials(model)
        self.frame = _check_frame(model, problem)
        mesh = model.mesh
        mechanical = model._domain_of("mechanical")  # noqa: SLF001
        domain = model._domain  # noqa: SLF001
        #: The areas with elements in each domain, mesh order.
        self.mechanical_regions = tuple(
            r for r in mesh.volumes if mechanical.active[mesh.elements(r)].any()
        )
        self.potential_regions = tuple(
            r for r in mesh.volumes if domain.active[mesh.elements(r)].any()
        )
        self._mesh = mesh
        self._mechanical_excluded = mechanical.excluded
        self._em_excluded = domain.excluded

    def forward(
        self,
        model: Model,
        *,
        keep: bool,  # noqa: ARG002
        request: SolveRequest | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, Any], None]:
        """Solve every frequency and terminal.

        Returns:
            ``{"admittance": (n_f, n_t, n_t) S/m, symmetrized,
            "admittance_raw": the same before the symmetrization,
            "capacitance": (n_t, n_t) F/m, "fields": the kept (frequency,
            terminal) columns}``, and no cache entry.
        """
        with task_manager(self.options.threads, self.comm):
            system = PiezoelectricSystem(model, self.problem, self.options, self)
            raw, kept = system.admittance(
                self.problem.frequency, self.problem.kept_frequencies
            )
            capacitance = system.clamped_capacitance()
        admittance = 0.5 * (raw + np.swapaxes(raw, 1, 2))
        outputs = {
            "admittance": admittance,
            "admittance_raw": raw,
            "capacitance": 0.5 * (capacitance + capacitance.T),
            "fields": kept,
        }
        return outputs, None

    def result(self, outputs: dict[str, Any]) -> PiezoelectricResult:
        """The public result from the forward outputs."""
        problem, order = self.problem, self.options.order
        displacement, potential, labels = [], [], []
        mesh = self._mesh
        mechanical = self._mechanical_excluded
        for (f, name), parts in outputs["fields"].items():
            omega = jnp.asarray(2 * np.pi * f)
            labels.append((f, name))
            displacement.append(
                FieldMap(
                    {
                        axis: Field(
                            coefficients=parts[axis],
                            mesh=mesh,
                            quantity=f"u{axis}",
                            unit="m",
                            space="h1",
                            order=order,
                            support=self.mechanical_regions,
                            omega=omega,
                            _excluded=mechanical,
                        )
                        for axis in "xyz"
                    },
                    label="component",
                )
            )
            potential.append(
                Field(
                    coefficients=parts["phi"],
                    mesh=mesh,
                    quantity="V",
                    unit="V",
                    space="h1",
                    order=order,
                    support=self.potential_regions,
                    omega=omega,
                    _excluded=self._em_excluded,
                )
            )
        return PiezoelectricResult(
            admittance_per_length=jnp.asarray(outputs["admittance"]),
            clamped_capacitance_per_length=jnp.asarray(outputs["capacitance"]),
            terminals=problem.terminal_names,
            frequencies=problem.frequency,
            displacement_fields=displacement,
            potential_fields=potential,
            kept=tuple(labels),
        )


# --- checks ------------------------------------------------------------------------


def _check_mechanical(model: Model) -> Any:
    """The mechanical domain; raise when it is empty.

    Raises:
        ValueError: Every volume is excluded from the mechanical domain.
    """
    domain = model._domain_of("mechanical")  # noqa: SLF001
    if not domain.active.any():
        raise ValueError(
            "Every volume is excluded from mechanical problems, so the "
            "Piezoelectric problem has nothing to solve on. Exclude fewer "
            'volumes for mechanical problems (excluded={"mechanical": ...}).'
        )
    return domain


def _lines_and_areas(model: Model, names: tuple[str, ...], what: str):
    """Split ``names`` into lines and areas of the 2D mesh.

    Raises:
        RegionNotFoundError: A name is not a region of the mesh.
        ValueError: An area outside the electromagnetic domain, or a line that
            does not border it.
    """
    mesh, domain = model.mesh, model._domain  # noqa: SLF001
    lines, areas = [], []
    for name in names:
        region = mesh.region(name)
        if region.dim == mesh.dim:
            if not domain.active[mesh.elements(name)].any():
                raise ValueError(
                    f"The {what} {name!r} is an area outside the electromagnetic "
                    'domain (excluded from "em"): a meshed electrode needs its '
                    "potential. Name its shell lines, or keep it in the "
                    "electromagnetic domain."
                )
            areas.append(name)
        else:
            if not domain.touches(name):
                raise ValueError(
                    f"The {what} line {name!r} borders no area of the "
                    "electromagnetic domain, so it has no potential to fix."
                )
            lines.append(name)
    return tuple(lines), tuple(areas)


class Electrodes:
    """The terminals' and the ground's lines and areas.

    Attributes:
        terminals: ``(name, lines, areas)`` per terminal, matrix order.
        ground_lines: The grounded lines (the ground, and every `gfem.PEC`
            line that is not a terminal).
        ground_areas: The grounded areas.
    """

    def __init__(self, terminals, ground_lines, ground_areas) -> None:
        self.terminals = terminals
        self.ground_lines = ground_lines
        self.ground_areas = ground_areas

    @property
    def lines(self) -> tuple[str, ...]:
        """Every electrode line: the Dirichlet lines of the potential."""
        out = [n for _, lines, _ in self.terminals for n in lines]
        return tuple(dict.fromkeys([*out, *self.ground_lines]))


def _check_electrodes(model: Model, problem: Piezoelectric) -> Electrodes:
    """The electrodes the problem names, checked against the model."""
    terminals = []
    named = set()
    for name, surfaces in problem.terminals:
        lines, areas = _lines_and_areas(model, surfaces, "terminal")
        terminals.append((name, lines, areas))
        named.update(surfaces)
    lines, areas = _lines_and_areas(model, problem.ground, "ground")
    pec = tuple(
        b.surface
        for b in model.boundaries
        if isinstance(b, PEC)
        and b.surface not in named
        and b.surface not in problem.ground
        and model._domain.touches(b.surface)  # noqa: SLF001
    )
    return Electrodes(tuple(terminals), tuple(dict.fromkeys([*lines, *pec])), areas)


def _check_clamped(model: Model, problem: Piezoelectric) -> None:
    """Raise for a clamped line that does not border the mechanical domain."""
    mechanical = model._domain_of("mechanical")  # noqa: SLF001
    for name in problem.clamped:
        region = model.mesh.region(name)
        if region.dim == model.mesh.dim or not mechanical.touches(name):
            raise ValueError(
                f"clamped {name!r} must be a line bordering the mechanical "
                "domain, where the displacement is fixed."
            )


def _check_absorbing(model: Model) -> None:
    """Raise for an `Absorbing` line that borders the mechanical domain."""
    mechanical = model._domain_of("mechanical")  # noqa: SLF001
    for b in model.boundaries:
        if isinstance(b, Absorbing) and mechanical.touches(b.surface):
            raise ValueError(
                f"gfem.Absorbing on {b.surface!r} absorbs electromagnetic waves; "
                "a Piezoelectric problem leaves that line traction-free. Absorb "
                "acoustic waves with damping_frame=gfem.DampingFrame(...)."
            )


def _check_materials(model: Model) -> None:
    """Raise for a mechanical region without stiffness or density.

    A piezoelectric region outside the electromagnetic domain raises too.

    Raises:
        MissingMaterialPropertyError: A region lacks ``stiffness`` or
            ``mass_density``.
        ValueError: A piezoelectric region is excluded from ``"em"``.
    """
    mesh = model.mesh
    mechanical = model._domain_of("mechanical").active  # noqa: SLF001
    domain = model._domain.active  # noqa: SLF001
    missing = []
    for region in mesh.volumes:
        elements = mesh.elements(region)
        active = elements[mechanical[elements]]
        if not active.size:
            continue
        material = model.materials.get(region)
        missing.extend(
            (region, prop)
            for prop in ("stiffness", "mass_density")
            if material is None or getattr(material, prop) is None
        )
        if (
            material is not None
            and getattr(material, "piezoelectric_tensor", None) is not None
            and not domain[active].all()
        ):
            raise ValueError(
                f'Region {region!r} is piezoelectric but excluded from "em": the '
                "potential that drives it has no domain there. Keep it in the "
                "electromagnetic domain."
            )
    if missing:
        raise MissingMaterialPropertyError(missing, physics="mechanical")


def _check_frame(model: Model, problem: Piezoelectric) -> Any:
    """The damping frame's regions checked; ``None`` without a frame.

    Raises:
        TypeError: A frame region is not an area.
        ValueError: A frame region is outside the mechanical domain, or no
            mechanical element lies outside the frame.
    """
    frame = problem.damping_frame
    if frame is None:
        return None
    mesh = model.mesh
    mechanical = model._domain_of("mechanical").active  # noqa: SLF001
    inside = mechanical.copy()
    for name in frame.regions:
        region = mesh.region(name)
        if region.dim != mesh.dim:
            raise TypeError(
                f"DampingFrame region {name!r} is a line: name areas of the mesh."
            )
        elements = mesh.elements(name)
        if not mechanical[elements].any():
            raise ValueError(
                f"DampingFrame region {name!r} is outside the mechanical domain: "
                "a frame damps the stiffness, so it must be a solid."
            )
        inside[elements] = False
    if not inside.any():
        raise ValueError(
            "DampingFrame names every mechanical region: nothing is left inside "
            "the frame, so it has no interior box to measure its depth from."
        )
    return frame


# --- material arrays ---------------------------------------------------------------


def _element_arrays(model: Model) -> dict[str, np.ndarray]:
    """Lab-frame stiffness, piezoelectric tensor, density and loss per element.

    Zero outside the regions that give them: ``c`` ``(n, 6, 6)`` Pa, ``e``
    ``(n, 3, 6)`` C/m^2, ``rho`` ``(n,)`` kg/m^3, ``loss`` ``(n,)`` the
    :math:`1/Q_m` (0: lossless).
    """
    mesh = model.mesh
    n = len(mesh.volume_elements)
    c = np.zeros((n, 6, 6))
    e = np.zeros((n, 3, 6))
    rho = np.zeros(n)
    loss = np.zeros(n)
    for region, material in model.materials.items():
        if material.stiffness is None:
            continue
        index = mesh.elements(region)
        shape = (len(index),)
        axes = material.axes
        c[index] = stiffness_in_frame(
            np.broadcast_to(np.asarray(material.stiffness, float), (*shape, 6, 6)),
            axes,
            None,
        )
        if material.piezoelectric_tensor is not None:
            e[index] = piezoelectric_in_frame(
                np.broadcast_to(
                    np.asarray(material.piezoelectric_tensor, float), (*shape, 3, 6)
                ),
                axes,
                None,
            )
        if material.mass_density is not None:
            rho[index] = np.asarray(material.mass_density, float)
        q = material.mechanical_quality_factor
        if q is not None:
            loss[index] = 1.0 / np.asarray(q, float)
    return {"c": c, "e": e, "rho": rho, "loss": loss}


def _tensor(built: Any, values: np.ndarray) -> ngs.CoefficientFunction:
    """Per-element ``(n, a, b)`` values as a P0 matrix coefficient."""
    _, a, b = values.shape
    parts = [p0(built, values[:, i, j]) for i in range(a) for j in range(b)]
    return ngs.CF(tuple(parts), dims=(a, b))


def _frame_depth(model: Model, built: Any, frame: Any) -> ngs.CoefficientFunction:
    r""":math:`\eta\,r^2` on the frame regions, zero elsewhere.

    The interior box is the bounding box of the mechanical triangles outside
    the frame, read from the imported nodes (the depth moves with the material
    points); a frame triangle beyond a side of the box has :math:`r` = the
    distance past it over the frame's extent on that side, the depths of a
    corner adding as squares.
    """
    mesh = model.mesh
    scale = built.scale
    nodes = np.asarray(mesh.imported_nodes, float)[:, :2] / scale
    cells = np.asarray(mesh.volume_elements)
    in_frame = np.zeros(len(cells), bool)
    for name in frame.regions:
        in_frame[mesh.elements(name)] = True
    mechanical = model._domain_of("mechanical").active  # noqa: SLF001
    interior = mechanical & ~in_frame
    box = nodes[np.unique(cells[interior])]
    lo, hi = box.min(axis=0), box.max(axis=0)
    ring = nodes[np.unique(cells[in_frame & mechanical])]
    low = lo - ring.min(axis=0)
    high = ring.max(axis=0) - hi
    space = ngs.H1(built.mesh, order=1)
    coordinates = []
    for axis in range(2):
        gf = ngs.GridFunction(space)
        gf.vec.FV().NumPy()[:] = nodes[built.node_order, axis]
        coordinates.append(gf)
    depth: Any = 0.0
    for axis, x in enumerate(coordinates):
        side = ngs.CF(0.0)
        if high[axis] > 0:
            side = ngs.IfPos(x - hi[axis], (x - hi[axis]) / high[axis], side)
        if low[axis] > 0:
            side = ngs.IfPos(lo[axis] - x, (lo[axis] - x) / low[axis], side)
        depth = depth + side * side
    return p0(built, in_frame.astype(float)) * frame.loss_factor * depth


# --- the system --------------------------------------------------------------------


class PiezoelectricSystem:
    """The assembled operator, the electrode liftings and the per-frequency solves.

    Attributes:
        space: The compound space ``[u_x, u_y, u_z, phi]``.
        ndof: Its size.
        offset: Where the potential block starts (3 displacement blocks of
            ``offset // 3`` DOFs).
        free: Boolean mask of the free DOFs.
        lifts: ``(ndof, n_terminals)`` real liftings.
        stiffness: The assembled :math:`K` (complex, with the loss).
        loss_form: The real symmetric matrix :math:`L` with
            :math:`K = K_r + iL` (the damping and :math:`Q_m` part).
    """

    def __init__(
        self,
        model: Model,
        problem: Piezoelectric,
        options: SolverOptions,
        rule: PiezoelectricRule,
    ) -> None:
        self.rule = rule
        self.model = model
        self.options = options
        order = options.order
        self.order = order
        disc = discretization(model.mesh)
        built = self.built = disc.built
        electrodes = rule.electrodes
        mech = em_region(model, built, "mechanical")
        diel = em_region(model, built, "em")
        self.mech, self.diel = mech, diel

        vu = h1_space(
            model,
            built,
            order,
            tuple(problem.clamped),
            physics="mechanical",
            complex_valued=True,
        )
        vp = h1_space(
            model, built, order, electrodes.lines, physics="em", complex_valued=True
        )
        self.vu, self.vp = vu, vp
        self.space = space = ngs.FESpace([vu, vu, vu, vp])
        self.offset = 3 * vu.ndof
        self.ndof = space.ndof

        self._assemble(model, rule)
        self.lifts, self.free = self._electrodes(vp, electrodes)
        self.potential_free = self._potential_free(vp, electrodes)
        self.free_bits = ngs.BitArray(self.free.tolist())

    def _assemble(self, model: Model, rule: PiezoelectricRule) -> None:
        """Assemble :math:`K` and :math:`M` (one pattern) and the loss matrix."""
        built, space, order = self.built, self.space, self.order
        mech, diel = self.mech, self.diel
        arrays = _element_arrays(model)
        eps = permittivity(model, built)
        if eps.dim == 1:
            eps = eps * ngs.Id(3)
        eps = EPS0 * eps
        c = _tensor(built, arrays["c"])
        e = _tensor(built, arrays["e"])
        rho = p0(built, arrays["rho"])
        loss = p0(built, arrays["loss"])
        if rule.frame is not None:
            loss = loss + _frame_depth(model, built, rule.frame)

        (ux, uy, uz, ph), (vx, vy, vz, ps) = space.TnT()
        su, sv = _strain(ux, uy, uz), _strain(vx, vy, vz)
        gp = ngs.CF((ngs.grad(ph)[0], ngs.grad(ph)[1], 0))
        gq = ngs.CF((ngs.grad(ps)[0], ngs.grad(ps)[1], 0))
        scale = built.scale
        dx_mech = quadrature.dx(order, definedon=mech)
        dx_diel = quadrature.dx(order, definedon=diel)
        stiff = sv * (c * su)
        coupling = sv * (e.trans * gp) + gq * (e * su)
        # one form, switched between K and M by two parameters: the two
        # matrices share one pattern, and neither is a difference of the other
        on, off = ngs.Parameter(1.0), ngs.Parameter(0.0)
        mass = scale**2 * rho * (ux * vx + uy * vy + uz * vz)
        with ngs.TaskManager():
            k = ngs.BilinearForm(space, symmetric=True)
            k += on * (1 + 1j * loss) * stiff * dx_mech
            k += on * coupling * dx_mech
            k += -on * (gq * (eps * gp)) * dx_diel
            k += -off * mass * dx_mech
            k.Assemble()
            self._k = k.mat.AsVector().FV().NumPy().copy()
            on.Set(0.0)
            off.Set(-1.0)
            k.Assemble()
            self._m = k.mat.AsVector().FV().NumPy().copy()
            # the loss part alone, real symmetric: K = K_r + i L
            on.Set(1.0)
            off.Set(0.0)
            loss_form = ngs.BilinearForm(space, symmetric=True)
            loss_form += loss * stiff * dx_mech
            loss_form.Assemble()
        self.loss_matrix = loss_form.mat
        self.matrix = k.mat.CreateMatrix()

    # --- electrodes ----------------------------------------------------------------

    def _fixed_area_dofs(
        self, vp: ngs.FESpace, electrodes: Electrodes
    ) -> list[np.ndarray]:
        """The potential DOFs on each electrode area, ``(terminals..., ground)``."""
        out = []
        areas = [a for _, _, a in electrodes.terminals] + [electrodes.ground_areas]
        for names in areas:
            mask = np.zeros(vp.ndof, bool)
            for name in names:
                region = region_domain(self.model, self.built, name)
                mask |= free_mask(vp.GetDofs(region), vp.ndof)
            out.append(mask)
        return out

    def _electrodes(
        self, vp: ngs.FESpace, electrodes: Electrodes
    ) -> tuple[np.ndarray, np.ndarray]:
        """The liftings and the free mask of the compound space."""
        free = free_mask(self.space.FreeDofs(), self.ndof)
        areas = self._fixed_area_dofs(vp, electrodes)
        for mask in areas:
            free[self.offset :][mask] = False
        lifts = np.zeros((self.ndof, len(electrodes.terminals)))
        for j, (_, lines, names) in enumerate(electrodes.terminals):
            g = ngs.GridFunction(vp)
            one = ngs.CF(1.0)
            for name in lines:
                g.Set(one, definedon=boundary_region(self.built, (name,)))
            for name in names:
                g.Set(one, definedon=region_domain(self.model, self.built, name))
            w = np.real(g.vec.FV().NumPy()).copy()
            w[np.abs(w) < 1e-10] = 0.0
            lifts[self.offset :, j] = w
            if free[lifts[:, j] != 0].any():
                raise RuntimeError("internal: a lifting is nonzero on a free DOF.")
        return lifts, free

    def _potential_free(self, vp: ngs.FESpace, electrodes: Electrodes) -> np.ndarray:
        """The free mask of the potential alone (real space, same numbering)."""
        free = free_mask(vp.FreeDofs(), vp.ndof)
        for mask in self._fixed_area_dofs(vp, electrodes):
            free[mask] = False
        return free

    # --- solves --------------------------------------------------------------------

    def state(self, frequency: float) -> tuple[np.ndarray, np.ndarray]:
        """``(x, A x)``: the terminals' states at ``frequency``, ``(ndof, n_t)``."""
        omega = 2 * np.pi * frequency
        self.matrix.AsVector().FV().NumPy()[:] = self._k - omega**2 * self._m
        handle = DirectHandle(
            self.matrix,
            self.free_bits,
            "complex_symmetric",
            is_complex=True,
            cholesky=True,
        )
        try:
            lifts = self.lifts.astype(complex)
            x = lifts + handle.solve(-handle.apply(lifts))
            return x, handle.apply(x)
        finally:
            handle.free()

    def admittance(
        self, frequencies: tuple[float, ...], kept: tuple[float, ...]
    ) -> tuple[np.ndarray, dict[tuple[float, str], dict[str, np.ndarray]]]:
        """:math:`Y'` over ``frequencies`` (before symmetrizing) and the kept fields."""
        nt = self.lifts.shape[1]
        out = np.zeros((len(frequencies), nt, nt), complex)
        names = self.rule.problem.terminal_names
        fields: dict[tuple[float, str], dict[str, np.ndarray]] = {}
        nu = self.offset // 3
        for k, f in enumerate(frequencies):
            x, ax = self.state(f)
            out[k] = -1j * 2 * np.pi * f * (self.lifts.T @ ax)
            if f in kept:
                for j, name in enumerate(names):
                    parts = {
                        axis: x[i * nu : (i + 1) * nu, j].copy()
                        for i, axis in enumerate("xyz")
                    }
                    parts["phi"] = x[self.offset :, j].copy()
                    fields[(f, name)] = parts
        return out, fields

    def clamped_capacitance(self) -> np.ndarray:
        r""":math:`C'_0`: the potential alone, with :math:`\varepsilon^S`, no motion."""
        model, built, order = self.model, self.built, self.order
        electrodes = self.rule.electrodes
        vr = h1_space(model, built, order, electrodes.lines, physics="em")
        eps = permittivity(model, built)
        if eps.dim == 1:
            eps = eps * ngs.Id(3)
        eps = EPS0 * eps
        p, q = vr.TnT()
        gp = ngs.CF((ngs.grad(p)[0], ngs.grad(p)[1], 0))
        gq = ngs.CF((ngs.grad(q)[0], ngs.grad(q)[1], 0))
        with ngs.TaskManager():
            form = ngs.BilinearForm(vr, symmetric=True)
            form += (gq * (eps * gp)) * quadrature.dx(order, definedon=self.diel)
            form.Assemble()
        free = ngs.BitArray(self.potential_free.tolist())
        handle = DirectHandle(
            form.mat, free, "hermitian", is_complex=False, cholesky=True
        )
        try:
            lifts = np.real(self.lifts[self.offset :])
            x = lifts + handle.solve(-handle.apply(lifts))
            return x.T @ handle.apply(x)
        finally:
            handle.free()


def _strain(gx: Any, gy: Any, gz: Any) -> ngs.CoefficientFunction:
    """Engineering strain (xx, yy, zz, yz, xz, xy), no variation along z."""
    ax, ay, az = ngs.grad(gx), ngs.grad(gy), ngs.grad(gz)
    return ngs.CF((ax[0], ay[1], 0, az[1], az[0], ax[1] + ay[0]))
