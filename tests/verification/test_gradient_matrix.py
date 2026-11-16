"""The gradient verification matrix: every v0.1 gradient against differences.

One cell per problem type x input kind x linear solver x frequency sweep:

- problems: electrostatic capacitance (and a potential functional, and the
  region energies), driven
  S on lumped ports and on wave ports (lossless, lossy, and two modes per
  shared port surface, read through cross-mode entries), driven S (and
  probed fields) on a periodic cell with plain and Bloch periodicity,
  eigenfrequency (frequency and loss of a simple mode);
- inputs: per-element permittivity ``eps_r`` and conductivity, the same per
  principal value of a field with axes (``eps_r_axes``,
  ``conductivity_axes``: biaxial values on the `TILTED` axes, so every
  entry of each projector is exercised), the permeability ``mu_r`` and the
  loss tangent ``tan_delta`` (driven and eigenmode, on the direct solver:
  the solver and sweep paths are the permittivity's), the fill of a
  `gfem.SheetField`,
  and the mesh node coordinates (through `gfem.shape.boundary_morph`);
- solvers: ``direct`` and ``bddc_gmres``; sweeps: uniform and adaptive (with
  frozen samples; the objective reads S at the samples, where the reduced
  model reproduces the full solve of the perturbed design too, so the
  differences of the adaptive pipeline check the frozen-basis gradient).

The PETSc cells (`PETSC_CELLS`: ``linear_solver="petsc"``, and SLEPc seeds
for the eigenmode cell) run the same checks where the PETSc stack is
installed, and are skipped elsewhere (the ``petsc`` / ``slepc`` markers).
They stay out of `CELLS`, from which the docs build derives the Capabilities
page's claims: the verification job that runs on every PR has no PETSc.

Each cell checks the gradient along 3 random unit directions (the third
along ``i`` for a complex permittivity of principal values) against
fourth-order central differences (tests/finite_differences.py, gate 1e-6)
at tight tolerance (iterative solves to 1e-12, eigenpairs to 1e-12), and
that the Taylor remainder falls at second order at production tolerance
(`gfem.SolverOptions` defaults: iterative solves to 1e-6, eigenpairs to
1e-10), with gradient and differences both carrying that solver error.

The periodic eigenmode cells (``eigenmode_periodic_plain`` and
``eigenmode_periodic_bloch``, `_eigen_periodic_cells`) solve a damped,
lossy cell periodic along x and y (tests/periodic_cases.py, `eigen_cell`):
plain on both solvers, Bloch on ``direct`` (the only linear solver of a
Bloch eigenmode solve), where the left eigenvectors carry the gradient. Its
node direction moves the twins of every pair together (a periodic
direction, the raw node gradient).

Then the consistency checks (JAX transformations and evicted cache entries
per input kind, reciprocity against an explicit adjoint, adaptive against
uniform at the samples) and the degenerate eigenmode clusters. Pairs this
version does not differentiate are rows of the raise table
(tests/verification/test_gradient_matrix_raise_table.py), not cells: the
electrostatic conductivity (an exact zero) and sheet fill (not an
electrostatic object), eigenmode field quantities, the port mode problem.

Run the gate with ``pytest -m verification -k gradient_matrix``. The slowest
shared-surface cells (`REFERENCE_CELLS`) also carry the ``reference`` marker, so
the default tiers (``-m "not reference ..."``) leave them to the reference
tier.
"""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import periodic_cases as pc
import pytest
from anisotropy import TILTED
from finite_differences import (
    assert_gradient_matches,
    assert_taylor_second_order,
    unit_directions,
)
from gradient_cases import (
    BIAXIAL,
    BLEND_DIELECTRIC,
    BLEND_FREQUENCIES,
    BLEND_METAL,
    CAVITY_TARGET,
    COVER_LAW,
    EIGEN_LAWS,
    EPS0,
    GUIDE_TARGET,
    JUNCTION_CHI_L,
    JUNCTION_L,
    JUNCTION_TARGET,
    KINETIC_CAPACITANCE,
    KINETIC_TARGET,
    LAW_FREQUENCIES,
    LAWS,
    LOSSLESS_LUMPED,
    LUMPED,
    LUMPED_TARGET,
    MIRRORED_TARGET,
    OMEGA_WEIGHT,
    RESONATOR_TARGET,
    SHEETS,
    STACK_PROBLEM,
    STATIC_LAW,
    absorbing_guide,
    capacitor,
    cavity_fill,
    cavity_model,
    core_design,
    dispersive_line,
    dispersive_mesh,
    dispersive_resonator,
    guide_model,
    hybrid_line,
    hybrid_mesh,
    junction_law_line,
    junction_line,
    junction_mesh,
    junction_values,
    kinetic_cavity,
    layered_line,
    layered_resonator,
    line_axes_design,
    line_design,
    lossless_lumped_line,
    lumped_line,
    mirrored_cavity,
    on_the_sheet,
    ramp,
    resonator_mesh,
    s_functional,
    scaled_law,
    shape_line,
    shared_surface_functional,
    sheet_cavity,
    sheet_fill,
    sheet_line,
    sheet_mesh,
    split_cavity,
    stack,
    stack_fields,
    terminated,
    tight,
    wave_line,
    wave_mesh,
    with_fields,
)
from periodic_cases import (
    EG_A,
    EG_B,
    EG_BLOCH_TARGET,
    EG_H,
    EG_K,
    EG_PLAIN_TARGET,
    EG_TAN_DELTA,
    MM,
    coordinate_classes,
    eigen_cell,
    eigen_cell_design,
    eigen_cell_mesh,
)

import gfem
from gfem.shape import boundary_morph

#: G's Bloch cells meet its absorbing top obliquely (`pc.G_OBLIQUE`).
pytestmark = pytest.mark.filterwarnings(pc.G_OBLIQUE)

SOLVERS = ("direct", "bddc_gmres")


@dataclasses.dataclass(frozen=True)
class Cell:
    """One cell: problem x input kind x solver x sweep (x variant)."""

    problem: str
    input: str
    solver: str
    sweep: str = ""
    variant: str = ""

    def __str__(self) -> str:
        parts = (self.problem, self.input, self.variant, self.solver, self.sweep)
        return "-".join(p for p in parts if p)


def _cells() -> list[Cell]:
    cells = []
    for solver in SOLVERS:
        cells += [
            Cell("electrostatic", "eps_r", solver),
            Cell("electrostatic", "eps_r_axes", solver),
            Cell("electrostatic", "nodes", solver, variant="capacitance"),
            Cell("electrostatic", "nodes", solver, variant="potential"),
            *(
                Cell("electrostatic", name, solver, variant="energy")
                for name in ("eps_r", "eps_r_axes", "nodes")
            ),
        ]
        for sweep in ("uniform", "adaptive"):
            cells += [
                Cell("driven", "eps_r", solver, sweep),
                Cell("driven", "conductivity", solver, sweep),
                # on first-order absorbing sides, owned by the design
                Cell("driven", "eps_r_axes", solver, sweep),
                Cell("driven", "conductivity_axes", solver, sweep),
                Cell("driven", "nodes", solver, sweep),
            ]
            sheets = (
                ("conductor", "impedance") if sweep == "uniform" else ("impedance",)
            )
            cells += [Cell("driven", "fill", solver, sweep, s) for s in sheets]
            # a lumped port's L and C, an impedance sheet's R and L
            cells.append(Cell("driven", "lumped", solver, sweep))
            # Wave ports: every port x input on the direct solver; on BDDC +
            # GMRES the lossy ports (the adjoint solve through the port
            # operator's Woodbury update, the adjoint loads' extra columns at
            # the samples) in the permittivity, whose conductivity cells take
            # the same code path, and in the nodes on the uniform sweep (the
            # adaptive backward is the solver's only through those solves).
            bddc = {"eps_r"} | ({"nodes"} if sweep == "uniform" else set())
            cells += [
                Cell("driven_wave", name, solver, sweep, port)
                for port in ("lossless", "lossy")
                for name in ("eps_r", "conductivity", "nodes")
                if solver == "direct" or (port == "lossy" and name in bddc)
            ]
            # principal values beside a lossy port: the port terms read no
            # field, so one cell shows the principal-value path through them
            if solver == "direct" and sweep == "uniform":
                cells.append(Cell("driven_wave", "eps_r_axes", solver, sweep, "lossy"))
            # A shared port surface (two modes per end): every input on the
            # direct solver, the permittivity on BDDC + GMRES
            cells += [
                Cell("driven_wave", name, solver, sweep, "shared")
                for name in ("eps_r", "conductivity", "nodes")
                if solver == "direct" or (name == "eps_r" and sweep == "uniform")
            ]
        cells += [
            Cell("eigenmode", name, solver)
            for name in (
                "eps_r",
                "conductivity",
                "eps_r_axes",
                "conductivity_axes",
                "fill",
                "lumped",
                "nodes",
            )
        ]
    # the curl (mu_r) and loss (tan_delta) paths: one solver and one sweep
    cells += [
        Cell("driven", f"{name}{form}", "direct", "uniform")
        for form in ("", "_axes")
        for name in ("mu_r", "tan_delta")
    ]
    cells += [Cell("eigenmode", name, "direct") for name in ("mu_r", "tan_delta")]
    return cells


def _driven_periodic_cells() -> list[Cell]:
    """The driven cells of a periodic model (the gradient cell G).

    Plain and Bloch periodicity on the uniform sweep for every input kind
    on the direct solver, Bloch on BDDC + GMRES too (its coarse LU and the
    transposed system), plain periodicity on the adaptive sweep (Bloch
    raises in its forward) in the permittivity and the nodes (conductivity
    and fill take the permittivity's code path), and a probed Bloch field
    (a field cotangent, loaded as P^T g on the transposed system). To keep
    the default tier's time, plain periodicity on BDDC + GMRES has the
    permittivity cell only (its pairs are the direct solver's).
    """
    inputs = ("eps_r", "conductivity", "fill", "nodes")
    return [
        *(
            Cell(problem, name, "direct", "uniform")
            for problem in ("driven_periodic_plain", "driven_periodic_bloch")
            for name in inputs
        ),
        Cell("driven_periodic_plain", "eps_r", "direct", "adaptive"),
        Cell("driven_periodic_plain", "nodes", "direct", "adaptive"),
        Cell("driven_periodic_bloch", "eps_r", "direct", "uniform", "field"),
        *(
            Cell("driven_periodic_bloch", name, "bddc_gmres", "uniform")
            for name in inputs
        ),
        Cell("driven_periodic_plain", "eps_r", "bddc_gmres", "uniform"),
    ]


# --- periodic eigenmode cells ------------------------------------------------------


def _eigen_periodic_cells() -> list[Cell]:
    """The periodic eigenmode cells: plain on both solvers, Bloch on ``direct``.

    Bloch: the damped cell in every input, and a linear lossy one (``tan
    delta``, no damping) in the permittivity.
    """
    inputs = ("eps_r", "conductivity", "fill", "nodes")
    return [
        *(Cell("eigenmode_periodic_plain", name, "direct") for name in inputs),
        Cell("eigenmode_periodic_plain", "eps_r", "bddc_gmres"),
        *(Cell("eigenmode_periodic_bloch", name, "direct") for name in inputs),
        Cell("eigenmode_periodic_bloch", "eps_r", "direct", variant="linear"),
    ]


# --- eigenmode mode-field cells (quantum metrics) ----------------------------------


def _participation_cells() -> list[Cell]:
    """The ratios of the result (``eigenmode_participation``), on ``direct``.

    By fixture variant (`_participation`): fixture J (`junction_line`)
    lossless, with a resistive port, lossy (complex permittivity, a
    conductivity, a static loss tangent), with a `gfem.Conductor` top, with
    a capacitive junction, with fields of principal values; the shorted
    guide's second-order absorbing wall; the glued line's resistive
    `gfem.Impedance` sheet; the mirrored cavity's pair split to a relative
    gap of 2.5e-4 (above the isolation threshold 1e-4). The node cells move
    fixture J's interface (`_participation_nodes`), with the static values
    of each variant (the lossy one's loss tangent among them). The sheet
    cavity's inductive `gfem.Impedance`, with a capacitance or without,
    gives the sheet participation and the sheet's share of the mode energy
    (`_kinetic`); its node cells move the sheet. The ``"chi"`` cells read
    the dispersive shift of `gfem.quantum.dressed_modes` between fixture J's
    two lowest modes (`_chi`), the whole chain from the solve. The lumped
    cells read one ratio each of the lossy glued line in its port's L and C
    and its sheet's R, L and C, and the dressed anharmonicity of the
    lossless line's junction mode in its port's and sheet's L and C, the
    port's L also given to `gfem.quantum.dressed_modes` (`_lumped_ratio`).
    """
    key = "eigenmode_participation"
    return [
        Cell(key, "eps_r", "direct", variant="inductive"),
        Cell(key, "nodes", "direct", variant="inductive"),
        Cell(key, "eps_r", "direct", variant="resistive"),
        Cell(key, "nodes", "direct", variant="resistive"),
        Cell(key, "eps_r", "direct", variant="lossy"),
        Cell(key, "conductivity", "direct", variant="lossy"),
        Cell(key, "nodes", "direct", variant="lossy"),
        Cell(key, "eps_r", "direct", variant="conductor"),
        Cell(key, "nodes", "direct", variant="conductor"),
        Cell(key, "eps_r", "direct", variant="absorbing2"),
        Cell(key, "eps_r", "direct", variant="capacitive_port"),
        Cell(key, "nodes", "direct", variant="capacitive_port"),
        Cell(key, "fill", "direct", variant="resistive_sheet"),
        Cell(key, "eps_r", "direct", variant="near_pair"),
        Cell(key, "eps_r_axes", "direct", variant="principal"),
        Cell(key, "nodes", "direct", variant="principal"),
        Cell(key, "fill", "direct", variant="sheet"),
        Cell(key, "eps_r", "direct", variant="sheet"),
        Cell(key, "nodes", "direct", variant="sheet"),
        Cell(key, "eps_r", "direct", variant="capacitive_sheet"),
        Cell(key, "fill", "direct", variant="capacitive_sheet"),
        Cell(key, "nodes", "direct", variant="capacitive_sheet"),
        Cell(key, "eps_r", "direct", variant="chi"),
        Cell(key, "nodes", "direct", variant="chi"),
        # the lumped values (ticket 21): one cell per ratio (`_lumped_ratio`)
        *(
            Cell(key, "lumped", "direct", variant=v)
            for v in (
                "participation",
                "q_port",
                "bulk",
                "sheet_participation",
                "anharmonicity",
            )
        ),
    ]


def _field_cells() -> list[Cell]:
    """The normalized mode field (``eigenmode_field``), on ``direct``.

    By fixture variant (`_field`): the interface participation of fixture
    J's layer interface (lossless) and of the sheet cavity's walls (a
    `gfem.Conductor` film, in its fill); a probed field component times a
    phase, so the phase rule's multiplier is nonzero, on fixture J with a
    resistive port and on the terminated resonator (in its conductivity),
    and on the lossy glued line in its lumped values (`_lumped_ratio`).
    """
    key = "eigenmode_field"
    return [
        Cell(key, "eps_r", "direct", variant="interface"),
        Cell(key, "eps_r", "direct", variant="probe"),
        Cell(key, "conductivity", "direct", variant="probe"),
        Cell(key, "fill", "direct", variant="interface"),
        Cell(key, "lumped", "direct", variant="probe"),
    ]


# --- dispersive materials ------------------------------------------------------------


def _law_cells() -> list[Cell]:
    """The driven cells of a permittivity law (ADR 0112), on ``direct``.

    Input ``"law"``: the law's parameters, normalized as p0 (1 + t), on the
    strip line (`dispersive_line`) for each law of `LAWS` (``poles``,
    ``relaxation``) and behind first- and second-order absorbing walls on the
    law region (``absorbing1``, ``absorbing2``: their face terms), through
    wave ports (``lossy``) and on the plain periodic cell. The existing inputs
    beside a law (variant ``"law"``): the host's per-element permittivity, and
    the nodes of the shape line whose cover holds a law (its susceptibility
    and absorbing face pieces transported).
    """
    return [
        *(
            Cell("driven", "law", "direct", "uniform", variant)
            for variant in ("poles", "relaxation", "absorbing1", "absorbing2")
        ),
        Cell("driven", "eps_r", "direct", "uniform", "law"),
        Cell("driven", "nodes", "direct", "uniform", "law"),
        Cell("driven_wave", "law", "direct", "uniform", "lossy"),
        Cell("driven_periodic_plain", "law", "direct", "uniform"),
        # eigenmode omega (frequency and Q) in the cover's law parameters
        *(
            Cell("eigenmode", "law", "direct", variant=variant)
            for variant in ("lorentz", "drude_lorentz", "relaxation", "drude_lossless")
        ),
        # beside the cover's lossy Lorentz law: omega in the sub layer's
        # permittivity and the interface's nodes (E5, E6); fixture J's ratios
        # and a probed field (P1-P4: the energy permittivity per mode, the
        # kappa term, the phase rule on eps_inf)
        Cell("eigenmode", "eps_r", "direct", variant="lorentz"),
        Cell("eigenmode", "nodes", "direct", variant="lorentz"),
        *(
            Cell("eigenmode_participation", name, "direct", variant="lorentz")
            for name in ("eps_r", "lumped", "nodes")
        ),
        Cell("eigenmode_field", "eps_r", "direct", variant="lorentz"),
        *_adaptive_law_cells(),
    ]


def _adaptive_law_cells() -> list[Cell]:
    """The adaptive sweep's law cells (ADR 0116), frozen samples.

    The ``poles`` law with its Lorentz pole moved into the band
    (`ADAPTIVE_LAWS`) and the ``lorentz`` law behind first-order absorbing
    walls, in their parameters; the host's per-element permittivity beside
    the moved ``poles`` law.
    """
    return [
        Cell("driven", "law", "direct", "adaptive", "poles"),
        Cell("driven", "law", "direct", "adaptive", "absorbing1"),
        Cell("driven", "eps_r", "direct", "adaptive", "law"),
    ]


def _blend_cells() -> list[Cell]:
    """The density of a metal-dielectric blend (ADR 0113): cells D11 and D12.

    The strip line's core blends a constant dielectric and a Drude metal by a
    density in [0.2, 0.8], under the index scheme and the linear one.
    """
    return [
        Cell("driven", "blend", "direct", "uniform", scheme)
        for scheme in ("index", "linear")
    ]


def _electrostatic_law_cells() -> list[Cell]:
    """Cell S1: the capacitance in a law's parameters (ADR 0111), on ``direct``.

    The capacitor's nitride holds `STATIC_LAW`: the static permittivity reads
    eps_inf and the strengths, and the frequency parameters are exact zeros.
    """
    return [Cell("electrostatic", "law", "direct", variant="static")]


CELLS = [
    *_cells(),
    *_driven_periodic_cells(),
    *_eigen_periodic_cells(),
    *_participation_cells(),
    *_field_cells(),
    *_law_cells(),
    *_blend_cells(),
    *_electrostatic_law_cells(),
]
#: The PETSc stack's cells, one per path its solves take: electrostatic,
#: driven on both sweeps, the adjoint through a wave port operator's
#: Woodbury update, and the eigenmode seed on SLEPc.
PETSC_CELLS = [
    Cell("electrostatic", "eps_r", "petsc"),
    Cell("driven", "eps_r", "petsc", "uniform"),
    Cell("driven", "eps_r", "petsc", "adaptive"),
    Cell("driven_wave", "eps_r", "petsc", "uniform", "lossy"),
    Cell("eigenmode", "eps_r", "petsc"),
    # plain periodicity keeps every backend (Bloch on PETSc raises)
    Cell("driven_periodic_plain", "eps_r", "petsc", "uniform"),
    # the mode-field adjoint after SLEPc seeds (its linear solver stays
    # direct: a PETSc one raises for a mode-field cotangent)
    Cell("eigenmode_participation", "eps_r", "direct", variant="slepc"),
    # a permittivity law's parameters (ADR 0112): the law pass after a PETSc
    # solve, and the eigenfrequency after SLEPc seeds
    Cell("driven", "law", "petsc", "uniform", "poles"),
    Cell("eigenmode", "law", "petsc", variant="lorentz"),
]
PETSC_PARAMS = [
    pytest.param(
        cell,
        marks=pytest.mark.slepc
        if cell.problem.startswith("eigenmode")
        else pytest.mark.petsc,
    )
    for cell in PETSC_CELLS
]
#: Cells run in the reference tier only: each shared-surface cell solves four
#: wave ports through two port mode sweeps, and these three take 2-4 min on
#: MN5 (8 cores). The default tiers keep the shared-surface permittivity on both
#: sweeps and every input on the uniform sweep.
REFERENCE_CELLS = {
    Cell("driven_wave", "eps_r", "bddc_gmres", "uniform", "shared"),
    Cell("driven_wave", "conductivity", "direct", "adaptive", "shared"),
    Cell("driven_wave", "nodes", "direct", "adaptive", "shared"),
}
CELL_PARAMS = [
    pytest.param(cell, marks=pytest.mark.reference, id=str(cell))
    if cell in REFERENCE_CELLS
    else cell
    for cell in CELLS
]


@dataclasses.dataclass
class Case:
    """A cell's objective, its input and how to perturb it.

    ``step``: fourth-order difference step along a unit direction (see
    tests/finite_differences.py); ``taylor``: the largest of the Taylor
    test's three steps along one unit direction, each ``taylor_ratio``
    times the next, from ``taylor_at`` (``p`` when None). At production
    tolerance the objective carries the solver's error (about 1e-7 relative
    for GMRES at 1e-6): the remainders must stay far above it.
    """

    objective: Callable
    p: jax.Array
    rng: np.random.Generator
    step: float
    taylor: float
    reference: Callable | None = None
    taylor_at: jax.Array | None = None
    taylor_ratio: float = 10**0.5
    mask: Callable = lambda directions: directions
    imaginary: bool = False  # the third direction along i (a complex input)

    def directions(self, n: int) -> list[np.ndarray]:
        out = self.mask(unit_directions(self.rng, self.p.shape, n))
        if self.imaginary and n >= 3:
            out[2] = 1j * out[2]
        return out


class Designs:
    """The cells' designs, each built on first use (module-scoped)."""

    def __init__(self, factory) -> None:
        self._factory = factory

    def _dir(self, name):
        return self._factory.mktemp(name)

    @functools.cached_property
    def stack(self):
        return stack(self._dir("stack"))

    @functools.cached_property
    def capacitor(self):
        return capacitor(self._dir("capacitor"))

    @functools.cached_property
    def line(self):
        return layered_line(self._dir("line"))

    @functools.cached_property
    def axes_line(self):
        return layered_line(self._dir("axes_line"), absorbing_order=1)

    @functools.cached_property
    def shape_line(self):
        return shape_line(self._dir("shape_line"))

    @functools.cached_property
    def wave_mesh(self):
        return wave_mesh(self._dir("wave"))

    @functools.cached_property
    def hybrid_mesh(self):
        # the coarsest grid that keeps the quadrant (96 tetrahedra): its two
        # modes and Gram coupling are the finer grids' to 3 digits
        return hybrid_mesh(self._dir("hybrid"), cells=(4, 2, 2))

    @functools.cached_property
    def sheet_mesh(self):
        return sheet_mesh(self._dir("sheet"))

    @functools.cached_property
    def sheet_cavity(self):
        return sheet_cavity(self._dir("sheet_cavity"))

    @functools.cached_property
    def resonator(self):
        return resonator_mesh(self._dir("resonator"))

    @functools.cached_property
    def layered_resonator(self):
        return layered_resonator(self._dir("layered_resonator"))

    @functools.cached_property
    def mirrored_cavity(self):
        return mirrored_cavity(self._dir("mirrored"))

    @functools.cached_property
    def corner_cavity(self):
        return mirrored_cavity(self._dir("corner"), corner=True)

    @functools.cached_property
    def junction(self):
        return junction_mesh(self._dir("junction"))

    @functools.cached_property
    def guide(self):
        return absorbing_guide(self._dir("guide"))

    @functools.cached_property
    def periodic(self):
        return pc.g_mesh(self._dir("periodic"))

    @functools.cached_property
    def eigen_cell(self):
        return eigen_cell_mesh(self._dir("eigen_cell"))

    @functools.cached_property
    def dispersive(self):
        return dispersive_mesh(self._dir("dispersive"))

    @functools.cached_property
    def law_shape_line(self):
        # the shape line's magnetic cover as a Lorentz medium (mu_r kept)
        line = shape_line(self._dir("law_shape_line"))
        return line.with_material("cover", gfem.Material(eps_r=COVER_LAW, mu_r=1.3))


@pytest.fixture(scope="module")
def designs(tmp_path_factory):
    return Designs(tmp_path_factory)


#: The mode-field cells' keys: their eigenpairs to 1e-13 (the forward polishes
#: the modes anyway; the differences stay clean).
MODE_FIELD_KEYS = ("eigenmode_participation", "eigenmode_field")


def _options(cell: Cell, *, production: bool) -> gfem.SolverOptions:
    extra = {}
    if cell.problem == "eigenmode" and cell.solver == "petsc":
        extra["eigen_solver"] = "slepc"  # a PETSc cell runs the whole PETSc stack
    if cell.problem in MODE_FIELD_KEYS and cell.variant == "slepc":
        extra["eigen_solver"] = "slepc"  # SLEPc seeds, the direct adjoint
    if production:
        return gfem.SolverOptions(order=2, linear_solver=cell.solver, **extra)
    if cell.problem in MODE_FIELD_KEYS:
        extra["eigen_tolerance"] = 1e-13
    elif cell.problem.startswith("eigenmode"):
        extra["eigen_tolerance"] = 1e-12
    return tight(cell.solver, **extra)


# --- the cells' objectives ---------------------------------------------------------

#: Steps on unit directions: fourth-order truncation and round-off both well
#: below the 1e-6 gate (measured worst errors 1e-8 .. 1e-10).
STEPS = {
    "eps_r": 1e-2,
    "mu_r": 1e-2,
    "tan_delta": 1e-3,
    "conductivity": 5e-2,
    "nodes": 2e-6,
    # a law's parameters, normalized: p0 (1 + t)
    "law": 1e-3,
    # a blend's density, in [0.2, 0.8] (the linear blend's S is strongly
    # curved: 1e-2 leaves 6e-7 of truncation, 5e-3 4e-8)
    "blend": 5e-3,
}
#: Ranges a cell draws its own ``mu_r`` or ``tan_delta`` from (the other
#: cells' designs carry neither).
RANGES = {"mu_r": (1.0, 2.0), "tan_delta": (0.01, 0.2)}
#: A principal-value input's suffix: ``eps_r_axes`` holds principal values of eps_r.
AXES = "_axes"


def _property(name: str) -> str:
    return name.removesuffix(AXES)


#: Grids of the driven cells: uniform frequencies, adaptive grid and samples.
LINE_FREQUENCIES = (18e9, 27e9)
LINE_GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 21))
LINE_SAMPLES = (LINE_GRID[0], LINE_GRID[20], LINE_GRID[10], LINE_GRID[4], LINE_GRID[16])
GRID = tuple(float(f) for f in np.linspace(10e9, 30e9, 11))
WAVE_FREQUENCIES = (12e9, 24e9)
WAVE_SAMPLES = (GRID[0], GRID[10], GRID[5], GRID[2])
#: The hybrid line's grids: modes 1 and 2 propagate, mode 3 is cut off.
HYBRID_FREQUENCIES = (55e9, 65e9)
HYBRID_GRID = tuple(float(f) for f in np.linspace(52e9, 68e9, 9))
HYBRID_SAMPLES = (HYBRID_GRID[0], HYBRID_GRID[8], HYBRID_GRID[4], HYBRID_GRID[2])
#: Max relative gradient difference, adjoint vs reciprocity, per solver
#: (measured 6.2e-15 and 1.7e-13; direct reached 2.8e-14 on MKL in CI).
SHARED_SURFACE_RECIPROCITY = {"direct": 1e-13, "bddc_gmres": 5e-13}
#: Relative value and gradient differences, adaptive vs full, at the samples
#: (measured 5.6e-11 and below 1e-9).
SHARED_SURFACE_ADAPTIVE = (2e-10, 3e-9)
SHAPE_FREQUENCIES = (GRID[2], GRID[8])
SHAPE_SAMPLES = (GRID[0], GRID[10], GRID[5])
SHEET_FREQUENCIES = (6e9, 9e9)
SHEET_GRID = tuple(float(f) for f in np.linspace(5e9, 10e9, 6))
SHEET_SAMPLES = (SHEET_GRID[0], SHEET_GRID[5], SHEET_GRID[2])


def _sweep(cell: Cell, frequencies, grid, samples, **kwargs):
    """The cell's problem and the grid rows its objective reads."""
    if cell.sweep == "uniform":
        problem = gfem.Driven(frequency=frequencies, **kwargs)
        return problem, list(range(len(frequencies)))
    problem = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples, **kwargs)
    return problem, [grid.index(f) for f in samples]


def _wave(d: Designs, cell: Cell):
    """A wave-port cell's mesh, model, problem and S objective at its rows.

    The ``"shared"`` variant excites p1_m1 of the hybrid line only: its
    cross-mode entries take adjoint loads through the shared surfaces' full
    modal corrections.
    """
    if cell.variant == "shared":
        mesh = d.hybrid_mesh
        problem, rows = _sweep(
            cell, HYBRID_FREQUENCIES, HYBRID_GRID, HYBRID_SAMPLES, excite="p1_m1"
        )
        rows = jnp.asarray(rows)
        return (
            mesh,
            hybrid_line(mesh),
            problem,
            lambda s: shared_surface_functional(s, rows),
        )
    mesh = d.wave_mesh
    problem, rows = _sweep(cell, WAVE_FREQUENCIES, GRID, WAVE_SAMPLES)
    rows = jnp.asarray(rows)
    return (
        mesh,
        wave_line(mesh, cell.variant),
        problem,
        lambda s: s_functional(s.values[rows]),
    )


#: Weights of the energy objective, (regions, terminals): all distinct.
ENERGY_WEIGHTS = np.array([[1.0, 0.3], [-0.7, 2.0]])


def _stack_functional(cell: Cell, result):
    """<W, C> of the stack, or <W, E> over its region energies (``energy``)."""
    weights = ENERGY_WEIGHTS  # not symmetric on purpose
    if cell.variant == "energy":
        value = jnp.stack(list(result.energy.values()))
    else:
        value = result.capacitance
    return jnp.sum(weights * value) / (EPS0 * 1e-6)  # O(1) values


def _electrostatic_eps(d: Designs, cell: Cell, options, seed: int) -> Case:
    """J(eps) = <W, C(eps)> on a two-terminal stack with a random P0 field.

    Per principal value (``eps_r_axes``): ``(n, 3)`` on the tilted axes.
    Variant ``energy``: ``<W, E(eps)>`` over the region energies.
    """
    model = d.stack
    axes = TILTED if cell.input.endswith(AXES) else None

    def objective(eps):
        fields = stack_fields(model, eps, axes)
        result = gfem.solve(fields, STACK_PROBLEM, options=options)
        return _stack_functional(cell, result)

    rng = np.random.default_rng(42 + seed)
    shape = (len(model.mesh.tetrahedra),) + ((3,) if axes else ())
    eps = jnp.asarray(rng.uniform(2.0, 9.0, shape))
    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


ELECTROSTATIC = gfem.Electrostatic(terminals={"top": "top"}, ground="bottom")


def _energy_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """<W, E> over the stack's region energies in every node coordinate."""
    rng = np.random.default_rng(7 + seed)
    eps = jnp.asarray(rng.uniform(2.0, 9.0, len(d.stack.mesh.tetrahedra)))
    model = stack_fields(d.stack, eps)
    structure = jax.tree_util.tree_structure(model.mesh)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(structure, [nodes])
        result = gfem.solve(model.replace(mesh=mesh), STACK_PROBLEM, options=options)
        return _stack_functional(cell, result)

    # the stack is in um: its layers' elements are 1.25e-7 m thick
    return Case(objective, jnp.asarray(model.mesh.nodes), rng, step=1e-9, taylor=1e-8)


def _electrostatic_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """C (pF) of a capacitor whose dielectric interface moves (+ a potential).

    Variant ``energy``: see `_energy_nodes`.
    """
    if cell.variant == "energy":
        return _energy_nodes(d, cell, options, seed)
    model = d.capacitor
    morph = boundary_morph(model, "oxide__nitride", grows="oxide")
    weights = np.random.default_rng(4).uniform(0.5, 1.5, 200)

    def objective(t):
        result = gfem.solve(
            model.replace(mesh=morph(t)), ELECTROSTATIC, options=options
        )
        value = result.capacitance[0, 0] * 1e12
        if cell.variant == "potential":
            phi = result.potential["top"].coefficients
            value = value + jnp.sum(weights[: phi.shape[0]] * phi[:200])
        return value

    t = jnp.zeros(morph.n_params)
    return Case(
        objective,
        t,
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _driven_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the two-layer line in one per-element property.

    Per principal value (``_axes``): biaxial values on the tilted axes, on
    the line with first-order absorbing sides (each side face's admittance
    tensor moves with the values of the element behind it). A ``mu_r`` or
    ``tan_delta`` cell adds its property to the design of the others.
    """
    name = _property(cell.input)
    if cell.input.endswith(AXES):
        line, axes = d.axes_line, TILTED
        values, rng = line_axes_design(line, seed)
    else:
        line, axes = d.line, None
        values, rng = line_design(line, seed)
    if name in RANGES:
        drawn = rng.uniform(*RANGES[name], values["eps_r"].shape)
        values = {**values, name: jnp.asarray(drawn)}
    problem, rows = _sweep(cell, LINE_FREQUENCIES, LINE_GRID, LINE_SAMPLES)

    def objective(p):
        model = with_fields(line, {**values, name: p}, axes)
        s = gfem.solve(model, problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    taylor = {"eps_r": 1e-1, "mu_r": 1e-1, "tan_delta": 3e-2, "conductivity": 1.0}
    p = values[name]
    return Case(
        objective,
        p,
        rng,
        step=STEPS[name],
        taylor=taylor[name],
        imaginary=axes is not None and jnp.iscomplexobj(p),
    )


def _driven_fill(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the glued line in the per-face fill of its sheet."""
    mesh = d.sheet_mesh
    problem, rows = _sweep(cell, SHEET_FREQUENCIES, SHEET_GRID, SHEET_SAMPLES)

    def objective_on(problem, rows):
        def objective(fill):
            model = sheet_line(mesh, SHEETS[cell.variant](gfem.SheetField(fill=fill)))
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    reference = None
    if cell.sweep == "adaptive":
        # the differences are taken on a uniform sweep of the samples
        reference = objective_on(gfem.Driven(frequency=SHEET_SAMPLES), [0, 1, 2])
    fill, rng = sheet_fill(mesh, seed)
    return Case(
        objective_on(problem, rows),
        fill,
        rng,
        step=5e-3,
        taylor=5e-2,
        reference=reference,
    )


def _lumped_values(t):
    """The lumped values `LUMPED` scaled by ``1 + t``: an O(1) input."""
    return jnp.asarray(LUMPED) * (1 + t)


def _driven_lumped(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the glued line in its port's L and C and its sheet's R, L and C."""
    mesh = d.sheet_mesh
    problem, rows = _sweep(cell, SHEET_FREQUENCIES, SHEET_GRID, SHEET_SAMPLES)

    def objective_on(problem, rows):
        def objective(t):
            model = lumped_line(mesh, _lumped_values(t))
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    reference = None
    if cell.sweep == "adaptive":
        # the differences are taken on a uniform sweep of the samples
        reference = objective_on(gfem.Driven(frequency=SHEET_SAMPLES), [0, 1, 2])
    return Case(
        objective_on(problem, rows),
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
        reference=reference,
    )


def _driven_nodes(d: Designs, cell: Cell, options, seed: int, *, line=None) -> Case:
    """S of the line with every kind of shape term; its interface moves."""
    line = d.shape_line if line is None else line
    morph = boundary_morph(line, "sub__cover", grows="sub")
    problem, rows = _sweep(cell, SHAPE_FREQUENCIES, GRID, SHAPE_SAMPLES)

    def objective(t):
        s = gfem.solve(line.replace(mesh=morph(t)), problem, options=options).s.values
        return s_functional(s[jnp.asarray(rows)])

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        # the line's curvature in t is small: its remainder reaches the
        # BDDC + GMRES error (1e-6) below 5e-5, a third-order bend above 4e-4
        taylor=2e-4,
        taylor_ratio=2.0,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _wave_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports in the core's per-element eps_r or conductivity.

    Per principal value (``_axes``): biaxial values on the tilted axes.
    """
    mesh, model, problem, functional = _wave(d, cell)
    values, rng = core_design(mesh, seed)
    name = _property(cell.input)
    axes = TILTED if cell.input.endswith(AXES) else None
    if axes:
        values = {k: v[:, None] * BIAXIAL for k, v in values.items()}

    def objective(p):
        field = gfem.MaterialField(**{**values, name: p}, axes=axes)
        m = model.with_material("core", field)
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return functional(gfem.solve(m, problem, options=options).s)

    # (the lossy uniform BDDC cell's remainder bends above 3e-2)
    taylor = {"eps_r": 3e-2, "conductivity": 1.0}[name]
    p = values[name]
    return Case(
        objective,
        p,
        rng,
        step=STEPS[name],
        taylor=taylor,
        imaginary=axes is not None and jnp.iscomplexobj(p),
    )


def _wave_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports; the top plate moves, held on the port faces."""
    _, model, problem, functional = _wave(d, cell)
    morph = boundary_morph(model, "top")

    def objective(t):
        moved = model.replace(mesh=morph(t))
        return functional(gfem.solve(moved, problem, options=options).s)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _omega(model, options, target):
    result = gfem.solve(model, gfem.Eigenmode(target=target), options=options)
    return jnp.real(result.omega[0] * OMEGA_WEIGHT)


def _eigen_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the terminated line resonator in one per-element property.

    Per principal value (``_axes``): the ramps times distinct factors per
    axis (biaxial), on the tilted axes. A ``mu_r`` or ``tan_delta`` cell
    adds its property, a ramp over its range, to the design of the others.
    """
    mesh = d.resonator
    values = {
        "eps_r": ramp(mesh, "dielectric", 3.5, 4.5),
        "conductivity": ramp(mesh, "dielectric", 0.2, 0.6),
    }
    name = _property(cell.input)
    if name in RANGES:
        values[name] = ramp(mesh, "dielectric", *RANGES[name])
    axes = TILTED if cell.input.endswith(AXES) else None
    if axes:
        values = {k: v[:, None] * BIAXIAL for k, v in values.items()}

    def objective(p):
        fields = values | {name: p}
        eps, sigma = fields.pop("eps_r"), fields.pop("conductivity")
        model = terminated(mesh, eps, sigma, axes, **fields)
        return _omega(model, options, RESONATOR_TARGET)

    # a unit step of 1 S/m would make the 0.2 S/m end of the ramp negative
    # (and one of 0.1 the 0.01 end of the loss tangent's)
    taylor = {"eps_r": 1e-1, "mu_r": 1e-1, "tan_delta": 3e-2, "conductivity": 3e-1}
    taylor = taylor[name]
    return Case(
        objective,
        values[name],
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor=taylor,
    )


def _eigen_fill(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the sheet cavity in the per-face fill of its resistive film."""
    mesh = d.sheet_cavity
    fill, rng = cavity_fill(mesh, seed=1 + seed)

    def objective(p):
        return _omega(cavity_model(mesh, p), options, CAVITY_TARGET)

    return Case(
        objective,
        fill,
        rng,
        step=5e-2,
        taylor=1e-1,
        mask=lambda directions: on_the_sheet(directions, fill),
    )


def _eigen_lumped(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the glued line in its port's L and C and its sheet's R, L and C."""
    mesh = d.sheet_mesh

    def objective(t):
        return _omega(lumped_line(mesh, _lumped_values(t)), options, LUMPED_TARGET)

    return Case(
        objective,
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


def _eigen_nodes(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """omega of the two-layer resonator; its interface moves."""
    model = d.layered_resonator
    morph = boundary_morph(model, "sub__cover", grows="sub")

    def objective(t):
        return _omega(model.replace(mesh=morph(t)), options, RESONATOR_TARGET)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- eigenmode ratios: participation, port Q, region and principal shares ----------

#: The split cavity's pair: a relative gap of 2.49e-4 (above the isolation
#: threshold 1e-4); differences take steps well below the gap.
NEAR_PAIR_DELTA, NEAR_PAIR_STEP = 1e-3, 5e-5
#: The R-only sheet's mode on the glued line (13.47 + 4.28i GHz, Q 1.6).
SHEET_LINE_TARGET = 13.5e9
#: Weights of the principal shares of a cell's principal objective.
PRINCIPAL_WEIGHTS = jnp.array([1.0, -0.6, 0.8])


def _ratios(result, variant: str):
    """A cell's objective: a weighted sum of mode 0's ratios.

    Every J variant reads the junction's participation and the sub layer's
    share; a resistive port adds ``1 / q_port`` (the loss term an objective
    uses, never ``q_port``); the principal variant the sub layer's shares
    along its axes.
    """
    out = result.participation["jj"][0] + 0.7 * result.bulk_participation["sub"][0]
    if "r" in result.q_port:
        out = out + 3.0 / result.q_port["r"][0]
    if variant == "principal":
        shares = result.principal_participation["sub"][0]
        out = out + jnp.dot(PRINCIPAL_WEIGHTS, shares)
    return out


def _kinetic(d: Designs, cell: Cell, options, seed: int) -> Case:
    """The kinetic cavity's mode 0 in its sheet's fill, "lower"'s eps_r or nodes.

    The ``"sheet"`` variant reads the sheet participation; the
    ``"capacitive_sheet"`` one adds a capacitance to the sheet and reads
    "lower"'s share of the mode energy beside it (both move with the
    sheet's term of the energy, ADR 0103). The node cells move the sheet
    with its glue plane, "lower" growing.
    """
    mesh = d.sheet_cavity
    fill, rng = cavity_fill(mesh, seed=1 + seed)
    capacitance = KINETIC_CAPACITANCE if cell.variant == "capacitive_sheet" else None
    problem = gfem.Eigenmode(target=KINETIC_TARGET)

    def read(model):
        result = gfem.solve(model, problem, options=options)
        out = result.sheet_participation["sheet"][0]
        if capacitance is not None:
            out = result.bulk_participation["lower"][0] + 0.5 * out
        return out

    def model(fill, eps=None, mesh=mesh):
        return kinetic_cavity(mesh, fill, eps, capacitance=capacitance)

    if cell.input == "fill":
        return Case(
            lambda p: read(model(p)),
            fill,
            rng,
            step=1e-2,
            taylor=1e-1,
            mask=lambda directions: on_the_sheet(directions, fill),
        )
    if cell.input == "eps_r":
        eps = ramp(mesh, "lower", 1.0, 2.0)
        return Case(
            lambda p: read(model(fill, p)),
            eps,
            rng,
            step=STEPS["eps_r"],
            taylor=1e-1,
        )
    morph = boundary_morph(model(fill), "sheet", grows="lower")
    return Case(
        lambda t: read(model(fill, mesh=morph(t))),
        jnp.zeros(morph.n_params),
        rng,
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


def _participation(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Mode 0's ratios in one input (`_participation_cells`)."""
    variant = cell.variant
    if variant in ("sheet", "capacitive_sheet"):
        return _kinetic(d, cell, options, seed)
    if variant == "absorbing2":
        mesh = d.guide
        core = ramp(mesh, "core", 9.0, 11.0)

        def objective(p):
            problem = gfem.Eigenmode(target=GUIDE_TARGET)
            result = gfem.solve(guide_model(mesh, p), problem, options=options)
            return result.bulk_participation["core"][0]

        return Case(
            objective, core, np.random.default_rng(seed), step=1e-2, taylor=1e-1
        )
    if variant == "resistive_sheet":
        mesh = d.sheet_mesh
        fill, rng = sheet_fill(mesh, seed=1 + seed)

        def objective(p):
            sheet = gfem.Impedance(
                "sheet", resistance=2e3, fill=gfem.SheetField(fill=p)
            )
            problem = gfem.Eigenmode(target=SHEET_LINE_TARGET)
            result = gfem.solve(sheet_line(mesh, sheet), problem, options=options)
            return 1.0 / result.q_port["p1"][0] + 0.5 / result.q_port["p2"][0]

        return Case(objective, fill, rng, step=5e-2, taylor=1e-1)
    if variant == "chi":
        return _chi(d, cell, options, seed)
    if variant == "near_pair":
        mesh = d.corner_cavity
        base = jnp.ones(len(mesh.tetrahedra))

        def objective(p):
            model = split_cavity(mesh, p, NEAR_PAIR_DELTA)
            problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)
            result = gfem.solve(model, problem, options=options)
            # mode 0, the nearer the target, is the lower of the pair
            return result.bulk_participation["corner"][0]

        return Case(
            objective,
            base,
            np.random.default_rng(seed),
            step=NEAR_PAIR_STEP,
            taylor=1e-4,
        )
    mesh = d.junction
    fixture = "inductive" if variant == "slepc" else variant
    values = junction_values(mesh, fixture)
    name = _property(cell.input)

    def objective(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"][name] = p
        model = junction_line(mesh, fixture, moved)
        problem = gfem.Eigenmode(target=JUNCTION_TARGET)
        return _ratios(gfem.solve(model, problem, options=options), fixture)

    p = values["sub"][name]
    return Case(
        objective,
        p,
        np.random.default_rng(seed),
        step=STEPS[name],
        taylor={"eps_r": 1e-1, "conductivity": 3e-1}[name],
        imaginary=bool(jnp.iscomplexobj(p)),
    )


# --- eigenmode normalized field: probes and interface participation ----------------

#: The probed point of the field cells, m: inside the lower layer of fixture
#: J and of the resonator (both 10 x 2 x 1 mm) and of the glued line (8 x 2 x
#: 1 mm), off every element face.
FIELD_POINT = np.array([3.37, 0.93, 0.41]) * 1e-3
#: The probe objective's phase multiplier: ``Re(exp(0.7i) E_z)`` reads the
#: imaginary part of a lossy mode's field, so the phase rule's ``mu2`` is
#: nonzero; and its scale (``|E|`` is about 1e9 V/m at 1 J).
FIELD_PHASE, FIELD_SCALE = np.exp(0.7j), 1e-9
#: The interface layers: J's sub / cover interface (``"default"``), and an
#: oxide on the cavity's metal walls (``"MA"``): thickness (m) and eps_r.
FIELD_LAYER = (3e-9, 10.0)


def _probe(result) -> jax.Array:
    """Mode 0's probed z component, times the phase, scaled."""
    e = result.e_fields[0].at(FIELD_POINT)
    return FIELD_SCALE * jnp.real(FIELD_PHASE * e[2])


def _field(d: Designs, cell: Cell, options, seed: int) -> Case:
    """A measurement of mode 0's normalized field in one input (`_field_cells`)."""
    thickness, eps_layer = FIELD_LAYER
    if cell.input == "fill":
        mesh = d.sheet_cavity
        fill, rng = cavity_fill(mesh, seed=1 + seed)

        def objective(p):
            result = gfem.solve(
                cavity_model(mesh, p),
                gfem.Eigenmode(target=CAVITY_TARGET),
                options=options,
            )
            return (
                1e3
                * gfem.interface_participation(
                    result, "pec", thickness, eps_layer, kind="MA"
                ).participation[0]
            )

        return Case(
            objective,
            fill,
            rng,
            step=5e-2,
            taylor=1e-1,
            mask=lambda directions: on_the_sheet(directions, fill),
        )
    if cell.input == "conductivity":
        mesh = d.resonator
        eps = ramp(mesh, "dielectric", 3.5, 4.5)
        sigma = ramp(mesh, "dielectric", 0.2, 0.6)

        def objective(p):
            problem = gfem.Eigenmode(target=RESONATOR_TARGET)
            return _probe(
                gfem.solve(terminated(mesh, eps, p), problem, options=options)
            )

        return Case(
            objective,
            sigma,
            np.random.default_rng(seed),
            step=STEPS["conductivity"],
            taylor=3e-1,
        )
    mesh = d.junction
    fixture = "resistive" if cell.variant == "probe" else "inductive"
    values = junction_values(mesh, fixture)

    def objective(p):
        moved = {r: dict(v) for r, v in values.items()}
        moved["sub"]["eps_r"] = p
        model = junction_line(mesh, fixture, moved)
        result = gfem.solve(
            model, gfem.Eigenmode(target=JUNCTION_TARGET), options=options
        )
        if cell.variant == "probe":
            return _probe(result)
        return (
            1e3
            * gfem.interface_participation(
                result, "sub__cover", thickness, eps_layer
            ).participation[0]
        )

    return Case(
        objective,
        values["sub"]["eps_r"],
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _lumped_ratio(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Mode 0 of the lossy glued line in its five lumped values (ticket 21).

    All five values scaled by ``1 + t`` (`_lumped_values`). By variant: port
    p1's participation, its port Q, "lower"'s share of the mode energy, the
    sheet's participation, or (``eigenmode_field``) the probed field's
    component along the plates (y) times a phase. The ``"anharmonicity"``
    variant is the lossless line's instead (`_lumped_anharmonicity`).
    """
    mesh = d.sheet_mesh
    problem = gfem.Eigenmode(target=LUMPED_TARGET)
    if cell.variant == "anharmonicity":
        return _lumped_anharmonicity(mesh, problem, options, seed)
    read = {
        "participation": lambda r: r.participation["p1"][0],
        "q_port": lambda r: r.q_port["p1"][0],
        "bulk": lambda r: r.bulk_participation["lower"][0],
        "sheet_participation": lambda r: r.sheet_participation["sheet"][0],
        "probe": lambda r: (
            FIELD_SCALE * jnp.real(FIELD_PHASE * r.e_fields[0].at(FIELD_POINT)[1])
        ),
    }[cell.variant]

    def objective(t):
        model = lumped_line(mesh, _lumped_values(t))
        return read(gfem.solve(model, problem, options=options))

    return Case(
        objective,
        jnp.zeros(len(LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


#: The dressed outputs' scale: rad/s to MHz.
TO_MHZ = 1 / (2 * np.pi * 1e6)


def _lumped_anharmonicity(mesh, problem, options, seed: int) -> Case:
    """Cell L6: the dressed anharmonicity of the lossless line's junction mode.

    Port p1 (L, C) is the linearized junction; its four values
    (`LOSSLESS_LUMPED`) are scaled by ``1 + t``, and the same traced port
    inductance is the ``inductance`` of `gfem.quantum.dressed_modes`, so the
    gradient runs through the solve (``omega``, ``participation``) and
    through the junction energy alike.
    """

    def objective(t):
        values = jnp.asarray(LOSSLESS_LUMPED) * (1 + t)
        result = gfem.solve(
            lossless_lumped_line(mesh, values), problem, options=options
        )
        dressed = gfem.quantum.dressed_modes(
            result.omega, result.participation["p1"], inductance=values[0]
        )
        return TO_MHZ * dressed.anharmonicity[0]

    return Case(
        objective,
        jnp.zeros(len(LOSSLESS_LUMPED)),
        np.random.default_rng(seed),
        step=1e-3,
        taylor=1e-1,
    )


def _dispersive_shift(result) -> jax.Array:
    """The dressed dispersive shift between fixture J's modes 0 and 1, MHz."""
    dressed = gfem.quantum.dressed_modes(
        result.omega, result.participation["jj"], inductance=JUNCTION_CHI_L
    )
    return TO_MHZ * dressed.dispersive_shift[0, 1]


def _chi(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Cells 15 and 16: the dispersive shift of fixture J's two lowest modes.

    The ``"chi"`` variant's junction (`JUNCTION_CHI_L`): both modes (1.70
    and 7.98 GHz) carry it (participations 0.81 and 0.11), and
    `gfem.quantum.dressed_modes` diagonalizes its cosine on them, with the
    solve's inductance. In the lower layer's permittivity, or as the
    layers' interface moves (the ports held). The node step is 5x the other
    node cells': the dressed step's round-off (4e-14 of the shift) then sits
    further under the gate (measured errors 1e-8 to 1e-7, against up to
    5e-7 at 2e-6 m).
    """
    mesh = d.junction
    problem = gfem.Eigenmode(target=JUNCTION_TARGET, n_modes=2)
    rng = np.random.default_rng(seed)
    if cell.input == "nodes":
        model = junction_line(mesh, "chi")
        morph = boundary_morph(model, "sub__cover", grows="sub")

        def moved(t):
            result = gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
            return _dispersive_shift(result)

        return Case(
            moved,
            jnp.zeros(morph.n_params),
            rng,
            step=5 * STEPS["nodes"],
            taylor=4e-5,
            taylor_at=jnp.full(morph.n_params, 1e-5),
        )
    values = junction_values(mesh)

    def objective(p):
        changed = {r: dict(v) for r, v in values.items()}
        changed["sub"]["eps_r"] = p
        model = junction_line(mesh, "chi", changed)
        result = gfem.solve(model, problem, options=options)
        return _dispersive_shift(result)

    return Case(
        objective,
        values["sub"]["eps_r"],
        rng,
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _participation_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Mode 0's ratios of fixture J as its interface "sub__cover" moves.

    The morph holds the ports' surfaces (both junction and resistive port
    sit on the line's ends), so the node gradient is exact along it.
    """
    variant = cell.variant
    if variant in ("sheet", "capacitive_sheet"):
        return _kinetic(d, cell, options, seed)
    if variant == "chi":
        return _chi(d, cell, options, seed)
    model = junction_line(d.junction, variant)
    morph = boundary_morph(model, "sub__cover", grows="sub")
    problem = gfem.Eigenmode(target=JUNCTION_TARGET)

    def objective(t):
        moved = model.replace(mesh=morph(t))
        return _ratios(gfem.solve(moved, problem, options=options), variant)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- driven on a periodic model (the gradient cell G) ------------------------------

#: The adaptive grid and samples of the periodic cells.
PERIODIC_GRID = tuple(float(f) for f in np.linspace(16e9, 32e9, 9))
PERIODIC_SAMPLES = (PERIODIC_GRID[0], PERIODIC_GRID[8], PERIODIC_GRID[4])
#: Probe points of the field cell, m (inside the cell, off the seams).
PERIODIC_PROBES = np.array([[1.3, 0.7, 0.5], [2.6, 4.2, 1.6], [3.4, 2.5, 2.4]]) * 1e-3


def _periodic_sweep(cell: Cell):
    """The periodic cell's problem and the grid rows its objective reads.

    ``wave_vector``: `pc.G_K` for the Bloch problem, (0, 0, 0) for its
    variant ``"k0"`` (the general path at the plain field), none for the
    plain one.
    """
    wave_vector = None
    if cell.problem == "driven_periodic_bloch":
        wave_vector = (0.0, 0.0, 0.0) if cell.variant == "k0" else pc.G_K
    if cell.sweep == "uniform":
        problem = gfem.Driven(
            frequency=pc.G_FREQUENCIES,
            wave_vector=wave_vector,
            keep_fields=pc.G_FREQUENCIES[1] if cell.variant == "field" else False,
        )
        return problem, list(range(len(pc.G_FREQUENCIES)))
    problem = gfem.Driven(
        frequency=PERIODIC_GRID, sweep="adaptive", samples=PERIODIC_SAMPLES
    )
    return problem, [PERIODIC_GRID.index(f) for f in PERIODIC_SAMPLES]


def _periodic_value(cell: Cell, result, rows):
    """S at ``rows``, or for the ``"field"`` variant a functional of probed E."""
    if cell.variant != "field":
        return s_functional(result.s.values[jnp.asarray(rows)])
    weights = np.array([[1.0, 0.5, -0.8], [0.3, 1.2, 0.7], [-0.4, 0.9, 1.1]])
    value = 0.0
    for port in ("p1", "p2"):
        e = result.e_field(pc.G_FREQUENCIES[1], port).at(PERIODIC_PROBES)
        value = value + jnp.sum(weights * jnp.abs(e) ** 2) / 1e4
        value = value + jnp.real((0.6 - 0.4j) * e[1, 2]) / 1e2
    return value


def _periodic_material(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S (or probed E) of G in the per-element eps_r or conductivity of its floor."""
    mesh = d.periodic
    rng = np.random.default_rng(10 + seed)
    n = len(mesh.elements("core")) + len(mesh.elements("sub"))
    values = {
        "eps_r": jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j)),
        "conductivity": jnp.asarray(rng.uniform(0.5, 2.0, n)),
    }
    problem, rows = _periodic_sweep(cell)
    name = cell.input

    def objective(p):
        model = pc.g(mesh, **{**values, name: p})
        return _periodic_value(cell, gfem.solve(model, problem, options=options), rows)

    # G's S bends little in eps_r: below 1e-1 the remainder reaches the
    # BDDC + GMRES error (1e-9 at 3e-2)
    taylor = {"eps_r": 3e-1, "conductivity": 1.0}[name]
    return Case(objective, values[name], rng, step=STEPS[name], taylor=taylor)


def _periodic_fill(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of G in the per-face fill of its sheet, which touches both pairs."""
    mesh = d.periodic
    rng = np.random.default_rng(20 + seed)
    fill = jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("sheet"))))
    problem, rows = _periodic_sweep(cell)

    def objective(p):
        model = pc.g(mesh, fill=p)
        return _periodic_value(cell, gfem.solve(model, problem, options=options), rows)

    # (the Bloch BDDC + GMRES remainder reaches the solver's error below 5e-2)
    return Case(objective, fill, rng, step=5e-3, taylor=1e-1)


def _periodic_nodes(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of G; its sidewall, across the x seam, moves (tied motion)."""
    model = pc.g(d.periodic)
    morph = boundary_morph(model, "wall", grows="core")
    problem, rows = _periodic_sweep(cell)

    def objective(t):
        result = gfem.solve(model.replace(mesh=morph(t)), problem, options=options)
        return _periodic_value(cell, result, rows)

    return Case(
        objective,
        jnp.zeros(morph.n_params),
        np.random.default_rng(seed),
        step=STEPS["nodes"],
        taylor=4e-5,
        taylor_at=jnp.full(morph.n_params, 1e-5),
    )


# --- the law cells' objectives --------------------------------------------------------

#: The smallest normalized law-gradient component a law cell admits, relative
#: to the gradient's norm: no parameter hides under the 1e-6 gate (ADR 0112).
LAW_COMPONENT = 1e-4


def _law_case(objective, law, seed: int) -> Case:
    """A law cell's case: ``objective`` of the normalized parameters t."""
    return Case(
        objective,
        jnp.zeros(len(jax.tree.leaves(law))),
        np.random.default_rng(seed),
        step=STEPS["law"],
        taylor=1e-1,
    )


def _driven_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the strip line in its core's law parameters (`LAWS`).

    ``absorbing1`` / ``absorbing2``: the ``lorentz`` law behind absorbing y
    walls of that order (their law face pieces' terms).
    """
    if cell.sweep == "adaptive":
        return _adaptive_law(d, cell, options, seed)
    name, absorbing = cell.variant, None
    if name.startswith("absorbing"):
        name, absorbing = "lorentz", int(name.removeprefix("absorbing"))
    mesh, law = d.dispersive, LAWS[name]
    problem = gfem.Driven(frequency=LAW_FREQUENCIES[name])

    def objective(t):
        model = dispersive_line(mesh, scaled_law(law, t), absorbing=absorbing)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    return _law_case(objective, law, seed)


def _electrostatic_law(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """C (pF) of the capacitor whose nitride holds `STATIC_LAW`, in its parameters."""
    model = d.capacitor

    def objective(t):
        law = scaled_law(STATIC_LAW, t)
        m = model.with_material("nitride", gfem.Material(eps_r=law))
        return gfem.solve(m, ELECTROSTATIC, options=options).capacitance[0, 0] * 1e12

    return _law_case(objective, STATIC_LAW, seed)


def _static_leaves(law: gfem.PermittivityLaw) -> np.ndarray:
    """Which leaves of ``law`` the static permittivity reads (eps_inf, strengths)."""
    paths = jax.tree_util.tree_flatten_with_path(law)[0]
    return np.array(
        [jax.tree_util.keystr(p).endswith(("eps_inf", "delta_eps")) for p, _ in paths]
    )


def _driven_host_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the strip line in the host's per-element eps_r, beside the ``poles`` law."""
    if cell.sweep == "adaptive":
        return _adaptive_host_law(d, cell, options, seed)
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("dielectric"))
    eps = jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j))
    problem = gfem.Driven(frequency=LAW_FREQUENCIES["poles"])

    def objective(p):
        host = gfem.MaterialField(eps_r=p)
        model = dispersive_line(mesh, LAWS["poles"], host=host)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    return Case(objective, eps, rng, step=STEPS["eps_r"], taylor=1e-1)


#: The adaptive law cells' laws: ``poles`` with its Lorentz pole moved from
#: 30 GHz into the band, and ``lorentz`` (behind the absorbing walls).
ADAPTIVE_LAWS = {
    "poles": dataclasses.replace(
        LAWS["poles"],
        terms=(
            LAWS["poles"].terms[0],
            dataclasses.replace(LAWS["poles"].terms[1], resonance_frequency=13.1e9),
        ),
    ),
    "absorbing1": LAWS["lorentz"],
}
#: Their grids and frozen samples: the endpoints, the two grid neighbours of
#: the pole, two more. Behind an absorbing face Re eps must stay > 0, so the
#: ``lorentz`` grid skips its metallic band (8 to 12.6 GHz) and its margin.
ADAPTIVE_LAW_GRIDS = {
    "poles": tuple(float(f) for f in np.linspace(4e9, 24e9, 21)),
    "absorbing1": tuple(
        float(f) for f in (*np.linspace(4e9, 7e9, 7), *np.linspace(15e9, 24e9, 10))
    ),
}
ADAPTIVE_LAW_SAMPLES = {
    "poles": (4e9, 24e9, 13e9, 14e9, 8e9, 19e9),
    "absorbing1": (4e9, 24e9, 7e9, 15e9, 5.5e9, 19e9),
}


def _adaptive_law_sweep(name: str) -> tuple[gfem.Driven, list[int], gfem.Driven]:
    """The adaptive problem, its sample rows, and the uniform sweep of the samples."""
    grid, samples = ADAPTIVE_LAW_GRIDS[name], ADAPTIVE_LAW_SAMPLES[name]
    problem = gfem.Driven(frequency=grid, sweep="adaptive", samples=samples)
    return problem, [grid.index(f) for f in samples], gfem.Driven(frequency=samples)


def _adaptive_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S at the samples of an adaptive sweep, in the core's law parameters.

    The differences are taken on a uniform sweep of the samples
    (`_driven_fill`'s pattern): at a sample the full solution lies in the
    frozen basis.
    """
    name = cell.variant
    law = ADAPTIVE_LAWS[name]
    absorbing = 1 if name == "absorbing1" else None
    problem, rows, uniform = _adaptive_law_sweep(name)

    def objective_on(problem, rows):
        def objective(t):
            model = dispersive_line(
                d.dispersive, scaled_law(law, t), absorbing=absorbing
            )
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    case = _law_case(objective_on(problem, rows), law, seed)
    case.reference = objective_on(uniform, list(range(len(rows))))
    if name == "absorbing1":
        # the 7 GHz sample sits 1 GHz under a pole 0.4 GHz wide: at 10 %
        # steps the third-order term still shows (slopes 1.66, 1.91)
        case.taylor = 1e-2
    return case


def _adaptive_host_law(d: Designs, _cell: Cell, options, seed: int) -> Case:
    """The host's per-element eps_r beside the moved ``poles`` law, adaptive."""
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    n = len(mesh.elements("dielectric"))
    eps = jnp.asarray(rng.uniform(2.0, 6.0, n) * (1 - 0.02j))
    problem, rows, uniform = _adaptive_law_sweep("poles")

    def objective_on(problem, rows):
        def objective(p):
            host = gfem.MaterialField(eps_r=p)
            model = dispersive_line(mesh, ADAPTIVE_LAWS["poles"], host=host)
            s = gfem.solve(model, problem, options=options).s.values
            return s_functional(s[jnp.asarray(rows)])

        return objective

    return Case(
        objective_on(problem, rows),
        eps,
        rng,
        step=STEPS["eps_r"],
        taylor=1e-1,
        reference=objective_on(uniform, list(range(len(rows)))),
    )


def _driven_blend(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the strip line whose core blends a dielectric and a Drude metal.

    The input is the density, one per core element, in [0.2, 0.8]; the
    variant is the scheme (``index`` or ``linear``).
    """
    mesh = d.dispersive
    rng = np.random.default_rng(seed)
    rho = jnp.asarray(rng.uniform(0.2, 0.8, len(mesh.elements("core"))))
    problem = gfem.Driven(frequency=BLEND_FREQUENCIES)

    def objective(p):
        core = gfem.MaterialField.interpolate(
            p, BLEND_DIELECTRIC, BLEND_METAL, scheme=cell.variant
        )
        model = dispersive_line(mesh, BLEND_DIELECTRIC.eps_r)
        model = model.with_material("core", core)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    # the linear blend's S is strongly curved: its third-order term shows at a
    # Taylor step of 1e-1 (slope 2.11), not from 3e-2
    return Case(objective, rho, rng, step=STEPS["blend"], taylor=3e-2)


def _driven_nodes_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of the shape line whose cover holds a law; its interface moves."""
    return _driven_nodes(d, cell, options, seed, line=d.law_shape_line)


def _wave_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S through wave ports in the law parameters of the core (``poles``)."""
    _, model, problem, functional = _wave(d, cell)
    law = LAWS["poles"]

    def objective(t):
        m = model.with_material("core", gfem.Material(eps_r=scaled_law(law, t)))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return functional(gfem.solve(m, problem, options=options).s)

    return _law_case(objective, law, seed)


def _periodic_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """S of G whose floor "core" holds the ``poles`` law, in its parameters."""
    mesh = d.periodic
    law = LAWS["poles"]
    problem, rows = _periodic_sweep(cell)

    def objective(t):
        model = pc.g(mesh).with_material(
            "core", gfem.Material(eps_r=scaled_law(law, t))
        )
        return _periodic_value(cell, gfem.solve(model, problem, options=options), rows)

    return _law_case(objective, law, seed)


def _eigen_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the layered resonator in its cover's law parameters (`EIGEN_LAWS`).

    ``drude_lossless`` (the polynomial path) moves its eps_inf and plasma
    frequency: its damping, 0, has no normalized direction.
    """
    law = EIGEN_LAWS[cell.variant]
    resonator = d.layered_resonator
    if cell.variant == "drude_lossless":
        (term,) = law.terms

        def build(t):
            drude = dataclasses.replace(
                term, plasma_frequency=term.plasma_frequency * (1 + t[1])
            )
            return dataclasses.replace(
                law, eps_inf=law.eps_inf * (1 + t[0]), terms=[drude]
            )

        size = 2
    else:

        def build(t):
            return scaled_law(law, t)

        size = len(jax.tree.leaves(law))

    def objective(t):
        model = dispersive_resonator(resonator, build(t))
        return _omega(model, options, RESONATOR_TARGET)

    return Case(
        objective,
        jnp.zeros(size),
        np.random.default_rng(seed),
        step=STEPS["law"],
        taylor=1e-1,
    )


def _eigen_beside_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """omega of the layered resonator beside its cover's Lorentz law (E5, E6).

    ``eps_r``: the sub layer as a field (a ramp, its conductivity kept);
    ``nodes``: the interface "sub__cover" moves, through the law region.
    """
    resonator = dispersive_resonator(d.layered_resonator, EIGEN_LAWS["lorentz"])
    if cell.input == "nodes":
        morph = boundary_morph(resonator, "sub__cover", grows="sub")

        def moved(t):
            return _omega(resonator.replace(mesh=morph(t)), options, RESONATOR_TARGET)

        return Case(
            moved,
            jnp.zeros(morph.n_params),
            np.random.default_rng(seed),
            step=STEPS["nodes"],
            taylor=4e-5,
            taylor_at=jnp.full(morph.n_params, 1e-5),
        )
    eps = ramp(resonator.mesh, "sub", 3.5, 4.5)

    def objective(p):
        sub = gfem.MaterialField(eps_r=p, conductivity=0.3)
        return _omega(resonator.with_material("sub", sub), options, RESONATOR_TARGET)

    return Case(
        objective,
        eps,
        np.random.default_rng(seed),
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


def _law_ratios(result):
    """P1-P3's objective: mode 0's ratios beside fixture J's law cover.

    The two shares are weighted unequally: they add up to 1 (no capacitor),
    so an equal sum would have no derivative.
    """
    bulk = result.bulk_participation
    return (
        result.participation["jj"][0]
        + 0.7 * bulk["sub"][0]
        - 0.4 * bulk["cover"][0]
        + 3.0 / result.q_port["r"][0]
    )


def _junction_beside_law(d: Designs, cell: Cell, options, seed: int) -> Case:
    """Fixture J with its cover's lossy Lorentz law (P1-P4, `junction_law_line`).

    Ratios (`_law_ratios`) in the sub layer's eps_r, the junction's L (scaled
    by ``1 + t``) or the interface's nodes; the probed field (`_probe`, the
    phase rule's multiplier nonzero) in the sub layer's eps_r.
    """
    mesh = d.junction
    problem = gfem.Eigenmode(target=JUNCTION_TARGET)
    read = _probe if cell.problem == "eigenmode_field" else _law_ratios

    def solve(model):
        return read(gfem.solve(model, problem, options=options))

    rng = np.random.default_rng(seed)
    if cell.input == "nodes":
        model = junction_law_line(mesh)
        morph = boundary_morph(model, "sub__cover", grows="sub")
        return Case(
            lambda t: solve(model.replace(mesh=morph(t))),
            jnp.zeros(morph.n_params),
            rng,
            step=STEPS["nodes"],
            taylor=4e-5,
            taylor_at=jnp.full(morph.n_params, 1e-5),
        )
    if cell.input == "lumped":
        return Case(
            lambda t: solve(
                junction_law_line(mesh, inductance=JUNCTION_L * (1 + t[0]))
            ),
            jnp.zeros(1),
            rng,
            step=1e-3,
            taylor=1e-1,
        )
    return Case(
        lambda p: solve(junction_law_line(mesh, p)),
        junction_values(mesh)["sub"]["eps_r"],
        rng,
        step=STEPS["eps_r"],
        taylor=1e-1,
    )


# --- the periodic eigenmode cells' objectives ---------------------------------------

#: The design inputs of the periodic eigen cell, by cell input.
EIGEN_CELL_INPUTS = {"eps_r": "eps", "conductivity": "sigma", "fill": "fill"}


def _eigen_periodic(
    d: Designs, cell: Cell, options, seed: int, *, wave_vector=None
) -> Case:
    """omega of the periodic eigen cell (plain or Bloch) in one input.

    ``wave_vector`` solves the plain cell's mode with that wave vector.
    """
    mesh = d.eigen_cell
    design = eigen_cell_design(mesh)
    bloch = cell.problem == "eigenmode_periodic_bloch"
    problem = gfem.Eigenmode(
        target=EG_BLOCH_TARGET if bloch else EG_PLAIN_TARGET,
        wave_vector=EG_K if bloch else wave_vector,
    )

    def omega(**changes):
        values = design | changes
        if cell.variant == "linear":
            model = eigen_cell(values["mesh"], values["eps"], tan_delta=EG_TAN_DELTA)
        else:
            model = eigen_cell(
                values["mesh"], values["eps"], values["sigma"], values["fill"]
            )
        result = gfem.solve(model, problem, options=options)
        return jnp.real(result.omega[0] * OMEGA_WEIGHT)

    design["mesh"] = mesh
    rng = np.random.default_rng(seed)
    if cell.input == "nodes":
        # the mid-plane nodes move along z, each tie class (twins across
        # the pairs, the corners of both) as one: a periodic direction
        nodes = np.asarray(mesh.imported_nodes)
        mid = np.isclose(nodes[:, 2], 0.5 * EG_H * MM)
        labels = coordinate_classes(mesh, (EG_A * MM, EG_B * MM, None))
        classes = np.unique(labels[mid])
        tied = jnp.asarray((labels[:, None] == classes[None, :]) & mid[:, None], float)
        x0 = jnp.asarray(nodes)
        tree = jax.tree_util.tree_structure(mesh)

        def objective(t):
            moved = x0.at[:, 2].add(tied @ t)
            return omega(mesh=jax.tree_util.tree_unflatten(tree, [moved]))

        return Case(
            objective,
            jnp.zeros(len(classes)),
            rng,
            step=STEPS["nodes"],
            # smaller steps' remainders reach the eigen tolerance's error (1e-9)
            taylor=2e-4,
            taylor_ratio=2.0,
            taylor_at=jnp.full(len(classes), 1e-5),
        )
    name = EIGEN_CELL_INPUTS[cell.input]

    def objective(p):
        return omega(**{name: p})

    step = {"eps": STEPS["eps_r"], "sigma": STEPS["conductivity"], "fill": 5e-3}
    # a unit step in the 0.02-0.06 S/m ramp would make it negative
    taylor = {"eps": 1e-1, "sigma": 3e-2, "fill": 5e-2}
    return Case(
        objective,
        jnp.asarray(design[name]),
        rng,
        step=step[name],
        taylor=taylor[name],
    )


BUILDERS = {
    ("electrostatic", "eps_r"): _electrostatic_eps,
    ("electrostatic", "eps_r_axes"): _electrostatic_eps,
    ("electrostatic", "nodes"): _electrostatic_nodes,
    ("driven", "eps_r"): _driven_material,
    ("driven", "conductivity"): _driven_material,
    ("driven", "eps_r_axes"): _driven_material,
    ("driven", "conductivity_axes"): _driven_material,
    **{
        ("driven", f"{name}{form}"): _driven_material
        for name in RANGES
        for form in ("", AXES)
    },
    ("driven", "fill"): _driven_fill,
    ("driven", "lumped"): _driven_lumped,
    ("driven", "nodes"): _driven_nodes,
    ("driven_wave", "eps_r"): _wave_material,
    ("driven_wave", "conductivity"): _wave_material,
    ("driven_wave", "eps_r_axes"): _wave_material,
    ("driven_wave", "nodes"): _wave_nodes,
    ("eigenmode", "eps_r"): _eigen_material,
    ("eigenmode", "conductivity"): _eigen_material,
    ("eigenmode", "eps_r_axes"): _eigen_material,
    ("eigenmode", "conductivity_axes"): _eigen_material,
    ("eigenmode", "mu_r"): _eigen_material,
    ("eigenmode", "tan_delta"): _eigen_material,
    ("eigenmode", "fill"): _eigen_fill,
    ("eigenmode", "lumped"): _eigen_lumped,
    ("eigenmode", "nodes"): _eigen_nodes,
    **{
        (problem, name): builder
        for problem in ("driven_periodic_plain", "driven_periodic_bloch")
        for name, builder in (
            ("eps_r", _periodic_material),
            ("conductivity", _periodic_material),
            ("fill", _periodic_fill),
            ("nodes", _periodic_nodes),
        )
    },
    **{
        (problem, name): _eigen_periodic
        for problem in ("eigenmode_periodic_plain", "eigenmode_periodic_bloch")
        for name in ("eps_r", "conductivity", "fill", "nodes")
    },
    **{
        ("eigenmode_participation", name): _participation
        for name in ("eps_r", "conductivity", "eps_r_axes", "fill")
    },
    **{("eigenmode_field", name): _field for name in ("eps_r", "conductivity", "fill")},
    ("eigenmode_participation", "nodes"): _participation_nodes,
    ("eigenmode_participation", "lumped"): _lumped_ratio,
    ("eigenmode_field", "lumped"): _lumped_ratio,
    ("driven", "law"): _driven_law,
    ("driven_wave", "law"): _wave_law,
    ("driven_periodic_plain", "law"): _periodic_law,
    ("eigenmode", "law"): _eigen_law,
    ("driven", "blend"): _driven_blend,
    ("electrostatic", "law"): _electrostatic_law,
}
#: Builders of the existing inputs beside a law, by (problem, input, variant).
VARIANT_BUILDERS = {
    ("driven", "eps_r", "law"): _driven_host_law,
    ("driven", "nodes", "law"): _driven_nodes_law,
    ("eigenmode", "eps_r", "lorentz"): _eigen_beside_law,
    ("eigenmode", "nodes", "lorentz"): _eigen_beside_law,
    **{
        ("eigenmode_participation", name, "lorentz"): _junction_beside_law
        for name in ("eps_r", "lumped", "nodes")
    },
    ("eigenmode_field", "eps_r", "lorentz"): _junction_beside_law,
}


def _builder(cell: Cell) -> Callable:
    variant = VARIANT_BUILDERS.get((cell.problem, cell.input, cell.variant))
    return variant or BUILDERS[cell.problem, cell.input]


def _case(designs: Designs, cell: Cell, *, production: bool, seed: int = 0) -> Case:
    build = _builder(cell)
    return build(designs, cell, _options(cell, production=production), seed)


# --- the cells -----------------------------------------------------------------------


@pytest.mark.parametrize("cell", [*CELL_PARAMS, *PETSC_PARAMS], ids=str)
def test_gradient_matrix_matches_central_differences(designs, cell):
    case = _case(designs, cell, production=False)
    grad = jax.grad(case.objective)(case.p)
    if cell.input == "law":
        # every normalized law parameter moves the objective; on the
        # electrostatic problem, every one its static permittivity reads, and
        # the frequency parameters are exact zeros
        g = np.abs(np.asarray(grad))
        if cell.problem == "electrostatic":
            read = _static_leaves(STATIC_LAW)
            assert not g[~read].any(), f"{cell}: {g}"
            g = g[read]
        assert g.min() >= LAW_COMPONENT * np.linalg.norm(g), f"{cell}: {g}"
    assert_gradient_matches(
        case.objective,
        case.p,
        case.directions(3),
        step=case.step,
        reference=case.reference,
        grad=grad,
    )


@pytest.mark.parametrize("cell", [*CELL_PARAMS, *PETSC_PARAMS], ids=str)
def test_gradient_matrix_taylor_remainder_is_second_order(designs, cell):
    case = _case(designs, cell, production=True, seed=1)
    (d,) = case.directions(1)
    at = case.p if case.taylor_at is None else case.taylor_at
    steps = [case.taylor / case.taylor_ratio**k for k in range(3)]
    assert_taylor_second_order(case.objective, at, d, steps)


# --- consistency: the same gradient however it is asked for --------------------------

#: One cell per input kind and problem (direct solver), for the transformations.
FAMILIES = [
    Cell("electrostatic", "eps_r", "direct"),
    Cell("electrostatic", "nodes", "direct", variant="potential"),
    Cell("driven", "eps_r", "direct", "uniform"),
    Cell("driven", "eps_r", "direct", "adaptive"),
    Cell("driven", "eps_r_axes", "direct", "uniform"),
    Cell("driven", "fill", "direct", "uniform", "conductor"),
    Cell("driven", "nodes", "direct", "uniform"),
    Cell("driven_wave", "eps_r", "direct", "uniform", "lossy"),
    Cell("eigenmode", "conductivity", "direct"),
    Cell("eigenmode", "fill", "direct"),
    Cell("eigenmode", "nodes", "direct"),
    # the mode-field adjoint: ratios (cell 1), a probed field (cell 26)
    Cell("eigenmode_participation", "eps_r", "direct", variant="inductive"),
    Cell("eigenmode_field", "eps_r", "direct", variant="probe"),
    Cell("eigenmode_participation", "nodes", "direct", variant="inductive"),
    Cell("driven_periodic_bloch", "eps_r", "direct", "uniform"),
    Cell("eigenmode_periodic_bloch", "eps_r", "direct"),
    # a permittivity law as a pytree inside `gfem.Material` (cells D1 and E1)
    Cell("driven", "law", "direct", "uniform", "poles"),
    Cell("eigenmode", "law", "direct", variant="lorentz"),
]
#: Two solves of one design agree to about 1 ulp (threaded assembly), so do
#: their gradients. An eigenmode gradient is only as reproducible as the
#: refined eigenvector, to the eigen tolerance (1e-12) times the operator's
#: condition: each solve's Arnoldi run starts from a random vector, so two
#: solves' gradients differ by up to ~1e-7 (the value by 1e-15).
MODE_FIELD = frozenset({"eigenmode_participation", "eigenmode_field"})

RTOL = {
    "eigenmode": 1e-6,
    "eigenmode_participation": 1e-6,
    "eigenmode_field": 1e-6,
    "eigenmode_periodic_plain": 1e-6,
    "eigenmode_periodic_bloch": 1e-6,
}


def _assert_same(got, want, cell):
    got, want = np.asarray(got), np.asarray(want)
    error = np.linalg.norm(got - want) / np.linalg.norm(want)
    assert error <= RTOL.get(cell.problem, 1e-9), f"{cell}: relative error {error:.2e}"


@pytest.mark.parametrize("cell", FAMILIES, ids=str)
def test_gradient_matrix_transformations_and_eviction_agree(designs, cell):
    # vmap(grad) is a loop of solves, a pullback can be called twice, and a
    # backward without its cache entry recomputes it from the residual design
    case = _case(designs, cell, production=False)
    (d,) = case.directions(1)
    batch = jnp.stack([case.p, case.p + 10 * case.step * d])
    grad = jax.grad(case.objective)
    looped = jnp.stack([grad(p) for p in batch])
    assert np.abs(np.asarray(looped)).max() > 0
    vmapped = jax.vmap(grad)(batch)
    for k in range(2):
        _assert_same(vmapped[k], looped[k], cell)

    _, pullback = jax.vjp(case.objective, case.p)
    first = pullback(1.0)[0]
    _assert_same(first, looped[0], cell)
    # the same kept entry: the second pullback repeats the first exactly, but
    # for a mode-field adjoint, an iterative solve on the kept operator that a
    # first pullback may have reshifted (ADR 0104): there it agrees to the
    # solve's acceptance (measured 1e-11 on MN5), not bit for bit
    repeat = 1e-9 if cell.problem in MODE_FIELD else 1e-12
    np.testing.assert_allclose(pullback(2.0)[0], 2 * first, rtol=repeat)

    options = dataclasses.replace(_options(cell, production=False), retain_memory=0)
    evicted = _builder(cell)(designs, cell, options, 0)
    _assert_same(jax.grad(evicted.objective)(case.p), looped[0], cell)


@pytest.mark.parametrize(
    "solver", [*SOLVERS, pytest.param("petsc", marks=pytest.mark.petsc)]
)
def test_gradient_matrix_reciprocity_equals_the_explicit_adjoint(designs, solver):
    # Exciting p1 only leaves S21 without p2's solution: its gradient takes
    # an adjoint solve. With both ports excited it comes from reciprocity.
    line = designs.line
    values, _ = line_design(line, seed=3)
    options = tight(solver)

    def s21_gradient(problem):
        def objective(eps):
            model = with_fields(line, {**values, "eps_r": eps})
            s = gfem.solve(model, problem, options=options).s
            return jnp.sum(jnp.abs(s["p2", "p1"]) ** 2)

        return jax.grad(objective)(values["eps_r"])

    full = s21_gradient(gfem.Driven(frequency=LINE_FREQUENCIES))
    adjoint = s21_gradient(gfem.Driven(frequency=LINE_FREQUENCIES, excite="p1"))
    np.testing.assert_allclose(adjoint, full, rtol=1e-8, atol=1e-10 * abs(full).max())


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("name", ["eps_r", "conductivity", "law"])
def test_gradient_matrix_adaptive_equals_the_full_solve_at_the_samples(
    designs, solver, name
):
    # at a sample the full solution lies in the frozen basis
    if name == "law":
        # the strip line's moved poles law (cell D3), in its parameters
        cell = Cell("driven", "law", solver, "adaptive", "poles")
        case = _adaptive_law(designs, cell, tight(solver), 0)
        value, grad = jax.value_and_grad(case.objective)(case.p)
        full_value, full_grad = jax.value_and_grad(case.reference)(case.p)
        assert float(value) == pytest.approx(float(full_value), rel=1e-9)
        assert _relative_norm(grad, full_grad) <= 1e-7
        return
    line = designs.line
    values, _ = line_design(line)
    options = tight(solver)
    adaptive_cell = Cell("driven", name, solver, "adaptive")
    adaptive = _driven_material(designs, adaptive_cell, options, 0).objective

    def full(p):
        model = with_fields(line, {**values, name: p})
        problem = gfem.Driven(frequency=LINE_SAMPLES)
        return s_functional(gfem.solve(model, problem, options=options).s.values)

    p = values[name]
    value, grad = jax.value_and_grad(adaptive)(p)
    full_value, full_grad = jax.value_and_grad(full)(p)
    assert float(value) == pytest.approx(float(full_value), rel=1e-9)
    assert _relative_norm(grad, full_grad) <= 1e-7


def _relative_norm(a, b) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def _shared_surface_core(designs, problem, options, *, seed=0):
    """The shared-surface objective in the core's eps_r, on ``problem``'s rows."""
    mesh = designs.hybrid_mesh
    model = hybrid_line(mesh)
    values, _ = core_design(mesh, seed)
    rows = jnp.arange(len(problem.frequency))

    def objective(eps):
        m = model.with_material("core", gfem.MaterialField(**{**values, "eps_r": eps}))
        m = m.replace(mesh=jax.lax.stop_gradient(m.mesh))
        return shared_surface_functional(
            gfem.solve(m, problem, options=options).s, rows
        )

    return objective, values["eps_r"]


@pytest.mark.parametrize("solver", SOLVERS)
def test_gradient_matrix_shared_surface_reciprocity_equals_the_explicit_adjoint(
    designs, solver
):
    # Exciting p1_m1 only leaves the cross-mode entries without p1_m2's and
    # p2_m2's solutions: their gradients take adjoint solves through the
    # shared surfaces' full modal corrections. With every port excited they
    # come from reciprocity, which holds only if each correction is symmetric.
    options = tight(solver)

    def gradient(problem):
        objective, eps = _shared_surface_core(designs, problem, options, seed=3)
        return jax.grad(objective)(eps)

    full = gradient(gfem.Driven(frequency=HYBRID_FREQUENCIES))
    adjoint = gradient(gfem.Driven(frequency=HYBRID_FREQUENCIES, excite="p1_m1"))
    error = np.abs(np.asarray(adjoint - full)).max() / np.abs(np.asarray(full)).max()
    assert error <= SHARED_SURFACE_RECIPROCITY[solver]


def test_gradient_matrix_shared_surface_adaptive_equals_the_full_solve_at_the_samples(
    designs,
):
    # at a sample the full solution lies in the frozen basis
    options = tight("direct")
    cell = Cell("driven_wave", "eps_r", "direct", "adaptive", "shared")
    adaptive = _wave_material(designs, cell, options, 0).objective
    problem = gfem.Driven(frequency=HYBRID_SAMPLES, excite="p1_m1")
    full, eps = _shared_surface_core(designs, problem, options)
    value, grad = jax.value_and_grad(adaptive)(eps)
    full_value, full_grad = jax.value_and_grad(full)(eps)
    value_error = abs(float(value) - float(full_value)) / abs(float(full_value))
    error = float(jnp.linalg.norm(grad - full_grad) / jnp.linalg.norm(full_grad))
    assert value_error <= SHARED_SURFACE_ADAPTIVE[0]
    assert error <= SHARED_SURFACE_ADAPTIVE[1]


# --- consistency on a periodic model (the gradient cell G) -------------------------


@pytest.mark.parametrize("name", ["eps_r", "nodes"])
def test_gradient_matrix_bloch_at_zero_wave_vector_equals_plain(designs, name):
    # k = 0 takes the general path (LU, the transposed system, no reciprocity
    # shortcut) to the plain periodic field
    plain_cell = Cell("driven_periodic_plain", name, "direct", "uniform")
    bloch_cell = Cell("driven_periodic_bloch", name, "direct", "uniform", "k0")
    plain = _case(designs, plain_cell, production=False)
    bloch = _case(designs, bloch_cell, production=False)
    want = np.asarray(jax.grad(plain.objective)(plain.p))
    got = np.asarray(jax.grad(bloch.objective)(plain.p))
    assert np.abs(want).max() > 0
    assert np.linalg.norm(got - want) <= 1e-8 * np.linalg.norm(want)


def _periodic_nodes_objective(model):
    """The Bloch S objective of ``model`` as a function of its raw node coordinates."""
    problem = gfem.Driven(frequency=pc.G_FREQUENCIES, wave_vector=pc.G_K)
    treedef = jax.tree_util.tree_structure(model.mesh)

    def objective(nodes):
        mesh = jax.tree_util.tree_unflatten(treedef, [nodes])
        result = gfem.solve(model.replace(mesh=mesh), problem, options=tight("direct"))
        return s_functional(result.s.values)

    return objective


def test_gradient_matrix_rigid_translation_of_a_periodic_cell_is_free(designs):
    # every term is invariant under a rigid move of the whole mesh, twins and
    # all: the raw node gradient has no component along it
    model = pc.g(designs.periodic)
    nodes = jnp.asarray(model.mesh.nodes)
    grad = np.asarray(jax.grad(_periodic_nodes_objective(model))(nodes))
    t = np.broadcast_to(np.array([0.3, -0.5, 0.8]), grad.shape)
    assert np.abs(grad).max() > 0
    assert abs(np.sum(grad * t)) <= 1e-10 * np.linalg.norm(grad) * np.linalg.norm(t)


def test_gradient_matrix_tied_morph_gradient_is_its_transpose_on_the_raw_one(designs):
    # the morph's shared columns sum each tie class's raw cotangents
    model = pc.g(designs.periodic)
    morph = boundary_morph(model, "wall", grows="core")
    objective = _periodic_nodes_objective(model)
    t = jnp.zeros(morph.n_params)
    tied = np.asarray(jax.grad(lambda t: objective(morph(t).nodes))(t))
    raw = np.asarray(jax.grad(objective)(jnp.asarray(model.mesh.nodes)))
    jacobian = np.asarray(jax.jacfwd(lambda t: morph(t).nodes)(t))  # (n, 3, p)
    want = np.einsum("nip,ni->p", jacobian, raw)
    assert np.abs(want).max() > 0
    np.testing.assert_allclose(tied, want, rtol=0, atol=1e-12 * np.abs(want).max())


# --- degenerate eigenmode clusters -------------------------------------------------
# The TE101 / TE011 pair of a square-based PEC box on a mesh with the
# square's symmetry: exactly degenerate. Only functions symmetric in the pair
# differentiate; one member alone raises gfem.DegenerateModeError (on the
# host, in the backward pass: JAX re-raises it as a JaxRuntimeError that
# keeps the message).

DEGENERATE = jax.errors.JaxRuntimeError
CLUSTER_OPTIONS = gfem.SolverOptions(eigen_tolerance=1e-12)


def _cavity_modes(mesh, eps, n_modes):
    model = gfem.Model(
        mesh,
        materials={"fill": gfem.MaterialField(eps_r=eps, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=n_modes)
    return gfem.solve(model, problem, options=CLUSTER_OPTIONS)


def _uniform(mesh):
    return jnp.ones(len(mesh.elements("fill")))


def test_gradient_matrix_degenerate_pair_is_labelled_as_one_cluster(designs):
    cavity = designs.mirrored_cavity
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    omega = np.asarray(result.omega)
    assert abs(omega[0] - omega[1]) <= 1e-12 * abs(omega[0])
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


@pytest.mark.parametrize("overlap", [0.95, 0.999])
def test_gradient_matrix_nearly_parallel_ritz_vectors_keep_the_pair(
    designs, monkeypatch, overlap
):
    # ARPACK returns an arbitrary basis of a multiple eigenvalue's eigenspace:
    # how parallel its two vectors are depends on round-off (process state)
    import gfem._eigen.rule as eigen_rule

    shift_invert = eigen_rule.shift_invert

    def nearly_parallel(*args, **kwargs):
        k, vectors = shift_invert(*args, **kwargs)
        i, j = np.argsort(np.where(k.real > 0, k.real, np.inf))[:2]
        assert abs(k[i] - k[j]) <= 1e-12 * abs(k[i])  # the pair
        q = np.linalg.qr(vectors[:, [i, j]])[0]
        vectors = vectors.copy()
        vectors[:, i] = q[:, 0]
        vectors[:, j] = overlap * q[:, 0] + np.sqrt(1 - overlap**2) * q[:, 1]
        return k, vectors

    monkeypatch.setattr(eigen_rule, "shift_invert", nearly_parallel)
    cavity = designs.mirrored_cavity
    result = _cavity_modes(cavity, _uniform(cavity), 3)
    np.testing.assert_array_equal(np.asarray(result.cluster), [0, 0, 1])


def test_gradient_matrix_cluster_sum_matches_central_differences(designs):
    cavity = designs.mirrored_cavity

    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(jnp.sum(omega) * OMEGA_WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(3)
    directions = unit_directions(rng, eps.shape, 3)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_cluster_smooth_max_matches_central_differences(designs):
    # any function symmetric in the pair: here a smooth max of the frequencies
    cavity = designs.mirrored_cavity

    def objective(eps):
        f = _cavity_modes(cavity, eps, 3).frequency[:2] / 1e9
        return jax.nn.logsumexp(10 * f) / 10

    eps = _uniform(cavity)
    rng = np.random.default_rng(4)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_single_mode_of_a_cluster_raises(designs):
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 2).omega[0])

    with pytest.raises(DEGENERATE, match="DegenerateModeError") as info:
        jax.grad(objective)(_uniform(cavity))
    message = str(info.value)
    assert "18.0" in message  # the cluster's frequencies
    assert "sum or mean" in message  # the remedy


def test_gradient_matrix_cluster_cut_by_n_modes_raises(designs):
    # n_modes=1 keeps one member of the pair: the guard mode shows the rest
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 1).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_conductor_cluster_cut_by_n_modes_raises(designs):
    # a conductor's seed asks for one pair beyond n_modes: the pair's other
    # member takes that slot and shows the cut
    cavity = designs.mirrored_cavity

    def objective(eps):
        model = gfem.Model(
            cavity,
            materials={"fill": gfem.MaterialField(eps_r=eps)},
            boundaries=[gfem.Conductor("wall", 5.8e7)],
        )
        problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=1)
        return jnp.real(gfem.solve(model, problem, options=CLUSTER_OPTIONS).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_conductor_cluster_cut_with_split_ritz_values_raises(
    designs, monkeypatch
):
    # the frozen seed's Ritz values of the pair split by 3e-8 (relative, at
    # seed tolerance 1e-3: CI run 41300599403) and its vectors were nearly
    # parallel: both seeds refined to one mode, and the guard showed no cut
    import gfem._eigen.rule as eigen_rule

    shift_invert = eigen_rule.shift_invert

    def split_and_parallel(*args, **kwargs):
        k, vectors = shift_invert(*args, **kwargs)
        i, j = np.argsort(np.where(k.real > 0, k.real, np.inf))[:2]
        assert abs(k[i] - k[j]) <= 1e-6 * abs(k[i])  # the pair
        k = k.copy()
        k[j] = k[i] * (1 + 3e-8)
        q = np.linalg.qr(vectors[:, [i, j]])[0]
        vectors = vectors.copy()
        vectors[:, i] = q[:, 0]
        vectors[:, j] = 0.95 * q[:, 0] + np.sqrt(1 - 0.95**2) * q[:, 1]
        return k, vectors

    monkeypatch.setattr(eigen_rule, "shift_invert", split_and_parallel)
    cavity = designs.mirrored_cavity

    def objective(eps):
        model = gfem.Model(
            cavity,
            materials={"fill": gfem.MaterialField(eps_r=eps)},
            boundaries=[gfem.Conductor("wall", 5.8e7)],
        )
        problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=1)
        return jnp.real(gfem.solve(model, problem, options=CLUSTER_OPTIONS).omega[0])

    with pytest.raises(DEGENERATE, match=r"DegenerateModeError.*n_modes"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_simple_mode_beside_a_cluster_differentiates(designs):
    # TM110 alone: no cotangent reaches the pair
    cavity = designs.mirrored_cavity

    def objective(eps):
        return jnp.real(_cavity_modes(cavity, eps, 3).omega[2] * OMEGA_WEIGHT)

    eps = _uniform(cavity)
    rng = np.random.default_rng(5)
    directions = unit_directions(rng, eps.shape, 2)
    assert_gradient_matches(objective, eps, directions, step=1e-2)


def test_gradient_matrix_stop_gradient_on_the_other_member_still_raises(designs):
    # the cotangents differ (one is zero): not symmetric, whatever the opt-out
    cavity = designs.mirrored_cavity

    def objective(eps):
        omega = _cavity_modes(cavity, eps, 2).omega
        return jnp.real(omega[0] + jax.lax.stop_gradient(omega[1]))

    with pytest.raises(DEGENERATE, match="DegenerateModeError"):
        jax.grad(objective)(_uniform(cavity))


def test_gradient_matrix_cluster_sum_in_the_node_coordinates(designs):
    # a shape design that keeps the square's symmetry: one side wall moves
    # (so the pair splits along the design direction); the cluster sum is
    # symmetric whatever the split
    cavity = designs.mirrored_cavity
    model = gfem.Model(
        cavity,
        materials={"fill": gfem.Material(eps_r=1.0, tan_delta=1e-3)},
        boundaries=[gfem.PEC("wall")],
    )
    morph = boundary_morph(model, "wall", where=lambda x: np.abs(x[:, 2]) < 1e-9)
    problem = gfem.Eigenmode(target=MIRRORED_TARGET, n_modes=2)

    def objective(t):
        moved = model.replace(mesh=morph(t))
        omega = gfem.solve(moved, problem, options=CLUSTER_OPTIONS).omega
        return jnp.real(jnp.sum(omega) * OMEGA_WEIGHT)

    t = jnp.zeros(morph.n_params)
    rng = np.random.default_rng(6)
    directions = unit_directions(rng, t.shape, 2)
    assert_gradient_matches(objective, t, directions, step=STEPS["nodes"])


# --- periodic eigenmodes: Bloch at k = 0 is plain periodicity ----------------------


@pytest.mark.parametrize("name", ["eps_r", "nodes"])
def test_gradient_matrix_bloch_eigen_at_k_zero_equals_plain(designs, name):
    # the general path (LU, two-sided refinement, complex projection) at
    # k = 0 and the plain one solve one operator; their gradients agree to
    # the eigen gradients' reproducibility (`RTOL`)
    cell = Cell("eigenmode_periodic_plain", name, "direct")
    options = _options(cell, production=False)
    plain = _eigen_periodic(designs, cell, options, 0)
    at_zero = _eigen_periodic(designs, cell, options, 0, wave_vector=(0.0, 0.0, 0.0))
    want = jax.grad(plain.objective)(plain.p)
    assert np.abs(np.asarray(want)).max() > 0
    _assert_same(jax.grad(at_zero.objective)(plain.p), want, cell)
