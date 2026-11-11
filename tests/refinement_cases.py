"""Singular cases shared by the refinement tests, with their reference values.

One case per problem type, each a 2D 270 degree re-entrant corner extruded by
one structured cell (`structured_meshes`, no gmsh). Uniform refinement
converges at h^(4/3) on all three (`RATE`, order 2), so adaptive refinement
has room to beat it. Every builder that writes a mesh takes the directory to
write it to and ``n``, the cells per unit length of the section; ``grade`` > 1
packs the cells towards the corner.

- Square coax, `coax_model`: a quarter of a square coax (inner half-side
  1 um, outer 2 um, extruded 0.5 um), `COAX_PROBLEM`; quantity
  `coax_capacitance`, the full coax's C / (eps0 len), reference `COAX_C`.
- Step line, `step_model`: a parallel-plate line in mm (L = 10, W = 2,
  H = 1, eps_r = 4) whose top plate steps down to H / 2 at L / 2, two 50 ohm
  lumped ports, `step_problem`; quantity `step_s`, reference `STEP_S`.
- L cavity, `cavity_model`: a PEC cavity of L-shaped section (side 2 mm,
  height 0.5 mm), `CAVITY_PROBLEM`; quantity `cavity_frequency`, reference
  `CAVITY_F1`.

Tests never recompute the references. Provenance: the coax and step values
are Richardson extrapolations (alpha = 4/3) of uniform order-2 runs, coax
n = 64 and 128, step n = 16 and 32, each checked against the fitted alpha
(1.335 and 1.345), the previous pair and an independent graded mesh (grade 3:
coax n = 64, step n = 32). The cavity's first mode is TM_z, so k^2 is the
first Dirichlet eigenvalue of the L (Trefethen and Betcke, "Computed
eigenmodes of planar regions", 2006); Richardson on uniform n = 32 and 64
lands 1.6e-6 from it, which validates the procedure used for the other two.

Errors of a uniform order-2 solve (relative), for orientation:

    n    coax      step S21 (3 GHz)   cavity f1
    2    8.54e-3   1.22e-3            5.66e-3
    4    3.32e-3   4.68e-4            2.22e-3
    8    1.30e-3   1.83e-4            9.55e-4
"""

from pathlib import Path

import numpy as np
import scipy.constants as const
import structured_meshes as sm

import gfem

#: Convergence rate of every case under uniform refinement, per halving of h.
RATE = 4 / 3

#: Full square coax, C / (eps0 len); the quarter model (extruded 0.5 um) has
#: C = COAX_C * eps0 * 0.5e-6 / 4 = 1.1326822e-17 F.
COAX_C = 10.2340924
#: Absolute bound of `COAX_C` (8e-8 relative).
COAX_C_BOUND = 8.4e-7

#: The step line's S entries per frequency (Hz), with absolute bounds.
STEP_S = {
    3e9: {
        "S21": 0.1831794 - 0.9146944j,
        "S21_bound": 5.5e-7,
        "S11": 0.2103088 + 0.2924889j,
        "S11_bound": 3.9e-7,
    },
    8e9: {
        "S21": -0.7821258 + 0.1740548j,
        "S21_bound": 1.4e-6,
        "S11": 0.5964552 - 0.0471740j,
        "S11_bound": 7.1e-7,
    },
}

#: First Dirichlet eigenvalue of the L of side 2, in mm^-2 (closed to ~1e-13).
CAVITY_LAMBDA1 = 9.6397238440219
#: The cavity's first resonance, Hz: c sqrt(CAVITY_LAMBDA1) / (2 pi), side in mm.
CAVITY_F1 = 148.14026965e9

#: Uniform 3D split (each tetrahedron into 8) from the n = 2 mesh: (DOF, error)
#: per level, order 2, direct. Step: |S21(3 GHz) - STEP_S|; cavity: relative
#: error of f1. Level 0 is the n = 2 mesh (pre-asymptotic; `uniform_split_error`
#: fits the later levels).
STEP_SPLIT = ((1706, 1.1377e-3), (11292, 4.5280e-4), (81464, 1.8045e-4))
CAVITY_SPLIT = (
    (674, 5.6633e-3),
    (4484, 2.4081e-3),
    (32456, 9.7531e-4),
    (246416, 3.8907e-4),
)

_EPS_R = 4.0
_QUARTER = 0.5e-6 / 4  # the coax model's extrusion over the full coax's 4 quarters

COAX_PROBLEM = gfem.Electrostatic(terminals={"inner": "inner"}, ground="outer")
CAVITY_PROBLEM = gfem.Eigenmode(target=CAVITY_F1, n_modes=1)


def uniform_split_error(split: tuple, dofs: int) -> float:
    """The uniform split's error at ``dofs``: a power law through levels 1 on."""
    n, e = np.log(np.asarray(split[1:])).T
    slope, offset = np.polyfit(n, e, 1)
    return float(np.exp(offset + slope * np.log(dofs)))


def step_problem(frequencies=tuple(STEP_S)) -> gfem.Driven:
    """The step line's uniform sweep, at the reference frequencies by default."""
    return gfem.Driven(frequency=list(frequencies))


def _mesh(directory: Path, name: str, mesh: tuple, unit: str) -> gfem.Mesh:
    path = sm.write_msh22(Path(directory) / f"{name}.msh", mesh)
    return gfem.Mesh.from_gmsh(path, length_unit=unit)


def coax_model(directory: Path, n: int, *, grade: float = 1.0) -> gfem.Model:
    """The square coax quarter, in um."""
    mesh = _mesh(directory, "coax", sm.square_coax_quarter(n, grade=grade), "um")
    return gfem.Model(mesh, materials={"dielectric": gfem.Material()})


def step_model(
    directory: Path, n: int, *, grade: float = 1.0, unit: str = "mm"
) -> gfem.Model:
    """The step line, in mm, PEC plates, 50 ohm lumped ports "p1" and "p2".

    ``unit`` reads the same mesh in another length unit (a chip-scale line
    in "um").
    """
    mesh = _mesh(directory, "step", sm.step_line(n, grade=grade), unit)
    return gfem.Model(
        mesh,
        materials={"dielectric": gfem.Material(eps_r=_EPS_R)},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def cavity_model(directory: Path, n: int, *, grade: float = 1.0) -> gfem.Model:
    """The L cavity, in mm, PEC walls."""
    mesh = _mesh(directory, "cavity", sm.l_cavity(n, grade=grade), "mm")
    return gfem.Model(
        mesh, materials={"air": gfem.Material()}, boundaries=[gfem.PEC("wall")]
    )


def solve(model: gfem.Model, problem, order: int = 2):
    """An order-``order`` direct solve, the setting the references were measured at."""
    options = gfem.SolverOptions(order=order, linear_solver="direct")
    return gfem.solve(model, problem, options=options)


def coax_capacitance(result: gfem.ElectrostaticResult) -> float:
    """The full coax's C / (eps0 len) from the quarter model's result."""
    return float(result.capacitance[0, 0]) / (const.epsilon_0 * _QUARTER)


def step_s(result: gfem.DrivenResult) -> np.ndarray:
    """The S matrices, shape (frequencies, 2, 2), ports in order p1, p2."""
    return np.asarray(result.s.values)


def cavity_frequency(result: gfem.EigenmodeResult) -> float:
    """The first mode's frequency, Hz (real part)."""
    return float(np.real(np.asarray(result.frequency)[0]))
