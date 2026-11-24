"""The cases of the reciprocal port waves: the terminated line with dipoles.

`eigen_cases.terminated_line` (two 25 ohm lumped ports, 20 hexahedra of the
10 x 2 x 1 mm strip line), solved at order 2 (1214 DOFs): a sweep of tens of
frequencies with a handful of dipoles takes a second. The dipoles are
literals (mm), so a failure reproduces without a random generator.
"""

import eigen_cases as ec
import numpy as np

import gfem

#: The band the line is swept over, Hz: its first resonances.
BAND = (4e9, 11e9)
ORDER2 = gfem.SolverOptions(order=2, linear_solver="direct")
#: Five positions inside the line, off the ends, mm.
POSITIONS_MM = (
    (6.287, 1.66, 0.69),
    (2.836, 1.171, 0.202),
    (0.685, 1.367, 0.714),
    (0.455, 1.07, 0.22),
    (7.945, 1.696, 0.638),
)
MOMENT = (0.0, 0.0, 1e-15)


def line_model(directory) -> gfem.Model:
    """The terminated line (ports "p1" and "p2", 25 ohm each)."""
    return ec.terminated_line(ec.line_mesh(directory, (20, 1, 1)), 25.0)


def dipoles(n: int = 5, *, magnetic: bool = True) -> list:
    """``n`` dipoles at the literal positions: electric ones, the last one magnetic.

    ``n`` above 5 repeats the positions, shifted along x by 0.011 mm each time
    so that every position is distinct.
    """
    out = []
    for k in range(n):
        x, y, z = POSITIONS_MM[k % 5]
        position = (x * 1e-3 + 1e-5 * (k // 5) * 1.1, y * 1e-3, z * 1e-3)
        if magnetic and k == n - 1 and n > 1:
            out.append(
                gfem.MagneticDipole(f"d{k}", position=position, moment=(0, 1e-12, 0))
            )
        else:
            out.append(gfem.ElectricDipole(f"d{k}", position=position, moment=MOMENT))
    return out


def grid(n: int = 41) -> tuple[float, ...]:
    return tuple(np.linspace(*BAND, n))
