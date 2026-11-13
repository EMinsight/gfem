"""The adaptive sweep with a dispersive region against the uniform sweep.

A 16 x 8 x 1 mm parallel-plate patch (host eps 4, PEC plates, first-order
`Absorbing` on the y walls and on the x ends outside the ports, 2 mm lumped
strip ports centred on each end) with a 4 x 4 mm block "core" off the axis,
swept over 1-20 GHz on 51 points. Every law enters the reduced model as its
susceptibility piece weighted by its exact theta(omega) (ADR 0010).

A lossless Lorentz pole inside the band is the case the plain greedy gets
wrong while reporting convergence: the field shapes next to the pole are
absent from a basis built away from it (on this patch, 1.3e-3 on S at the
default tolerance). The greedy therefore solves the grid neighbours of the
pole right after the endpoints (ADR 0116). The smooth laws need no seed: a
Djordjevic-Sarkar substrate and a Debye block behind the absorbing wall (its
face pieces read sqrt(Re eps(omega))) converge like a constant material.
"""

import warnings

import numpy as np
import pytest
import structured_meshes as sm

import gfem

GHZ = 1e9
LX, LY, LZ = 16.0, 8.0, 1.0
CELLS = (16, 8, 1)
GRID = [float(f) for f in np.linspace(1 * GHZ, 20 * GHZ, 51)]
DIRECT = gfem.SolverOptions(linear_solver="direct", tolerance=1e-10)

#: The pole sits off the grid, between 7.84 and 8.22 GHz.
LOSSLESS = gfem.PermittivityLaw(
    eps_inf=2.0,
    terms=[gfem.Lorentz(delta_eps=3.0, resonance_frequency=8.1 * GHZ, damping=0.0)],
)


def _mesh(directory, block):
    """The patch with "core" over ``block`` = ((x0, x1), (y0, y1)), mm."""
    edges = tuple(
        np.linspace(0, length, n + 1)
        for length, n in zip((LX, LY, LZ), CELLS, strict=True)
    )
    (x0, x1), (y0, y1) = block
    tol = 1e-9

    def volume(c):
        return "core" if x0 < c[0] < x1 and y0 < c[1] < y1 else "dielectric"

    def surface(c):
        if abs(c[2]) < tol:
            return "bottom"
        if abs(c[2] - LZ) < tol:
            return "top"
        for x, name in ((0.0, "port1"), (LX, "port2")):
            if abs(c[0] - x) < tol:
                return name if 3 < c[1] < 5 else "side"
        if abs(c[1]) < tol or abs(c[1] - LY) < tol:
            return "side"
        return ""  # interior faces

    path = sm.write_msh22(
        directory / "patch.msh", sm.structured(edges, volume, surface)
    )
    return gfem.Mesh.from_gmsh(path, length_unit="mm")


def _patch(mesh, host, core):
    return gfem.Model(
        mesh,
        materials={"dielectric": host, "core": core},
        boundaries=[gfem.PEC("bottom"), gfem.PEC("top"), gfem.Absorbing("side")],
        ports=[
            gfem.LumpedPort("p1", "port1", "+z"),
            gfem.LumpedPort("p2", "port2", "+z"),
        ],
    )


def _error(model, tolerance=None):
    """max |S_adaptive - S_uniform| over the grid, and the adaptive result."""
    uniform = gfem.solve(model, gfem.Driven(frequency=GRID), options=DIRECT)
    options = DIRECT
    if tolerance is not None:
        options = gfem.SolverOptions(
            linear_solver="direct", tolerance=1e-10, adaptive_tolerance=tolerance
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", gfem.SweepNotConvergedWarning)
        adaptive = gfem.solve(
            model, gfem.Driven(frequency=GRID, sweep="adaptive"), options=options
        )
    difference = np.asarray(adaptive.s.values) - np.asarray(uniform.s.values)
    return float(np.abs(difference).max()), adaptive


@pytest.fixture(scope="module")
def off_axis(tmp_path_factory):
    return _mesh(tmp_path_factory.mktemp("off_axis"), ((6, 10), (1, 5)))


def test_a_lossless_pole_in_band_is_sampled_on_both_sides(off_axis):
    model = _patch(off_axis, gfem.Material(eps_r=4.0), gfem.Material(eps_r=LOSSLESS))
    error, adaptive = _error(model)  # the default adaptive_tolerance
    assert adaptive.sweep.converged
    j = int(np.searchsorted(GRID, 8.1 * GHZ))
    assert tuple(adaptive.sweep.sampled_frequencies[2:4]) == (GRID[j - 1], GRID[j])
    assert error < 1e-4  # 9e-7 measured; 1.3e-3 without the seeds


def test_a_djordjevic_sarkar_substrate_matches_the_uniform_sweep(off_axis):
    host = gfem.Material(
        eps_r=gfem.PermittivityLaw.djordjevic_sarkar(4.3, 0.02, frequency=1 * GHZ)
    )
    error, _ = _error(_patch(off_axis, host, gfem.Material(eps_r=10.0)), 1e-6)
    assert error < 1e-6  # 4e-14 measured


def test_a_law_behind_the_absorbing_wall_matches_the_uniform_sweep(tmp_path):
    # the block touches the y = 0 wall: its order-1 face pieces carry the law
    mesh = _mesh(tmp_path, ((6, 10), (0, 4)))
    debye = gfem.PermittivityLaw(
        eps_inf=3.0, terms=[gfem.Debye(delta_eps=5.0, relaxation_frequency=5 * GHZ)]
    )
    model = _patch(mesh, gfem.Material(eps_r=4.0), gfem.Material(eps_r=debye))
    error, _ = _error(model, 1e-6)
    assert error < 1e-6  # 4e-14 measured
