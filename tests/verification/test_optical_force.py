"""`gfem.optical_force` against closed forms.

Three physical cases, each with the discretization error of its mesh, measured
and gated at a multiple of it: the radiation pressure on the wall of a PEC TE101
cavity at 1 J, the radiation-pressure efficiency of the free-space sphere of
`free_space_cases` against the Mie series, and the force between two silicon
slabs read as `jax.grad` of the effective index of a port mode through a node
map (the mode force of two coupled waveguides). The exactness of the force
(the derivative of the discrete Lagrangian) is tested in
tests/unit/test_optical_force.py; these are the convergence to the physical
force.
"""

import math

import free_space_cases as fsc
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import structured_meshes as sm
from scipy.optimize import brentq

import gfem

C0 = fsc.C0

# --- PEC TE101 cavity: the wall force at 1 J --------------------------------------

TE_A, TE_B, TE_D, TE_T = 22.86, 10.16, 30.0, 2.0  # mm, WR-90 section
TE_TARGET = 8.2e9
GRID = 1e-9  # mm: a classifier tolerance
#: The force on the end wall x = a of the TE101 mode at 1 J:
#: :math:`U / (a (1 + (a/d)^2))`, N (the radiation pressure of the two
#: travelling waves of the mode on its wall).
TE_FORCE = 1.0 / (TE_A * 1e-3 * (1 + (TE_A / TE_D) ** 2))
#: Relative error of the wall force on the 8 x 3 x 10-cell mesh, measured at
#: order 2 (4.45e-5) and order 3 (1.3e-7): the error falls with the order, as
#: the field's does. Gate: three times each.
TE_TOLERANCE = {2: 1.4e-4, 3: 4e-7}


@pytest.fixture(scope="module", params=[2, 3])
def te101(request, tmp_path_factory):
    nx = 8
    edges = (
        np.concatenate([np.linspace(0, TE_A, nx + 1), [TE_A + TE_T]]),
        np.linspace(0, TE_B, 4),
        np.linspace(0, TE_D, nx + 2),
    )

    def volume(c):
        return "metal" if c[0] > TE_A else "air"

    def surface(c):
        if abs(c[0] - TE_A) < GRID:
            return "front"  # the cavity's end wall: the slab's face
        if c[0] > TE_A:
            return ""
        near = [abs(c[0]), abs(c[1]), abs(c[1] - TE_B), abs(c[2]), abs(c[2] - TE_D)]
        return "wall" if min(near) < GRID else ""

    path = tmp_path_factory.mktemp("te101") / "te101.msh"
    sm.write_msh22(path, sm.structured(edges, volume, surface))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material()},
        excluded=["metal"],
        boundaries=[gfem.PEC("wall"), gfem.PEC("front")],
    )
    options = gfem.SolverOptions(order=request.param, linear_solver="direct")
    result = gfem.solve(
        model, gfem.Eigenmode(target=TE_TARGET, n_modes=1), options=options
    )
    return request.param, model, result.e_field(0)


def test_the_wall_force_of_a_te101_mode_is_its_radiation_pressure(te101):
    order, model, field = te101
    force = np.asarray(gfem.optical_force(field, "metal", model=model).force)
    assert force[0] == pytest.approx(TE_FORCE, rel=TE_TOLERANCE[order])


# --- the free-space sphere: radiation pressure against Mie ------------------------

#: Relative gate on :math:`Q_\mathrm{pr}` of the sphere (order 3, plane wave at
#: 30 degrees, the cross sections' mesh, Mie at the volume-equivalent radius).
#: Measured: lossless -0.72 % (TE), -0.94 % (TM); lossy -0.22 % (TE), -0.38 %
#: (TM): the error of the cross sections (`free_space_cases`) at the same mesh.
#: Gate 3 %, above three times the largest.
PR_TOLERANCE = 1.5 * fsc.TOLERANCE
OPTIONS = gfem.SolverOptions(order=3, linear_solver="direct")


@pytest.fixture(scope="module")
def sphere(tmp_path_factory):
    return fsc.sphere(tmp_path_factory.mktemp("force") / "sphere.msh", mirrored=True)


def _pressure(mesh, material, radius):
    r"""Per polarization, the force over :math:`I \pi a^2 / c` (a 3-vector)."""
    model = fsc.model(mesh, material)
    problem = gfem.Driven(fsc.F0, sources=fsc.SOURCES, keep_fields=True)
    result = gfem.solve(model, problem, options=OPTIONS)
    area = math.pi * (radius * fsc.UM) ** 2
    out = {}
    for source in fsc.SOURCES:
        field = result.e_field(fsc.F0, source.name)
        force = np.asarray(gfem.optical_force(field, "sphere", model=model).force)
        xs = gfem.cross_sections(result, source.name, model=model)
        out[source.name] = force * C0 / (float(np.asarray(xs.intensity)[0]) * area)
    return out


K_HAT = np.array([math.sin(fsc.THETA), 0.0, math.cos(fsc.THETA)])


@pytest.mark.parametrize(
    ("eps", "material"), [(4.0, fsc.LOSSLESS), (4 - 1j, fsc.LOSSY)]
)
def test_the_radiation_pressure_on_a_sphere_matches_mie(sphere, eps, material):
    mesh, radius = sphere
    reference = fsc.mie_radiation_pressure(eps, fsc.K0 * radius * fsc.UM)
    for q in _pressure(mesh, material, radius).values():
        assert q @ K_HAT == pytest.approx(reference, rel=PR_TOLERANCE)
        # the mirror plane y = 0 of the mesh and the wave: no force across it,
        # to the round-off of the solve (5e-13 measured)
        assert abs(q[1]) < 1e-8 * reference


# --- two silicon slabs: the mode force ---------------------------------------------

SLAB = 0.22  # um
WAVELENGTH = 1.55  # um
INDEX = 3.48


def _mode_force_exact(gap):
    r"""The force on a slab per unit power per width.

    It is :math:`c^{-1}\partial n_\mathrm{eff}/\partial g`, the gap :math:`g`.

    The even (bonding) TE supermode of two slabs of width 220 nm and index 3.48
    in air, from the field in the gap :math:`\cosh(\gamma x)`, a cosine in the
    core and the decay in the cladding at its right face; the force on the
    outer slab along the gap's growth is the central difference of
    :math:`n_\mathrm{eff}(g)` in the gap divided by :math:`c`
    (step :math:`10^{-4} g`, error :math:`10^{-8}` relative; the same number
    follows from the Maxwell stress in the gap to :math:`5 \times 10^{-8}`).
    """
    k0 = 2 * math.pi / WAVELENGTH

    def n_eff(g):
        def residual(beta):
            gamma = math.sqrt(beta**2 - k0**2)
            kappa = math.sqrt((k0 * INDEX) ** 2 - beta**2)
            u, du = math.cosh(gamma * g / 2), gamma * math.sinh(gamma * g / 2)
            s, c = math.sin(kappa * SLAB), math.cos(kappa * SLAB)
            return (-u * kappa * s + du * c) + gamma * (u * c + du / kappa * s)

        betas = np.linspace(k0 * (1 + 1e-9), k0 * INDEX * (1 - 1e-9), 4000)
        values = [residual(b) for b in betas]
        flips = [i for i in range(len(betas) - 1) if values[i] * values[i + 1] < 0]
        i = flips[-1]  # the fundamental has the largest beta
        return brentq(residual, betas[i], betas[i + 1], xtol=1e-14 * k0) / k0

    h = 1e-4 * gap
    return (n_eff(gap + h) - n_eff(gap - h)) / (2 * h) * 1e6 / C0


#: Reference values of `_mode_force_exact` at 100 and 300 nm, computed by
#: transfer matrices on the same structure, checked here as the series' own.
MODE_REFERENCE = {0.1: -4.760315e-3, 0.3: -5.914489e-4}
#: Relative error of the mode force of the 10 nm-cell, order-3 mesh, measured
#: 7e-7 (gap 100 nm) and 5e-6 (300 nm); gate three times it and more.
MODE_TOLERANCE = {0.1: 3e-6, 0.3: 2e-5}


@pytest.mark.parametrize("gap", [0.1, 0.3])
def test_the_closed_form_of_the_mode_force_is_the_reference(gap):
    assert _mode_force_exact(gap) == pytest.approx(MODE_REFERENCE[gap], rel=1e-5)


def _slab_mesh(path, gap, step=0.01):
    """Two slabs of the gap, one cell thick in y and z, the port at z = 0 (um)."""
    left = (-gap / 2 - SLAB, -gap / 2)
    right = (gap / 2, gap / 2 + SLAB)
    fine = np.arange(-0.45 - gap / 2, 0.45 + gap / 2 + 1e-12, step)
    fine = np.unique(np.round(np.concatenate([fine, [*left, *right]]), 9))
    outer = np.array([0.6, 0.8, 1.05, 1.4, 1.8]) + gap / 2
    xe = np.unique(np.concatenate([-outer[::-1], fine, outer]))
    edges = (xe, np.linspace(0, 0.1, 2), np.linspace(0, 0.05, 2))

    def volume(c):
        if left[0] < c[0] < left[1]:
            return "si1"
        if right[0] < c[0] < right[1]:
            return "si2"
        return "air"

    def surface(c):
        if abs(c[2]) < 1e-9:
            return "port"
        if abs(c[1]) < 1e-9 or abs(c[1] - 0.1) < 1e-9 or abs(abs(c[0]) - xe[-1]) < 1e-9:
            return "wall"
        return ""

    sm.write_msh22(path, sm.structured(edges, volume, surface))
    return gfem.Mesh.from_gmsh(path)


@pytest.mark.parametrize("gap", [0.1, 0.3])
def test_the_mode_force_between_two_slabs_is_the_derivative_of_the_effective_index(
    tmp_path, gap
):
    mesh = _slab_mesh(tmp_path / "slabs.msh", gap)
    silicon = gfem.Material(eps_r=INDEX**2)
    model = gfem.Model(
        mesh,
        materials={"air": gfem.Material(), "si1": silicon, "si2": silicon},
        boundaries=[gfem.PEC("wall")],
        ports=[gfem.WavePort("p", surface="port")],
    )
    nodes = np.asarray(mesh.nodes)
    moving = np.unique(np.asarray(mesh.tetrahedra)[mesh.elements("si2")])
    motion = np.zeros_like(nodes)
    motion[moving, 0] = 1.0  # slab 2 away from slab 1: the gap grows

    def n_eff(q):
        moved = model.replace(mesh=mesh.with_nodes(jnp.asarray(nodes) + q * motion))
        mode = gfem.solve(
            moved,
            gfem.PortMode("p", C0 / (WAVELENGTH * 1e-6)),
            options=gfem.SolverOptions(order=3),
        )
        return jnp.real(mode.n_eff[0])

    # with_nodes reads metres: the derivative per m of the gap, over c
    force = float(jax.grad(n_eff)(0.0)) / C0
    assert force == pytest.approx(_mode_force_exact(gap), rel=MODE_TOLERANCE[gap])
