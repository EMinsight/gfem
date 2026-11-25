"""`gfem.optical_force`: the force and torque a solved field exerts on a body.

The force is the exact derivative of the discrete time-averaged Lagrangian
under a rigid motion of the body's nodes, so on an eigenmode (1 J stored) it
equals -(1 J / omega) d omega / d xi, read from `jax.grad` of the
eigenfrequency through a node map, whatever the mesh error of the field. That
identity is the exactness test; physical closed forms (a plane wave on a
cube, the radiation pressure on the wall of a TE101 cavity) are tests of
the convergence to the physical force and carry their discretization error.

Meshes are structured (`structured_meshes`), in mm for the cavities and um for
the plane wave.
"""

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pml_cases as pc
import pytest
import structured_meshes as sm
from strip2d import write_strip

import gfem
from gfem import _mpi

EPS = np.finfo(float).eps

# --- a closed cavity holding a dielectric block (eigenmode, 1 J) ------------------

A, B, D = 1.0, 0.8, 1.2  # mm
BLOCK = ((0.2, 0.6), (0.2, 0.6), (0.4, 0.8))  # mm, on the 0.2 mm grid
CENTER = np.array([0.4, 0.4, 0.6]) * 1e-3  # m: the block's volume centroid
GRID = 1e-9  # mm: a classifier tolerance
OPTIONS = gfem.SolverOptions(order=2, linear_solver="direct")
TARGET = 150e9  # Hz, near the lowest modes of the loaded cavity
# Gate of the eigen identity: the eigen solve accepts a backward error of
# eigen_tolerance = 1e-10, so the eigenvalue derivative (the perturbation
# formula at the accepted mode) is right to that order; the gate is 100 times
# it, relative to the largest component of the force or torque (the probe
# measured 6e-15 to 3e-11, and 1e-12 here).
IDENTITY = 100 * gfem.SolverOptions().eigen_tolerance


def _cavity_volume(c):
    if all(lo < x < hi for x, (lo, hi) in zip(c, BLOCK, strict=True)):
        return "block"
    return "gas" if c[0] < 0.2 else "air"


def _cavity_surface(c):
    if abs(c[0]) < GRID:
        return "px0"
    if abs(c[0] - A) < GRID:
        return "px1"
    if min(abs(c[1]), abs(c[1] - B), abs(c[2]), abs(c[2] - D)) < GRID:
        return "wall"
    if abs(c[0] - 0.2) < GRID and 0.4 < c[2] < 0.8:
        # the x = 0.2 interface: on the block, or touching it at its edge
        return "whole" if 0.2 < c[1] < 0.6 else ("partial" if c[1] < 0.2 else "")
    return ""


@pytest.fixture(scope="module")
def cavity(tmp_path_factory):
    edges = (np.linspace(0, A, 6), np.linspace(0, B, 5), np.linspace(0, D, 7))
    path = tmp_path_factory.mktemp("cavity") / "cavity.msh"
    sm.write_msh22(path, sm.structured(edges, _cavity_volume, _cavity_surface))
    mesh = gfem.Mesh.from_gmsh(path, length_unit="mm")
    model = gfem.Model(
        mesh,
        materials={
            "air": gfem.Material(),
            "gas": gfem.Material(),
            "block": gfem.Material(eps_r=4.0),
        },
        boundaries=[gfem.PEC("wall"), gfem.PEC("px0"), gfem.PEC("px1")],
    )
    result = gfem.solve(
        model, gfem.Eigenmode(target=TARGET, n_modes=1), options=OPTIONS
    )
    return types.SimpleNamespace(mesh=mesh, model=model, field=result.e_field(0))


def _block_nodes(mesh):
    return np.unique(np.asarray(mesh.tetrahedra)[mesh.elements("block")])


def _motion_matrix(mesh, nodes, center):
    """Node displacements ``(6, n_nodes, 3)`` of the six unit rigid motions.

    The three translations, then the rotations about the axes through
    ``center``, of the body's nodes: x -> x + a + b x (x - c).
    """
    x = np.asarray(mesh.nodes)
    motion = np.zeros((6, *x.shape))
    for axis in range(3):
        unit = np.eye(3)[axis]
        motion[axis, nodes] = unit
        motion[3 + axis, nodes] = np.cross(unit, x[nodes] - center)
    return motion


def _frequency_gradient(cavity, motion):
    """d omega / d (the six rigid motions' amplitudes), from `jax.grad`."""
    x = jnp.asarray(cavity.mesh.nodes)

    def omega(a):
        moved = cavity.model.replace(
            mesh=cavity.mesh.with_nodes(x + jnp.tensordot(a, jnp.asarray(motion), 1))
        )
        eigen = gfem.solve(
            moved, gfem.Eigenmode(target=TARGET, n_modes=1), options=OPTIONS
        )
        return jnp.real(eigen.omega[0])

    return np.asarray(jax.grad(omega)(jnp.zeros(6)))


def test_the_force_and_torque_of_an_eigenmode_are_minus_one_joule_over_omega_d_omega(
    cavity,
):
    nodes = _block_nodes(cavity.mesh)
    got = gfem.optical_force(cavity.field, "block", model=cavity.model)
    motion = _motion_matrix(cavity.mesh, nodes, np.asarray(got.origin))
    omega = float(np.asarray(cavity.field.omega).real)
    expected = -_frequency_gradient(cavity, motion) / omega  # at 1 J
    force, torque = expected[:3], expected[3:]
    assert np.abs(np.asarray(got.force) - force).max() <= IDENTITY * np.abs(force).max()
    assert (
        np.abs(np.asarray(got.torque) - torque).max() <= IDENTITY * np.abs(torque).max()
    )
    # the field is not symmetric: every component is a real test
    assert np.abs(force).min() > 1e-4 * np.abs(force).max()


def test_the_default_origin_is_the_body_volume_centroid(cavity):
    got = gfem.optical_force(cavity.field, "block", model=cavity.model)
    assert got.origin.shape == (3,)
    assert np.abs(np.asarray(got.origin) - CENTER).max() < 1e-16
    # a predicate body selects the same elements
    by_predicate = gfem.optical_force(
        cavity.field,
        lambda c: (
            (c > np.array(BLOCK)[:, 0] * 1e-3) & (c < np.array(BLOCK)[:, 1] * 1e-3)
        ).all(axis=1),
        model=cavity.model,
    )
    np.testing.assert_array_equal(by_predicate.force, got.force)
    np.testing.assert_array_equal(by_predicate.torque, got.torque)


def test_the_torque_about_another_origin_adds_the_moment_of_the_force(cavity):
    first = gfem.optical_force(cavity.field, "block", model=cavity.model)
    other = np.array([0.7e-3, -0.2e-3, 1.1e-3])
    second = gfem.optical_force(cavity.field, "block", model=cavity.model, origin=other)
    np.testing.assert_array_equal(second.force, first.force)
    np.testing.assert_array_equal(second.origin, other)
    expected = np.asarray(first.torque) + np.cross(
        np.asarray(first.origin) - other, np.asarray(first.force)
    )
    # one cross product and one sum in float64: a few rounding errors of
    # the largest term
    scale = np.abs(first.torque).max() + np.linalg.norm(first.origin - other) * (
        np.linalg.norm(first.force)
    )
    assert np.abs(np.asarray(second.torque) - expected).max() <= 16 * EPS * scale
    # and the origin is a traced value without a change of the result
    traced = jax.jit(
        lambda o: (
            gfem.optical_force(
                cavity.field, "block", model=cavity.model, origin=o
            ).torque
        )
    )(other)
    np.testing.assert_allclose(traced, second.torque, rtol=0, atol=16 * EPS * scale)


def test_the_result_is_a_pytree_of_three_float_arrays(cavity):
    got = gfem.optical_force(cavity.field, "block", model=cavity.model)
    leaves, tree = jax.tree_util.tree_flatten(got)
    assert len(leaves) == 3
    assert all(leaf.shape == (3,) and leaf.dtype == jnp.float64 for leaf in leaves)
    assert jax.tree_util.tree_unflatten(tree, leaves).force is leaves[0]
    assert "N m" in repr(got)
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.force = leaves[1]  # ty: ignore[invalid-assignment]


def test_a_body_may_have_its_own_sheet_but_not_one_that_it_touches(cavity):
    # the interface at x = 0.2 on the block's face lies on the body
    model = cavity.model.replace(
        boundaries=[*cavity.model.boundaries, gfem.Impedance("whole", resistance=50.0)]
    )
    on = gfem.optical_force(cavity.field, "block", model=model)
    assert np.isfinite(on.force).all()
    touching = cavity.model.replace(
        boundaries=[
            *cavity.model.boundaries,
            gfem.Impedance("partial", resistance=50.0),
        ]
    )
    with pytest.raises(ValueError, match=r"'partial'.*touches the body"):
        gfem.optical_force(cavity.field, "block", model=touching)


# --- a plane wave on a lossless cube in a perfectly matched layer -----------------


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    sides = pc.SIDES
    lo, hi = 1 / 3, 2 / 3

    def inside(c, a, b):
        return bool(((c > a) & (c < b)).all())

    def volume(c):
        if inside(c, lo, hi):
            return "scat"
        if inside(c, 0, 1):
            return "gas" if c[0] < 0.5 else "air"
        return "pml"

    path = pc.box(
        tmp_path_factory.mktemp("cube") / "cube.msh",
        sides,
        thickness=1 / 3,
        cells=6,
        volume=volume,
    )
    mesh = gfem.Mesh.from_gmsh(path)
    air = gfem.Material()
    model = gfem.Model(
        mesh,
        materials={
            "air": air,
            "gas": air,
            "pml": air,
            "scat": gfem.Material(eps_r=4.0),
        },
        pml=gfem.PML("pml"),
        background=air,
    )
    wave = gfem.PlaneWave("pw", theta=0.0, phi=0.0, polarization="TM")
    result = gfem.solve(
        model,
        gfem.Driven(pc.F0, sources=[wave], keep_fields=True),
        options=gfem.SolverOptions(order=2, linear_solver="direct"),
    )
    return types.SimpleNamespace(mesh=mesh, model=model, result=result, wave=wave)


def test_a_plane_wave_pushes_a_lossless_cube_along_its_direction(cube):
    field = cube.result.e_field(pc.F0, "pw")
    got = gfem.optical_force(field, "scat", model=cube.model)
    force = np.asarray(got.force)
    xs = gfem.cross_sections(cube.result, "pw", model=cube.model)
    sigma = float(np.asarray(xs.scattering)[0])
    intensity = float(np.asarray(xs.intensity)[0])
    pressure = force[2] * 299792458.0 / (intensity * sigma)
    # F_z = (I/c) sigma_sca (1 - <cos theta>) lies between 0 and 2 I sigma_sca / c
    assert 0 < pressure < 2
    # by symmetry the transverse components vanish: the Kuhn mesh is not
    # mirror symmetric, so they are of the discretization error of F_z
    # (measured 0.6 % of it, gated at 5 %)
    assert np.abs(force[:2]).max() < 0.05 * force[2]
    # a lossless cube takes no torque from a linearly polarized wave about
    # its centre, up to the same asymmetry (measured 0.1 % of F_z times the
    # cube's half-width, gated at 2 %)
    assert np.abs(got.torque).max() < 0.02 * force[2] * 0.5e-6
    np.testing.assert_allclose(got.origin, np.full(3, 0.5e-6), atol=1e-18)


# --- the radiation pressure on the wall of a TE101 cavity -------------------------

TE_A, TE_B, TE_D, TE_T = 22.86, 10.16, 30.0, 2.0  # mm
TE_TARGET = 8.2e9
TE_GRID = 6


@pytest.fixture(scope="module")
def te101(tmp_path_factory):
    edges = (
        np.concatenate([np.linspace(0, TE_A, TE_GRID + 1), [TE_A + TE_T]]),
        np.linspace(0, TE_B, 4),
        np.linspace(0, TE_D, TE_GRID + 2),
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
    result = gfem.solve(
        model, gfem.Eigenmode(target=TE_TARGET, n_modes=1), options=OPTIONS
    )
    return types.SimpleNamespace(mesh=mesh, model=model, field=result.e_field(0))


def test_a_body_sliding_on_a_wall_has_nan_for_the_components_along_it(te101):
    got = gfem.optical_force(te101.field, "metal", model=te101.model)
    force = np.asarray(got.force)
    # the slab spans the walls y = 0, y = b, z = 0 and z = d: it may move
    # along x only, whatever else its motion would be
    assert np.isfinite(force[0])
    assert np.isnan(force[1:]).all()
    assert np.isnan(np.asarray(got.torque)).all()
    # the slab's own front face is not a wall it slides on: the force is the
    # radiation pressure of the TE101 mode, U / (a (1 + (a / d)^2)) at 1 J,
    # to the discretization error of the 6-cell mesh (measured 7e-5 here;
    # 4.5e-5 at 8 cells), gated at 5 times it
    exact = 1.0 / (TE_A * 1e-3 * (1 + (TE_A / TE_D) ** 2))
    assert abs(force[0] - exact) < 4e-4 * exact


def test_the_wall_force_equals_minus_one_joule_over_omega_d_omega(te101):
    mesh, model = te101.mesh, te101.model
    nodes = np.unique(np.asarray(mesh.tetrahedra)[mesh.elements("metal")])
    x = jnp.asarray(mesh.nodes)
    direction = np.zeros(x.shape)
    direction[nodes, 0] = 1.0

    def omega(a):
        moved = model.replace(mesh=mesh.with_nodes(x + a * jnp.asarray(direction)))
        eigen = gfem.solve(
            moved, gfem.Eigenmode(target=TE_TARGET, n_modes=1), options=OPTIONS
        )
        return jnp.real(eigen.omega[0])

    expected = -float(jax.grad(omega)(0.0)) / float(np.asarray(te101.field.omega).real)
    got = float(gfem.optical_force(te101.field, "metal", model=model).force[0])
    assert abs(got - expected) <= IDENTITY * abs(expected)


# --- refusals ---------------------------------------------------------------------


def _force(cavity, body="block", *, model=None, field=None, **kwargs):
    return gfem.optical_force(
        field or cavity.field, body, model=model or cavity.model, **kwargs
    )


def test_wrong_argument_types_raise_type_errors(cavity):
    with pytest.raises(TypeError, match=r"needs a gfem\.Field"):
        gfem.optical_force(np.zeros(3), "block", model=cavity.model)
    with pytest.raises(TypeError, match=r"needs model= a gfem\.Model"):
        gfem.optical_force(cavity.field, "block", model=cavity.model.mesh)
    for body in (3, [], [1, 2], None):
        with pytest.raises(TypeError, match="body must be a volume region name"):
            _force(cavity, body)
    for origin in ([0.0, 0.0], 1.0, "abc", [0j, 0j, 0j], [True, False, True]):
        with pytest.raises(TypeError, match="origin must be None or three real"):
            _force(cavity, origin=origin)


def test_an_unknown_region_raises_by_name(cavity):
    with pytest.raises(gfem.RegionNotFoundError, match="nowhere"):
        _force(cavity, "nowhere")


def test_a_field_that_is_not_a_3d_time_harmonic_electric_field_is_refused(cavity):
    field = cavity.field
    with pytest.raises(ValueError, match="electric field"):
        _force(cavity, field=dataclasses.replace(field, quantity="B", space="hdiv"))
    with pytest.raises(ValueError, match="static"):
        _force(cavity, field=dataclasses.replace(field, omega=None))
    real = dataclasses.replace(field, coefficients=jnp.real(field.coefficients))
    with pytest.raises(ValueError, match="complex phasor"):
        _force(cavity, field=real)


def test_a_field_not_solved_on_the_model_is_refused(cavity):
    moved = cavity.mesh.with_nodes(1.01 * np.asarray(cavity.mesh.nodes))
    with pytest.raises(ValueError, match="another mesh"):
        _force(cavity, model=cavity.model.replace(mesh=moved))
    other = cavity.model.replace(excluded=["block"])
    with pytest.raises(ValueError, match="solve domain"):
        _force(cavity, model=other)


def test_a_body_that_selects_no_element_or_has_no_shell_is_refused(cavity):
    with pytest.raises(ValueError, match="surface region"):
        _force(cavity, "whole")
    with pytest.raises(ValueError, match="one boolean per tetrahedron"):
        _force(cavity, lambda _c: np.zeros(3, bool))
    with pytest.raises(ValueError, match="selects no element"):
        _force(cavity, lambda c: np.zeros(len(c), bool))
    with pytest.raises(ValueError, match="no stress shell"):
        _force(cavity, lambda c: np.ones(len(c), bool))


def _gas(material):
    """The cavity's model with its "gas" region (in the shell) another material."""

    def make(cavity):
        return cavity.model.replace(
            materials={**cavity.model.materials, "gas": material}
        )

    return make


def test_a_shell_that_is_not_one_plain_medium_is_refused(cavity):
    n = len(cavity.mesh.elements("gas"))
    cases = {
        "MaterialField": gfem.MaterialField(eps_r=np.ones(n)),
        "lossy": gfem.Material(tan_delta=0.01),
        "conducting": gfem.Material(conductivity=1.0),
        "anisotropic": gfem.Material(eps_r=(1.0, 1.0, 2.0)),
        "law": gfem.Material(
            eps_r=gfem.PermittivityLaw(
                terms=[gfem.Lorentz(delta_eps=1.0, resonance_frequency=1e12, damping=0)]
            )
        ),
    }
    expected = {
        "MaterialField": "MaterialField",
        "lossy": "lossy",
        "conducting": "lossy",
        "anisotropic": "anisotropic",
        "law": "dispersive",
    }
    for name, material in cases.items():
        with pytest.raises(ValueError, match=expected[name]):
            _force(cavity, model=_gas(material)(cavity))
    with pytest.raises(ValueError, match="two media"):
        _force(cavity, model=_gas(gfem.Material(eps_r=1.5))(cavity))
    with pytest.raises(ValueError, match="two media"):
        _force(cavity, model=_gas(gfem.Material(mu_r=1.5))(cavity))


def test_a_body_with_a_lossy_or_anisotropic_or_designed_body_is_accepted(cavity):
    for material in (
        gfem.Material(eps_r=4.0, tan_delta=0.01),
        gfem.Material(eps_r=(4.0, 3.0, 5.0)),
    ):
        model = cavity.model.replace(
            materials={**cavity.model.materials, "block": material}
        )
        assert np.isfinite(_force(cavity, model=model).force).all()


def test_a_port_an_absorbing_surface_or_a_layer_beside_the_shell_is_refused(cavity):
    boundaries = [gfem.PEC("wall"), gfem.PEC("px1")]
    absorbing = cavity.model.replace(boundaries=[*boundaries, gfem.Absorbing("px0")])
    with pytest.raises(ValueError, match="absorbing surface 'px0'"):
        _force(cavity, model=absorbing)
    port = gfem.LumpedPort("p", "px0", "+y", resistance=50.0)
    ported = cavity.model.replace(boundaries=boundaries, ports=[port])
    with pytest.raises(ValueError, match=r"port 'p'.*surface 'px0'"):
        _force(cavity, model=ported)


def test_a_body_node_on_a_periodic_pair_is_refused(cavity):
    # a body on the periodic wall: the cells of the first x layer below the block
    low = cavity.model.replace(
        materials={**cavity.model.materials, "block": gfem.Material()},
        boundaries=[gfem.PEC("wall"), gfem.Periodic(("px0", "px1"))],
    )
    with pytest.raises(ValueError, match="Periodic pair"):
        _force(cavity, lambda c: (c[:, 0] < 0.2e-3) & (c[:, 2] < 0.4e-3), model=low)


def test_a_dipole_in_the_shell_is_refused(cavity):
    near = ("dipole", (0.1e-3, 0.5e-3, 0.7e-3))
    field = dataclasses.replace(cavity.field, _point_sources=(near,))
    with pytest.raises(ValueError, match="dipole 'dipole'"):
        _force(cavity, field=field)
    far = ("dipole", (0.9e-3, 0.1e-3, 0.1e-3))
    assert np.isfinite(
        _force(
            cavity, field=dataclasses.replace(cavity.field, _point_sources=(far,))
        ).force
    ).all()


def test_the_layer_and_the_parts_of_an_incident_column_are_refused(cube):
    near = lambda c: ((c > 1 / 6 * 1e-6) & (c < 5 / 6 * 1e-6)).all(axis=1)  # noqa: E731
    total = cube.result.e_field(pc.F0, "pw")
    with pytest.raises(ValueError, match="perfectly matched layer"):
        gfem.optical_force(total, near, model=cube.model)
    for part, word in (("incident", "incident"), ("scattered", "scattered")):
        field = cube.result.e_field(pc.F0, "pw", part=part)
        with pytest.raises(ValueError, match=f"the {word} part"):
            gfem.optical_force(field, "scat", model=cube.model)


def test_a_distributed_field_the_recorder_and_a_2d_mesh_are_refused(
    cavity, monkeypatch
):
    across = dataclasses.replace(
        cavity.field, _comm=_mpi.StaticComm(5, types.SimpleNamespace(rank=0, size=4))
    )
    with pytest.raises(gfem.DistributedUnsupportedError) as caught:
        _force(cavity, field=across)
    assert caught.value.feature == "optical_force"
    from gfem import measurements

    monkeypatch.setattr(measurements._objective, "recording", object)
    with pytest.raises(gfem.MeshRefinementUnsupportedError, match="optical_force"):
        _force(cavity)


def test_a_2d_mesh_is_refused_by_name(tmp_path):
    mesh = gfem.Mesh.from_gmsh(write_strip(tmp_path / "strip.msh"), length_unit="um")
    model = gfem.Model(
        mesh,
        materials={name: gfem.Material(eps_r=2.0) for name in "abc"},
        boundaries=(gfem.PEC("outer"),),
    )
    field = gfem.Field(
        coefficients=jnp.zeros(len(mesh.nodes), complex),
        mesh=mesh,
        quantity="E",
        unit="V/m",
        space="h1",
        order=1,
        support=("a", "b", "c"),
        omega=1e12,
    )
    with pytest.raises(NotImplementedError, match=r"optical_force.*2D mesh"):
        gfem.optical_force(field, "a", model=model)


# --- gradients ---------------------------------------------------------------------


def _wrench(field, model):
    """Force then torque about the centroid, ``(6,)``."""
    got = gfem.optical_force(field, "block", model=model)
    return jnp.concatenate([got.force, got.torque])


def _fd4(f, x, h):
    """The five-point derivative of ``f`` at ``x`` with step ``h``."""
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / (12 * h)


def _check_design_gradient(solve, eps, h, rtol, components):
    """``jax.vjp`` of a design's wrench against the five-point FD of two solves.

    The derivative of ``f`` is ``f'(eps) = FD4 + h^4/30 f^(5) + O(round-off/h)``.
    """
    wrench, pull = jax.vjp(solve, eps)
    fd = _fd4(lambda x: np.asarray(solve(x)), eps, h)
    scale = np.abs(fd).max()
    for k in components:
        ct = jnp.zeros(6).at[k].set(1.0)
        (grad,) = pull(ct)
        assert abs(float(grad) - fd[k]) <= rtol * scale, (k, float(grad), fd[k])
    return wrench


def test_the_gradient_of_an_eigenmode_force_and_torque_in_the_body_permittivity(
    cavity,
):
    n_block = len(cavity.mesh.elements("block"))

    def solve(eps):
        model = cavity.model.replace(
            materials={
                **cavity.model.materials,
                "block": gfem.MaterialField(eps_r=eps * jnp.ones(n_block)),
            }
        )
        result = gfem.solve(
            model, gfem.Eigenmode(target=TARGET, n_modes=1), options=OPTIONS
        )
        return _wrench(result.e_field(0), model)

    # Truncation of the five-point derivative is h^4 f^(5) / 30 and the
    # force varies on the scale of eps = 4, so f^(5) / f' ~ 4! / eps^4 and h =
    # 0.05 leaves 2e-8; the round-off of an eigen solve (backward error
    # eigen_tolerance = 1e-10 on the mode) over 12 h is 1e-9 at worst. Both are
    # far below the gate; the probe measured 4.5e-10 at h = 1e-4. Without the
    # frequency cotangent the gradient is 60 % off.
    _check_design_gradient(solve, 4.0, 0.05, 1e-6, components=(0, 1, 2, 3, 4, 5))


def test_the_gradient_of_a_driven_force_in_the_body_permittivity(cube):
    n_scat = len(cube.mesh.elements("scat"))

    def solve(eps):
        model = cube.model.replace(
            materials={
                **cube.model.materials,
                "scat": gfem.MaterialField(eps_r=eps * jnp.ones(n_scat)),
            }
        )
        result = gfem.solve(
            model,
            gfem.Driven(pc.F0, sources=[cube.wave], keep_fields=True),
            options=gfem.SolverOptions(order=2, linear_solver="direct"),
        )
        got = gfem.optical_force(result.e_field(pc.F0, "pw"), "scat", model=model)
        return jnp.concatenate([got.force, got.torque])

    # A direct solve is exact to round-off, so the five-point step is set by the
    # truncation alone: h = 0.05 as above (the probe: 3.9e-8 at h = 1e-4).
    # The force along the wave is the component that does not vanish by symmetry.
    _check_design_gradient(solve, 4.0, 0.05, 1e-6, components=(2,))


def _field_function(cavity, direction):
    """``f(t)``: the force at the coefficients ``e + t direction``."""

    def f(t):
        field = dataclasses.replace(
            cavity.field, coefficients=cavity.field.coefficients + t * direction
        )
        return _wrench(field, cavity.model)

    return f


def test_the_cotangent_of_the_coefficients_is_twice_the_conjugate_of_the_form(cavity):
    e = np.asarray(cavity.field.coefficients)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(e.shape) + 1j * rng.standard_normal(e.shape)
    direction *= np.linalg.norm(e) / np.linalg.norm(direction)
    f = _field_function(cavity, jnp.asarray(direction))
    _, pull = jax.vjp(
        lambda c: _wrench(
            dataclasses.replace(cavity.field, coefficients=c), cavity.model
        ),
        jnp.asarray(e),
    )
    # The wrench is a quadratic form of the coefficients, so the central
    # difference over t = +-1 is its exact directional derivative; what remains
    # is round-off of the form, a few eps times the cancellation between its curl
    # and mass parts (the probe measured 3e-14, 1e-12 here).
    exact = (np.asarray(f(1.0)) - np.asarray(f(-1.0))) / 2
    for k in range(6):
        (g,) = pull(jnp.zeros(6).at[k].set(1.0))
        assert float(np.real(np.sum(np.asarray(g) * direction))) == pytest.approx(
            exact[k], abs=1e-12 * np.abs(exact).max()
        )


def test_the_cotangent_of_the_frequency_is_minus_twice_the_curl_force_over_omega(
    cavity,
):
    def f(s):
        field = dataclasses.replace(cavity.field, omega=s * cavity.field.omega)
        return _wrench(field, cavity.model)

    # F(omega) = A / omega^2 + B at fixed coefficients: the five-point
    # truncation is 12 (h / omega)^4 relative, 1.2e-7 at h = 0.01 omega, and
    # the round-off over h is 1e-12.
    _, pull = jax.vjp(f, 1.0)
    fd = _fd4(lambda s: np.asarray(f(s)), 1.0, 1e-2)
    for k in range(6):
        (g,) = pull(jnp.zeros(6).at[k].set(1.0))
        assert float(g) == pytest.approx(fd[k], abs=1e-6 * np.abs(fd).max())


def test_the_wrench_maps_over_fields_and_compiles(cavity):
    scales = jnp.asarray([1.0, 2.0, 0.5j])

    def scaled(a):
        field = dataclasses.replace(
            cavity.field, coefficients=a * cavity.field.coefficients
        )
        return _wrench(field, cavity.model)

    base = np.asarray(_wrench(cavity.field, cavity.model))
    # a quadratic form: |a|^2 times the force, to the round-off of the form
    np.testing.assert_allclose(
        jax.vmap(scaled)(scales),
        base * np.abs(np.asarray(scales))[:, None] ** 2,
        rtol=1e-10,
        atol=1e-10 * np.abs(base).max(),
    )
    np.testing.assert_allclose(
        jax.jit(scaled)(2.0), 4 * base, rtol=1e-10, atol=1e-10 * np.abs(base).max()
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda a: scaled(a)[0]))(1.0),
        2 * base[0],
        rtol=1e-10,
    )


def test_the_node_coordinates_the_shell_material_and_a_reduced_field_raise_by_name(
    cavity,
):
    x = jnp.asarray(cavity.mesh.nodes)

    def moved(s):
        model = cavity.model.replace(mesh=cavity.mesh.with_nodes(s * x))
        return gfem.optical_force(cavity.field, "block", model=model).force[0]

    with pytest.raises(gfem.NonDifferentiableError, match=r"model\.mesh\.nodes\)"):
        jax.grad(moved)(1.0)

    n_gas = len(cavity.mesh.elements("gas"))

    def shell_eps(s):
        model = cavity.model.replace(
            materials={
                **cavity.model.materials,
                "gas": gfem.MaterialField(eps_r=s * jnp.ones(n_gas)),
            }
        )
        return gfem.optical_force(cavity.field, "block", model=model).force[0]

    with pytest.raises(
        gfem.NonDifferentiableError, match=r"materials\['gas'\]\.eps_r.*stress shell"
    ):
        jax.grad(shell_eps)(1.0)

    reduced = dataclasses.replace(cavity.field, _reduced=True)

    def through(c):
        field = dataclasses.replace(reduced, coefficients=c)
        return gfem.optical_force(field, "block", model=cavity.model).force[0]

    with pytest.raises(gfem.NonDifferentiableError, match="adaptive sweep"):
        jax.grad(through)(reduced.coefficients)


def test_an_input_the_force_does_not_read_has_an_exact_zero_gradient(cavity):
    n_block = len(cavity.mesh.elements("block"))

    def thermal(s):
        model = cavity.model.replace(
            materials={
                **cavity.model.materials,
                "block": gfem.MaterialField(
                    eps_r=4.0 * jnp.ones(n_block),
                    thermal_conductivity=s * jnp.ones(n_block),
                ),
            }
        )
        return gfem.optical_force(cavity.field, "block", model=model).force[0]

    assert float(jax.grad(thermal)(1.0)) == 0.0
