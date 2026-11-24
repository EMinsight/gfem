"""Exact gradients of a piezoelectric solve with respect to its material leaves.

Each leaf (the stiffness and piezoelectric tensor in the crystal frame, the
clamped permittivity's principal values, the density, the quality factor, and
an electrode's Young's modulus and Poisson's ratio through
`gfem.isotropic_stiffness`) is checked against fourth-order central differences
at the gate of `finite_differences`, for the admittance, the clamped
capacitance and a reading of the kept fields, on both electrode variants.
"""

import dataclasses

import finite_differences as fd
import jax
import jax.numpy as jnp
import numpy as np
import piezo_cases as pc
import pytest

import gfem

VARIANTS = ("lines", "meshed")
LN_REGIONS = ("ln", "ln_frame")
NAMES = ("Re Y", "Im Y", "C0", "field")


@pytest.fixture(scope="module", params=VARIANTS)
def device(request, tmp_path_factory):
    variant = request.param
    path = tmp_path_factory.mktemp(variant) / "idt.msh"
    return variant, pc.mesh(path, variant)


def _model(mesh, variant, edit):
    """The device with every solid a `gfem.MaterialField`, ``edit`` applied."""
    base = pc.model(mesh, variant)
    fields = {
        region: gfem.MaterialField.uniform(region, material)
        for region, material in base.materials.items()
        if region != "air"
    }
    edit(fields)
    return gfem.Model(
        mesh, materials={**base.materials, **fields}, excluded={"mechanical": ["air"]}
    )


def _replace(fields, regions, **values):
    for region in regions:
        fields[region] = dataclasses.replace(fields[region], **values)


# name -> (edit(fields, p), p shape, step per unit of p, directions, depends on C0)
def _stiffness(fields, p):
    _replace(fields, LN_REGIONS, stiffness=pc.C_LN + 1e10 * p)


def _piezoelectric(fields, p):
    _replace(fields, LN_REGIONS, piezoelectric_tensor=pc.E_LN + 0.1 * p)


def _permittivity(fields, p):
    _replace(fields, LN_REGIONS, eps_r=jnp.asarray(pc.EPS_LN) + p)


def _density(fields, p):
    _replace(fields, LN_REGIONS, mass_density=pc.RHO_LN + 100.0 * p)


def _quality(fields, p):
    _replace(fields, LN_REGIONS, mechanical_quality_factor=pc.QM + 10.0 * p)


def _metal(fields, p):
    stiffness = gfem.isotropic_stiffness(pc.E_AL + 1e9 * p[0], pc.NU_AL + 0.01 * p[1])
    _replace(fields, ("hot", "gnd"), stiffness=stiffness)


LEAVES = {
    # edit, shape of p, directions, whether the clamped capacitance depends on it
    "stiffness": (_stiffness, (6, 6), 2, False),
    "piezoelectric_tensor": (_piezoelectric, (3, 6), 2, False),
    "eps_r": (_permittivity, (3,), 1, True),
    "mass_density": (_density, (), 1, False),
    "mechanical_quality_factor": (_quality, (), 1, False),
    "isotropic_stiffness": (_metal, (2,), 1, False),
}


def _direction(rng, name, shape):
    d = rng.standard_normal(shape)
    if name == "stiffness":  # a stiffness is symmetric
        d = d + d.T
    return d / np.linalg.norm(d)


def _objectives(model, weights):
    """``[Re Y', Im Y', C0, a reading of the kept fields]`` of the model."""
    result = gfem.solve(model, pc.problem(keep_fields=True))
    y = result.admittance_per_length[0, 0, 0]
    uz = result.displacement_fields[0]["z"].coefficients
    phi = result.potential_fields[0].coefficients
    reading = jnp.real(uz @ weights[0]) + jnp.imag(phi @ weights[1])
    return jnp.stack(
        [y.real, y.imag, result.clamped_capacitance_per_length[0, 0], reading]
    )


@pytest.fixture(scope="module")
def weights(device):
    """Fixed random weights on the kept coefficients."""
    variant, mesh = device
    model = _model(mesh, variant, lambda _: None)
    result = gfem.solve(model, pc.problem(keep_fields=True))
    rng = np.random.default_rng(3)
    sizes = (
        result.displacement_fields[0]["z"].coefficients.shape,
        result.potential_fields[0].coefficients.shape,
    )
    return tuple(rng.standard_normal(s) for s in sizes)


@pytest.mark.parametrize("name", LEAVES)
def test_a_material_gradient_matches_finite_differences(device, weights, name):
    variant, mesh = device
    if name == "isotropic_stiffness" and variant == "lines":
        pytest.skip("the lines variant has no metal")
    edit, shape, count, depends_on_c0 = LEAVES[name]

    def objectives(p):
        return _objectives(
            _model(mesh, variant, lambda fields: edit(fields, p)), weights
        )

    p0 = jnp.zeros(shape)
    jacobian = np.asarray(fd._alone(jax.jacrev(objectives), p0))
    rng = np.random.default_rng(11)
    step = 1e-2
    for d in (_direction(rng, name, shape) for _ in range(count)):
        samples = np.array(
            [
                np.asarray(fd._alone(objectives, p0 + k * step * d))
                for k in (-2, -1, 1, 2)
            ]
        )
        difference = (8 * (samples[2] - samples[1]) - (samples[3] - samples[0])) / (
            12 * step
        )
        derivative = jacobian.reshape(len(NAMES), -1) @ d.reshape(-1)
        for i, label in enumerate(NAMES):
            if label == "C0" and not depends_on_c0:
                # the capacitance does not read it: exact zeros on both sides
                assert derivative[i] == 0.0
                assert abs(difference[i]) < 1e-12 * abs(samples[0, i]) / step
                continue
            error = abs(derivative[i] - difference[i]) / abs(difference[i])
            assert error < fd.RTOL, (
                f"{name} {label} direction: gradient {derivative[i]:.12e} vs "
                f"difference {difference[i]:.12e} (relative error {error:.2e})"
            )


# --- what raises ----------------------------------------------------------------------


def test_a_traced_permittivity_law_parameter_raises(device):
    variant, mesh = device
    base = pc.model(mesh, variant)

    def objective(eps_inf):
        law = gfem.PermittivityLaw(
            eps_inf=eps_inf,
            terms=[gfem.Debye(delta_eps=1.0, relaxation_frequency=1e9)],
        )
        materials = {**base.materials, "air": gfem.Material(eps_r=law)}
        model = gfem.Model(mesh, materials=materials, excluded={"mechanical": ["air"]})
        result = gfem.solve(model, pc.problem())
        return result.clamped_capacitance_per_length[0, 0]

    with pytest.raises(gfem.NonDifferentiableError, match="permittivity law"):
        jax.grad(objective)(1.0)


def test_an_unread_property_has_an_exact_zero_gradient(device):
    variant, mesh = device

    def objective(sigma):
        model = _model(
            mesh,
            variant,
            lambda fields: _replace(fields, ("ln",), thermal_conductivity=sigma),
        )
        result = gfem.solve(model, pc.problem())
        return result.clamped_capacitance_per_length[0, 0]

    assert float(jax.grad(objective)(1.0)) == 0.0


def test_traced_node_coordinates_raise_until_node_gradients_exist(device):
    variant, mesh = device
    model = pc.model(mesh, variant)

    def objective(scale):
        moved = mesh.with_nodes(mesh.nodes * scale)
        result = gfem.solve(model.replace(mesh=moved), pc.problem())
        return result.clamped_capacitance_per_length[0, 0]

    with pytest.raises(gfem.NonDifferentiableError, match="not available yet"):
        jax.grad(objective)(1.0)
