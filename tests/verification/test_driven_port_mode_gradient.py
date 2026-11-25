"""A driven solve's port modes follow a design at the port face as `PortMode` does.

The modes of a driven solve are the port mode problem's, swept once over the
grid: a design at the face of a wave port moves ``gamma`` and ``z_ref`` of the
driven result, and the derivatives are `gfem.PortMode`'s, whichever route
reaches them (the driven port pass pulls its output cotangents back to the
same 2D adjoint). This catches a port pass that routes a cotangent on
``gamma`` or ``Z_PV`` differently from the port mode rule's.
"""

import jax
import jax.numpy as jnp
import numpy as np
from gradient_cases import face_line, ramp, tight, wave_mesh

import gfem

FREQUENCIES = (12e9, 24e9)


def test_driven_gamma_and_z_ref_differentiate_like_the_port_mode(tmp_path):
    mesh = wave_mesh(tmp_path)
    model = face_line(mesh, "face_path")  # p1 has a voltage path: Z_PV is defined
    n = len(mesh.elements("lo"))
    values = {"eps_r": ramp(mesh, "lo", 3.5, 4.5), "tan_delta": jnp.full(n, 0.05)}
    options = tight("direct", port_mode_tolerance=1e-12)
    rng = np.random.default_rng(2)
    c_gamma = jnp.asarray(rng.standard_normal(2) + 1j * rng.standard_normal(2))
    c_z = jnp.asarray(rng.standard_normal(2) + 1j * rng.standard_normal(2))

    def functional(gamma, z):
        return jnp.sum(jnp.real(c_gamma * gamma)) + jnp.sum(jnp.real(c_z * z)) / 50.0

    def designed(p):
        field = gfem.MaterialField(**{**values, "eps_r": p})
        moved = model.with_material("lo", field)
        return moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))

    def driven(p):
        result = gfem.solve(designed(p), gfem.Driven(FREQUENCIES), options=options)
        return functional(result.port_modes["p1"].gamma, result.s.z_ref[:, 0])

    def port_mode(p):
        result = gfem.solve(
            designed(p), gfem.PortMode("p1", FREQUENCIES), options=options
        )
        return functional(result.gamma, result.Z_PV)

    p = values["eps_r"]
    by_driven, by_mode = jax.grad(driven)(p), jax.grad(port_mode)(p)
    assert np.abs(np.asarray(by_mode)).max() > 0
    np.testing.assert_allclose(
        by_driven, by_mode, rtol=0, atol=1e-10 * np.abs(np.asarray(by_mode)).max()
    )


def test_a_constant_anisotropic_owner_at_an_absorbing_edge_spares_other_regions(
    tmp_path,
):
    # the absorbing edge's admittance beside a constant anisotropic material has
    # no leaf: it must not refuse the design of another region, in the port mode
    # as in a driven solve
    from finite_differences import assert_gradient_matches, unit_directions

    mesh = wave_mesh(tmp_path)
    model = face_line(mesh, "face_absorbing").with_material(
        "hi", gfem.Material(eps_r=(2.0, 2.3, 2.6))
    )
    options = tight("direct", port_mode_tolerance=1e-12)
    p = ramp(mesh, "lo", 3.5, 4.5)

    def objective(q):
        moved = model.with_material("lo", gfem.MaterialField(eps_r=q))
        moved = moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))
        result = gfem.solve(moved, gfem.PortMode("p1", (24e9, 30e9)), options=options)
        return jnp.sum(jnp.imag(result.gamma) + 0.3 * jnp.real(result.gamma)) / 100.0

    assert np.abs(np.asarray(jax.grad(objective)(p))).max() > 0
    assert_gradient_matches(
        objective, p, unit_directions(np.random.default_rng(3), p.shape, 2), step=1e-3
    )


def test_a_gamma_cotangent_alone_polishes_a_reduced_row_as_the_driven_route_does(
    tmp_path,
):
    # a row the sweep did not solve exactly takes the Newton polish before the
    # Hellmann-Feynman term, whatever the cotangent: at the default tolerance
    # the unpolished Ritz pair would differ from the driven pass's
    mesh = wave_mesh(tmp_path)
    model = face_line(mesh, "face_lossy")
    band = tuple(np.linspace(21e9, 30e9, 7))
    options = tight("direct")
    p = ramp(mesh, "lo", 3.5, 4.5)

    def designed(q):
        moved = model.with_material("lo", gfem.MaterialField(eps_r=q))
        return moved.replace(mesh=jax.lax.stop_gradient(moved.mesh))

    def driven(q):
        result = gfem.solve(designed(q), gfem.Driven(band), options=options)
        return jnp.sum(jnp.imag(result.port_modes["p1"].gamma))

    def port_mode(q):
        result = gfem.solve(designed(q), gfem.PortMode("p1", band), options=options)
        return jnp.sum(jnp.imag(result.gamma))

    solved = gfem.solve(designed(p), gfem.PortMode("p1", band), options=options).solved
    assert not np.asarray(solved).all()
    by_driven, by_mode = jax.grad(driven)(p), jax.grad(port_mode)(p)
    np.testing.assert_allclose(
        by_mode, by_driven, rtol=0, atol=1e-9 * np.abs(np.asarray(by_driven)).max()
    )
