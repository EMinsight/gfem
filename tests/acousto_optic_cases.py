"""Lithium niobate's photoelastic tensor, shared by the acousto-optic tests."""

import numpy as np

#: Photoelastic constants p^E of lithium niobate (dimensionless), the set the
#: push-pull modulator reference uses; p14 is negative with a positive e22 and r22.
P11, P12, P13, P14 = -0.026, 0.090, 0.133, -0.075
P31, P33, P41, P44 = 0.179, 0.071, -0.151, 0.146


def tensor_3m_p(
    *, p11=P11, p12=P12, p13=P13, p14=P14, p31=P31, p33=P33, p41=P41, p44=P44
) -> np.ndarray:
    """The contracted 6x6 tensor of class 3m, crystal frame, Voigt xx yy zz yz xz xy.

    Not symmetric: p14 and p41 differ.
    """
    return np.array(
        [
            [p11, p12, p13, p14, 0, 0],
            [p12, p11, p13, -p14, 0, 0],
            [p31, p31, p33, 0, 0, 0],
            [p41, -p41, 0, p44, 0, 0],
            [0, 0, 0, 0, p44, p41],
            [0, 0, 0, 0, p14, (p11 - p12) / 2],
        ]
    )


P_LN = tensor_3m_p()


# --- a manufactured acoustic result on a 2D section ----------------------------------

#: The acoustic section's regions, the optical slab's names (its x, y are the same).
ACOUSTIC_REGIONS = ("below", "film", "above")


def acoustic_mesh(path, maxh: float = 1.0, margin: float = 2.0):
    """A 2D section around the layered slab's x, y extent, ``margin`` um larger."""
    import electro_optic_cases as eo
    import thermal_cases as tc

    import gfem

    h = eo.SLAB_HALF + margin
    rects = [
        (-h, -2.0 - margin, h, 0.0, "below", None),
        (-h, 0.0, h, eo.SLAB_FILM, "film", None),
        (-h, eo.SLAB_FILM, h, 2.6 + margin, "above", None),
    ]
    nodes, tris, lines = tc._mesh2d(rects, maxh, [])
    tc._write_section(path, nodes, tris, lines)
    return gfem.Mesh.from_gmsh(path, length_unit="um")


def affine_field(
    mesh, support, quantity, unit, gradient, *, offset=0.0, order=1, curvature=0.0
):
    """A real `gfem.Field` ``offset + gradient . (x, y) + curvature x^2`` (SI).

    Exact at order 1 for the affine part and at order 2 with a ``curvature``.
    """
    import jax.numpy as jnp
    import ngsolve as ngs

    import gfem
    from gfem._engine.fields import fe_space

    built, fes, _ = fe_space(mesh, tuple(support), (), "h1", order)
    gf = ngs.GridFunction(fes)
    # engine coordinates are in mesh length units, `scale` metres each
    s = built.scale
    gf.Set(
        offset
        + s * (gradient[0] * ngs.x + gradient[1] * ngs.y)
        + curvature * (s * ngs.x) ** 2
    )
    return gfem.Field(
        coefficients=jnp.asarray(gf.vec.FV().NumPy().copy()),
        mesh=mesh,
        quantity=quantity,
        unit=unit,
        space="h1",
        order=order,
        support=tuple(support),
    )


def complex_field(
    mesh, support, quantity, unit, gradient, *, offset=0.0, order=1, curvature=0.0
):
    """A complex field ``offset + gradient . (x, y) + curvature x^2`` (complex data)."""
    import dataclasses

    gradient = np.asarray(gradient, complex)
    parts = [
        affine_field(
            mesh,
            support,
            quantity,
            unit,
            part(gradient),
            offset=part(offset),
            order=order,
            curvature=part(curvature),
        )
        for part in (np.real, np.imag)
    ]
    return dataclasses.replace(
        parts[0], coefficients=parts[0].coefficients + 1j * parts[1].coefficients
    )


def manufactured_result(
    mesh,
    frequencies,
    displacement_gradients,
    potential_gradients,
    *,
    mechanical=ACOUSTIC_REGIONS,
    electric=ACOUSTIC_REGIONS,
    terminals=("hot",),
    order=1,
    curvature=None,
):
    """A `gfem.PiezoelectricResult` holding exact fields, one per frequency.

    ``displacement_gradients[k]`` is the complex ``(3, 2)`` array of the
    gradients (per metre) of :math:`u_x, u_y, u_z` and ``potential_gradients[k]``
    the complex ``(2,)`` gradient of the potential, at ``frequencies[k]``, for
    each terminal. ``curvature[k]`` optionally adds :math:`c_i x^2` to the
    components of the displacement (order 2).
    """
    import jax.numpy as jnp

    import gfem
    from gfem._field import FieldMap

    displacements, potentials, kept = [], [], []
    for k, f in enumerate(frequencies):
        for terminal in terminals:
            parts = {
                c: complex_field(
                    mesh,
                    mechanical,
                    f"u{c}",
                    "m",
                    displacement_gradients[k][i],
                    order=order,
                    curvature=0.0 if curvature is None else curvature[k][i],
                )
                for i, c in enumerate("xyz")
            }
            displacements.append(FieldMap(parts, label="component"))
            potentials.append(
                complex_field(
                    mesh, electric, "V", "V", potential_gradients[k], order=order
                )
            )
            kept.append((f, terminal))
    n = len(terminals)
    return gfem.PiezoelectricResult(
        admittance_per_length=jnp.zeros((len(frequencies), n, n), complex),
        clamped_capacitance_per_length=jnp.zeros((n, n)),
        terminals=tuple(terminals),
        frequencies=tuple(frequencies),
        displacement_fields=displacements,
        potential_fields=potentials,
        kept=tuple(kept),
    )


def te_slab(eps_film):
    """The 1D TE slab's effective index and the film's share of the field energy.

    The layered slab's film between oxide, TE mode (E along x), exact.
    """
    import electro_optic_cases as eo
    from scipy.integrate import quad
    from scipy.optimize import brentq

    k0 = 2 * np.pi / eo.WAVELENGTH
    d, n_ox = eo.SLAB_FILM * 1e-6, eo.N_OX

    def kappa(n):
        return k0 * np.sqrt(eps_film - n * n)

    def gamma(n):
        return k0 * np.sqrt(n * n - n_ox * n_ox)

    def even(n):
        return np.tan(kappa(n) * d / 2) - gamma(n) / kappa(n)

    lowest = np.sqrt(max(eps_film - (np.pi / (k0 * d)) ** 2, n_ox**2)) + 1e-9
    n = brentq(even, lowest, np.sqrt(eps_film) - 1e-12, xtol=1e-15)
    film = quad(lambda x: np.cos(kappa(n) * x) ** 2, -d / 2, d / 2)[0]
    cladding = np.cos(kappa(n) * d / 2) ** 2 / gamma(n)
    return n, film / (film + cladding)
