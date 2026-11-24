r"""The Purcell factor of a point dipole: its emitted power over its bulk power.

`purcell_factor` divides ``DrivenResult.emitted_power`` by the closed-form
power of the same dipole in the unbounded, lossless material at its
position (`gfem.conventions`, Point dipoles), read from the model at each
frequency. The solve refuses a position where that material is lossy,
anisotropic or a `gfem.MaterialField`, so it is always defined.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const

from gfem._materials import PermittivityLaw
from gfem._sources import DIPOLES, MagneticDipole

if TYPE_CHECKING:
    from gfem.model import Model
    from gfem.results import DrivenResult

__all__ = ["bulk_power", "purcell_factor"]


def bulk_power(
    dipole: Any, omega: Any, *, eps_r: Any = 1.0, mu_r: float = 1.0
) -> jax.Array:
    r"""The power of ``dipole`` in an unbounded lossless medium, W.

    :math:`\mu_0\mu_r n\,\omega^4|p|^2/(12\pi c)` for an electric dipole,
    :math:`\mu_0\varepsilon_r n\,\omega^4|m|^2/(12\pi c^3)` for a magnetic
    one, :math:`n = \sqrt{\varepsilon_r\mu_r}`.
    """
    moment = np.asarray(dipole.moment, complex)
    size = float(np.vdot(moment, moment).real)
    n = jnp.sqrt(eps_r * mu_r)
    c = const.c
    if isinstance(dipole, MagneticDipole):
        return const.mu_0 * eps_r * n * omega**4 * size / (12 * math.pi * c**3)
    return const.mu_0 * mu_r * n * omega**4 * size / (12 * math.pi * c)


def purcell_factor(result: DrivenResult, name: str, *, model: Model) -> jax.Array:
    r"""The Purcell factor of a point dipole: :math:`F = P / P_0`, ``(n_freq,)``.

    :math:`P` is the dipole's emitted power (``result.emitted_power``) and
    :math:`P_0` the power of the same dipole in the unbounded material at its
    position, lossless and isotropic there (the solve refuses any other):

    .. math::

        P_0^{(p)} = \frac{\mu_0 \mu_r n\, \omega^4 |p|^2}{12 \pi c}, \qquad
        P_0^{(m)} = \frac{\mu_0 \varepsilon_r n\, \omega^4 |m|^2}{12 \pi c^3},

    with :math:`n = \sqrt{\varepsilon_r\mu_r}` read at each frequency (a
    permittivity law's value there). In a non-magnetic medium the partial
    local density of states along the moment is :math:`\rho_0 F`, with
    :math:`\rho_0 = \omega^2 n / (\pi^2 c^3)` (`gfem.conventions`, Point
    dipoles).

    Differentiable through the emitted power: :math:`P_0` is a constant of
    the design (the material at the dipole is a fixed `gfem.Material`).

    Example:
        A dipole 50 nm above a substrate::

            dipole = gfem.ElectricDipole("d", position=r0, moment=(0, 0, 1e-30))
            result = gfem.solve(model, gfem.Driven(f, sources=dipole))
            F = gfem.purcell_factor(result, "d", model=model)

    Args:
        result: The `gfem.DrivenResult` of a solve with the dipole among its
            sources.
        name: The dipole's name.
        model: The model it was solved on: the material at the dipole.

    Returns:
        ``(n_freq,)``, dimensionless.

    Raises:
        KeyError: ``name`` is not a point dipole of the result (a port, or
            not a source).
        ValueError: ``name`` is an incident-field source, which has no
            emitted power, or a dipole of a ``Driven(source_fields=False)``
            solve, whose own field was not solved.
    """
    sources = result.sources
    source = sources.get(name)
    if source is None:
        dipoles = [n for n, s in sources.items() if isinstance(s, DIPOLES)]
        listed = ", ".join(map(repr, dipoles)) or "none"
        raise KeyError(
            f"{name!r} is not a point dipole of the result; its dipoles are {listed}."
        )
    if not isinstance(source, DIPOLES):
        # a value error: the source exists, its kind has no emitted power
        raise ValueError(  # noqa: TRY004
            f"{name!r} is a {type(source).__name__} column, which emits no power "
            "of its own: the Purcell factor is a point dipole's (gfem."
            "ElectricDipole or gfem.MagneticDipole)."
        )
    result._refuse_unsolved_source(  # noqa: SLF001
        name, "so it has no emitted power and no Purcell factor"
    )
    material = model.materials[_region(model, source)]
    omega = 2 * math.pi * np.asarray(result.frequency)
    eps = material.eps_r
    if isinstance(eps, PermittivityLaw):  # real at the solve's frequencies
        eps_r = jnp.real(jnp.stack([jnp.asarray(eps(f)) for f in result.frequency]))
    else:
        eps_r = jnp.asarray(eps)
    # the solve refused an anisotropic material at the dipole
    mu_r = float(np.asarray(material.mu_r))
    bulk = bulk_power(source, omega, eps_r=eps_r, mu_r=mu_r)
    return result.emitted_power[name] / bulk


def _region(model: Model, source: Any) -> str:
    """The volume region of the element the solve placed ``source`` in."""
    from gfem._engine.probe import Layout, locate

    mesh = model.mesh
    layout = Layout(
        support=tuple(mesh.volumes),
        excluded=tuple(model._domain.excluded),  # noqa: SLF001
        space="hcurl",
        order=1,
    )
    located = locate(mesh, layout, np.array([source.position], float))
    tag = int(mesh._topology.volume_tags[located.tetrahedra[0]])  # noqa: SLF001
    return mesh._region_names()[tag]  # noqa: SLF001
