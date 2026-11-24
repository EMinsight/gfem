r"""The reverse mode of one port mode over its frequency grid.

Given the cotangents of a mode's outputs (the propagation constant, the
power-voltage impedance and the field's DOF vector at each frequency), `mode_pass`
returns the cotangents of the 2D section's coefficients
(`gfem._engine.port_mode_sensitivity.SectionCotangents`), summed over the
frequencies, plus the law parameters' gradient (which needs each row's
frequency).

A cotangent on :math:`\gamma` alone is the Hellmann-Feynman term and needs no
solve. A cotangent on the field or on :math:`Z_{PV}` runs the normalization's
pullback (`gfem._engine.port_mode_normalization`) over the rows in reverse
order, then one deflated adjoint solve per row
(`gfem._engine.port_mode_adjoint`). A reduced row is first polished by one
Newton step, so that its derivative is the exact discrete mode's; the
normalization then runs on the whole free-DOF vector.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from gfem._engine.port_mode import FullReduction, PortModeSpace
from gfem._engine.port_mode_adjoint import (
    POLISH_RTOL,
    pencil_cotangent,
    polish,
    shift_invert_factor,
)
from gfem._engine.port_mode_normalization import normalize_rows, normalize_vjp
from gfem._engine.port_mode_sensitivity import (
    SectionCotangents,
    SectionKernel,
    law_cotangents,
)

__all__ = ["ModePass", "mode_pass"]

# A new direction joins the extended basis unless it is below this fraction of
# the vector it came from (rounding noise).
_NEW_DIRECTION = 1e-12


@dataclasses.dataclass(eq=False)
class ModePass:
    """What a mode's backward gives the section.

    Attributes:
        section: The coefficients' cotangents, summed over the rows (the
            normalization's explicit form terms included).
        laws: The gradient by each permittivity-law parameter, by input kind.
    """

    section: SectionCotangents
    laws: dict[tuple, float]


def mode_pass(  # noqa: PLR0915
    space: PortModeSpace,
    *,
    frequency: np.ndarray,
    lam: np.ndarray,
    y: np.ndarray,
    v: np.ndarray,
    solved: np.ndarray,
    wavenumbers: Callable[[float], tuple[float, float, float]],
    volts: np.ndarray | None,
    ct_gamma: np.ndarray | None,
    ct_z_pv: np.ndarray | None,
    ct_field: np.ndarray | None,
    who: str,
) -> ModePass:
    r"""The section cotangents of a mode's outputs.

    Args:
        space: The port mode problem.
        frequency: ``(n,)`` Hz, in the order of the other arrays.
        lam: ``(n,)`` the pencil eigenvalue of each row.
        y: ``(n, m)`` the pencil vector of each row in the basis ``v``.
        v: ``(n_free, m)`` the real reduced basis.
        solved: ``(n,)`` whether a row is an exact solve.
        wavenumbers: ``frequency -> (k0, k_max, shift)`` per local unit.
        volts: The voltage functional on the free DOFs, ``None`` without a
            voltage path.
        ct_gamma: ``(n,)`` the cotangent of :math:`\gamma` in 1/m, or ``None``.
        ct_z_pv: ``(n,)`` the cotangent of :math:`Z_{PV}`, or ``None``.
        ct_field: ``(n, n_dof)`` the cotangent of the field vector, or ``None``.
        who: The port, for messages.

    Raises:
        PortModeError: An adjoint solve missed (`gfem._engine.port_mode_adjoint`).
    """
    n = len(frequency)
    scale = space.section.scale
    order = np.argsort(frequency, kind="stable")
    f = np.asarray(frequency, float)[order]
    lam_s = np.asarray(lam)[order]
    y_s = np.asarray(y)[order]
    solved_s = np.asarray(solved)[order]
    omega = 2 * math.pi * f
    cg = (
        np.zeros(n, complex)
        if ct_gamma is None
        else np.asarray(ct_gamma, complex)[order] / scale
    )
    waves = [wavenumbers(float(fi)) for fi in f]
    k0 = np.array([w[0] for w in waves])
    shift = np.array([w[2] for w in waves])
    exact = np.flatnonzero(solved_s)
    factors: dict[int, Any] = {}

    def factor_of(i: int) -> Any:
        j = int(exact[np.argmin(np.abs(f[exact] - f[i]))])
        if j not in factors:
            a, b = space.pencil.at(k0[j])
            factors[j] = shift_invert_factor(a, b, shift[j])
        return factors[j]

    x = np.array([v @ yi for yi in y_s], complex)
    lam_r = lam_s.astype(complex).copy()
    has_path = volts is not None
    needs = (ct_field is not None and np.any(ct_field)) or (
        has_path and ct_z_pv is not None and np.any(ct_z_pv)
    )
    x_bar = np.zeros_like(x)
    pairs_forms: list = []
    if needs:
        # polish the reduced rows: the derivative is the exact mode's
        for i in range(n):
            if solved_s[i]:
                continue
            a, b = space.pencil.at(k0[i])
            lam_r[i], x[i] = polish(
                a, b, lam_r[i], x[i], factor=factor_of(i), rtol=POLISH_RTOL, who=who
            )
        # the normalization on the whole free-DOF vector: a cotangent on the
        # field points anywhere in the space, not only along the reduced basis
        fields = FullReduction(space)
        mass = space.mass
        gamma = 1j * np.sqrt(-lam_r + 0j)
        norm_rows = normalize_rows(fields, mass, x, gamma, omega, volts=volts)
        z_bar = np.zeros(norm_rows.z.shape, complex)
        if ct_field is not None:
            z_bar = np.asarray(ct_field, complex)[order][:, space.free]
        v_bar = np.zeros(n, complex)
        p_bar = np.zeros(n, complex)
        if has_path and ct_z_pv is not None:
            ct = np.asarray(ct_z_pv, complex)[order]
            volt = norm_rows.voltage
            phase = norm_rows.phase
            z_bar_v = ct * np.conj(volt) / (2 * np.conj(phase))
            v_bar = z_bar_v + np.conj(ct * volt / (2 * np.conj(phase)))
            p_bar = -np.conj(ct * np.abs(volt) ** 2 / (2 * np.conj(phase) ** 2))
        pulled = normalize_vjp(
            fields,
            mass,
            norm_rows,
            x,
            gamma,
            omega,
            z_bar=z_bar,
            voltage_bar=v_bar,
            phase_bar=p_bar,
            volts=volts,
        )
        pairs_forms = pulled.pairs
        x_bar = pulled.y
        cg = cg + pulled.gamma
    gamma = 1j * np.sqrt(-lam_r + 0j)
    lam_bar = cg / (2 * gamma)

    kernel = SectionKernel(space, float(k0[0]))
    total = SectionCotangents.zeros(space.section)
    laws: dict[tuple, float] = {}
    for i in range(n):
        if lam_bar[i] == 0 and not np.any(x_bar[i]):
            continue
        a, b = space.pencil.at(k0[i])
        q = pencil_cotangent(
            a,
            b,
            lam_r[i],
            x[i],
            lam_bar=lam_bar[i],
            x_bar=x_bar[i],
            factor=factor_of(i),
            who=who,
        )
        row = kernel.at(k0[i]).cotangents([(q, x[i], lam_r[i])])
        for key, value in law_cotangents(space.section, row, omega[i]).items():
            laws[key] = laws.get(key, 0.0) + value
        total.add(row)
    if pairs_forms:
        total.add(kernel.explicit(pairs_forms))
    return ModePass(total, laws)
