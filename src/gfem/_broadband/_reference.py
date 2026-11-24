r"""The one real reference impedance per port that the fit and time views need.

The rules are those of `NetworkParameters.to_touchstone`. ``z0`` given: S is
renormalized to it. Otherwise each port's ``z_ref`` must be real and constant
over frequency. A network in which no port has a reference impedance (wave
ports without ``voltage_path``, S on unit modal power) is taken as it is:
the time views give the response on unit modal power and refuse ``z0``, the
fit needs ``z0`` as a nominal label. Some ports without one: refused.

A reference impedance counts as real and constant over frequency when its
imaginary part and its spread across frequency are both within a relative
:math:`10^{-9}` of its magnitude: a TEM wave port's :math:`Z_{PV}`, real and
frequency independent in theory, differs between frequencies by about
:math:`10^{-13}` relative from the mode solve's round-off, and exact equality
would refuse it. The port's reference is then its mean real value.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._text import hertz

if TYPE_CHECKING:
    from gfem.results import NetworkParameters

#: Relative imaginary part and spread over frequency still read as constant.
RTOL = 1e-9


def constant_real(z_ref: np.ndarray) -> np.ndarray | None:
    r"""Each port's mean real reference ``(n_ports,)``, or None when not constant.

    Args:
        z_ref: Reference impedances ``(n_freq, n_ports)``, no NaN.

    Returns:
        :math:`\overline{\operatorname{Re} Z_p}` over frequency for each port
        :math:`p`, if :math:`\max_f |\operatorname{Im} Z_p|` and
        :math:`\max_f \operatorname{Re} Z_p - \min_f \operatorname{Re} Z_p`
        are both at most :math:`10^{-9} \max_f |Z_p|`; None otherwise.
    """
    z = np.asarray(z_ref)
    real = np.real(z)
    scale = RTOL * np.abs(z).max(axis=0)
    spread = real.max(axis=0) - real.min(axis=0)
    if (np.abs(np.imag(z)).max(axis=0) > scale).any() or (spread > scale).any():
        return None
    return real.mean(axis=0).astype(float)


@dataclasses.dataclass(frozen=True)
class Reference:
    """S on one real reference per port, and that reference.

    Attributes:
        network: The network on the reference: renormalized when ``z0`` was
            given, the original one otherwise.
        z_ref: The reference ``(n_ports,)``, ohm; NaN for every port of a
            network without reference impedances taken on unit modal power.
        nominal: Whether ``z_ref`` is a label: S was left on unit modal power.
    """

    network: NetworkParameters
    z_ref: Any
    nominal: bool


def resolve(
    net: NetworkParameters, z0: Any, what: str, *, nominal_z0: bool
) -> Reference:
    """S on one real reference per port, by the rules of the module docstring.

    Args:
        net: The network.
        z0: The reference to renormalize to, or None.
        what: The caller's name, for messages (e.g. ``"step_response"``).
        nominal_z0: How a network without reference impedances is taken:
            True, it needs ``z0`` as a nominal label (the fit); False, it is
            on unit modal power and refuses ``z0`` (the time views).

    Raises:
        ValueError: The reference cannot be resolved, or ``z0`` is not one
            real value > 0 or one per port; the message gives the remedy.
    """
    n_ports = len(net.ports)
    z_ref = net.z_ref
    if not is_traced(z_ref):
        arr = np.asarray(z_ref)
        if _modal(net, arr, z0, what, nominal_z0=nominal_z0):
            if nominal_z0:
                return Reference(net, _z0(z0, net.ports, what), nominal=True)
            return Reference(net, jnp.full(n_ports, jnp.nan), nominal=False)
    if z0 is not None:
        if not is_traced(z0):
            _z0(z0, net.ports, what)
        z = jnp.broadcast_to(jnp.asarray(z0, dtype=float), (n_ports,))
        return Reference(net.renormalize(z), z, nominal=False)
    if is_traced(z_ref):
        # a traced reference cannot be checked to be one real value per port,
        # and reading it as one would be wrong physics for a complex Z_PV
        raise ValueError(
            f"{what} needs one real reference impedance per port, constant over "
            "frequency, but z_ref is traced by JAX (jax.jit, jax.grad or "
            "jax.vmap) and cannot be checked. Pass z0= (e.g. z0=50.0) to "
            "renormalize S to one fixed real reference first."
        )
    constant = constant_real(arr)
    if constant is None:
        found = np.unique(np.round(arr, 12))
        raise ValueError(
            f"{what} needs one real reference impedance per port, constant over "
            f"frequency, but z_ref takes the values {found.tolist()} ohm (a "
            "WavePort's Z_PV is complex and frequency dependent). Pass z0= "
            "(e.g. z0=50.0) to renormalize S to one reference first."
        )
    return Reference(net, jnp.asarray(constant), nominal=False)


def _modal(
    net: NetworkParameters,
    z_ref: np.ndarray,
    z0: Any,
    what: str,
    *,
    nominal_z0: bool,
) -> bool:
    """Whether no port has a reference impedance (and S is finite).

    Raises:
        ValueError: Some ports have none, or none has one and ``z0`` is
            missing (``nominal_z0``) or given (otherwise).
    """
    nan = np.isnan(z_ref).any(axis=0)
    if not nan.any():
        return False
    finite = is_traced(net.values) or np.isfinite(np.asarray(net.values)).all()
    if np.isnan(z_ref).all() and finite:
        if nominal_z0 and z0 is None:
            raise ValueError(
                "No port has a reference impedance (z_ref is NaN: wave ports "
                "without voltage_path), so S is normalized to unit modal power "
                f"and has no impedance for {what}. Pass z0= (e.g. z0=50.0) to "
                "use S unchanged under that nominal label, or give each "
                "WavePort a voltage_path."
            )
        if not nominal_z0 and z0 is not None:
            raise ValueError(
                "No port has a reference impedance (z_ref is NaN: wave ports "
                "without voltage_path), so S cannot be renormalized to z0. Call "
                f"{what} without z0 for the response of S on unit modal power, "
                "or give each WavePort a voltage_path."
            )
        return True
    missing = [p for p, bad in zip(net.ports, nan, strict=True) if bad]
    where = np.isnan(z_ref[:, nan]).any(axis=1)
    freqs = ", ".join(
        hertz(float(f))
        for f, bad in zip(np.asarray(net.frequency), where, strict=True)
        if bad
    )
    raise ValueError(
        f"Port {', '.join(map(repr, missing))} has no reference impedance "
        "(z_ref is NaN): a LumpedPort without a resistance, a WavePort "
        "without a voltage_path beside ports with one, or a FloquetPort whose "
        f"order is evanescent at {freqs}. Its waves are undefined, so {what} "
        "cannot use it; give the port a resistance or a voltage_path, or for "
        "a FloquetPort drop the port or narrow the grid."
    )


def _z0(z0: Any, ports: tuple[str, ...], what: str) -> np.ndarray:
    """``z0`` checked: one real value > 0, or one per port; ``(n_ports,)``."""
    try:
        z = np.asarray(z0)
        ok = (
            not np.iscomplexobj(z)
            and z.dtype != bool
            and z.ndim <= 1
            and (z.ndim == 0 or z.shape == (len(ports),))
        )
        z = z.astype(float) if ok else z
    except (TypeError, ValueError):
        ok = False
    if not ok or not (np.isfinite(z).all() and (z > 0).all()):
        raise ValueError(
            f"{what} z0 must be one real reference impedance > 0 ohm, or one "
            f"per port ({len(ports)}: {', '.join(map(repr, ports))}); got {z0!r}."
        )
    return np.broadcast_to(z, (len(ports),)).copy()
