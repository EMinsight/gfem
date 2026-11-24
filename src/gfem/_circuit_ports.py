"""Circuit port names, the circuit kind and the RF reference, for SAX circuits.

Shared by `gfem.NetworkParameters.to_sdict` and `gfem.SaxComponent`, which
check them the same way.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import jax.numpy as jnp

from gfem._checks import is_traced

Circuit = Literal["optical", "rf"]

#: The reference impedance of an RF circuit when none is given, ohm: SAX's
#: RF default.
DEFAULT_Z0 = 50.0


def check_circuit(circuit: Any, sets: str) -> None:
    """Raise unless ``circuit`` is a circuit kind; ``sets`` says what it sets.

    Raises:
        ValueError: ``circuit`` is neither ``"optical"`` nor ``"rf"``.
    """
    if circuit not in ("optical", "rf"):
        raise ValueError(
            f"circuit must be 'optical' or 'rf', got {circuit!r}: it sets {sets}, "
            "so it is never inferred."
        )


def circuit_ports(
    ports: Mapping[str, str] | None, names: Sequence[str]
) -> dict[str, str]:
    """``ports`` checked: gfem port name -> circuit port name, in the given order.

    ``None`` keeps every port under its own name. A circuit port name is an
    identifier, or ``identifier@mode`` (SAX multimode) for every port or none.

    Raises:
        TypeError: ``ports`` is not a mapping.
        ValueError: ``ports`` is empty, names a port that does not exist, or
            gives an invalid or repeated circuit port name.
    """
    listed = ", ".join(map(repr, names))
    if ports is None:
        bad = [p for p in names if not _is_circuit_name(p)]
        if bad:
            raise ValueError(
                f"Port {bad[0]!r} cannot be a circuit port name (an identifier, "
                "or name@mode); rename it in the circuit with ports=, e.g. "
                f"ports={{{bad[0]!r}: 'o1'}}."
            )
        exposed = {p: p for p in names}
    else:
        if not isinstance(ports, Mapping):
            raise TypeError(
                "ports= maps gfem port names to circuit port names, e.g. "
                f"{{{names[0]!r}: 'o1'}}; got {type(ports).__name__}."
            )
        exposed = dict(ports)
        if not exposed:
            raise ValueError(f"ports= exposes no port; the ports are {listed}.")
        unknown = [p for p in exposed if p not in names]
        if unknown:
            raise ValueError(
                f"No port {', '.join(map(repr, unknown))}; the ports are {listed}."
            )
        bad = [c for c in exposed.values() if not _is_circuit_name(c)]
        if bad:
            raise ValueError(
                f"Circuit port name {bad[0]!r} is not an identifier or name@mode "
                "(SAX needs one, e.g. 'o1' or 'o1@TE0')."
            )
        names_given = list(exposed.values())
        repeated = sorted({c for c in names_given if names_given.count(c) > 1})
        if repeated:
            raise ValueError(
                f"Circuit port name {', '.join(map(repr, repeated))} is given to "
                "more than one port; each exposed port needs its own."
            )
    moded = ["@" in c for c in exposed.values()]
    if any(moded) and not all(moded):
        raise ValueError(
            "Either every circuit port name is name@mode or none is (SAX cannot "
            f"join single-mode and multimode ports); got {list(exposed.values())}."
        )
    return exposed


def _is_circuit_name(name: object) -> bool:
    """``name`` is an identifier, or ``identifier@mode``."""
    if not isinstance(name, str):
        return False
    base, at, mode = name.partition("@")
    return base.isidentifier() and (not at or re.fullmatch(r"\w+", mode) is not None)


def circuit_z0(z0: Any) -> Any:
    r"""An RF circuit's ``z0`` checked: one real value :math:`> 0`.

    `DEFAULT_Z0` if ``None``. A traced value is checked for its dtype and
    shape only.

    Raises:
        ValueError: ``z0`` is complex, not a number, not a scalar, or a
            concrete value that is not finite and :math:`> 0`.
    """
    if z0 is None:
        return DEFAULT_Z0
    try:
        real = jnp.result_type(z0).kind in "iuf" and jnp.ndim(z0) == 0
    except TypeError:
        real = False
    if real and not is_traced(z0):
        value = float(z0)
        real = math.isfinite(value) and value > 0
    if not real or isinstance(z0, bool):
        raise ValueError(
            "z0 must be one real reference impedance > 0 ohm, the one every "
            f"port of the RF circuit shares, got {z0!r}."
        )
    return z0
