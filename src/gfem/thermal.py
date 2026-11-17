"""Heat sources of thermal problems: `HeatSource` and `joule_heating`.

A `HeatSource` is the heat a thermal problem is driven by, given per
volume region: a uniform amount (`HeatSource.uniform`) or the Joule heat
of a DC conduction solve (`joule_heating`). It is a frozen pytree: its
amount (or its power density per element) is a leaf, so a temperature is
differentiable with respect to it. A thermal solve resolves every source
to one power density per tetrahedron (W/m^3, `HeatSource._density`).

On a cross-section mesh (`gfem.Mesh.from_gmsh_cross_section`) extensive
amounts are given per unit length (``power_per_length``, W/m): ``power``
and ``voltage`` raise there, since the slab's thickness is a numerical
choice the result does not depend on (ADR 0170).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced
from gfem._pytree import register_fields
from gfem.errors import RegionNotFoundError

if TYPE_CHECKING:
    import jax

    from gfem.mesh import Mesh

__all__ = ["HeatSource", "joule_heating"]

_UNITS: dict[str | None, str] = {
    "power": "W",
    "power_per_length": "W/m",
    "power_density": "W/m^3",
}


def _regions(region: Any) -> tuple[str, ...]:
    if isinstance(region, str):
        return (region,)
    if (
        isinstance(region, Sequence)
        and region
        and all(isinstance(r, str) for r in region)
    ):
        return tuple(dict.fromkeys(region))
    raise TypeError(
        "HeatSource.uniform: region must be a volume region name or a non-empty "
        f"sequence of names, got {region!r}."
    )


def _check_finite(value: Any, what: str) -> Any:
    """``value`` as a float64 JAX scalar; finite when concrete."""
    if not is_traced(value):
        try:
            number = float(np.asarray(value))
        except (TypeError, ValueError):
            raise TypeError(f"{what} must be a real number, got {value!r}.") from None
        if not math.isfinite(number):
            raise ValueError(f"{what} must be finite, got {number}.")
    out = jnp.asarray(value, dtype=jnp.float64)
    if out.shape != ():
        raise TypeError(f"{what} must be a scalar, got shape {out.shape}.")
    return out


def _per_length_remedy(name: str, owner: str) -> ValueError:
    """The error of an extensive amount given on a cross-section (ADR 0170)."""
    if name == "voltage":
        remedy = (
            "drive the heater by current instead: current=V / "
            "(resistance_per_length * L) for a heater of length L and voltage V, "
            "or by power_per_length"
        )
    else:
        remedy = "give power_per_length=P / L for a heater of length L instead"
    return ValueError(
        f"{owner}: {name}= is not accepted on a cross-section mesh, where the "
        "solve stands for a slab whose thickness is a numerical choice (a slab "
        f"total is not the device's): {remedy}."
    )


def _tet_volumes(mesh: Mesh) -> np.ndarray:
    """Volume of each tetrahedron in m^3, from host node positions."""
    nodes = mesh.imported_nodes if is_traced(mesh.nodes) else np.asarray(mesh.nodes)
    p = nodes[mesh.tetrahedra]
    a, b, c = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
    return np.abs(np.einsum("ij,ij->i", np.cross(a, b), c)) / 6.0


@dataclasses.dataclass(frozen=True, eq=False)
class HeatSource:
    """Heat delivered to volume regions of a thermal problem.

    Build one with `HeatSource.uniform` (an amount spread evenly over
    regions) or `gfem.joule_heating` (the Joule heat of a DC conduction
    solve under a drive). Sources superpose: a thermal problem takes one or
    a sequence. The amount is differentiable.

    Attributes:
        regions: The volume regions that receive the heat.
    """

    regions: tuple[str, ...]
    #: ``"uniform"`` or ``"joule"``.
    _kind: str = "uniform"
    #: For a uniform source: which amount `_amount` is.
    _quantity: str | None = None
    #: For a uniform source: the amount, a scalar (W, W/m or W/m^3).
    _amount: Any = None
    #: For a joule source: the power density per tetrahedron, W/m^3.
    _density: Any = None
    #: For a joule source: its total power in W (slab total on a
    #: cross-section), when concrete; for the repr.
    _total: float | None = None
    #: For a joule source: the cross-section thickness in m, else ``None``.
    _thickness: float | None = None

    @classmethod
    def uniform(
        cls,
        region: str | Sequence[str],
        *,
        power: float | jax.Array | None = None,
        power_per_length: float | jax.Array | None = None,
        power_density: float | jax.Array | None = None,
    ) -> HeatSource:
        """Heat spread evenly over the volume of one or more regions.

        Exactly one amount is given. It needs no mesh: the regions and the
        amount are checked against the model's mesh when a thermal problem
        is solved.

        Example:
            A heater dissipating 10 mW, and a cross-section heater at 5 W/m::

                gfem.HeatSource.uniform("heater", power=10e-3)
                gfem.HeatSource.uniform("heater", power_per_length=5.0)

        Args:
            region: A volume region name, or a sequence of names (their
                union, spread by volume).
            power: The total power in W over the regions (3D meshes).
            power_per_length: The power per unit length in W/m, on a
                cross-section mesh only.
            power_density: The power density in W/m^3, on any mesh.

        Raises:
            TypeError: ``region`` is not a name or names, or an amount is
                not a real scalar.
            ValueError: Not exactly one amount, or it is not finite.
        """
        regions = _regions(region)
        given = {
            name: value
            for name, value in (
                ("power", power),
                ("power_per_length", power_per_length),
                ("power_density", power_density),
            )
            if value is not None
        }
        if len(given) != 1:
            raise ValueError(
                "HeatSource.uniform takes exactly one amount: power= (W), "
                "power_per_length= (W/m, on a cross-section) or power_density= "
                f"(W/m^3); got {', '.join(given) or 'none'}."
            )
        ((quantity, value),) = given.items()
        amount = _check_finite(value, f"HeatSource.uniform {quantity}")
        return cls(regions=regions, _kind="uniform", _quantity=quantity, _amount=amount)

    def _density_on(self, mesh: Mesh) -> jax.Array:
        """The power density per tetrahedron of ``mesh`` (W/m^3), in JAX.

        Region volumes are host constants, so the density is linear in the
        amount (or is the joule density itself).

        Raises:
            RegionNotFoundError: A region is not in the mesh.
            ValueError: A region is not a volume; ``power`` on a
                cross-section or ``power_per_length`` on a 3D mesh; a joule
                source from another mesh.
        """
        n = len(mesh.tetrahedra)
        if self._kind == "joule":
            if self._density.shape != (n,):
                raise ValueError(
                    f"This Joule heat source holds {self._density.shape[0]} "
                    f"elements, but the mesh has {n}: a DC conduction result heats "
                    "the mesh it was solved on. Solve DCConduction on the thermal "
                    "model's mesh."
                )
            return self._density
        for name in self.regions:
            if name not in mesh.volumes:
                if name in mesh.surfaces:
                    raise ValueError(
                        f"HeatSource.uniform: {name!r} is a surface, not a volume "
                        "region. Heat through a surface is a gfem.HeatFlux."
                    )
                raise RegionNotFoundError(name, mesh.volumes, kind="volume")
        section = mesh.section_thickness is not None
        if self._quantity == "power" and section:
            raise _per_length_remedy("power", "HeatSource.uniform")
        if self._quantity == "power_per_length" and not section:
            raise ValueError(
                "HeatSource.uniform: power_per_length= is for a cross-section mesh "
                "(gfem.Mesh.from_gmsh_cross_section); this mesh is 3D: give the "
                "total power= in W instead."
            )
        inside = np.zeros(n, bool)
        for name in self.regions:
            inside[mesh.elements(name)] = True
        if self._quantity == "power_density":
            return jnp.where(jnp.asarray(inside), self._amount, 0.0)
        volume = float(_tet_volumes(mesh)[inside].sum())
        if section:
            # the slab total is the per-length power times the thickness
            volume /= mesh.section_thickness
        return jnp.asarray(inside, jnp.float64) * (self._amount / volume)

    def __repr__(self) -> str:
        regions = list(self.regions)
        if self._kind == "joule":
            if self._total is None:
                return f"HeatSource(joule, regions={regions})"
            if self._thickness is None:
                return (
                    f"HeatSource(joule, regions={regions}, total {self._total:.6g} W)"
                )
            per_length = self._total / self._thickness
            return f"HeatSource(joule, regions={regions}, total {per_length:.6g} W/m)"
        try:
            amount = f"{float(np.asarray(self._amount)):.6g} {_UNITS[self._quantity]}"
        except (TypeError, ValueError, Exception):  # noqa: BLE001 - a tracer
            amount = self._quantity
        return f"HeatSource(uniform, regions={regions}, {self._quantity}={amount})"


register_fields(HeatSource, ("_amount", "_density"))


def _drive(value: Any, terminals: tuple[str, ...], what: str) -> jax.Array:
    """A per-terminal drive vector from a number or a mapping (missing: 0)."""
    if isinstance(value, Mapping):
        unknown = [k for k in value if k not in terminals]
        if unknown:
            raise KeyError(
                f"joule_heating: {what} names {unknown[0]!r}, which is not a "
                f"terminal; the terminals are {', '.join(map(repr, terminals))}."
            )
        return jnp.stack(
            [
                _check_finite(value[t], f"joule_heating {what}[{t!r}]")
                if t in value
                else jnp.asarray(0.0)
                for t in terminals
            ]
        )
    if len(terminals) != 1:
        raise ValueError(
            f"joule_heating: the result has {len(terminals)} terminals "
            f"({', '.join(map(repr, terminals))}), so give {what} as a mapping "
            f"from terminal name to value, e.g. {what}={{{terminals[0]!r}: ...}}."
        )
    return _check_finite(value, f"joule_heating {what}")[None]


def joule_heating(
    result: Any,
    *,
    voltage: float | Mapping[str, float] | jax.Array | None = None,
    current: float | Mapping[str, float] | jax.Array | None = None,
    power: float | jax.Array | None = None,
    power_per_length: float | jax.Array | None = None,
) -> HeatSource:
    """The Joule heat of a DC conduction result under a drive.

    The drive is applied to the unit-voltage solve with no new solve, so
    one `gfem.DCConduction` result serves every drive, and the heat is
    differentiable with respect to the drive. Each tetrahedron of a
    conductor gets its exact mean power density ``V^T D_e V``
    (`gfem.DCConductionResult.dissipation`); elsewhere it is zero.

    Exactly one drive is given:

    - ``voltage`` in V, or ``current`` in A into the terminal: a number for
      a result with one terminal, or a mapping from terminal name to value
      (missing terminals at 0 V or 0 A).
    - ``power`` in W, or ``power_per_length`` in W/m: for one terminal and
      a ground only.

    On a cross-section mesh, drive by ``current`` (the same number as in
    the device) or ``power_per_length``; ``voltage`` and ``power`` raise.
    ``power_per_length`` raises on a 3D mesh.

    Example:
        A 2D heater at 25 mA, then a steady thermal solve::

            dc = gfem.solve(model, gfem.DCConduction({"heater": "front"}, "back"))
            heat = gfem.joule_heating(dc, current=25e-3)

    Args:
        result: A `gfem.DCConductionResult`.
        voltage: Terminal voltage(s) in V.
        current: Terminal current(s) in A.
        power: The heater's total power in W.
        power_per_length: The heater's power per unit length in W/m.

    Returns:
        A `gfem.HeatSource` on the conductors.

    Raises:
        NotImplementedError: ``result`` is a `gfem.DrivenResult`.
        TypeError: ``result`` is not a DC conduction result.
        ValueError: Not exactly one drive; a drive that does not fit the
            result or the mesh (the message gives the remedy).
        KeyError: A mapping names an unknown terminal.
    """
    from gfem.results import DCConductionResult, DrivenResult

    if isinstance(result, DrivenResult):
        raise NotImplementedError(
            "heating from electromagnetic loss is not available in this version; "
            "give the dissipated power per region with gfem.HeatSource.uniform"
        )
    if not isinstance(result, DCConductionResult):
        raise TypeError(
            "joule_heating takes the result of gfem.solve(model, "
            f"gfem.DCConduction(...)), got {type(result).__name__}."
        )
    given = {
        name: value
        for name, value in (
            ("voltage", voltage),
            ("current", current),
            ("power", power),
            ("power_per_length", power_per_length),
        )
        if value is not None
    }
    if len(given) != 1:
        raise ValueError(
            "joule_heating takes exactly one drive: voltage=, current=, power= or "
            f"power_per_length=; got {', '.join(given) or 'none'}."
        )
    ((name, value),) = given.items()
    thickness = result._thickness  # noqa: SLF001
    if thickness is not None and name in ("voltage", "power"):
        raise _per_length_remedy(name, "joule_heating")
    if thickness is None and name == "power_per_length":
        raise ValueError(
            "joule_heating: power_per_length= is for a cross-section mesh "
            "(gfem.Mesh.from_gmsh_cross_section); this result was solved on a 3D "
            "mesh: give the total power= in W instead."
        )
    terminals = result.terminals
    g = jnp.asarray(result.conductance)
    if name == "voltage":
        v = _drive(value, terminals, "voltage")
    elif name == "current":
        v = jnp.linalg.solve(g, _drive(value, terminals, "current"))
    else:
        if len(terminals) != 1 or not result._grounded:  # noqa: SLF001
            raise ValueError(
                f"joule_heating: {name}= fixes the drive for one terminal and a "
                f"ground only, but the result has terminals "
                f"{', '.join(map(repr, terminals))}"
                f"{'' if result._grounded else ' and no ground'}. Give the "  # noqa: SLF001
                "terminal voltages or currents instead."
            )
        total = _check_finite(value, f"joule_heating {name}")
        if not is_traced(total) and float(total) < 0:
            raise ValueError(f"joule_heating: {name} must be >= 0, got {float(total)}.")
        if name == "power_per_length":
            total = total * thickness
        v = jnp.sqrt(total / g[0, 0])[None]
    density = jnp.einsum("i,eij,j->e", v, jnp.asarray(result.dissipation), v)
    total_power = None
    if not is_traced((v, g)):
        total_power = float(v @ g @ v)
    return HeatSource(
        regions=result._regions,  # noqa: SLF001
        _kind="joule",
        _density=density,
        _total=total_power,
        _thickness=thickness,
    )
