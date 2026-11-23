"""Heat sources, traces and switching times of thermal problems.

`HeatSource`, `joule_heating` and `electromagnetic_heating` give the heat;
`Trace` names what a `gfem.TransientThermal` solve records at every time
step; `rise_time` and `time_constant` measure a recorded trace.

A `HeatSource` is the heat a thermal problem is driven by, given per
volume region: a uniform amount (`HeatSource.uniform`), the Joule heat
of a DC conduction solve (`joule_heating`) or the electromagnetic loss of a
driven solve (`electromagnetic_heating`). It is a frozen pytree: its
amount (or its power density per element) is a leaf, so a temperature is
differentiable with respect to it. A thermal solve resolves every source
to one power density per tetrahedron, in W/m^3, and one per sheet face, in
W/m^2.

On a cross-section mesh (`gfem.Mesh.from_gmsh_cross_section`) extensive
amounts are given per unit length (``power_per_length``, W/m): ``power``
and ``voltage`` raise there, since the slab's thickness is a numerical
choice the result does not depend on.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import check_fixed, is_traced, static_float
from gfem._pytree import register_fields
from gfem.errors import (
    NonDifferentiableError,
    RegionNotFoundError,
    UnsettledTraceWarning,
)

if TYPE_CHECKING:
    from gfem._field import Field
    from gfem.mesh import Mesh

__all__ = [
    "HeatSource",
    "Trace",
    "electromagnetic_heating",
    "joule_heating",
    "rise_time",
    "time_constant",
]

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


def _per_length_remedy(
    name: str, owner: str, per_length: str = "power_per_length"
) -> ValueError:
    """The error of an extensive amount given on a cross-section.

    ``per_length`` is the keyword the remedy names.
    """
    if name == "voltage":
        remedy = (
            "drive the heater by current instead: current=V / "
            "(resistance_per_length * L) for a heater of length L and voltage V, "
            "or by power_per_length"
        )
    else:
        device = "heater" if per_length == "power_per_length" else "device"
        remedy = f"give {per_length}=P / L for a {device} of length L instead"
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


def _triangle_areas(mesh: Mesh) -> np.ndarray:
    """Area of each gfem triangle in m^2, from host node positions."""
    nodes = mesh.imported_nodes if is_traced(mesh.nodes) else np.asarray(mesh.nodes)
    p = nodes[mesh.triangles]
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) / 2


@dataclasses.dataclass(frozen=True, eq=False)
class HeatSource:
    """Heat delivered to volume regions of a thermal problem.

    Build one with `HeatSource.uniform` (an amount spread evenly over
    regions), `gfem.joule_heating` (the Joule heat of a DC conduction
    solve under a drive) or `gfem.electromagnetic_heating` (the loss of a
    driven solve under a drive, with sheet heat on its faces). Sources
    superpose: a thermal problem takes one or a sequence. The amount is
    differentiable.

    Attributes:
        regions: The volume regions that receive the heat.
    """

    regions: tuple[str, ...]
    #: ``"uniform"``, ``"joule"`` or ``"electromagnetic"``.
    _kind: str = "uniform"
    #: For a uniform source: which amount `_amount` is.
    _quantity: str | None = None
    #: For a uniform source: the amount, a scalar (W, W/m or W/m^3).
    _amount: Any = None
    #: For a source built from a solve: the power density per tetrahedron,
    #: W/m^3.
    _density: Any = None
    #: For a source built from a solve: the power density per gfem triangle
    #: (`gfem.Mesh.triangles` order), W/m^2, or ``None`` (no face heat).
    _face_density: Any = None
    #: For a source built from a solve: its total power in W (slab total on a
    #: cross-section), when concrete; for the repr.
    _total: float | None = None
    #: The part of `_total` carried by faces (sheet heat), when concrete.
    _sheet_total: float | None = None
    #: For a source built from a solve: the cross-section thickness in m,
    #: else ``None``.
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
        if self._density is not None:
            if self._density.shape != (n,):
                raise self._other_mesh(self._density.shape[0], n)
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

    def _face_density_on(self, mesh: Mesh) -> jax.Array:
        """The power density per gfem triangle of ``mesh`` (W/m^2), in JAX.

        All zero for a source without face heat.

        Raises:
            ValueError: The face density is from another mesh.
        """
        n = len(mesh.triangles)
        if self._face_density is None:
            return jnp.zeros(n, jnp.float64)
        if self._face_density.shape != (n,):
            raise self._other_mesh(self._face_density.shape[0], n, "faces")
        return self._face_density

    def _other_mesh(self, held: int, n: int, what: str = "elements") -> ValueError:
        """The error of a source built on another mesh than the thermal one."""
        if self._kind == "joule":
            return ValueError(
                f"This Joule heat source holds {held} {what}, but the mesh has "
                f"{n}: a DC conduction result heats the mesh it was solved on. "
                "Solve DCConduction on the thermal model's mesh."
            )
        return ValueError(
            f"This {self._kind} heat source holds {held} {what}, but the mesh has "
            f"{n}: a heat source built from a solve heats the mesh it was solved "
            "on. Solve on the thermal model's mesh."
        )

    def __repr__(self) -> str:
        regions = list(self.regions)
        if self._kind != "uniform":
            head = f"HeatSource({self._kind}, regions={regions}"
            if self._total is None:
                return head + ")"
            per = 1.0 if self._thickness is None else self._thickness
            unit = "W" if self._thickness is None else "W/m"
            total = f"{head}, total {self._total / per:.6g} {unit}"
            sheet = self._sheet_total
            if not sheet:
                return total + ")"
            volume = (self._total - sheet) / per
            return (
                f"{total}: volume {volume:.6g} {unit}, sheet {sheet / per:.6g} {unit})"
            )
        try:
            amount = f"{float(np.asarray(self._amount)):.6g} {_UNITS[self._quantity]}"
        except (TypeError, ValueError, Exception):  # noqa: BLE001 - a tracer
            amount = self._quantity
        return f"HeatSource(uniform, regions={regions}, {self._quantity}={amount})"


register_fields(HeatSource, ("_amount", "_density", "_face_density"))


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
    conductor gets its exact mean power density :math:`V^T D_e V`
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
        TypeError: ``result`` is not a DC conduction result (a
            `gfem.DrivenResult` heats through `electromagnetic_heating`).
        ValueError: Not exactly one drive; a drive that does not fit the
            result or the mesh (the message gives the remedy).
        KeyError: A mapping names an unknown terminal.
    """
    from gfem.results import DCConductionResult, DrivenResult

    if isinstance(result, DrivenResult):
        raise TypeError(
            "joule_heating takes a DC conduction result; a driven result's "
            "electromagnetic loss becomes heat through "
            "gfem.electromagnetic_heating(result, model=, frequency=, "
            "incident_power=)."
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


_EM = "electromagnetic_heating"


@jax.custom_vjp
def _amplitude(power: jax.Array) -> jax.Array:
    """``sqrt(power)``, whose cotangent is 0 where none arrives.

    A cross term goes as ``sqrt(P_p) sqrt(P_q)``: at ``P_p = 0`` its
    derivative is unbounded next to a driven partner (``inf``, as the heat
    grows like ``sqrt(P_p)``), and exactly 0 when every partner is at 0 W,
    where plain ``jnp.sqrt`` would give ``0 * inf = nan``.
    """
    return jnp.sqrt(power)


def _amplitude_fwd(power: jax.Array) -> tuple[jax.Array, jax.Array]:
    root = jnp.sqrt(power)
    return root, root


def _amplitude_bwd(root: jax.Array, g: jax.Array) -> tuple[jax.Array]:
    safe = jnp.where(g == 0, 1.0, 2 * root)
    return (jnp.where(g == 0, 0.0, g / safe),)


_amplitude.defvjp(_amplitude_fwd, _amplitude_bwd)


def _incident(value: Any, excited: tuple[str, ...], name: str) -> dict[str, Any]:
    """The incident power of each driven port, in port order (a JAX scalar each).

    Raises:
        KeyError: A port that is not excited.
        ValueError: A number with several excited ports, or a negative power.
        TypeError: A value that is not a real scalar.
    """
    if isinstance(value, Mapping):
        unknown = [k for k in value if k not in excited]
        if unknown:
            raise KeyError(
                f"{_EM}: {name} names port {unknown[0]!r}, which was not excited; "
                f"the excited ports are {', '.join(map(repr, excited))}. Solve "
                "with gfem.Driven(excite=...) naming every port the drive feeds."
            )
        if not value:
            raise ValueError(f"{_EM}: {name} is empty: give each driven port's power.")
        ports = [p for p in excited if p in value]
        powers = {p: _check_finite(value[p], f"{_EM} {name}[{p!r}]") for p in ports}
    elif len(excited) != 1:
        raise ValueError(
            f"{_EM}: the result has {len(excited)} excited ports "
            f"({', '.join(map(repr, excited))}), so give {name} as a mapping from "
            f"port name to value, e.g. {name}={{{excited[0]!r}: ...}}; ports left "
            "out are at 0."
        )
    else:
        powers = {excited[0]: _check_finite(value, f"{_EM} {name}")}
    for port, power in powers.items():
        if not is_traced(power) and float(power) < 0:
            raise ValueError(
                f"{_EM}: the {name} of port {port!r} must be >= 0, got {float(power)}."
            )
    return powers


def _phases(value: Any, ports: tuple[str, ...]) -> dict[str, Any]:
    """The incident phase of each driven port, rad (0 where not given).

    Raises:
        TypeError: Not a mapping.
        KeyError: A port that is not driven.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{_EM}: incident_phase is relative, so one number cannot carry it: give "
            f"a mapping from port name to phase in rad, e.g. incident_phase="
            f"{{{ports[-1]!r}: math.pi}}, got {value!r}."
        )
    unknown = [k for k in value if k not in ports]
    if unknown:
        raise KeyError(
            f"{_EM}: incident_phase names port {unknown[0]!r}, which the drive does "
            f"not feed; the driven ports are {', '.join(map(repr, ports))}. Give "
            "its incident power too."
        )
    return {
        p: _check_finite(v, f"{_EM} incident_phase[{p!r}]") for p, v in value.items()
    }


def _check_heated_result(result: Any, model: Any) -> None:
    """Raise unless ``result`` is a 3D driven result and ``model`` a model.

    Raises:
        TypeError: Another result type, or ``model`` not a `gfem.Model`.
        NotImplementedError: A port mode result, or a 2D model or result.
    """
    from gfem._dim2 import entry_refusal, refuse_entry
    from gfem.model import Model
    from gfem.results import (
        DCConductionResult,
        DrivenResult,
        EigenmodeResult,
        PortModeResult,
    )

    if isinstance(result, DCConductionResult):
        raise TypeError(
            f"{_EM} takes a driven result; a DC conduction result's Joule heat is "
            "gfem.joule_heating(result, voltage=, current=, power=)."
        )
    if isinstance(result, PortModeResult):
        raise NotImplementedError(
            f"{_EM} does not heat from a port mode (heat per length along a "
            "line): give the line's loss per length with "
            "gfem.HeatSource.uniform(..., power_per_length=)."
        )
    if isinstance(result, EigenmodeResult):
        raise TypeError(
            f"{_EM}: an eigenmode has no drive, so its amplitude and its heat are "
            "undefined; solve a driven problem at the mode's frequency."
        )
    if not isinstance(result, DrivenResult):
        raise TypeError(
            f"{_EM} takes the result of gfem.solve(model, gfem.Driven(..., "
            f"keep_fields=...)), got {type(result).__name__}."
        )
    refuse_entry(getattr(model, "mesh", None), _EM)
    if result.polarization is not None:
        raise entry_refusal(_EM)
    if not isinstance(model, Model):
        raise TypeError(f"{_EM} needs model= a gfem.Model, got {type(model).__name__}.")


def _waves(
    result: Any, frequency: float, powers: dict[str, Any], phases: dict[str, Any]
) -> tuple[jax.Array, jax.Array]:
    """The driven ports' incident waves at the port surfaces, and ``|a_p|^2``.

    The fields are fed at the port surfaces, the drive is given at the
    reference planes: a plane at ``d`` inside the domain sees the incident
    wave of the surface times ``exp(-gamma d)``, as S does
    (`gfem.DrivenResult.with_offsets`). ``|a_p|^2`` is formed from the power,
    linear in it, so a 0 W port takes no square root.
    """
    index = next(
        k
        for k, g in enumerate(result.frequency)
        if math.isclose(float(g), frequency, rel_tol=1e-9)
    )
    offsets = dict(result.offsets)
    shift = jnp.stack(
        [
            result._plane_shift(p, offsets[p])[index]  # noqa: SLF001
            if offsets.get(p, 0.0)
            else jnp.asarray(1.0 + 0j)
            for p in powers
        ]
    )
    power = jnp.stack(list(powers.values()))
    phase = jnp.stack([jnp.asarray(phases.get(p, 0.0), jnp.float64) for p in powers])
    a = _amplitude(power) * jnp.exp(1j * phase) * shift
    return a, power * jnp.abs(shift) ** 2


def _coherent(
    re: jax.Array, im: jax.Array, a: jax.Array, squared: jax.Array
) -> jax.Array:
    """``sum_pq conj(a_p) a_q D[p, q]`` per element, the diagonal as ``|a_p|^2``."""
    off = 1.0 - jnp.eye(len(a))
    cross = jnp.einsum("p,epq,q->e", jnp.conj(a), (re + 1j * im) * off, a)
    own = jnp.einsum("p,ep->e", squared, jnp.diagonal(re, axis1=1, axis2=2))
    return own + jnp.real(cross)


def electromagnetic_heating(
    result: Any,
    *,
    model: Any,
    frequency: float,
    incident_power: float | Mapping[str, float] | jax.Array | None = None,
    incident_power_per_length: float | Mapping[str, float] | jax.Array | None = None,
    incident_phase: Mapping[str, float] | None = None,
) -> HeatSource:
    r"""The heat a driven result's electromagnetic loss gives under a drive.

    The kept fields of the driven ports at ``frequency`` are combined with
    no new solve, so one `gfem.Driven` result serves every drive. The ports
    of one call are fed coherently: with incident waves
    :math:`a_p = \sqrt{P_p}\, e^{i \phi_p}`, each tetrahedron and sheet face
    receives

    .. math::

        q = \sum_{pq} \overline{a_p} a_q D[p, q],

    :math:`D[p, q]` the power the fields of ports :math:`p` and :math:`q`
    dissipate together there (`gfem.dissipated_power` gives its diagonal, per
    region and sheet). A tetrahedron gets its exact mean power density, a
    `gfem.Conductor` or `gfem.Impedance` sheet face its exact mean power per
    area, which the thermal problem loads at the face. What is not heat gets
    none: a perfectly matched layer, an `gfem.Absorbing` face, a port and a
    lumped port's resistance. Incoherent drives (other tones, unrelated
    sources) are separate heat sources, which a thermal problem adds.

    Exactly one of ``incident_power`` (W) and ``incident_power_per_length``
    (W/m, on a cross-section slab from `gfem.Mesh.from_gmsh_cross_section`)
    is given: a number when one port was excited, else a mapping from port
    name to power, ports left out at 0. ``incident_phase`` gives each driven
    port's relative phase, in rad, at its reference plane: a plane moved by
    ``WavePort(offset=)`` or `gfem.DrivenResult.with_offsets` moves the
    phase (and, on a lossy port mode, the power) with it, as for S. With a
    complex port reference impedance a field is per unit :math:`|a|^2`,
    which is not exactly 1 W incident.

    Differentiable with respect to the drive (powers and phases) and through
    the driven solve, as `gfem.dissipated_power` is. A diagonal term is
    linear in its port's power; a cross term grows as
    :math:`\sqrt{P_p P_q}`, so its derivative with respect to a power of
    exactly 0 W next to a driven partner is unbounded (``inf``).

    Example:
        A differential feed of 1 mW per port, then a steady thermal solve::

            driven = gfem.solve(model, gfem.Driven(f, keep_fields=True))
            heat = gfem.electromagnetic_heating(
                driven,
                model=model,
                frequency=f,
                incident_power={"p1": 1e-3, "p2": 1e-3},
                incident_phase={"p2": math.pi},
            )
            steady = gfem.solve(model, gfem.SteadyThermal(heat))

    Args:
        result: A `gfem.DrivenResult` with fields kept at ``frequency`` for
            every driven port.
        model: The model the result was solved on.
        frequency: A kept frequency, Hz (matched to 1e-9 relative).
        incident_power: The incident power in W (3D meshes).
        incident_power_per_length: The incident power per unit length in
            W/m (cross-section slabs).
        incident_phase: The relative phase of each driven port, rad, 0 where
            left out.

    Returns:
        A `gfem.HeatSource` on the solve domain's volume regions outside a
        perfectly matched layer, with face heat on the sheets.

    Raises:
        TypeError: ``result`` is not a driven result (a
            `gfem.DCConductionResult` heats through `joule_heating`), or an
            `gfem.EigenmodeResult` (a mode has no drive); ``incident_phase``
            is not a mapping.
        NotImplementedError: A `gfem.PortModeResult`, a 2D model (pose the
            driven solve on the slab), or what `gfem.dissipated_power`
            refuses (complex ``mu_r``, a law blend).
        ValueError: Not exactly one drive, the wrong one for the mesh, a
            number for several excited ports, a negative power, or no field
            kept at ``frequency``.
        KeyError: A port that was not excited, or a phase for a port the
            drive does not feed.
        NonDifferentiableError: ``frequency`` is traced; or what
            `gfem.dissipated_power` raises under `jax.grad`.
    """
    from gfem._engine.heat import sheet_surfaces
    from gfem._engine.power_flow import _check_support
    from gfem._heat import loss_matrices
    from gfem._pml import physical_regions
    from gfem.measurements import check_heat_field

    _check_heated_result(result, model)
    f = static_float(
        frequency,
        f"{_EM}: frequency",
        "The frequency is a fixed input of the driven solve: give a kept frequency.",
    )
    given = {
        name: value
        for name, value in (
            ("incident_power", incident_power),
            ("incident_power_per_length", incident_power_per_length),
        )
        if value is not None
    }
    if len(given) != 1:
        raise ValueError(
            f"{_EM} takes exactly one drive: incident_power= (W) or "
            "incident_power_per_length= (W/m, on a cross-section); got "
            f"{', '.join(given) or 'none'}."
        )
    ((name, value),) = given.items()
    thickness = model.mesh.section_thickness
    if thickness is not None and name == "incident_power":
        raise _per_length_remedy(name, _EM, "incident_power_per_length")
    if thickness is None and name == "incident_power_per_length":
        raise ValueError(
            f"{_EM}: incident_power_per_length= is for a cross-section mesh "
            "(gfem.Mesh.from_gmsh_cross_section); this model's mesh is 3D: give "
            "the total incident_power= in W instead."
        )
    powers = _incident(value, tuple(result.s.excited), name)
    ports = tuple(powers)
    phases = _phases(incident_phase, ports)
    fields = [result.e_field(f, p) for p in ports]
    for field in fields:
        check_heat_field(field, model, _EM)
    matrices = loss_matrices(fields, model)

    scale = thickness if thickness is not None else 1.0
    waves = _waves(result, f, {p: w * scale for p, w in powers.items()}, phases)
    tets = _coherent(matrices["tetrahedra_re"], matrices["tetrahedra_im"], *waves)
    faces = _coherent(matrices["triangles_re"], matrices["triangles_im"], *waves)
    mesh = model.mesh
    density = tets / _tet_volumes(mesh)
    face_density = faces / _triangle_areas(mesh) if sheet_surfaces(model) else None
    total = sheet_total = None
    if not is_traced((tets, faces)):
        sheet_total = float(jnp.sum(faces))
        total = float(jnp.sum(tets)) + sheet_total
    support = _check_support(fields[0], model)
    regions = tuple(
        r
        for r in physical_regions(mesh.volumes, model.pml)
        if support[mesh.elements(r)].any()
    )
    return HeatSource(
        regions=regions,
        _kind="electromagnetic",
        _density=density,
        _face_density=face_density,
        _total=total,
        _sheet_total=sheet_total,
        _thickness=thickness,
    )


class Trace:
    """A linear functional of the temperature, recorded at every time step.

    A `gfem.TransientThermal` problem records each of its traces at all
    ``steps + 1`` grid times, ``t = 0`` included
    (`gfem.TransientThermalResult.traces`), without keeping the fields. A
    trace is a value object: what it reads (a point, a region, a section's
    thermo-optic weights) is fixed when it is built, and its weights on the
    solve's degrees of freedom are private, built at solve time.

    Build one with `Trace.point` (the temperature at a point, K) or
    `Trace.mean` (the exact mean temperature over volume regions, K);
    `gfem.ThermoOpticShift.trace` gives the thermo-optic phase per length
    of a section, a functional of the rise (rad/m).

    Example:
        The core's mean temperature and the temperature under a heater::

            traces = {
                "core": gfem.Trace.mean("core"),
                "top": gfem.Trace.point((0.0, 2.2e-6, 0.0)),
            }

    Attributes:
        unit: The unit of the trace's values (``"K"`` for a temperature).
    """

    unit: str = "K"

    def _sample(self, temperature: Field | None, rise: Field) -> jax.Array:
        """The trace's value on one temperature and its rise (linear in both).

        A solve records it at every step through its weights on the
        coefficients of both fields, found once by differentiating this
        method (`gfem._engine.thermal.trace_rows`).

        Args:
            temperature: The absolute temperature field, K, or None for a
                trace of the rise alone (a thermo-optic phase).
            rise: The temperature rise above the source-free temperature, K.
        """
        raise NotImplementedError(f"{type(self).__name__} has no value.")

    @staticmethod
    def point(point: Any) -> Trace:
        """The temperature at a point, in K.

        Args:
            point: The point ``(x, y, z)`` in m, inside the thermal domain
                (checked when the problem is solved). On a cross-section
                mesh, z is through the slab's thickness.

        Raises:
            ValueError: ``point`` is not three finite numbers.
            NonDifferentiableError: ``point`` is traced by JAX.
        """
        check_fixed(point, "Trace.point takes a fixed point")
        try:
            p = np.asarray(point, dtype=float)
        except (TypeError, ValueError):
            p = np.empty(0)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError(
                f"Trace.point takes one point (x, y, z) in m, got {point!r}."
            )
        return _FieldTrace(_point=(float(p[0]), float(p[1]), float(p[2])))

    @staticmethod
    def mean(region: str | Sequence[str]) -> Trace:
        """The exact mean temperature over volume regions, in K.

        The mean of the solved field over the regions' elements in the
        thermal domain, as `gfem.region_mean` computes it.

        Args:
            region: A volume region name, or a sequence of names (their
                union); checked when the problem is solved.

        Raises:
            TypeError: ``region`` is not a name or names.
        """
        try:
            regions = _regions(region)
        except TypeError:
            raise TypeError(
                "Trace.mean: region must be a volume region name or a non-empty "
                f"sequence of names, got {region!r}."
            ) from None
        return _FieldTrace(_regions=regions)


@dataclasses.dataclass(frozen=True, eq=False)
class _FieldTrace(Trace):
    """The temperature at a point (`Trace.point`) or its mean over regions."""

    #: For a point trace: the point in m.
    _point: tuple[float, float, float] | None = None
    #: For a mean trace: the volume regions.
    _regions: tuple[str, ...] = ()
    unit: str = "K"

    def _sample(self, temperature: Field | None, rise: Field) -> jax.Array:
        """The temperature at the point, or its mean over the regions."""
        del rise
        if temperature is None:
            raise ValueError(f"{self!r} reads the temperature, and none was given.")
        if self._point is not None:
            return temperature.at(self._point)
        from gfem.measurements import region_mean

        return region_mean(temperature, self._regions)

    def __repr__(self) -> str:
        if self._point is not None:
            return f"Trace.point({self._point!r})"
        regions = self._regions[0] if len(self._regions) == 1 else self._regions
        return f"Trace.mean({regions!r})"


register_fields(_FieldTrace, ())


# --- switching times (copied from gsim) ------------------------------

#: The tail still to come, relative to the swing, that warns.
_SETTLED = 1e-2


def _normalised(time: Any, trace: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The time and the trace normalised to its first and last sample."""
    if is_traced((time, trace)):
        raise NonDifferentiableError(
            f"gfem.{name} is a measurement on recorded samples (a crossing time "
            "found by search), and gfem does not differentiate it. Wrap its "
            "arguments in jax.lax.stop_gradient, or differentiate a trace value "
            "instead."
        )
    t = np.asarray(time, dtype=float).reshape(-1)
    y = np.asarray(trace, dtype=float).reshape(-1)
    if t.shape != y.shape or t.size < 2:
        raise ValueError(
            f"{name}: time and trace must be 1D arrays of the same length (at least "
            f"2 samples), got shapes {np.shape(time)} and {np.shape(trace)}."
        )
    swing = y[-1] - y[0]
    if swing == 0 or not np.isfinite(swing):
        raise ValueError(
            f"{name}: the trace ends where it starts ({y[0]:g}), so it has no "
            "swing to normalise; measure a trace that changes."
        )
    return t, (y - y[0]) / swing


def _crossing_time(t: np.ndarray, s: np.ndarray, level: float, name: str) -> float:
    """The first time ``s`` reaches ``level``, interpolated from the previous sample."""
    above = np.flatnonzero(s >= level)
    if not above.size:  # pragma: no cover - s[-1] = 1 reaches every level < 1
        raise ValueError(f"{name}: the trace never reaches {level:g} of its swing.")
    i = int(above[0])
    if i == 0:
        return float(t[0])
    frac = (level - s[i - 1]) / (s[i] - s[i - 1])
    return float(t[i - 1] + frac * (t[i] - t[i - 1]))


def _check_settled(t: np.ndarray, s: np.ndarray, switch_times: Any, name: str) -> None:
    """Warn when the normalised trace ``s`` has not settled.

    The tail still to come is the geometric sum of the increments after the
    last sample, from the last two increments after the last switch. Only
    switches inside the samples count: one at or after the last sample (the
    switch that ends a slice of a longer run) has not acted on them yet.
    """
    check_fixed(switch_times, f"{name}: switch_times must be fixed times")
    switches = np.asarray(switch_times, dtype=float).reshape(-1)
    inside = switches[(switches >= t[0]) & (switches < t[-1])]
    after = t > inside.max() if inside.size else np.ones(t.size, bool)
    where = (
        f" after the last waveform switch at {inside.max():g} s" if inside.size else ""
    )
    if after.sum() < 3:
        warnings.warn(
            f"{name}: the trace has fewer than three samples{where}, so whether it "
            f"has settled cannot be estimated; the time is normalised to the last "
            "sample and may be biased. Run longer after the switch.",
            UnsettledTraceWarning,
            stacklevel=3,
        )
        return
    tail = s[after]
    d_last, d_prev = tail[-1] - tail[-2], tail[-2] - tail[-3]
    if d_last == 0:
        return
    r = d_last / d_prev if d_prev != 0 else math.inf
    remaining = abs(d_last * r / (1 - r)) if 0 < r < 1 else math.inf
    if remaining > _SETTLED:
        estimate = (
            f"about {100 * remaining:.2g} % of its swing is still to come"
            if math.isfinite(remaining)
            else f"its last increments{where} do not decay geometrically"
        )
        warnings.warn(
            f"{name}: the trace has not settled: {estimate} (more than 1 %), and "
            "the time is normalised to the last sample, so it is biased short. "
            "Run longer (a larger duration at the same time step).",
            UnsettledTraceWarning,
            stacklevel=3,
        )


def rise_time(time: Any, trace: Any, *, switch_times: Any = ()) -> float:
    """The 10-90 % rise time of a trace, in the unit of ``time``.

    The trace is normalised to its first and last samples, ``s = (y -
    y[0]) / (y[-1] - y[0])``, so a falling trace works the same; each
    level's time is the first sample at or above it, interpolated linearly
    from the previous sample; the rise time is ``t(0.9) - t(0.1)``. The
    definition is gsim's. Because it normalises to the last sample, a run
    that ends before the trace settles gives a time biased short: it warns
    then (`gfem.UnsettledTraceWarning`).

    Example:
        The core's rise time after a heater step::

            result = gfem.solve(model, problem)
            gfem.rise_time(
                result.time, result.traces["core"], switch_times=result.switch_times
            )

    Args:
        time: The sample times, ``(n,)``, increasing.
        trace: The samples, ``(n,)``.
        switch_times: The waveform's switch times
            (`gfem.TransientThermalResult.switch_times`): the settling
            check reads only the samples after the last one.

    Raises:
        ValueError: The shapes differ, or the trace ends where it starts.
        NonDifferentiableError: An argument is traced by JAX.
    """
    t, s = _normalised(time, trace, "rise_time")
    _check_settled(t, s, switch_times, "rise_time")
    return _crossing_time(t, s, 0.9, "rise_time") - _crossing_time(
        t, s, 0.1, "rise_time"
    )


def time_constant(time: Any, trace: Any, *, switch_times: Any = ()) -> float:
    """The time a trace takes to cover 1 - 1/e (63.2 %) of its swing.

    The trace is normalised to its first and last samples as in
    `rise_time`, and the time is ``t(1 - 1/e) - t[0]``, the crossing
    interpolated linearly from the previous sample; the definition is
    gsim's. For a single exponential it is the exponential's time
    constant. Normalising to the last sample biases it short when the run
    ends before the trace settles: it warns then
    (`gfem.UnsettledTraceWarning`).

    Args:
        time: The sample times, ``(n,)``, increasing.
        trace: The samples, ``(n,)``.
        switch_times: The waveform's switch times
            (`gfem.TransientThermalResult.switch_times`): the settling
            check reads only the samples after the last one.

    Raises:
        ValueError: The shapes differ, or the trace ends where it starts.
        NonDifferentiableError: An argument is traced by JAX.
    """
    t, s = _normalised(time, trace, "time_constant")
    _check_settled(t, s, switch_times, "time_constant")
    return _crossing_time(t, s, 1.0 - math.exp(-1.0), "time_constant") - float(t[0])
