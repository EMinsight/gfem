r"""gfem driven solves as components of SAX circuits.

`SaxComponent` wraps a function that builds a `gfem.Model` and a
`gfem.Driven` template as a model callable for a
`SAX <https://flaport.github.io/sax/>`__ netlist: the function's keyword
defaults become the component's settings, the template's frequencies its
band, and each call returns S between the component's circuit ports as a
plain dictionary (an ``SDict``). gfem does not import SAX.

Per call the component solves the whole band as a function of the design
settings alone, converts S on the band (`gfem.NetworkParameters.to_sdict`)
and reads it at the circuit's wavelength by a cubic Hermite interpolation in
frequency, so `jax.jit`, `jax.vmap` over the wavelength and `jax.grad` all
work, and the solve count does not grow with the number of wavelengths.
"""

from __future__ import annotations

import collections
import dataclasses
import inspect
import os
import types
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants

from gfem._checks import is_traced
from gfem._text import hertz_band
from gfem.model import FloquetPort, LumpedPort, Model, WavePort
from gfem.options import SolverOptions
from gfem.problems import Driven
from gfem.results import NetworkParameters, _circuit_ports, _circuit_z0
from gfem.solve import solve

__all__ = ["SaxComponent"]

Circuit = Literal["optical", "rf"]

#: Setting names the component owns: a model function may not use them.
_RESERVED = ("wl", "f", "z0")
#: Solved networks a component keeps, the least recently used dropped first.
_MEMO_SIZE = 64
#: Relative slack that admits a frequency at the band's edges, as
#: ``Driven``'s own grid matching.
_SLACK = 1e-9


@dataclasses.dataclass(frozen=True, eq=False, init=False, repr=False)
class SaxComponent:
    r"""A gfem driven solve as a SAX model, for ``sax.circuit(models=...)``.

    ``SaxComponent(model, problem, circuit="optical")`` is called like
    SAX's analytic models: with keyword settings, the model function's
    parameters plus ``wl``, the wavelength in µm. It returns
    ``{(input, output): S}`` for every pair of its circuit ports, zeros
    included (SAX reads a missing key as zero); ``sd["o1", "o2"]`` is
    :math:`S_{21}`. The values have the shape of ``wl``.

    **Settings.** ``model`` is a function of keyword settings returning a
    `gfem.Model`, every parameter with a default, or a bare `gfem.Model`
    for a component without design settings. Settings arrive as Python
    numbers or JAX arrays (SAX passes a default unchanged and a value set on
    an instance as an array); build the model from them, e.g. a
    `gfem.MaterialField` of a permittivity array, a boundary morph or a
    lumped value. The component's signature is the function's parameters,
    keyword-only, plus ``wl``; `settings` holds the defaults.

    **Band.** The template's frequencies (any order) are the component's
    band. Every evaluation solves the whole band for the design settings,
    converts S there and interpolates each entry's real and imaginary parts
    with a cubic Hermite polynomial in frequency, at
    :math:`f = c / \lambda` (slopes from the three-point second-order
    difference, one-sided at the ends; linear on two frequencies). At a
    band frequency the value is the solve's S; between them the error is
    about :math:`\theta^3/16`, :math:`\theta` the largest phase change of an
    entry between neighbouring band frequencies, so about 25 band
    frequencies per :math:`2\pi` of phase keep it near :math:`10^{-3}`. A
    one-frequency template admits only that frequency. The default ``wl``
    is the band frequency nearest the band's midpoint, so a call without
    ``wl`` reads a solved value. A wavelength outside the band raises
    `ValueError`; under `jax.jit`, `jax.grad` or `jax.vmap`, where it may be
    unknown, it gives NaN in the value and the gradient instead. The
    gradient with respect to ``wl`` is the interpolant's.

    **Conventions.** SAX's photonic models are :math:`e^{-i\omega t}`, so
    the component returns :math:`\overline{S}`, gfem's S in that
    convention, with unit modal power on each wave port, at the reference
    planes the ports declare (``WavePort(offset=)``). Each exposed wave port
    needs a ``polarity`` reference (or a ``voltage_path``), because the
    relative signs of a block's ports decide the interference around a
    circuit loop and the default mode sign depends on the mesh and on the
    model's orientation. Ports left out of ``ports`` terminate the device
    as in a solve: a wave port absorbs its mode, a lumped port is its own
    R, L and C.

    **RF.** With ``circuit="rf"`` the settings are ``f``, the frequency in
    Hz, and ``z0``, the reference impedance in ohm (one real scalar, 50 by
    default), instead of ``wl``. S keeps gfem's :math:`e^{+i\omega t}`,
    like SAX's RF models, and is renormalized from the exposed ports' own
    reference impedances to ``z0`` on the band, before the interpolation in
    ``f``; it is no solve input, so a new ``z0`` costs no solve. Every
    exposed port needs a reference: a `gfem.LumpedPort` with a resistance
    (its inductance and capacitance stay a load at the port) or a
    `gfem.WavePort` with a ``voltage_path``, whose impedance is
    :math:`Z_{PV}`. A lumped port with only an inductance or a capacitance
    stays out of ``ports`` as an internal lumped load, its value a design
    setting. The mesh is 3D. Pass ``z0`` once to the whole circuit, e.g.
    ``circ(f=f, z0=50.0)``, never to one instance: SAX's lumped RF models
    then share it, and a model with another reference does not match.
    Models that are matched to their own impedance (SAX's coplanar and
    microstrip lines, which take no ``z0``) are right beside the block only
    where that impedance equals ``z0``.

    **Solves.** The constructor solves once at the defaults. A component
    keeps the S of its last 64 distinct design settings (exact values; the
    wavelength is not part of it), so SAX's no-argument call when the
    circuit is built, calls at new wavelengths and a block that is not
    differentiated cost no solve. A design computed inside `jax.grad`,
    `jax.jit` or `jax.vmap` is solved on every call, except that instances
    fed the very same traced value (one tracer object, not an equal one)
    share one forward and one backward solve, kept only while that tracer
    lives. A call inside a transformation keeps no concrete entry. With an
    adaptive template (``sweep="adaptive"``) without ``samples``, the
    constructor's solve is the greedy sweep: its sampled frequencies are
    frozen into `problem`, so every later solve, traced ones included, uses
    them.

    Build the circuit once, outside the transformed function, and pass
    designs as instance settings.

    Example:
        An interferometer arm of a gfem slab whose design patch is a
        permittivity array, beside SAX's straight waveguide::

            def slab_at(*, eps=eps0):
                return model.with_material("design", gfem.MaterialField(eps_r=eps))

            wl = np.linspace(1.50, 1.60, 21)  # um
            arm = gfem.SaxComponent(
                slab_at,
                gfem.Driven(c / (wl * 1e-6), polarization="Ez"),
                circuit="optical",
                ports={"p1": "o1", "p2": "o2"},
            )
            mzi, _ = sax.circuit(netlist, models={"slab": arm, ...})
            jax.grad(lambda eps: abs(mzi(wl=1.55, top={"eps": eps})["in", "out"]) ** 2)

    Args:
        model: A function of keyword settings returning a `gfem.Model`, or
            a `gfem.Model`.
        problem: The `gfem.Driven` template: its frequencies are the band;
            its polarization (2D), sweep and samples are used as given. Its
            ``excite`` is ``None`` or the exposed ports, ``keep_fields``
            ``False``, with no sources.
        circuit: ``"optical"``: a ``wl`` setting in µm and SAX's photonic
            :math:`e^{-i\omega t}`. ``"rf"``: settings ``f`` in Hz and
            ``z0`` in ohm, :math:`e^{+i\omega t}` as gfem, and S
            renormalized to ``z0`` (see **RF** above). Never inferred,
            because the wrong kind still composes, with wrong interference.
        ports: gfem port name -> circuit port name, e.g. ``{"p1": "o1"}``. A
            circuit port name is an identifier, or ``name@mode`` (SAX
            multimode) for every port or none. ``None`` exposes every port
            under its own name.
        options: The numerics of every solve.
        cache: A directory for the solve cache, as for `gfem.solve`, used by
            every solve that is not inside a JAX transformation.

    Attributes:
        model: The model function or model, as given.
        problem: The template, as given, with the samples of an adaptive
            sweep frozen by the constructor.
        circuit: The circuit kind.
        ports: The exposed ports, gfem name -> circuit name.
        options: The solver options.
        cache: The solve cache directory, or ``None``.
        settings: The default of every setting, ``wl`` (optical) or ``f``
            and ``z0`` (RF) included.

    Raises:
        TypeError: ``model`` is neither a `gfem.Model` nor a function
            returning one; a parameter of the function has no default, is
            not a keyword parameter (``*args``, ``**kwargs``, positional
            only) or has a dictionary default; ``problem`` is not a
            `gfem.Driven`, ``options`` not a `gfem.SolverOptions` or
            ``ports`` not a mapping.
        ValueError: ``circuit`` is unknown; a parameter is named ``wl``,
            ``f`` or ``z0``; ``ports`` names an unknown port or repeats or
            misspells a circuit port name; optical: an exposed
            `gfem.WavePort` has neither ``polarity`` nor ``voltage_path``; RF:
            the mesh is 2D, or an exposed port is a `gfem.FloquetPort`, a
            `gfem.LumpedPort` without a resistance or a `gfem.WavePort`
            without a ``voltage_path``; or the template's ``excite``,
            ``keep_fields`` or ``sources`` is set.
            The solve at the defaults raises what `gfem.solve` raises.
    """

    model: Model | Callable[..., Model]
    problem: Driven
    circuit: Circuit
    ports: Mapping[str, str]
    options: SolverOptions | None
    cache: str | os.PathLike | None
    settings: Mapping[str, Any]
    _function: Callable[..., Model]
    _design: tuple[str, ...]
    _band: np.ndarray
    _solved: Driven
    _memo: collections.OrderedDict[tuple, NetworkParameters]
    _traced: dict[tuple, tuple[list[weakref.ref], NetworkParameters]]

    def __init__(
        self,
        model: Model | Callable[..., Model],
        problem: Driven,
        *,
        circuit: Circuit,
        ports: Mapping[str, str] | None = None,
        options: SolverOptions | None = None,
        cache: str | os.PathLike | None = None,
    ) -> None:
        """Validate, then solve at the defaults (see the class docstring)."""
        if circuit not in ("optical", "rf"):
            raise ValueError(
                f"circuit must be 'optical' or 'rf', got {circuit!r}: it sets the "
                "circuit's frequency setting, the time convention and the "
                "normalization of S, so it is never inferred."
            )
        if options is not None and not isinstance(options, SolverOptions):
            raise TypeError(
                "SaxComponent options must be a gfem.SolverOptions, got "
                f"{type(options).__name__}."
            )
        function = _model_function(model)
        defaults = _defaults(function)
        built = _build(function, defaults)
        exposed = _circuit_ports(ports, built.port_names)
        _check_references(built, exposed, circuit)
        band, solved = _template(problem, exposed)
        greedy = None
        if solved.sweep == "adaptive" and solved.samples is None:
            # the greedy solve is the constructor's solve at the defaults;
            # its samples are the band's from now on, in every later solve
            greedy = solve(built, solved, options=options, cache=cache)
            sweep = greedy.sweep
            if sweep is None:  # an adaptive solve always reports its sweep
                raise RuntimeError("An adaptive solve returned no sweep report.")
            samples = tuple(sweep.sampled_frequencies)
            problem = dataclasses.replace(problem, samples=samples)
            solved = dataclasses.replace(solved, samples=samples)
        middle = 0.5 * (band[0] + band[-1])
        default_f = float(band[np.argmin(np.abs(band - middle))])
        if circuit == "rf":
            settings = {**defaults, "f": default_f, "z0": 50.0}
        else:
            settings = {**defaults, "wl": _wavelength(default_f)}
        fields = {
            "model": model,
            "problem": problem,
            "circuit": circuit,
            "ports": types.MappingProxyType(exposed),
            "options": options,
            "cache": cache,
            "settings": types.MappingProxyType(settings),
            "_function": function,
            "_design": tuple(defaults),
            "_band": band,
            "_solved": solved,
            "_memo": collections.OrderedDict(),
            "_traced": {},
            # SAX reads the settings from the signature; no return annotation,
            # since SAX takes one naming a model or callable for a factory
            "__signature__": inspect.Signature(
                [
                    inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
                    for k, v in settings.items()
                ]
            ),
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)
        if greedy is None:
            self._network(defaults, built)
        else:
            self._keep(self._key(defaults), greedy.s)

    def __call__(self, **settings: Any) -> dict[tuple[str, str], jax.Array]:
        """S between the circuit ports at ``wl``, for these settings.

        Settings not given take their defaults (`settings`).

        Returns:
            ``{(input, output): S}`` for every pair of exposed ports, each of
            the shape of ``wl``.

        Raises:
            TypeError: A setting is not one of `settings`.
            ValueError: ``wl`` is outside the band (outside JAX
                transformations; NaN inside them).
        """
        unknown = [k for k in settings if k not in self.settings]
        if unknown:
            raise TypeError(
                f"SaxComponent got no setting {', '.join(map(repr, unknown))}; its "
                f"settings are {', '.join(map(repr, self.settings))}."
            )
        values = {**self.settings, **settings}
        # z0 is no solve input: checked first, so a bad one costs no solve
        z0 = _circuit_z0(values["z0"]) if self.circuit == "rf" else None
        network = self._network({k: values[k] for k in self._design})
        if self.circuit == "rf":
            # renormalized on the band grid, before the interpolation
            grid = network.to_sdict(circuit="rf", ports=self.ports, z0=z0)
            return self._at(grid, jnp.asarray(values["f"], dtype=float))
        grid = network.to_sdict(circuit="optical", ports=self.ports)
        # the circuit's wavelength in um, as a frequency in Hz
        frequency = scipy.constants.c / (jnp.asarray(values["wl"], dtype=float) * 1e-6)
        return self._at(grid, frequency)

    def __repr__(self) -> str:
        """Circuit, ports, settings and band."""
        settings = ", ".join(f"{k!r}: {_brief(v)}" for k, v in self.settings.items())
        n = len(self._band)
        return (
            f"SaxComponent(circuit={self.circuit!r}, ports={dict(self.ports)!r}, "
            f"settings={{{settings}}}, frequency={n} value{'s' * (n > 1)} "
            f"{hertz_band(self._band[0], self._band[-1], 4)})"
        )

    # -- solves -------------------------------------------------------------------

    def _network(
        self, design: dict[str, Any], model: Model | None = None
    ) -> NetworkParameters:
        """S over the band for ``design``: from the memo, or solved and kept.

        ``model`` is the function's model at ``design`` when the caller has
        built it already.
        """
        key = self._key(design)
        if key is not None:
            hit = self._lookup(key)
            if hit is not None:
                return hit
        if model is None:
            model = _build(self._function, design)
        network = solve(model, self._solved, options=self.options, cache=self.cache).s
        self._keep(key, network)
        return network

    def _key(self, design: dict[str, Any]) -> tuple[tuple, list[Any]] | None:
        """The memo key of ``design`` and the band, with its tracers, or ``None``.

        A concrete setting is keyed by its exact value, a traced one by the
        identity of its tracer (the tracers are returned, for the weak
        references that guard the identities); a setting that is not a
        number, an array of numbers, a string, a bool or ``None`` is never
        keyed, so it is solved every time rather than mistaken.
        """
        parts: list[Any] = [tuple(self._band)]
        tracers: list[Any] = []
        for name in self._design:
            value = design[name]
            if is_traced(value):
                leaves = jax.tree.leaves(value)
                if not all(is_traced(leaf) for leaf in leaves):
                    return None
                parts.append((name, "tracer", tuple(id(leaf) for leaf in leaves)))
                tracers.extend(leaves)
                continue
            part = _value_key(value)
            if part is None:
                return None
            parts.append((name, part))
        return tuple(parts), tracers

    def _lookup(self, key: tuple[tuple, list[Any]]) -> NetworkParameters | None:
        """The kept network for ``key``, or ``None``."""
        parts, tracers = key
        if tracers:
            hit = self._traced.get(parts)
            # an id is reused once its tracer dies: the entry must hold these
            if hit is not None and all(
                ref() is tracer for ref, tracer in zip(hit[0], tracers, strict=True)
            ):
                return hit[1]
            return None
        if parts in self._memo:
            self._memo.move_to_end(parts)
            return self._memo[parts]
        return None

    def _keep(
        self, key: tuple[tuple, list[Any]] | None, network: NetworkParameters
    ) -> None:
        """Keep ``network`` under ``key``, least recent out.

        A concrete key keeps NumPy arrays only, so a solve staged inside
        `jax.jit` (tracers) is dropped. A traced key keeps the network
        until one of its tracers dies, which is when the trace ends.
        """
        if key is None:
            return
        parts, tracers = key
        if tracers:

            def drop(_ref: weakref.ref, parts: tuple = parts) -> None:
                self._traced.pop(parts, None)

            self._traced[parts] = ([weakref.ref(t, drop) for t in tracers], network)
            return
        if is_traced((network.values, network.z_ref)):
            return
        self._memo[parts] = dataclasses.replace(
            network, values=np.asarray(network.values), z_ref=np.asarray(network.z_ref)
        )
        while len(self._memo) > _MEMO_SIZE:
            self._memo.popitem(last=False)

    # -- the band -----------------------------------------------------------------

    def _at(
        self, grid: dict[tuple[str, str], jax.Array], frequency: jax.Array
    ) -> dict[tuple[str, str], jax.Array]:
        """``grid`` (S over the band) read at ``frequency`` (Hz, any shape)."""
        if not is_traced(frequency):
            self._check_in_band(np.asarray(frequency))
        keys = list(grid)
        stacked = jnp.stack([grid[k] for k in keys], axis=-1)
        values = _hermite(frequency, jnp.asarray(self._band), stacked)
        return {k: values[..., i] for i, k in enumerate(keys)}

    def _check_in_band(self, frequency: np.ndarray) -> None:
        """Raise if a concrete frequency lies outside the band."""
        lo, hi = self._band[0], self._band[-1]
        outside = (frequency < lo * (1 - _SLACK)) | (frequency > hi * (1 + _SLACK))
        if outside.any():
            if self.circuit == "rf":
                bad = ", ".join(hertz_band(f, f) for f in frequency[outside][:3])
                raise ValueError(
                    f"f = {bad} is outside the component's band, "
                    f"{hertz_band(lo, hi)}; widen the frequencies of its Driven "
                    "template to cover it."
                )
            bad = ", ".join(f"{_wavelength(f):.6g}" for f in frequency[outside][:3])
            raise ValueError(
                f"wl = {bad} um is outside the component's band, "
                f"{_wavelength(hi):.6g}-{_wavelength(lo):.6g} um "
                f"({hertz_band(lo, hi)}); widen the frequencies of its Driven "
                "template to cover it."
            )


def _model_function(model: Any) -> Callable[..., Any]:
    """``model`` as a function of keyword settings.

    Raises:
        TypeError: ``model`` is neither a `gfem.Model` nor callable.
    """
    if isinstance(model, Model):
        return lambda: model
    if not callable(model):
        raise TypeError(
            "SaxComponent needs a gfem.Model or a function of keyword settings "
            f"returning one, e.g. def device(*, width=0.5): ...; got "
            f"{type(model).__name__}."
        )
    return model


def _defaults(function: Callable[..., Any]) -> dict[str, Any]:
    """The function's parameters and their defaults, in order.

    Raises:
        TypeError: A parameter has no default, is not a keyword parameter or
            has a dictionary default.
        ValueError: A parameter is named like a setting the component owns.
    """
    out = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise TypeError(
                f"The model function takes {parameter}; SAX reads a model's settings "
                "from its named parameters, so give each setting its own keyword "
                "parameter with a default, e.g. def device(*, width=0.5): ..."
            )
        if parameter.kind is parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"The model function's parameter {name!r} is positional-only; SAX "
                "passes settings by keyword, so drop the '/', e.g. "
                f"def device(*, {name}=...): ..."
            )
        if parameter.default is parameter.empty:
            raise TypeError(
                f"The model function's parameter {name!r} has no default; SAX "
                "builds a circuit by calling each model without arguments, so give "
                f"it one, e.g. def device(*, {name}=0.5): ..."
            )
        if name in _RESERVED:
            raise ValueError(
                f"The model function's parameter {name!r} is a setting of the "
                "component itself (wl, f and z0 are), and the solve owns the "
                "frequency; rename it."
            )
        if isinstance(parameter.default, Mapping):
            raise TypeError(
                f"The model function's parameter {name!r} has a dictionary default; "
                "SAX reads a dictionary setting as settings of a sub-model, so pass "
                "its entries as separate parameters."
            )
        out[name] = parameter.default
    return out


def _build(function: Callable[..., Any], design: dict[str, Any]) -> Model:
    """The function's model at ``design``.

    Raises:
        TypeError: It returns something other than a `gfem.Model`.
    """
    model = function(**design)
    if not isinstance(model, Model):
        raise TypeError(
            "The SaxComponent model function must return a gfem.Model, got "
            f"{type(model).__name__}."
        )
    return model


def _check_references(model: Model, exposed: dict[str, str], circuit: Circuit) -> None:
    """Raise if an exposed port has no reference a circuit of this kind can use.

    Raises:
        ValueError: Optical: an exposed `gfem.WavePort` has neither
            ``polarity`` nor ``voltage_path``. RF: the mesh is 2D, or an
            exposed port is a `gfem.FloquetPort`, a `gfem.LumpedPort` with
            no resistance or a `gfem.WavePort` with no ``voltage_path``.
    """
    if circuit == "rf":
        _check_rf_references(model, exposed)
        return
    for port in model.ports:
        if (
            port.name in exposed
            and isinstance(port, WavePort)
            and port.polarity is None
            and port.voltage_path is None
        ):
            raise ValueError(
                f"WavePort {port.name!r} has no polarity reference: the sign of its "
                "mode, and so the interference a circuit computes around it, would "
                "depend on the mesh and on the model's orientation. Give it "
                "polarity=(point, direction), a point on the port where its field "
                "is strong and the direction along which that field is positive "
                f"(e.g. WavePort({port.name!r}, ..., polarity=((x, y), '+z')) on a "
                "2D line port), or a voltage_path; or leave it out of ports= to "
                "terminate it."
            )


def _check_rf_references(model: Model, exposed: dict[str, str]) -> None:
    """Raise if an RF circuit cannot take ``model`` and its exposed ports."""
    if model.mesh.dim == 2:
        raise ValueError(
            "An RF component needs a 3D model: a 2D model has no lumped port or "
            "voltage_path to give a port its reference impedance. Use "
            "circuit='optical' for a 2D model, or a 3D mesh for RF."
        )
    for port in model.ports:
        if port.name not in exposed:
            continue
        if isinstance(port, FloquetPort):
            raise ValueError(  # noqa: TRY004
                f"FloquetPort {port.name!r} cannot be an RF circuit port: its "
                "orders have no reference impedance. Leave it out of ports=, or "
                "use circuit='optical'."
            )
        if isinstance(port, LumpedPort) and port.rlc[0] is None:
            raise ValueError(
                f"LumpedPort {port.name!r} has no resistance, so no reference "
                "impedance (its inductance and capacitance are a load, not a "
                "port): give it a resistance (e.g. LumpedPort("
                f"{port.name!r}, ..., resistance=50.0, capacitance=c)) to make it "
                "a circuit port, or leave it out of ports= to keep it as a lumped "
                "load inside the device."
            )
        if isinstance(port, WavePort) and port.voltage_path is None:
            raise ValueError(
                f"WavePort {port.name!r} has no voltage_path, so its mode has no "
                "reference impedance for an RF circuit: give it "
                f"WavePort({port.name!r}, ..., voltage_path=(start, end)), from "
                "the ground to the signal conductor, or leave it out of ports= "
                "to terminate it."
            )


def _template(problem: Any, exposed: dict[str, str]) -> tuple[np.ndarray, Driven]:
    """The band in Hz, increasing, and the problem every solve solves.

    Raises:
        TypeError: ``problem`` is not a `gfem.Driven`.
        ValueError: Its ``excite`` is neither ``None`` nor the exposed ports,
            or it keeps fields or has sources.
    """
    if not isinstance(problem, Driven):
        raise TypeError(
            "SaxComponent needs a gfem.Driven template, whose frequencies are its "
            f"band, got {type(problem).__name__}."
        )
    if problem.excite is not None and set(problem.excite) != set(exposed):
        raise ValueError(
            f"The template's excite={problem.excite!r} differs from the exposed "
            f"ports {tuple(exposed)!r}: the component excites exactly those, so "
            "pass excite=None."
        )
    if problem.keep_fields is not False:
        raise ValueError(
            "The template keeps fields, but a circuit reads S only; drop "
            "keep_fields from the Driven template."
        )
    if problem.sources:
        raise ValueError(
            "The template has sources, but a circuit component's S has a column "
            "per port only; drop sources from the Driven template."
        )
    band = np.sort(np.asarray(problem.frequency, dtype=float))
    solved = dataclasses.replace(problem, frequency=tuple(band), excite=tuple(exposed))
    return band, solved


def _wavelength(frequency: float) -> float:
    """The vacuum wavelength in µm of ``frequency`` in Hz."""
    return scipy.constants.c / frequency * 1e6


def _value_key(value: Any) -> tuple | None:
    """A setting's exact value as a hashable key, or ``None`` if it has none."""
    if is_traced(value):
        return None
    if value is None or isinstance(value, str | bool):
        return (type(value).__name__, value)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biufc":
        return None
    return (array.dtype.str, array.shape, array.tobytes())


def _brief(value: Any) -> str:
    """A setting's value for a repr: an array by its shape."""
    if not isinstance(value, str) and np.ndim(value) > 0:
        return f"array{np.shape(value)}"
    return repr(value)


def _hermite(x: jax.Array, xp: jax.Array, fp: jax.Array) -> jax.Array:
    """``fp`` (rows over increasing ``xp``) interpolated at ``x``, cubic Hermite.

    Real and imaginary parts alike (the weights are real); slopes from the
    three-point second-order difference, one-sided at the ends; linear on
    two points; one point admits only itself. Out of ``[xp[0], xp[-1]]``
    (relative slack ``_SLACK``) the value and its gradient are NaN: the
    result is multiplied by NaN there, which a ``jnp.where`` would not do
    for the gradient. Returns shape ``x.shape + fp.shape[1:]``.
    """
    flat = x.reshape(-1)
    n = xp.shape[0]
    tail = fp.shape[1:]
    column = (-1,) + (1,) * len(tail)
    inside = (flat >= xp[0] * (1 - _SLACK)) & (flat <= xp[-1] * (1 + _SLACK))
    if n == 1:
        inside = jnp.abs(flat - xp[0]) <= _SLACK * xp[0]
        out = jnp.broadcast_to(fp[0], flat.shape + tail)
    else:
        xc = jnp.clip(flat, xp[0], xp[-1])
        i = jnp.clip(jnp.searchsorted(xp, xc, side="right") - 1, 0, n - 2)
        h = (xp[i + 1] - xp[i]).reshape(column)
        t = ((xc - xp[i]) / (xp[i + 1] - xp[i])).reshape(column)
        m = _slopes(xp, fp)
        t2, t3 = t * t, t * t * t
        out = (
            (2 * t3 - 3 * t2 + 1) * fp[i]
            + (t3 - 2 * t2 + t) * h * m[i]
            + (-2 * t3 + 3 * t2) * fp[i + 1]
            + (t3 - t2) * h * m[i + 1]
        )
    out = out * jnp.where(inside, 1.0, jnp.nan).reshape(column)
    return out.reshape(x.shape + tail)


def _slopes(xp: jax.Array, fp: jax.Array) -> jax.Array:
    """The derivative of ``fp`` at each ``xp``, second-order differences."""
    column = (-1,) + (1,) * (fp.ndim - 1)
    h = jnp.diff(xp)
    d = jnp.diff(fp, axis=0) / h.reshape(column)
    if xp.shape[0] == 2:
        return jnp.stack([d[0], d[0]])
    left, right = h[:-1].reshape(column), h[1:].reshape(column)
    inner = (right * d[:-1] + left * d[1:]) / (left + right)
    first = ((2 * h[0] + h[1]) * d[0] - h[0] * d[1]) / (h[0] + h[1])
    last = ((2 * h[-1] + h[-2]) * d[-1] - h[-1] * d[-2]) / (h[-1] + h[-2])
    return jnp.concatenate([first[None], inner, last[None]], axis=0)
