r"""The response of a periodic cell to one meta-atom parameter, and its interpolation.

`cell_response` solves a cell at a grid of parameter values and keeps S with
$\partial S/\partial p$ at each; `CellResponse` interpolates them by piecewise
cubic Hermite polynomials in pure JAX.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from gfem import _codec, _progress
from gfem._checks import is_traced
from gfem._pytree import readonly, register_fields
from gfem.errors import NonDifferentiableError

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gfem.model import Model
    from gfem.options import SolverOptions
    from gfem.results import DiffractionOrderResult, NetworkParameters

__all__ = ["CellResponse", "cell_response"]

_FORMAT = "cell response"

_TRACED = (
    "cell_response: the cell's design is traced by JAX (make_cell(p, 0.0) returned "
    "a model with traced values, or a sample p is traced): a cell response "
    "tabulates fixed cells, and differentiating its stored dS/dp in the design "
    "would need second derivatives of the solve, which gfem does not compute. "
    "Build the response outside jax.grad, jax.jit and jax.vmap; to optimize the "
    "meta-atom itself, differentiate gfem.solve of the cell directly."
)
_STATIC = (
    "CellResponse: {name} must be fixed numbers: the sample grid, frequencies and "
    "reference impedances are constants of the response. Build them outside "
    "jax.grad, jax.jit or jax.vmap."
)


# --- validation --------------------------------------------------------------------


def _samples(parameter: Any, who: str) -> np.ndarray:
    """``parameter`` as a read-only strictly increasing finite 1D float array."""
    bad = ValueError(
        f"{who}: parameter must be a flat, strictly increasing sequence of finite "
        f"samples, got {parameter!r}."
    )
    try:
        arr = np.array(parameter, dtype=float)
    except (TypeError, ValueError):
        raise bad from None
    if arr.ndim != 1 or arr.size < 1 or not np.all(np.isfinite(arr)):
        raise bad
    if np.any(np.diff(arr) <= 0):
        raise bad
    return readonly(arr)


def _vectors(value: Any) -> tuple[tuple[float, float, float], ...] | None:
    """Two finite 3-vectors as tuples of floats, or ``None``."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (2, 3) or not np.all(np.isfinite(arr)):
        return None
    return tuple(tuple(float(v) for v in row) for row in arr)  # ty: ignore[invalid-return-type]


def _check_lattice(who: str, lattice: Mapping[str, Any], surfaces: set[str]) -> dict:
    """The lattice vectors of every Floquet surface as tuples, or CR9."""
    out = {}
    for surface in sorted(surfaces | set(lattice)):
        vec = _vectors(lattice[surface]) if surface in lattice else None
        if vec is None:
            raise ValueError(
                f"{who}: lattice vectors of {surface!r} must be two finite "
                "3-vectors in m."
            )
        out[surface] = vec
    return out


def _check_direction(
    who: str, direction: Mapping[str, Any], surfaces: set[str]
) -> dict:
    """``direction`` as ints, one +1 or -1 per Floquet surface, or CR9."""
    ok = set(direction) == surfaces and all(
        isinstance(h, (int, np.integer))
        and not isinstance(h, (bool, np.bool_))
        and h in (1, -1)
        for h in direction.values()
    )
    if not ok:
        raise ValueError(
            f"{who}: direction must map each Floquet surface to +1 or -1, got "
            f"{dict(direction)!r}."
        )
    return {s: int(h) for s, h in direction.items()}


def _check_wave_vector(who: str, k: Any) -> tuple[float, float, float] | None:
    """``k`` as a tuple of floats, ``None`` kept."""
    if k is None:
        return None
    try:
        arr = np.asarray(k, dtype=float)
    except (TypeError, ValueError):
        arr = np.zeros(0)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"{who}: wave_vector must be None or a finite 3-vector in rad/m, got {k!r}."
        )
    return (float(arr[0]), float(arr[1]), float(arr[2]))


@dataclasses.dataclass(frozen=True, eq=False)
class CellResponse:
    r"""S of a periodic cell, sampled against one parameter, with its derivative.

    Built by `cell_response`, or directly for an analytic response. Calling
    it evaluates S at a parameter value by the piecewise cubic Hermite
    interpolant of the samples: on the interval :math:`[p_k, p_{k+1}]` of
    width :math:`h`, with :math:`s = (p - p_k)/h`,

    .. math::

        S(p) = h_{00}(s) S_k + h\,h_{10}(s) S'_k + h_{01}(s) S_{k+1}
               + h\,h_{11}(s) S'_{k+1},

    where :math:`h_{00} = 2s^3 - 3s^2 + 1`, :math:`h_{10} = s^3 - 2s^2 + s`,
    :math:`h_{01} = -2s^3 + 3s^2`, :math:`h_{11} = s^3 - s^2` and
    :math:`S'_k = \partial S/\partial p` at :math:`p_k`. Real and imaginary
    parts interpolate alike, so there is no phase to unwrap. The interpolant
    reproduces S and its derivative at every sample, and between samples
    ``jax.grad`` returns the derivative of the interpolant. The evaluation is
    pure JAX: it differentiates in :math:`p` and in ``values`` and
    ``derivative``, and works under ``jax.jit`` and ``jax.vmap``.

    Attributes:
        parameter: ``(n_p,)`` the samples, strictly increasing, in the user's
            units.
        frequency: ``(n_f,)`` Hz.
        values: ``(n_p, n_f, n_ports, n_excited)`` complex, S.
        derivative: The same shape, :math:`\partial S/\partial p` per unit of
            ``parameter``.
        z_ref: ``(n_f, n_ports)`` reference impedances, constant in
            :math:`p`.
        ports: The Floquet ports, in the row order of S.
        excited: The excited ports, in the column order of S.
        diffraction_orders: The `gfem.DiffractionOrderResult` of each
            Floquet port, of the first sample.
        lattice: Per Floquet surface, the lattice vectors
            :math:`(a_1, a_2)` in m.
        direction: Per Floquet surface, the direction sign :math:`\pm1` of
            the beams leaving through it along :math:`\hat e_3`.
        wave_vector: The incidence's Bloch wave vector, rad/m, or ``None``
            for plain periodicity.

    Raises:
        ValueError: The samples are not strictly increasing and finite, the
            arrays do not have the shape ``(n_p, n_f, n_ports, n_excited)``,
            a Floquet port has no diffraction order, ``direction`` is not
            :math:`\pm1` per surface, or a lattice vector or the frequencies
            are not finite numbers.
        NonDifferentiableError: A static field is traced by JAX.
    """

    parameter: np.ndarray
    frequency: np.ndarray
    values: jax.Array
    derivative: jax.Array
    z_ref: np.ndarray
    ports: tuple[str, ...]
    excited: tuple[str, ...]
    diffraction_orders: Mapping[str, DiffractionOrderResult]
    lattice: Mapping[str, tuple[tuple[float, float, float], tuple[float, float, float]]]
    direction: Mapping[str, int]
    wave_vector: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        who = "CellResponse"
        for name in ("parameter", "frequency", "z_ref", "wave_vector"):
            if is_traced(getattr(self, name)):
                raise NonDifferentiableError(_STATIC.format(name=name))
        parameter = _samples(self.parameter, who)
        values = jnp.asarray(self.values)
        derivative = jnp.asarray(self.derivative)
        ports = tuple(self.ports)
        excited = tuple(self.excited)
        n_f = values.shape[1] if values.ndim == 4 else np.size(self.frequency)
        expected = (len(parameter), n_f, len(ports), len(excited))
        if tuple(values.shape) != expected or tuple(derivative.shape) != expected:
            raise ValueError(
                f"{who}: values and derivative must both have shape "
                "(n_parameter, n_frequency, n_ports, n_excited) = "
                f"{expected}, got {tuple(values.shape)} and "
                f"{tuple(derivative.shape)}."
            )
        freq = np.asarray(self.frequency, dtype=float)
        if freq.shape != (n_f,) or not np.all(np.isfinite(freq) & (freq > 0)):
            raise ValueError(
                f"{who}: frequency must hold {n_f} frequencies > 0 Hz, got "
                f"{self.frequency!r}."
            )
        orders = dict(self.diffraction_orders)
        missing = [n for n in ports if n not in orders]
        if missing:
            raise ValueError(
                f"{who}: diffraction_orders lacks Floquet port(s) {missing}."
            )
        surfaces = {orders[n].surface for n in ports}
        direction = _check_direction(who, self.direction, surfaces)
        lattice = _check_lattice(who, self.lattice, surfaces)
        k = _check_wave_vector(who, self.wave_vector)
        z_ref = np.asarray(self.z_ref)
        if z_ref.shape != (n_f, len(ports)):
            raise ValueError(
                f"{who}: z_ref must have shape (n_frequency, n_ports) = "
                f"{(n_f, len(ports))}, got {z_ref.shape}."
            )
        for name, arr in (("values", values), ("derivative", derivative)):
            as_complex = (
                arr
                if jnp.issubdtype(arr.dtype, jnp.complexfloating)
                else arr.astype(jnp.result_type(arr.dtype, 1j))
            )
            object.__setattr__(self, name, as_complex)
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "frequency", readonly(freq))
        object.__setattr__(self, "z_ref", readonly(z_ref))
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "excited", excited)
        object.__setattr__(self, "diffraction_orders", orders)
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "wave_vector", k)

    def __call__(self, p: ArrayLike) -> NetworkParameters:
        r"""S at the parameter value ``p``, for every frequency.

        Args:
            p: A scalar in the sampled range ``[parameter[0], parameter[-1]]``.
                A one-sample response evaluates only at its sample.

        Returns:
            The `gfem.NetworkParameters` at ``p`` (the type of ``result.s``),
            so the cell-output helpers that read S apply unchanged. Where ``p``
            is traced and out of range, every entry and its gradient are NaN.

        Raises:
            ValueError: ``p`` is not a scalar, or is concrete and outside the
                sampled range.
        """
        from gfem.results import NetworkParameters

        if np.ndim(p) != 0:
            raise ValueError(
                f"CellResponse: p must be a scalar, got shape {np.shape(p)}; "
                "evaluate several values with jax.vmap(response)."
            )
        lo, hi = float(self.parameter[0]), float(self.parameter[-1])
        if is_traced(p):
            p = jnp.asarray(p, dtype=float)
            outside = ~((p >= lo) & (p <= hi))
            p = p + jnp.where(outside, jnp.nan, 0.0)
        else:
            check_range(self, p)
            p = jnp.asarray(np.asarray(p, dtype=float))
        if len(self.parameter) == 1:
            values = self.values[0] + self.derivative[0] * (p - lo)
        else:
            values = _hermite(p, self.parameter, self.values, self.derivative)
        return NetworkParameters(
            values=values,
            z_ref=jnp.asarray(self.z_ref),
            frequency=tuple(float(f) for f in self.frequency),
            ports=self.ports,
            excited=self.excited,
        )

    def to_json(self, path: str | os.PathLike) -> Path:
        """Write the response as JSON, with an ``.npz`` sidecar for its arrays.

        The sidecar, ``<stem>.npz`` next to the JSON file, holds ``values``,
        ``derivative`` and the other arrays. Floats and complex numbers
        round-trip bitwise.

        Args:
            path: The JSON file to write; its directory is created.

        Returns:
            The JSON file's path.

        Raises:
            TypeError: The response holds values computed inside `jax.grad`,
                `jax.jit` or `jax.vmap`.
        """
        import json

        from gfem import job

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoder = _codec.Encoder(_codec.Context(path.parent.resolve()))
        body = encoder.encode(self)
        sidecar = path.with_suffix(".npz")
        doc = _codec.header(_FORMAT) | {
            "arrays": sidecar.name if encoder.arrays else None,
            "response": body,
        }
        if encoder.arrays:
            job._save_npz(sidecar, encoder.arrays)  # noqa: SLF001
        path.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
        return path

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> CellResponse:
        """Read a response written by `to_json`.

        Args:
            path: The JSON file.

        Raises:
            FileNotFoundError: The JSON file or its sidecar is missing.
            ValueError: Not a cell response file, a newer schema version, or
                a type or an entry this gfem does not know.
        """
        from gfem import job

        path = Path(path)
        head = job._read_json(path)  # noqa: SLF001
        doc = _codec.check_header(head, _FORMAT, "gfem cell response file", path)
        arrays = job._read_arrays(doc, path)  # noqa: SLF001
        if "response" not in doc:
            raise ValueError(f"{path} has no 'response' entry.")
        written_by = doc.get("gfem_version")
        decoder = _codec.Decoder(
            _codec.Context(path.parent.resolve()),
            arrays,
            path=path,
            written_by=written_by if isinstance(written_by, str) else None,
        )
        value = decoder.decode(doc["response"])
        if not isinstance(value, cls):
            raise ValueError(f"{path} is not a gfem cell response file.")  # noqa: TRY004
        return value


register_fields(CellResponse, ("values", "derivative"))
_codec.register(
    CellResponse,
    tag="CellResponse",
)


def check_range(response: CellResponse, p: ArrayLike) -> None:
    """Raise CR7 when a concrete ``p`` (scalar or array) leaves the sampled range."""
    arr = np.asarray(p, dtype=float)
    lo, hi = float(response.parameter[0]), float(response.parameter[-1])
    outside = ~((arr >= lo) & (arr <= hi))
    if np.any(outside):
        raise ValueError(
            f"CellResponse: {int(outside.sum())} of {arr.size} parameter values lie "
            f"outside the sampled range [{lo:g}, {hi:g}] (min {np.nanmin(arr):g}, "
            f"max {np.nanmax(arr):g}); the cell response does not extrapolate. Map "
            "an unbounded design variable onto the range, e.g. lo + (hi - lo) * "
            "jax.nn.sigmoid(x)."
        )


def _hermite(p: jax.Array, xs: np.ndarray, ys: jax.Array, dys: jax.Array) -> jax.Array:
    """Piecewise cubic Hermite through ``(xs, ys, dys)`` at a scalar ``p``."""
    x = jnp.asarray(xs)
    k = jnp.clip(jnp.searchsorted(x, p, side="right") - 1, 0, len(xs) - 2)
    h = x[k + 1] - x[k]
    s = (p - x[k]) / h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * ys[k] + h * h10 * dys[k] + h01 * ys[k + 1] + h * h11 * dys[k + 1]


# --- the build ---------------------------------------------------------------------


@dataclasses.dataclass
class _Signature:
    """The static labels of a cell that the parameter may not change."""

    ports: tuple[tuple[str, str, tuple[int, int], str], ...]
    lattice: dict[str, np.ndarray]
    media: dict[str, tuple[float, float]]
    offsets: tuple[tuple[str, float], ...]
    propagating: dict[str, tuple[tuple[tuple[int, int], ...], ...]]
    direction: dict[str, int]


def _signature(model: Model, frequency: np.ndarray, wave_vector: Any) -> _Signature:
    from gfem._floquet import floquet_ports_in

    fports = floquet_ports_in(model.ports)
    surfaces = tuple(dict.fromkeys(p.surface for p in fports))
    lattice, media, propagating, direction = {}, {}, {}, {}
    for s in surfaces:
        face = model._face(s, "cell_response")  # noqa: SLF001
        lattice[s] = np.stack([face.a1, face.a2])
        media[s] = (float(face.eps_r), float(face.mu_r))
        direction[s] = int(np.sign(-face.normal @ face.frame[2]))
        propagating[s] = tuple(
            model.propagating_orders(s, float(f), wave_vector=wave_vector)
            for f in frequency
        )
    return _Signature(
        ports=tuple(
            (p.name, p.surface, tuple(p.order), p.polarization) for p in fports
        ),
        lattice=lattice,
        media=media,
        offsets=tuple((p.name, float(p.offset)) for p in fports),
        propagating=propagating,
        direction=direction,
    )


def _differs(a: _Signature, b: _Signature) -> str | None:
    """What of the cell ``b`` differs from ``a``, in the order of CR5, else ``None``."""
    if a.ports != b.ports:
        return "ports"
    if a.lattice.keys() != b.lattice.keys() or any(
        not np.allclose(a.lattice[s], b.lattice[s], rtol=1e-9, atol=0.0)
        for s in a.lattice
    ):
        return "lattice"
    if a.media.keys() != b.media.keys() or any(
        not np.allclose(a.media[s], b.media[s], rtol=1e-12, atol=0.0) for s in a.media
    ):
        return "half-space media"
    if a.offsets != b.offsets:
        return "port offsets"
    if a.propagating != b.propagating:
        return "propagating orders"
    return None


def _order_arrays(parts: Sequence[DiffractionOrderResult]) -> DiffractionOrderResult:
    """One `DiffractionOrderResult` over several one-frequency ones."""
    first = parts[0]
    cat = {
        name: np.concatenate([np.atleast_1d(getattr(d, name)) for d in parts])
        for name in ("frequency", "wave_vector", "gamma", "propagating", "theta", "phi")
    }
    return dataclasses.replace(first, **cat)


def cell_response(
    make_cell: Callable[[float, Any], Model],
    parameter: ArrayLike,
    *,
    frequency: ArrayLike,
    excite: Sequence[str],
    options: SolverOptions | None = None,
    out: str | os.PathLike | None = None,
) -> CellResponse:
    r"""Tabulate a cell's S and :math:`\partial S/\partial p` against a parameter.

    ``make_cell(p, dp)`` returns the cell at the meta-atom parameter ``p``,
    its mesh moved by ``dp``: ``p`` is a plain float (the cell may be meshed
    at it), ``dp`` a JAX scalar of value 0 that moves the mesh, typically
    through `gfem.shape.boundary_morph`. For each sample and each frequency
    the builder runs one `gfem.solve` and takes
    :math:`\partial S/\partial\,\mathrm{dp}` at 0 by reverse mode, one
    backward pass per real entry of S. ``dp`` should move the cell by one unit
    of ``p`` per unit of ``dp``, so that the stored derivative is per unit of
    ``p``.

    One mesh moved by a boundary morph over the whole range is smooth in
    :math:`p` but bounded: check the range with `gfem.shape.quality`. Across
    a wider range, mesh each sample (``make_cell`` builds the cell at ``p``
    and moves it by ``dp``): the table is then as smooth as the meshes are
    converged. The parameter may move the meta-atom only, not the lattice,
    the half-spaces or the ports.

    Only the rows of the Floquet ports are kept. With ``out=`` the response
    is written after each sample, and a rerun with the same ``out``,
    ``parameter``, ``frequency`` and ``excite`` keeps the finished samples.

    Args:
        make_cell: ``(p, dp) -> gfem.Model``.
        parameter: The samples, strictly increasing, in the user's units.
        frequency: Frequencies, Hz.
        excite: Names of the ports to excite.
        options: Solver options, passed to `gfem.solve`.
        out: A JSON path to write after each sample and resume from.

    Returns:
        The `CellResponse`.

    Raises:
        TypeError: ``make_cell`` is not callable or does not return a model.
        ValueError: The samples are not strictly increasing, ``excite`` names
            no port of the cell, the cell has no Floquet port, a sample
            changes the lattice, half-spaces, orders or ports, or ``out``
            holds another response.
        NonDifferentiableError: The cell's design or a sample is traced by
            JAX; a cell response tabulates fixed cells.
    """
    from gfem.model import Model
    from gfem.problems import Driven

    who = "cell_response"
    if not callable(make_cell):
        raise TypeError(
            "cell_response needs make_cell, a function (p, dp) -> gfem.Model, got "
            f"{type(make_cell).__name__}."
        )
    if is_traced(parameter):
        raise NonDifferentiableError(_TRACED)
    samples = _samples(parameter, who)
    # validates the frequencies with Driven's own messages
    freqs = np.asarray(Driven(frequency=frequency).frequency, dtype=float)
    p0 = float(samples[0])

    def concrete(p: float) -> Model:
        model = make_cell(p, 0.0)
        if not isinstance(model, Model):
            raise TypeError(
                f"cell_response: make_cell({p:g}, dp) returned "
                f"{type(model).__name__}, not a gfem.Model."
            )
        if is_traced(model):
            raise NonDifferentiableError(_TRACED)
        return model

    model0 = concrete(p0)
    names = [p.name for p in model0.ports]
    if (
        isinstance(excite, str)
        or not isinstance(excite, Sequence)
        or len(excite) == 0
        or any(n not in names for n in excite)
    ):
        raise ValueError(
            "cell_response: excite must name at least one port of the cell, got "
            f"{excite!r}."
        )
    excite = tuple(excite)
    sig0 = _signature(model0, freqs, None)
    if not sig0.ports:
        raise ValueError(
            "cell_response: the cell has no Floquet port; a cell response keeps "
            "the rows of S of the Floquet ports (gfem.FloquetPort, "
            "gfem.diffraction.floquet_ports)."
        )
    ports = tuple(n for n, *_ in sig0.ports)
    old = _resume(out, samples, freqs, excite, ports, sig0)
    values = [] if old is None else list(np.asarray(old.values))
    derivative = [] if old is None else list(np.asarray(old.derivative))
    meta: dict[str, Any] = {}
    if old is not None:
        meta = {"z_ref": np.asarray(old.z_ref), "orders": dict(old.diffraction_orders)}

    def assemble(n: int) -> CellResponse:
        return CellResponse(
            parameter=samples[:n],
            frequency=freqs,
            values=jnp.asarray(np.stack(values[:n])),
            derivative=jnp.asarray(np.stack(derivative[:n])),
            z_ref=meta["z_ref"],
            ports=ports,
            excited=excite,
            diffraction_orders=meta["orders"],
            lattice={s: tuple(map(tuple, v)) for s, v in sig0.lattice.items()},
            direction=sig0.direction,
            wave_vector=None,
        )

    done = len(values)
    with _progress.task(_progress.CELL_RESPONSE, total=len(samples)) as row:
        row.advance(done)
        for i in range(done, len(samples)):
            p = float(samples[i])
            what = _differs(sig0, _signature(concrete(p), freqs, None)) if i else None
            if what is not None:
                raise ValueError(
                    f"cell_response: the cell at parameter {p:g} differs from the "
                    f"cell at {p0:g} in its {what}; the parameter may move the "
                    "meta-atom only, not the lattice, the half-spaces or the "
                    "ports."
                )
            solved = [
                _solve_one(make_cell, p, float(f), excite, ports, options)
                for f in freqs
            ]
            values.append(np.stack([v for v, _, _, _ in solved]))
            derivative.append(np.stack([d for _, d, _, _ in solved]))
            if not meta:
                meta["z_ref"] = np.stack([z for _, _, z, _ in solved])
                meta["orders"] = {
                    n: _order_arrays([o[n] for _, _, _, o in solved]) for n in ports
                }
            if out is not None:
                assemble(i + 1).to_json(out)
            row.advance(parameter=p)
    return assemble(len(samples))


def _resume(  # noqa: PLR0917
    out: str | os.PathLike | None,
    samples: np.ndarray,
    freqs: np.ndarray,
    excite: tuple[str, ...],
    ports: tuple[str, ...],
    sig0: _Signature,
) -> CellResponse | None:
    """The finished samples of ``out``, or ``None``; CR6 if it is another build."""
    if out is None or not Path(out).exists():
        return None
    old = CellResponse.from_json(out)
    same_lattice = old.lattice.keys() == sig0.lattice.keys() and all(
        np.allclose(old.lattice[s], sig0.lattice[s], rtol=1e-9, atol=0.0)
        for s in sig0.lattice
    )
    for what, same in (
        (
            "parameter values",
            np.array_equal(old.parameter, samples[: len(old.parameter)]),
        ),
        ("frequencies", np.array_equal(old.frequency, freqs)),
        ("excited ports", old.excited == excite),
        ("ports", old.ports == ports),
        ("lattice", same_lattice),
    ):
        if not same:
            raise ValueError(
                f"cell_response: {out} holds a cell response with other {what}; "
                "give a new out= path, or the same parameter, frequency and "
                "excite to resume."
            )
    return old


def _solve_one(  # noqa: PLR0917
    make_cell: Callable[[float, Any], Model],
    p: float,
    frequency: float,
    excite: tuple[str, ...],
    ports: tuple[str, ...],
    options: SolverOptions | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, DiffractionOrderResult]]:
    """S and dS/ddp at 0 of the Floquet rows at one frequency.

    One solve, then one backward pass per real entry.
    """
    import gfem
    from gfem.problems import Driven

    problem = Driven(frequency=[frequency], excite=list(excite))
    labels: dict[str, Any] = {}

    def stacked(dp: jax.Array) -> tuple[jax.Array, np.ndarray]:
        result = gfem.solve(make_cell(p, dp), problem, options=options)
        rows = jnp.asarray([result.s.ports.index(n) for n in ports])
        cols = jnp.asarray([result.s.excited.index(n) for n in excite])
        s = result.s.values[0][rows][:, cols]
        labels["orders"] = {n: result.diffraction_orders[n] for n in ports}
        z = np.asarray(jax.lax.stop_gradient(result.s.z_ref))[0][np.asarray(rows)]
        return jnp.stack([s.real, s.imag]), z

    value, vjp, z = jax.vjp(stacked, jnp.asarray(0.0), has_aux=True)
    grad = np.empty(value.size)
    for j in range(value.size):
        cotangent = np.zeros(value.size)
        cotangent[j] = 1.0
        grad[j] = float(vjp(jnp.asarray(cotangent.reshape(value.shape)))[0])
    v = np.asarray(value)
    g = grad.reshape(v.shape)
    return v[0] + 1j * v[1], g[0] + 1j * g[1], np.asarray(z), labels["orders"]
