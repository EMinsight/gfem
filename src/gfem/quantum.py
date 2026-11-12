"""Dressed modes of a junction on the solved linear modes.

One junction of linear inductance ``L_J`` turns the linear modes of an
eigenmode solve into the dressed modes of a superconducting circuit. From the
modes' angular frequencies ``omega_m`` and their signed junction participation
``p_m`` (`gfem.EigenmodeResult.participation` of the junction's `LumpedPort`),
`dressed_modes` returns the dressed frequencies, Lamb shifts, anharmonicities
and dispersive shifts::

    modes = gfem.solve(model, gfem.Eigenmode(target=5e9, n_modes=2))
    dressed = gfem.quantum.dressed_modes(
        modes.omega, modes.participation["junction"], inductance=L_J
    )
    chi = dressed.dispersive_shift[0, 1]  # rad/s

Every output is a signed angular frequency in rad/s, a difference of
transition frequencies (see the "Quantum metrics" section of
`gfem.conventions`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Sequence
from typing import Any, Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
import scipy.constants as const
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.typing import ArrayLike

from gfem._checks import is_traced
from gfem._pytree import register_fields
from gfem.errors import ModeLabelError, NonDifferentiableError

__all__ = ["MAX_DIMENSION", "DressedModes", "Method", "dressed_modes"]

#: The largest Hilbert-space dimension, the product of ``levels`` over the
#: modes, that `dressed_modes` diagonalizes.
MAX_DIMENSION = 2000

#: How `dressed_modes` computes the levels: ``"diagonalization"`` of the
#: junction cosine in a truncated Fock basis, or the ``"first_order"`` closed
#: forms.
Method = Literal["diagonalization", "first_order"]

#: The reduced flux quantum ``hbar / (2 e)``, Wb.
_PHI_0 = const.hbar / (2 * const.e)

_SECOND_ORDER = (
    "gfem.quantum.dressed_modes is differentiable to first order only: its levels "
    "are Rayleigh quotients on fixed eigenvectors. Use first derivatives (jax.grad, "
    "jax.jvp); if you differentiate a function that itself takes a gradient of the "
    "dressed modes (even one held with jax.lax.stop_gradient), compute that inner "
    "gradient outside the function being differentiated; or use "
    "method='first_order', which differentiates to any order."
)

_OVERLAP = (
    "DressedModes.overlap has no derivative: it is the largest overlap of a dressed "
    "level with a Fock state, read for its value (how close the modes are to "
    "hybridizing), not differentiated. Wrap it in jax.lax.stop_gradient to use it "
    "in a differentiated function."
)


@dataclasses.dataclass(frozen=True, eq=False)
class DressedModes:
    """The dressed modes of a junction on the solved linear modes.

    Every array is in the input mode order, in rad/s, and is a difference of
    the levels ``E`` of the junction Hamiltonian (``E(0)`` the ground state,
    ``E(1_m)`` one photon in mode ``m``).

    Attributes:
        omega: ``(n,)`` dressed transition frequency ``E(1_m) - E(0)``.
        lamb_shift: ``(n,)`` ``omega`` minus the real part of the input
            ``omega``; negative for a transmon.
        anharmonicity: ``(n,)`` ``E(2_m) - 2 E(1_m) + E(0)``, the second
            transition frequency minus the first; negative for a transmon.
        dispersive_shift: ``(n, n)`` symmetric,
            ``E(1_m 1_n) - E(1_m) - E(1_n) + E(0)``: the full shift of mode
            ``n``'s frequency per photon in mode ``m``. Its diagonal is
            ``anharmonicity``.
        overlap: The smallest overlap, in (1/2, 1], of a used level (the
            ground state, ``1_m``, ``2_m``, ``1_m 1_n``) with the Fock state
            it is labelled by; ``None`` for ``method="first_order"``. It has
            no derivative: a gradient through it raises
            `gfem.NonDifferentiableError`.
    """

    omega: jax.Array
    lamb_shift: jax.Array
    anharmonicity: jax.Array
    dispersive_shift: jax.Array
    overlap: jax.Array | None = None


register_fields(
    DressedModes,
    ("omega", "lamb_shift", "anharmonicity", "dispersive_shift", "overlap"),
)


def dressed_modes(
    omega: ArrayLike,
    participation: ArrayLike,
    *,
    inductance: ArrayLike,
    levels: int | Sequence[int] = 12,
    method: Method = "diagonalization",
) -> DressedModes:
    """The dressed modes of one junction on the given linear modes.

    The model is the junction's cosine potential on top of the solved
    linear modes::

        H / hbar = sum_m omega_m a_m† a_m - (E_J / hbar) [cos(phi) - 1 + phi^2 / 2]
        phi = sum_m phi_m (a_m + a_m†)
        phi_m = sign(p_m) sqrt(|p_m| hbar omega_m / (2 E_J))
        E_J = (hbar / 2e)^2 / L_J

    built on ``levels`` Fock states per mode. The cosine is exact (not a
    Taylor series). Each level used is labelled by its largest overlap with
    a bare Fock state. With one junction only the product of the
    participation signs of two modes enters, and it cancels in every
    output: the signs are accepted so that ``participation`` can be
    `gfem.EigenmodeResult.participation` of the junction's port as it is.

    ``method="first_order"`` returns the closed forms instead, with
    ``K_mn = hbar omega_m omega_n |p_m p_n| / (4 E_J)``: ``dispersive_shift
    = -K`` off the diagonal, ``anharmonicity = -K_mm / 2``, ``lamb_shift =
    -(1/2) sum_n K_mn``, ``omega = Re(omega) + lamb_shift``. It is for
    comparison with published first-order numbers: it overestimates or
    underestimates the dispersive shift by about ``3 alpha / Delta`` (1.5 to
    1.9 times at ordinary detunings) and never changes its sign.

    Validity:

    - The model is the junction's cosine on the solved linear modes. It has
      no charge dispersion: it is for the transmon regime.
    - It uses ``Re omega`` only, so it is meaningful for ``Q >> 1``.
    - Modes not passed in (not solved, or not selected) contribute no Lamb
      shift and no Kerr term.
    - The solve's qubit frequency is the linearized one: the dressed
      frequency is lower by about ``|alpha|``.
    - Labels need an overlap above 1/2 (`gfem.ModeLabelError` otherwise).
    - ``inductance`` must be the value the solve used; pass the same traced
      value to the `gfem.LumpedPort` to get the derivative through the
      solve. gfem cannot detect two different values passed to the two
      places.

    Differentiation: first derivatives with respect to ``omega``,
    ``participation`` and ``inductance`` are exact, in reverse and forward
    mode (`jax.grad`, `jax.jvp`, `jax.jacfwd`, `jax.jacrev`, under
    `jax.jit` and `jax.vmap`); ``levels`` and ``method`` are static. Levels
    are Rayleigh quotients on fixed eigenvectors, so a second
    differentiation of ``method="diagonalization"`` raises
    `gfem.NonDifferentiableError` at trace time (also, by over-refusal,
    when the inner gradient is held with `jax.lax.stop_gradient`: compute it
    outside the differentiated function). ``method="first_order"``
    differentiates to any order. A reverse-mode gradient through
    ``overlap`` raises `gfem.NonDifferentiableError`; its forward-mode
    tangent is NaN.

    Args:
        omega: ``(n,)`` angular frequencies of the linear modes, rad/s;
            complex values are accepted and their real part is used.
        participation: ``(n,)`` signed junction participation of each mode,
            as ``EigenmodeResult.participation[port]``; none may be zero
            (leave such a mode out).
        inductance: The junction's linear inductance ``L_J``, H: the value
            given to its `gfem.LumpedPort`.
        levels: Fock levels per mode, one int for all modes or one per
            mode, each at least 3.
        method: ``"diagonalization"`` or ``"first_order"``.

    Returns:
        The dressed modes, in the input mode order.

    Raises:
        ValueError: ``omega`` and ``participation`` are not 1-D of equal
            length; ``participation`` is 2-D (several junctions); a
            ``levels`` entry is below 3 or a ``levels`` sequence has the
            wrong length; the product of ``levels`` exceeds
            `MAX_DIMENSION`; ``method`` is unknown. For concrete inputs also:
            a non-finite value, ``Re omega <= 0``, ``inductance <= 0``, a
            participation of magnitude above 1 or exactly zero (a traced zero
            participation is refused by the host check below).
        gfem.ModeLabelError: A used level's largest overlap with a Fock
            state is 1/2 or less. Under a JAX transformation it arrives, like
            a traced zero participation's `ValueError`, as
            ``jax.errors.JaxRuntimeError`` with the class name and message.
    """
    if method not in get_args(Method):
        raise ValueError(f"method must be one of {get_args(Method)}, got {method!r}.")
    n = _check_shapes(omega, participation, inductance)
    dims = _dims(levels, n)
    if method == "diagonalization" and math.prod(dims) > MAX_DIMENSION:
        raise ValueError(
            f"dressed_modes: the Hilbert space of {n} modes at levels {dims} has "
            f"dimension {math.prod(dims)}, above gfem.quantum.MAX_DIMENSION = "
            f"{MAX_DIMENSION}; select fewer modes or lower levels."
        )
    traced = is_traced((omega, participation, inductance))
    if not traced:
        _check_values(omega, participation, inductance)
    w = jnp.real(jnp.asarray(omega))
    p = jnp.asarray(participation)
    ej = _PHI_0**2 / jnp.asarray(inductance) / const.hbar  # E_J / hbar, rad/s
    if method == "first_order":
        return _first_order(w, p, ej, traced)
    return _diagonalized(w, p, ej, dims, traced)


# --- argument checks ---------------------------------------------------------


def _check_shapes(
    omega: ArrayLike, participation: ArrayLike, inductance: ArrayLike
) -> int:
    p_shape, w_shape = np.shape(participation), np.shape(omega)
    if len(p_shape) == 2:
        raise ValueError(
            "dressed_modes takes one junction: participation must be 1-D (one value "
            f"per mode), got shape {p_shape}. Several junctions are not supported."
        )
    if len(w_shape) != 1 or w_shape != p_shape or w_shape[0] == 0:
        raise ValueError(
            "omega and participation must be 1-D arrays of the same nonzero length "
            f"(one value per mode), got shapes {w_shape} and {p_shape}."
        )
    if np.shape(inductance) != ():
        raise ValueError(
            f"inductance must be one number (H), got shape {np.shape(inductance)}."
        )
    return w_shape[0]


def _dims(levels: int | Sequence[int], n: int) -> tuple[int, ...]:
    dims = (levels,) * n if isinstance(levels, int | np.integer) else tuple(levels)
    if len(dims) != n:
        raise ValueError(
            f"levels must be one int or one int per mode ({n}), got {levels!r}."
        )
    if any(not isinstance(d, int | np.integer) or d < 3 for d in dims):
        raise ValueError(
            "every levels entry must be an int of at least 3 (a two-photon state "
            f"must exist), got {levels!r}."
        )
    return tuple(int(d) for d in dims)


def _check_values(
    omega: ArrayLike, participation: ArrayLike, inductance: ArrayLike
) -> None:
    w, p, lj = np.asarray(omega), np.asarray(participation), np.asarray(inductance)
    if not np.all(np.isfinite(w)) or np.any(np.real(w) <= 0):
        raise ValueError(f"omega must be finite with a positive real part, got {w!r}.")
    if (
        not np.isrealobj(p)
        or not np.all(np.isfinite(p))
        or np.any(np.abs(p) > 1 + 1e-12)
    ):
        raise ValueError(
            "participation must be finite real values in [-1, 1] (to 1e-12), "
            f"got {p!r}."
        )
    if not np.isrealobj(lj) or not np.isfinite(lj) or lj <= 0:
        raise ValueError(f"inductance must be one finite number > 0 (H), got {lj!r}.")
    _check_nonzero(p)


def _check_nonzero(p: np.ndarray) -> None:
    zero = np.flatnonzero(p == 0)
    if zero.size:
        raise ValueError(
            f"participation of mode {int(zero[0])} is exactly zero: its zero-point "
            "flux has an infinite slope there, so no derivative exists. Leave the "
            "mode out (it does not couple to the junction)."
        )


# --- first order -------------------------------------------------------------


def _first_order(
    w: jax.Array, p: jax.Array, ej: jax.Array, traced: bool
) -> DressedModes:
    if traced:  # a traced zero participation is refused on the host
        w = w + _host_check(jax.lax.stop_gradient(p))
    a = w * jnp.abs(p)
    k = jnp.outer(a, a) / (4 * ej)
    alpha = -jnp.diagonal(k) / 2
    chi = -k + jnp.diag(jnp.diagonal(k) + alpha)
    lamb = -jnp.sum(k, axis=1) / 2
    return DressedModes(
        omega=w + lamb,
        lamb_shift=lamb,
        anharmonicity=alpha,
        dispersive_shift=chi,
        overlap=None,
    )


# --- diagonalization ---------------------------------------------------------


@functools.cache
def _operators(dims: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, list]:
    """Constants of the Fock basis: occupations, embedded ``a + a†``, eigensystems."""
    occupation = np.array(list(np.ndindex(*dims)), dtype=float)  # (dim, n), row-major
    xs, eig = [], []
    for m, d in enumerate(dims):
        a = np.diag(np.sqrt(np.arange(1.0, d)), 1)
        x = a + a.T
        eig.append(np.linalg.eigh(x))
        out = np.eye(1)
        for k, dk in enumerate(dims):
            out = np.kron(out, x if k == m else np.eye(dk))
        xs.append(out)
    return occupation, np.stack(xs), eig


def _cosine(phi: jax.Array, dims: tuple[int, ...]) -> jax.Array:
    """``cos(x) - 1 + x^2 / 2`` of ``x = sum_m phi_m X_m``, analytic in ``phi``.

    ``cos(x) = Re kron_m exp(i phi_m X_m)`` with each ``exp`` from the
    constant eigendecomposition of the truncated ``X_m = a + a†``: no
    `jnp.linalg.eigh` of a traced matrix, whose derivative is NaN when two
    modes have equal or zero ``phi``.
    """
    _, xs, eig = _operators(dims)
    u = jnp.ones((1, 1), dtype=complex)
    for m, (xi, um) in enumerate(eig):
        u = jnp.kron(u, (um * jnp.exp(1j * phi[m] * xi)) @ um.T)
    x = jnp.einsum("m,mij->ij", phi, xs)
    return jnp.real(u) - jnp.eye(u.shape[0]) + x @ x / 2


@jax.custom_jvp
def _first_order_only(h: jax.Array) -> jax.Array:
    """Zero; its derivative is the refusal of a second differentiation."""
    return jnp.zeros((), h.dtype)


@_first_order_only.defjvp
def _first_order_only_jvp(primals, tangents):  # noqa: ARG001
    raise NonDifferentiableError(_SECOND_ORDER)


@jax.custom_jvp
def _dressing(
    w: jax.Array, pot: jax.Array, v: jax.Array, steps: jax.Array
) -> jax.Array:
    """Each used level minus its bare energy: a Rayleigh quotient on fixed ``v``.

    ``H = diag(occupation @ w) + pot``. Column ``r`` of the fixed eigenvectors
    ``v`` is the level labelled by Fock state ``s_r``, and ``steps[r, i] =
    occupation_i - occupation(s_r)``, so ``v_r^T H v_r / v_r^T v_r`` minus the
    bare energy of ``s_r`` is computed without subtracting large numbers.
    """
    weight = v.T**2  # (n_used, dim)
    bare = jnp.einsum("rin,n,ri->r", steps, w, weight)
    return (bare + jnp.einsum("ir,ij,jr->r", v, pot, v)) / jnp.sum(weight, axis=1)


@_dressing.defjvp
def _dressing_jvp(primals, tangents):
    w, pot, v, steps = primals
    dw, dpot, _, _ = tangents
    weight = v.T**2
    bare = jnp.einsum("rin,n,ri->r", steps, dw, weight)
    norm = jnp.sum(weight, axis=1)
    # Hellmann-Feynman: the exact first derivative, with no 1 / (E_i - E_j). The
    # bare energy's own derivative is dw . occupation(s_r), added by the caller.
    # The guards' zeros are added so that jit cannot drop them: in a second
    # differentiation their primals are themselves differentiated and they raise.
    tangent = (bare + jnp.einsum("ir,ij,jr->r", v, dpot, v)) / norm
    guard = _first_order_only(w) + _first_order_only(pot)
    return _dressing(w, pot, v, steps), tangent + guard


# A linear map that evaluates to NaN and whose transpose raises on a nonzero
# cotangent: the tangent of `overlap`. Reverse mode then raises exactly when a
# gradient flows through `overlap`, and leaves every other output alone.
_overlap_tangent_p = Primitive("gfem_overlap_tangent")
_overlap_tangent_p.def_impl(lambda t: jnp.full_like(t, jnp.nan))
_overlap_tangent_p.def_abstract_eval(lambda t: t)


def _overlap_tangent_transpose(ct, t):
    if type(ct) is ad.Zero:
        return [ad.Zero(t.aval)]
    raise NonDifferentiableError(_OVERLAP)


def _overlap_tangent_jvp(primals, tangents):
    (t,), (dt,) = primals, tangents
    out = _overlap_tangent_p.bind(t)
    if type(dt) is ad.Zero:
        return out, ad.Zero.from_primal_value(out)
    return out, _overlap_tangent_p.bind(dt)


ad.primitive_transposes[_overlap_tangent_p] = _overlap_tangent_transpose
ad.primitive_jvps[_overlap_tangent_p] = _overlap_tangent_jvp
batching.primitive_batchers[_overlap_tangent_p] = lambda args, dims: (
    _overlap_tangent_p.bind(*args),
    dims[0],
)
mlir.register_lowering(
    _overlap_tangent_p,
    mlir.lower_fun(lambda t: t * jnp.nan, multiple_results=False),
)


@jax.custom_jvp
def _no_derivative(value: jax.Array, depends_on: jax.Array) -> jax.Array:  # noqa: ARG001
    """``value``, whose derivative with respect to ``depends_on`` is refused."""
    return value


@_no_derivative.defjvp
def _no_derivative_jvp(primals, tangents):
    value, _ = primals
    _, dt = tangents
    return value, _overlap_tangent_p.bind(jnp.zeros_like(value) + jnp.sum(dt))


def _labels(dims: tuple[int, ...]) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """The Fock states used and where each output reads them."""
    n = len(dims)
    states: list[tuple[int, ...]] = [(0,) * n]

    def at(occ: tuple[int, ...]) -> int:
        if occ not in states:
            states.append(occ)
        return states.index(occ)

    one = [at(tuple(int(k == m) for k in range(n))) for m in range(n)]
    two = [at(tuple(2 * int(k == m) for k in range(n))) for m in range(n)]
    pair = {
        (m, q): at(tuple(int(k in (m, q)) for k in range(n)))
        for m in range(n)
        for q in range(m + 1, n)
    }
    return states, {"one": one, "two": two, "pair": pair}


def _diagonalized(
    w: jax.Array, p: jax.Array, ej: jax.Array, dims: tuple[int, ...], traced: bool
) -> DressedModes:
    # One compiled function per (dims, traced): eager calls compile once.
    modes, check = _levels(w, p, ej, dims=dims, host=traced)
    if not traced:
        _check_labels(*(np.asarray(c) for c in check), dims=dims)
    return modes


@functools.partial(jax.jit, static_argnames=("dims", "host"))
def _levels(
    w: jax.Array, p: jax.Array, ej: jax.Array, *, dims: tuple[int, ...], host: bool
) -> tuple[DressedModes, tuple[jax.Array, ...]]:
    occupation, _, _ = _operators(dims)
    phi = jnp.sign(p) * jnp.sqrt(jnp.abs(p) * w / (2 * ej))
    pot = -ej * _cosine(phi, dims)
    h = jnp.diag(jnp.asarray(occupation) @ w) + pot
    states, where = _labels(dims)
    rows = np.array([np.ravel_multi_index(s, dims) for s in states])
    _, v = jnp.linalg.eigh(jax.lax.stop_gradient(h))
    weight = v[rows] ** 2  # (n_used, dim): overlap of each eigenvector with each state
    k = jnp.argmax(weight, axis=1)
    best = jnp.take_along_axis(weight, k[:, None], axis=1)[:, 0]
    # The other Fock state that the labelled eigenvector holds most of.
    rival = jnp.argmax(
        v[:, k].T ** 2 - 2.0 * (jnp.arange(v.shape[0]) == rows[:, None]), 1
    )
    bare = jax.lax.stop_gradient(jnp.asarray(occupation) @ w)
    check = (bare, best, jnp.asarray(rows), rival, jax.lax.stop_gradient(p))
    # Under a transformation the labels are checked on the host; the check's
    # zero is added to every level so that jit cannot drop it.
    zero = _host_check(*check, dims=dims) if host else jnp.zeros(())
    steps = occupation[None, :, :] - occupation[rows][:, None, :]
    # Levels minus their bare energies; the bare energies cancel exactly in every
    # output (one_m = w_m + d(1_m) - d(0), and so on).
    d = _dressing(w, pot, jax.lax.stop_gradient(v[:, k]), jnp.asarray(steps)) + zero

    d0 = d[0]
    shift = d[jnp.asarray(where["one"])] - d0
    one = w + shift
    alpha = d[jnp.asarray(where["two"])] - d0 - 2 * shift
    chi = jnp.diag(alpha)
    for (m, q), i in where["pair"].items():
        pair = d[i] - d0 - shift[m] - shift[q]
        chi = chi.at[m, q].set(pair).at[q, m].set(pair)
    overlap = _no_derivative(
        jax.lax.stop_gradient(jnp.min(best)), jnp.concatenate([w, p, ej[None]])
    )
    modes = DressedModes(
        omega=one,
        lamb_shift=shift,
        anharmonicity=alpha,
        dispersive_shift=chi,
        overlap=overlap,
    )
    return modes, check


# --- the host check ----------------------------------------------------------


def _host_check(*arrays: jax.Array, dims: tuple[int, ...] | None = None) -> jax.Array:
    """A zero computed on the host after the checks, raising there if one fails."""
    if dims is None:  # first order: the zero-participation check alone
        (p,) = arrays

        def host(p_):
            _check_nonzero(np.asarray(p_))
            return np.zeros((), np.float64)

        return jax.pure_callback(
            host, jax.ShapeDtypeStruct((), jnp.float64), p, vmap_method="sequential"
        )

    def host_labels(*values):
        _check_labels(*(np.asarray(x) for x in values), dims=dims)
        return np.zeros((), np.float64)

    return jax.pure_callback(
        host_labels,
        jax.ShapeDtypeStruct((), jnp.float64),
        *arrays,
        vmap_method="sequential",
    )


def _state_name(occ: tuple[int, ...]) -> str:
    modes = [m for m, c in enumerate(occ) if c]
    if not modes:
        return "the ground state"
    if len(modes) == 2:
        return f"one photon in each of modes {modes[0]} and {modes[1]}"
    count = "one photon" if occ[modes[0]] == 1 else "two photons"
    return f"{count} in mode {modes[0]}"


def _check_labels(
    bare: np.ndarray,
    best: np.ndarray,
    rows: np.ndarray,
    rival: np.ndarray,
    p: np.ndarray,
    *,
    dims: tuple[int, ...],
) -> None:
    _check_nonzero(p)
    bad = np.flatnonzero(best <= 0.5)
    if not bad.size:
        return
    i = int(bad[np.argmin(best[bad])])
    state = tuple(int(c) for c in np.unravel_index(int(rows[i]), dims))
    other = tuple(int(c) for c in np.unravel_index(int(rival[i]), dims))
    ghz = 1 / (2 * np.pi * 1e9)
    raise ModeLabelError(
        f"dressed_modes: the level labelled {_state_name(state)} overlaps its Fock "
        f"state by {float(best[i]):.3f} only (at most 1/2): it has hybridized with "
        f"Fock state {other}. Bare frequencies involved: "
        f"{float(bare[rows[i]]) * ghz:.6g} GHz for {state} and "
        f"{float(bare[rival[i]]) * ghz:.6g} GHz for {other}. The "
        "modes are no longer separate objects. Move the modes apart (change the "
        "design), or leave the hybridized mode out of the call. "
        "method='first_order' is not a remedy: it gives a number for a picture "
        "that no longer holds."
    )
