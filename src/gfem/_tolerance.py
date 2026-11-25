"""Tolerance report: the linearized spread, effects and yield of named quantities.

A forward model is a JAX function of a few named real scalars (an edge bias, a
permittivity, a gap) returning named real quantities (a transmission, a
frequency). One reverse-mode pass per quantity gives the Jacobian at the
nominal point; the parameters' tolerances, read as three standard deviations of
independent or correlated Gaussians, then give each quantity's spread, the share
of each parameter in it and the fraction of devices that meet a specification.
All statistics run in NumPy at report time, nothing is traced.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtr

from gfem._checks import is_traced

__all__ = ["ToleranceReport", "tolerance_report"]

_OWNER = "tolerance_report"

# The joint yield is a Monte Carlo on the linear model with 2**18 samples, in
# chunks of 2**16, from a generator seeded with 0: a report is reproducible.
_SAMPLES = 2**18
_CHUNK = 2**16
_SEED = 0

# A correlation matrix with an eigenvalue below minus this is not positive
# semidefinite; the rounding of a set of pairwise coefficients stays far above.
_PSD_FLOOR = -1e-10

# A quantity is flagged when the linear model misses a forward evaluation at
# plus or minus the tolerance by more than this fraction of its three sigma.
_NONLINEAR = 0.1


@dataclasses.dataclass(frozen=True, eq=False)
class ToleranceReport:
    r"""The linearized spread, per-parameter effects and yield of named quantities.

    Built by `gfem.tolerance_report`; ``str(report)`` is the text report and
    `to_markdown` the same as pipe tables. With the Jacobian
    :math:`J_{ij} = \partial f_i / \partial p_j` at the nominal point and a
    tolerance :math:`t_j` read as :math:`3\sigma_j`, the spread of quantity
    :math:`i` is :math:`s_i^2 = (J \Sigma J^\top)_{ii}` with
    :math:`\Sigma = D R D`, :math:`D = \mathrm{diag}(\sigma)` and :math:`R` the
    correlation matrix (the identity by default).

    Attributes:
        parameters: The parameter names, in the order of ``nominal``.
        quantities: The quantity names, in the order the function returned them.
        nominal: The nominal value of each parameter.
        tolerance: The tolerance :math:`\pm t` of each parameter, :math:`t = 3\sigma`.
        values: Each quantity at the nominal point.
        jacobian: :math:`J_{ij}`, ``(n_quantities, n_parameters)``, per unit of
            the parameter.
        effect: :math:`J_{ij} t_j`, the signed change of a quantity at
            ``+tolerance`` of a parameter, ``(n_quantities, n_parameters)``.
        share: :math:`J_{ij}^2 \sigma_j^2 / s_i^2`, the share of each parameter
            in the variance of a quantity, ``(n_quantities, n_parameters)``;
            zero for a quantity with no spread.
        correlation_share: The rest of each quantity's variance, the part of
            :math:`s_i^2` the correlation adds (it may be negative), so that a
            row of ``share`` and this remainder sum to one; ``None`` without
            correlation.
        std: The spread :math:`s_i` of each quantity (the text prints
            :math:`\pm 3 s_i`).
        worst_case: Each quantity's range when every parameter sits at the end
            of its tolerance that hurts most, :math:`f_i \pm \sum_j |J_{ij}| t_j`
            (it ignores the correlation).
        specs: The specification window ``(low, high)`` of each quantity that
            has one, ``None`` for an open side.
        yields: The probability that each quantity with a spec lies in its
            window, in :math:`[0, 1]`, exact on the linear model.
        cpk: The process capability :math:`\min((u - f)/3s, (f - l)/3s)` of each
            quantity with a spec, over its finite sides.
        joint_yield: The probability that every spec is met; ``None`` without
            specs. Exact with one spec, otherwise a seeded Monte Carlo on the
            linear model.
        joint_yield_error: The Monte Carlo standard error of ``joint_yield``
            (0 with one spec); ``None`` without specs.
        joint_yield_bounds: The Bonferroni lower and Frechet upper bounds on
            the joint yield; ``None`` without specs.
        checked: The parameters whose linearity was checked, largest variance
            share first; empty when the check is off.
        linear_error: :math:`e_{ij} = \max_\pm |f_i^\pm - f_i \mp J_{ij} t_j|`
            for the forward evaluations :math:`f^\pm` at :math:`p_j \pm t_j`,
            ``(n_quantities, len(checked))``.
        mean_shift: The shift of the mean of each quantity the curvature of the
            checked parameters causes, :math:`\tfrac12 \sum_j h_{ij}\sigma_j^2`
            with :math:`h_{ij} = (f_i^+ + f_i^- - 2 f_i)/t_j^2`; empty when the
            check is off.
        nonlinear: The quantities whose linear error exceeds a tenth of
            :math:`3 s_i` for a checked parameter: their yield is not reliable.
    """

    parameters: tuple[str, ...]
    quantities: tuple[str, ...]
    nominal: Mapping[str, float]
    tolerance: Mapping[str, float]
    values: Mapping[str, float]
    jacobian: np.ndarray
    effect: np.ndarray
    share: np.ndarray
    correlation_share: Mapping[str, float] | None
    std: Mapping[str, float]
    worst_case: Mapping[str, tuple[float, float]]
    specs: Mapping[str, tuple[float | None, float | None]]
    yields: Mapping[str, float]
    cpk: Mapping[str, float]
    joint_yield: float | None
    joint_yield_error: float | None
    joint_yield_bounds: tuple[float, float] | None
    checked: tuple[str, ...]
    linear_error: np.ndarray
    mean_shift: Mapping[str, float]
    nonlinear: tuple[str, ...]

    def __str__(self) -> str:
        return self._render(markdown=False)

    def to_markdown(self) -> str:
        """The report as Markdown: the same tables as pipe tables."""
        return self._render(markdown=True)

    def __repr__(self) -> str:
        head = (
            f"ToleranceReport({_count(len(self.quantities), 'quantity', 'quantities')}, "
            f"{_count(len(self.parameters), 'parameter', 'parameters')}"
        )
        if self.joint_yield is None:
            return head + ")"
        return f"{head}, joint yield {_percent(self.joint_yield)})"

    # --- rendering ---------------------------------------------------------------

    def _render(self, *, markdown: bool) -> str:
        blocks: list[list[str]] = []
        title = (
            f"Tolerance report: {_count(len(self.quantities), 'quantity', 'quantities')}, "
            f"{_count(len(self.parameters), 'parameter', 'parameters')}"
        )
        if self.correlation_share is None:
            who = "parameters independent."
        else:
            who = "parameters correlated as given."
        blocks.append(
            [
                f"### {title}" if markdown else title,
                "Linearized at the nominal design. Each tolerance ±t is 3 standard "
                "deviations of a Gaussian;" + (" " if markdown else "\n") + who,
            ]
        )
        parameter_rows = [
            [name, _g(self.nominal[name], 4), "±" + _g(self.tolerance[name], 4)]
            for name in self.parameters
        ]
        blocks.append(
            _table(["Parameter", "nominal", "tolerance"], parameter_rows, markdown)
        )
        blocks.append(self._quantity_block(markdown))
        blocks.append(self._effect_block(markdown))
        blocks.append(self._remark_block())
        blocks.append(self._linearity_block())
        return "\n\n".join("\n".join(b) for b in blocks if b) + "\n"

    def _starred(self, name: str) -> bool:
        """Whether a finite spec limit of ``name`` lies beyond three sigma."""
        low, high = self.specs[name]
        reach = 3 * self.std[name]
        f = self.values[name]
        return (low is not None and f - low > reach) or (
            high is not None and high - f > reach
        )

    def _quantity_block(self, markdown: bool) -> list[str]:
        has_specs = bool(self.specs)
        header = ["Quantity", "nominal", "±3σ", "worst case"]
        if has_specs:
            header += ["spec", "yield", "Cpk"]
        rows = []
        for name in self.quantities:
            lo, hi = self.worst_case[name]
            row = [
                name,
                _g(self.values[name], 3),
                "±" + _g(3 * self.std[name], 2),
            ]
            worst = f"[{_g(lo, 3)}, {_g(hi, 3)}]"
            if name in self.specs:
                low, high = self.specs[name]
                met = (low is None or lo >= low) and (high is None or hi <= high)
                worst += " pass" if met else " fail"
            row.append(worst)
            if has_specs:
                if name in self.specs:
                    star = "*" if self._starred(name) else ""
                    row += [
                        _spec(*self.specs[name]),
                        _percent(self.yields[name]) + star,
                        _g(self.cpk[name], 2),
                    ]
                else:
                    row += ["", "", ""]
            rows.append(row)
        lines = _table(header, rows, markdown)
        if self.joint_yield is not None:
            if len(self.yields) > 1:
                low, high = self.joint_yield_bounds  # ty: ignore[not-iterable]
                tail = (
                    f"{_percent(self.joint_yield)} ± "
                    f"{100 * self.joint_yield_error:.2f} % (Monte Carlo on the "
                    f"linear model; bounds {_percent(low)} to {_percent(high)})"
                )
            else:
                tail = f"{_percent(self.joint_yield)} (exact on the linear model)"
            lines += ["", f"All specs: {tail}"] if markdown else [f"All specs: {tail}"]
        if any(self._starred(name) for name in self.specs):
            lines.append(
                "* the spec limit lies beyond ±3σ, where the linear model was not checked."
            )
        return lines

    def _effect_block(self, markdown: bool) -> list[str]:
        title = "Effect at +tolerance (share of variance), largest first"
        corr = self.correlation_share
        if markdown:
            header = ["Quantity", *self.parameters] + (["correlation"] if corr else [])
            rows = []
            for i, name in enumerate(self.quantities):
                row = [name] + [
                    f"{self.effect[i, j]:+.3g} ({100 * self.share[i, j]:.0f} %)"
                    for j in range(len(self.parameters))
                ]
                if corr is not None:
                    row.append(f"{100 * corr[name]:+.0f} %")
                rows.append(row)
            return [title, "", *_table(header, rows, markdown)]
        width = max(len(q) for q in self.quantities)
        lines = [title]
        for i, name in enumerate(self.quantities):
            order = sorted(
                range(len(self.parameters)), key=lambda j, i=i: -self.share[i, j]
            )
            parts = [
                f"{self.parameters[j]} {self.effect[i, j]:+.3g} "
                f"({100 * self.share[i, j]:.0f} %)"
                for j in order
            ]
            if corr is not None:
                parts.append(f"correlation {100 * corr[name]:+.0f} %")
            lines.append(f"{name.ljust(width)}   " + "   ".join(parts))
        return lines

    def _remark_block(self) -> list[str]:
        silent = [
            name
            for j, name in enumerate(self.parameters)
            if self.tolerance[name] > 0 and not np.any(self.jacobian[:, j])
        ]
        return [
            f"{name} has no effect on any quantity: check that the function reads it."
            for name in silent
        ]

    def _linearity_block(self) -> list[str]:
        if not self.checked:
            return ["Linearity not checked."]
        steps = ", ".join("±" + _g(self.tolerance[name], 4) for name in self.checked)
        where = ", ".join(self.checked)
        error = self.linear_error
        i, j = np.unravel_index(int(np.argmax(error)), error.shape)
        head = (
            f"Linearity, checked on {where} (forward solves at {steps}): linear "
            f"error at most {_g(error[i, j], 2)} ({self.quantities[i]})"
        )
        lines = []
        if self.nonlinear:
            lines.append(head + ".")
            for name in self.nonlinear:
                row = self.quantities.index(name)
                col = int(np.argmax(error[row]))
                limit = 3 * _NONLINEAR * self.std[name]
                lines.append(
                    f"{name}: nonlinear within ±{_g(self.tolerance[self.checked[col]], 4)} "
                    f"of {self.checked[col]} (linear error {_g(error[row, col], 2)} > "
                    f"{_g(limit, 2)}); its yield is not reliable."
                )
        else:
            lines.append(head + ", within a tenth of ±3σ for every quantity.")
        shifts = ", ".join(f"{name} {self.mean_shift[name]:+.2g}" for name in self.quantities)
        lines.append(f"Mean shift: {shifts}.")
        return lines


def _count(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _g(x: float, digits: int) -> str:
    return f"{x:.{digits}g}"


def _percent(y: float) -> str:
    return f"{100 * y:.1f} %"


def _spec(low: float | None, high: float | None) -> str:
    if low is None:
        return f"<= {_g(high, 4)}"  # ty: ignore[invalid-argument-type]
    if high is None:
        return f">= {_g(low, 4)}"
    return f"{_g(low, 4)} to {_g(high, 4)}"


def _table(header: list[str], rows: list[list[str]], markdown: bool) -> list[str]:
    """Lines of a table: padded columns, or Markdown pipes."""
    if markdown:

        def cell(text: str) -> str:
            return text.replace("|", r"\|")

        lines = [
            "| " + " | ".join(cell(c) for c in header) + " |",
            "|" + "|".join(" --- " for _ in header) + "|",
        ]
        lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
        return lines
    widths = [max(len(r[k]) for r in [header, *rows]) for k in range(len(header))]
    return [
        "   ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        for row in [header, *rows]
    ]


# --- inputs -------------------------------------------------------------------------


def _parameters(
    nominal: Mapping[str, float], tolerance: Mapping[str, float]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """The parameter names, nominal values and tolerances, as validated arrays."""
    if not isinstance(nominal, Mapping) or not nominal:
        msg = f"{_OWNER}: nominal must be a non-empty mapping of parameter name to value."
        raise TypeError(msg)
    if not isinstance(tolerance, Mapping):
        msg = f"{_OWNER}: tolerance must be a mapping of parameter name to ±t."
        raise TypeError(msg)
    for what, mapping in (("nominal", nominal), ("tolerance", tolerance)):
        if is_traced(dict(mapping)):
            msg = (
                f"{_OWNER} is not differentiable: its {what} is computed inside "
                "jax.jit, jax.grad or jax.vmap. Call it outside them; it returns "
                "NumPy values."
            )
            raise TypeError(msg)
    names = tuple(nominal)
    for name in names:
        if not isinstance(name, str):
            msg = f"{_OWNER}: parameter names must be str, got {name!r}."
            raise TypeError(msg)
    if set(tolerance) != set(names):
        msg = (
            f"{_OWNER}: tolerance has keys {sorted(map(str, tolerance))} but nominal "
            f"has keys {sorted(names)}: give a tolerance (0 holds a parameter) for "
            "every parameter."
        )
        raise ValueError(msg)
    x0 = np.empty(len(names))
    for k, name in enumerate(names):
        value = np.asarray(nominal[name])
        if value.shape != () or np.iscomplexobj(value):
            msg = (
                f"{_OWNER}: nominal[{name!r}] must be a real scalar, got shape "
                f"{value.shape}. A density design is not a tolerance parameter: "
                "use a FabricationModel bias."
            )
            raise ValueError(msg)
        if not np.isfinite(value):
            msg = f"{_OWNER}: nominal[{name!r}] must be finite, got {value}."
            raise ValueError(msg)
        x0[k] = value
    t = np.empty(len(names))
    for k, name in enumerate(names):
        value = np.asarray(tolerance[name], dtype=float)
        if value.shape != () or not np.isfinite(value) or value < 0:
            msg = (
                f"{_OWNER}: tolerance[{name!r}] must be a finite scalar, not "
                f"negative, got {tolerance[name]!r}."
            )
            raise ValueError(msg)
        t[k] = value
    return names, x0, t


def _specs(
    specs: Mapping[str, tuple[float | None, float | None]] | None,
) -> dict[str, tuple[float | None, float | None]]:
    """The specs with their windows checked, ``None`` for an open side."""
    out: dict[str, tuple[float | None, float | None]] = {}
    for name, window in (specs or {}).items():
        try:
            low, high = window
        except (TypeError, ValueError):
            msg = f"{_OWNER}: the spec of {name!r} must be (low, high), got {window!r}."
            raise ValueError(msg) from None
        if low is None and high is None:
            msg = (
                f"{_OWNER}: the spec of {name!r} is open on both sides: leave the "
                "quantity out of specs."
            )
            raise ValueError(msg)
        for limit in (low, high):
            if limit is not None and math.isnan(limit):
                msg = f"{_OWNER}: the spec of {name!r} has a NaN limit."
                raise ValueError(msg)
        if low is not None and high is not None and not low < high:
            msg = f"{_OWNER}: the spec of {name!r} needs low < high, got {window!r}."
            raise ValueError(msg)
        out[name] = (None if low is None else float(low), None if high is None else float(high))
    return out


def _correlation(
    correlation: Mapping[tuple[str, str], float] | None, names: tuple[str, ...]
) -> np.ndarray | None:
    """The correlation matrix of named pairs, ``None`` without any."""
    if not correlation:
        return None
    pairs: dict[frozenset[str], float] = {}
    for key, rho in correlation.items():
        try:
            a, b = key
        except (TypeError, ValueError):
            msg = f"{_OWNER}: a correlation key is a pair of parameter names, got {key!r}."
            raise ValueError(msg) from None
        for name in (a, b):
            if name not in names:
                msg = (
                    f"{_OWNER}: correlation names {name!r}, which is not a parameter "
                    f"(parameters {list(names)})."
                )
                raise ValueError(msg)
        if a == b:
            msg = f"{_OWNER}: correlation pairs {a!r} with itself."
            raise ValueError(msg)
        rho = float(rho)
        if not -1 <= rho <= 1:
            msg = (
                f"{_OWNER}: the correlation of {a!r} and {b!r} must be in [-1, 1], "
                f"got {rho}."
            )
            raise ValueError(msg)
        pair = frozenset(key)
        if pair in pairs and pairs[pair] != rho:
            msg = (
                f"{_OWNER}: the correlation of {a!r} and {b!r} is given twice "
                f"with different values ({pairs[pair]} and {rho})."
            )
            raise ValueError(msg)
        pairs[pair] = rho
    r = np.eye(len(names))
    for pair, rho in pairs.items():
        i, j = (names.index(n) for n in pair)
        r[i, j] = r[j, i] = rho
    if np.linalg.eigvalsh(r).min() < _PSD_FLOOR:
        msg = (
            f"{_OWNER}: the correlation matrix is not positive semidefinite "
            f"(smallest eigenvalue {np.linalg.eigvalsh(r).min():.3g}): the "
            "pairwise coefficients contradict each other."
        )
        raise ValueError(msg)
    return r


# --- evaluation ---------------------------------------------------------------------


def _read(out: Any) -> tuple[tuple[str, ...], jax.Array]:
    """The quantity names and their values stacked, from a function's output."""
    if not isinstance(out, Mapping):
        msg = (
            f"{_OWNER}: quantities must return a mapping of name to real scalar, "
            f"got {type(out).__name__}."
        )
        raise TypeError(msg)
    if not out:
        msg = f"{_OWNER}: quantities returned no quantity."
        raise ValueError(msg)
    values = []
    for name, value in out.items():
        if not isinstance(name, str):
            msg = f"{_OWNER}: quantity names must be str, got {name!r}."
            raise TypeError(msg)
        arr = jnp.asarray(value)
        if jnp.issubdtype(arr.dtype, jnp.complexfloating):
            msg = (
                f"{_OWNER}: quantity {name!r} is complex: take jnp.abs, .real, "
                "jnp.angle or s.db() to give a real quantity."
            )
            raise TypeError(msg)
        if arr.shape != ():
            msg = (
                f"{_OWNER}: quantity {name!r} has shape {arr.shape}: give one name "
                "per scalar (a dict comprehension over the array's entries)."
            )
            raise ValueError(msg)
        values.append(arr.astype(jnp.float64))
    return tuple(out), jnp.stack(values)


def _evaluate(
    quantities: Callable[[dict[str, jax.Array]], Mapping[str, jax.Array]],
    names: tuple[str, ...],
    x: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    out = quantities({n: jnp.asarray(x[k]) for k, n in enumerate(names)})
    q, vec = _read(out)
    return q, np.asarray(vec)


def _finite_values(q: tuple[str, ...], f: np.ndarray, where: str) -> None:
    bad = ~np.isfinite(f)
    if bad.any():
        msg = f"{_OWNER}: quantity {q[int(np.argmax(bad))]!r} {where} is not finite."
        raise ValueError(msg)


# --- statistics ---------------------------------------------------------------------


def _yield(f: float, s: float, low: float | None, high: float | None) -> float:
    """The probability that a Gaussian :math:`N(f, s^2)` lies in ``[low, high]``."""
    if s == 0:
        return float((low is None or f >= low) and (high is None or f <= high))
    a = -math.inf if low is None else (low - f) / s
    b = math.inf if high is None else (high - f) / s
    if a > 0:  # both limits in the upper tail: subtract the small tails, not 1 - 1
        return float(ndtr(-a) - ndtr(-b))
    return float(ndtr(b) - ndtr(a))


def _cpk(f: float, s: float, low: float | None, high: float | None) -> float:
    margins = []
    if low is not None:
        margins.append(f - low)
    if high is not None:
        margins.append(high - f)
    margin = min(margins)
    if s == 0:
        return math.inf if margin >= 0 else -math.inf
    return margin / (3 * s)


def _joint(
    f0: np.ndarray,
    jac: np.ndarray,
    sigma: np.ndarray,
    r: np.ndarray | None,
    low: np.ndarray,
    high: np.ndarray,
) -> float:
    """The fraction of samples of the linear model inside every window."""
    p = jac.shape[1]
    corr = np.eye(p) if r is None else r
    w, v = np.linalg.eigh(corr)
    root = v * np.sqrt(np.clip(w, 0.0, None))
    # f = f0 + J D L z, z standard normal
    a = jac @ (sigma[:, None] * root)
    rng = np.random.default_rng(_SEED)
    inside = 0
    for start in range(0, _SAMPLES, _CHUNK):
        z = rng.standard_normal((min(_CHUNK, _SAMPLES - start), p))
        f = f0 + z @ a.T
        inside += int(np.count_nonzero(np.all((f >= low) & (f <= high), axis=1)))
    return inside / _SAMPLES


def tolerance_report(
    quantities: Callable[[dict[str, jax.Array]], Mapping[str, jax.Array]],
    nominal: Mapping[str, float],
    *,
    tolerance: Mapping[str, float],
    specs: Mapping[str, tuple[float | None, float | None]] | None = None,
    correlation: Mapping[tuple[str, str], float] | None = None,
    check_linearity: int = 1,
) -> ToleranceReport:
    r"""The linearized spread, per-parameter effects and fabrication yield of named quantities.

    ``quantities`` is a JAX function of a dict of named real scalars (the
    parameters) returning a dict of named real scalars (the quantities); gfem
    differentiates it once at ``nominal`` with ``jax.jacrev``. Each parameter
    :math:`p_j` varies by :math:`\pm t_j`, read as three standard deviations of a
    Gaussian, :math:`\sigma_j = t_j/3`, so a foundry's "±10 nm" goes in as
    written. For a uniform spread :math:`\pm a`, give :math:`t = \sqrt{3}\,a`
    (the yield is then approximate).

    With the Jacobian :math:`J` and the covariance
    :math:`\Sigma = D R D` of the parameters (:math:`D = \mathrm{diag}(\sigma)`,
    :math:`R` the correlation matrix), the spread of quantity :math:`i` is
    :math:`s_i = \sqrt{(J \Sigma J^\top)_{ii}}`, and a spec window
    :math:`[l_i, u_i]` is met with probability
    :math:`\Phi((u_i - f_i)/s_i) - \Phi((l_i - f_i)/s_i)`. The joint yield of
    several specs is a Monte Carlo on the linear model with :math:`2^{18}`
    samples from ``numpy.random.default_rng(0)``, so it is reproducible. The
    linearity of the checked parameters is tested by two forward evaluations
    at :math:`p_j \pm t_j`.

    Cost: :math:`1 + 2k` forward evaluations (:math:`k` = ``check_linearity``)
    and one backward pass per quantity, whatever the number of parameters.
    The function is reverse-mode differentiated: it must go through
    `gfem.solve` or other reverse-mode code. ``tolerance_report`` itself is not
    differentiable; call it outside ``jax.jit`` and ``jax.grad``.

    Args:
        quantities: The forward model, ``p -> {name: real scalar}``, with ``p``
            a dict of the parameters of ``nominal``. Complex values and arrays
            are refused: take ``jnp.abs``, ``.real``, ``jnp.angle`` or ``s.db()``,
            and give one name per scalar.
        nominal: The nominal value of each named parameter, in the parameter's
            own unit.
        tolerance: The tolerance :math:`t_j \ge 0` of each parameter, same keys
            as ``nominal`` (0 holds a parameter).
        specs: ``{quantity: (low, high)}`` windows in the quantity's unit, ``None``
            for an open side. A quantity with no spec gets a spread and effects
            but no yield.
        correlation: Correlation coefficients of named pairs of parameters,
            ``{("a", "b"): 0.5}``; unlisted pairs are 0 and the order within a
            pair is free.
        check_linearity: How many parameters to check for linearity, the ones
            with the largest share of the variance; 0 turns the check off.

    Returns:
        The report: ``print`` it, or read its fields.

    Raises:
        TypeError: ``quantities`` returns no mapping, a non-str name or a complex
            value, or ``nominal`` or ``tolerance`` is traced.
        ValueError: A quantity is not a scalar or not finite, an input is
            non-finite, negative where it must not be, inconsistent (keys,
            specs, correlation not positive semidefinite) or out of range
            (``check_linearity``).
    """
    names, x0, t = _parameters(nominal, tolerance)
    n_params = len(names)
    spec_windows = _specs(specs)
    r = _correlation(correlation, names)
    if isinstance(check_linearity, bool) or not 0 <= check_linearity <= n_params:
        msg = (
            f"{_OWNER}: check_linearity must be an integer from 0 to the number of "
            f"parameters ({n_params}), got {check_linearity!r}."
        )
        raise ValueError(msg)

    seen: list[tuple[str, ...]] = []

    def stacked(x):
        q, vec = _read(quantities({n: x[k] for k, n in enumerate(names)}))
        seen.append(q)
        return vec, vec

    jac, f0 = jax.jacrev(stacked, has_aux=True)(jnp.asarray(x0, dtype=jnp.float64))
    qnames = seen[-1]
    jac, f0 = np.asarray(jac, dtype=float), np.asarray(f0, dtype=float)
    m = len(qnames)
    _finite_values(qnames, f0, "at the nominal point")
    for i, j in zip(*np.nonzero(~np.isfinite(jac)), strict=True):
        msg = (
            f"{_OWNER}: the derivative of quantity {qnames[i]!r} with respect to "
            f"parameter {names[j]!r} is not finite."
        )
        raise ValueError(msg)
    for name in spec_windows:
        if name not in qnames:
            msg = (
                f"{_OWNER}: spec for {name!r}, which is not a quantity "
                f"(quantities {list(qnames)})."
            )
            raise ValueError(msg)

    sigma = t / 3
    corr = np.eye(n_params) if r is None else r
    cov_p = sigma[:, None] * corr * sigma[None, :]
    var = np.maximum(np.einsum("ip,pq,iq->i", jac, cov_p, jac), 0.0)
    s = np.sqrt(var)
    contribution = (jac * sigma) ** 2
    safe = np.where(var > 0, var, 1.0)
    share = np.where(var[:, None] > 0, contribution / safe[:, None], 0.0)
    effect = jac * t
    if r is None:
        correlation_share = None
    else:
        rest = np.where(var > 0, (var - contribution.sum(axis=1)) / safe, 0.0)
        correlation_share = {q: float(rest[i]) for i, q in enumerate(qnames)}
    reach = (np.abs(jac) * t).sum(axis=1)

    yields = {}
    cpk = {}
    for name, (low, high) in spec_windows.items():
        i = qnames.index(name)
        yields[name] = _yield(f0[i], s[i], low, high)
        cpk[name] = _cpk(f0[i], s[i], low, high)

    joint = error = bounds = None
    if yields:
        ys = list(yields.values())
        bounds = (max(0.0, 1.0 - sum(1.0 - y for y in ys)), min(ys))
        if len(ys) == 1:
            joint, error = ys[0], 0.0
        else:
            rows = [qnames.index(name) for name in spec_windows]
            low = np.array([-math.inf if w[0] is None else w[0] for w in spec_windows.values()])
            high = np.array([math.inf if w[1] is None else w[1] for w in spec_windows.values()])
            joint = _joint(f0[rows], jac[rows], sigma, r, low, high)
            error = math.sqrt(joint * (1 - joint) / _SAMPLES)

    # The check: parameters ranked by their summed variance share, held ones out.
    score = share.sum(axis=0)
    ranked = [j for j in np.argsort(-score, kind="stable") if t[j] > 0]
    chosen = ranked[:check_linearity]
    linear_error = np.zeros((m, len(chosen)))
    shift = np.zeros(m)
    for c, j in enumerate(chosen):
        for sign in (1.0, -1.0):
            x = x0.copy()
            x[j] += sign * t[j]
            q, f = _evaluate(quantities, names, x)
            if q != qnames:
                msg = f"{_OWNER}: quantities returned different names at {names[j]!r} ± tolerance."
                raise ValueError(msg)
            _finite_values(qnames, f, f"at parameter {names[j]!r} = {x[j]:g}")
            linear_error[:, c] = np.maximum(
                linear_error[:, c], np.abs(f - f0 - sign * jac[:, j] * t[j])
            )
            shift += 0.5 * (f - f0) * sigma[j] ** 2 / t[j] ** 2
    mean_shift = {q: float(shift[i]) for i, q in enumerate(qnames)} if chosen else {}
    flagged = (
        (linear_error > 3 * _NONLINEAR * s[:, None]).any(axis=1)
        if chosen
        else np.zeros(m, dtype=bool)
    )

    return ToleranceReport(
        parameters=names,
        quantities=qnames,
        nominal={n: float(x0[k]) for k, n in enumerate(names)},
        tolerance={n: float(t[k]) for k, n in enumerate(names)},
        values={q: float(f0[i]) for i, q in enumerate(qnames)},
        jacobian=jac,
        effect=effect,
        share=share,
        correlation_share=correlation_share,
        std={q: float(s[i]) for i, q in enumerate(qnames)},
        worst_case={
            q: (float(f0[i] - reach[i]), float(f0[i] + reach[i]))
            for i, q in enumerate(qnames)
        },
        specs=spec_windows,
        yields=yields,
        cpk=cpk,
        joint_yield=joint,
        joint_yield_error=error,
        joint_yield_bounds=bounds,
        checked=tuple(names[j] for j in chosen),
        linear_error=linear_error,
        mean_shift=mean_shift,
        nonlinear=tuple(q for i, q in enumerate(qnames) if flagged[i]),
    )
