"""Length-scale constraints on a layer grid, and the robust triple's filter radius."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from gfem._checks import is_traced, static_float

if TYPE_CHECKING:
    from gfem.design import HelmholtzFilter, LayerGrid

#: Sharpness of the solid / void indicator inside the constraints. It only says
#: which side of the threshold a pixel's filtered value lies on, the question the
#: exported contour answers: from 16 on the values match a step to three digits,
#: and 32 changes most smoothly when a feature appears or vanishes. It does not
#: follow the optimizer's continuation: at 8 grey void would count as solid.
_INDICATOR_BETA = 32.0

#: The decay of the inflection-point weight, in units of the squared filter
#: radius (the PDE filter's value of Arrieta, Romano and Johnson 2025).
_DECAY_PER_RADIUS_SQ = 10.0

#: Largest rule per filter radius: beyond it the calibration lets lines far
#: narrower than the rule pass (at 1.5 R, a line of 0.8 of the rule).
_MAX_RULE_PER_RADIUS = 1.25

#: Coarsest pitch per filter radius: at R/2 the threshold drifts to 1.3 times
#: the rule; from R/3 on it is within a few percent.
_PITCHES_PER_RADIUS = 3

#: Relative slack on the range limits (round-off of lengths given as fractions).
_SLACK = 1e-9


class LengthScaleValues(NamedTuple):
    """The solid and void length-scale constraint values of one design.

    A named tuple: ``jnp.stack(values)`` gives the 2-vector an optimizer's
    inequality constraint takes.

    Attributes:
        solid: At most 0 when the printed lines are at least ``min_width``
            wide.
        void: At most 0 when the printed spaces are at least ``min_spacing``
            wide.
    """

    solid: jax.Array
    void: jax.Array


class LengthScaleConstraints:
    r"""Minimum width and spacing of a layer grid's design, as two constraints.

    Call it on the filtered density (``rules(blur(rho))``);
    `length_scale_constraints` builds it. It returns
    ``LengthScaleValues(solid, void)``, each at most 0 when the printed
    pattern, the contour of the filtered density at ``eta``, keeps its rule.

    Attributes:
        min_width: The minimum width of a printed line (m).
        min_spacing: The minimum width of a printed space (m).
        eta: The threshold the pattern is printed at.
        radius: The filter radius (m).
        eta_solid: :math:`\eta_e`, the lowest filtered value a line of width
            ``min_width`` keeps along its middle.
        eta_void: :math:`\eta_d`, the highest filtered value a space of width
            ``min_spacing`` keeps along its middle.
        decay: :math:`c` (m²), how fast the weight of a pixel falls with the
            squared gradient of the filtered density.
        epsilon_solid: :math:`\epsilon_s`, the solid constraint's scale.
        epsilon_void: :math:`\epsilon_v`, the void constraint's scale.
    """

    def __init__(
        self,
        blur: HelmholtzFilter,
        *,
        min_width: float,
        min_spacing: float,
        eta: float,
    ) -> None:
        """Derive the thresholds; use `length_scale_constraints` to build one."""
        grid = _grid_of(blur)
        radius = blur.radius
        self.min_width = min_width
        self.min_spacing = min_spacing
        self.eta = eta
        self.radius = radius
        self.eta_solid = 1 - (1 - eta) / math.cosh(math.sqrt(3) * min_width / radius)
        self.eta_void = eta / math.cosh(math.sqrt(3) * min_spacing / radius)
        self.decay = _DECAY_PER_RADIUS_SQ * radius**2
        n_sampled = int(np.sum(grid.sampled))
        area = n_sampled * grid.pitch**2
        # One critical feature as long as the design is wide: the value of a
        # line printed exactly at its rule (Arrieta, Romano and Johnson 2025,
        # Eq. 55), with the curvature of the filtered profile across it.
        scale = (
            radius
            / math.sqrt(area)
            * 3
            * math.sqrt(math.pi)
            / (128 * _DECAY_PER_RADIUS_SQ**2.5)
        )
        self.epsilon_solid = scale / (6 * (1 - self.eta_solid)) ** 3
        self.epsilon_void = scale / (6 * self.eta_void) ** 3
        self._grid = grid
        self._mask = None if grid.sampled.all() else grid.sampled.astype(np.float64)
        self._n_sampled = n_sampled

    def __call__(self, rho: jax.Array) -> LengthScaleValues:
        r"""The constraint values of a filtered density.

        Args:
            rho: The filtered density, ``blur(rho)``, shape ``grid.shape``.

        Returns:
            ``LengthScaleValues(solid, void)``, each
            :math:`g/\epsilon - 1`, at most 0 when the rule holds.

        Raises:
            ValueError: ``rho`` has the wrong shape or is complex.
        """
        # imported here: gfem.design imports this module
        from gfem.design import project

        grid = self._grid
        rho = jnp.asarray(rho)
        if rho.shape != grid.shape:
            raise ValueError(
                f"length_scale_constraints: rho has shape {rho.shape}, but the "
                f"grid on {grid.region!r} has shape {grid.shape} (ny, nx). Pass "
                "the filtered pixel array, blur(rho)."
            )
        if jnp.iscomplexobj(rho):
            raise ValueError("length_scale_constraints: densities must be real.")
        rho = rho.astype(jnp.float64)
        solid = project(rho, _INDICATOR_BETA, self.eta)
        weight = jnp.exp(-self.decay * _gradient_squared(rho, grid))
        short = jnp.minimum(rho - self.eta_solid, 0.0) ** 2
        narrow = jnp.minimum(self.eta_void - rho, 0.0) ** 2
        g_solid = self._mean(solid * weight * short)
        g_void = self._mean((1 - solid) * weight * narrow)
        return LengthScaleValues(
            solid=g_solid / self.epsilon_solid - 1,
            void=g_void / self.epsilon_void - 1,
        )

    def _mean(self, values: jax.Array) -> jax.Array:
        """The mean over the sampled pixels (the others are not the device)."""
        if self._mask is None:
            return jnp.mean(values)
        return jnp.sum(values * self._mask) / self._n_sampled

    def __repr__(self) -> str:
        return (
            f"LengthScaleConstraints({self._grid!r}, min_width={self.min_width:g} m, "
            f"min_spacing={self.min_spacing:g} m, eta={self.eta:g})"
        )


jax.tree_util.register_static(LengthScaleConstraints)


def _gradient_squared(rho: jax.Array, grid: LayerGrid) -> jax.Array:
    """Squared gradient of a pixel array, central differences on pixel centres.

    The ghost pixels follow the filter's edges: at a wall the edge value (the
    half-sample mirror of the zero-flux filter), across a seam the other edge.
    """
    out = jnp.zeros_like(rho)
    for axis, wrap in ((0, grid.periodic[1]), (1, grid.periodic[0])):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (1, 1)
        padded = jnp.pad(rho, pad, mode="wrap" if wrap else "edge")
        n = padded.shape[axis]
        hi = jax.lax.slice_in_dim(padded, 2, n, axis=axis)
        lo = jax.lax.slice_in_dim(padded, 0, n - 2, axis=axis)
        out = out + ((hi - lo) / (2 * grid.pitch)) ** 2
    return out


def length_scale_constraints(
    blur: HelmholtzFilter, *, min_width: float, min_spacing: float, eta: float = 0.5
) -> LengthScaleConstraints:
    r"""Minimum width and spacing rules as two differentiable constraints.

    The printed pattern is the contour of the filtered density
    :math:`\tilde\rho` at ``eta``. A line printed at least ``min_width``
    wide keeps :math:`\tilde\rho \ge \eta_e` along its middle, where
    :math:`\tilde\rho` is flat; a space at least ``min_spacing`` wide keeps
    :math:`\tilde\rho \le \eta_d`. The constraints measure the shortfall
    there (Zhou, Lazarov, Wang and Sigmund 2015):

    .. math::

        g_s = \frac{1}{|\Omega|}\int_\Omega \hat\rho\,
        e^{-c|\nabla\tilde\rho|^2} \min(\tilde\rho - \eta_e, 0)^2,
        \qquad
        g_v = \frac{1}{|\Omega|}\int_\Omega (1 - \hat\rho)\,
        e^{-c|\nabla\tilde\rho|^2} \min(\eta_d - \tilde\rho, 0)^2,

    with :math:`\hat\rho` the tanh projection of :math:`\tilde\rho` at a fixed
    sharpness 32, whatever the optimizer's current one. The returned values
    are :math:`g/\epsilon - 1`, at most 0 when the rule holds. The
    parameters follow from the filter radius :math:`R` with no tuning
    (Arrieta, Romano and Johnson 2025, for the PDE filter, here with
    separate width and spacing and any threshold):

    .. math::

        \eta_e = 1 - (1 - \eta)\operatorname{sech}\frac{\sqrt3\,w}{R},\quad
        \eta_d = \eta\operatorname{sech}\frac{\sqrt3\,s}{R},\quad
        c = 10R^2,\quad
        \epsilon = \frac{R}{\sqrt{|\Omega|}}
        \frac{3\sqrt\pi}{128 \cdot 10^{5/2}}\,\gamma^{-3},

    with :math:`w` = ``min_width``, :math:`s` = ``min_spacing``,
    :math:`\gamma = 6(1-\eta_e)` for solid and :math:`6\eta_d` for void, and
    :math:`|\Omega|` the area of the grid's sampled pixels (the means run
    over them only; unsampled pixels are not part of the device).

    What :math:`g \le 0` guarantees: straight lines and spaces print at
    least about 0.97 of their rule (0.94 at ``eta`` other than 1/2); a
    compact feature (a dot, a hole) must print about 1.1 times it. After an
    optimization a few percent of a phase's area can still sit below the
    rule: to clear a foundry rule, constrain about 10 % above it and confirm
    on the pattern with `LayerPattern.check`.

    Switch the constraints on once the topology has formed, at the last
    sharpness of the continuation (16 or more), never from a grey start::

        blur = gfem.design.helmholtz_filter(grid, radius=165e-9)
        rules = gfem.design.length_scale_constraints(
            blur, min_width=165e-9, min_spacing=165e-9
        )


        def constraints(rho):  # each entry <= 0 when held
            return jnp.stack(rules(blur(rho)))

    The gradient of :math:`\tilde\rho` is a central difference that follows
    the filter's edges: across a seam of a periodic grid it wraps, and at a
    wall it sees the mirror image, so half a line against a wall reads as a
    whole one. Neighbours outside the grid (a feed guide) are not seen: pin
    them as pixels inside the grid. On a 2D model the lengths are in the
    model's xy plane.

    Args:
        blur: The filter the design is filtered with, built on a `LayerGrid`
            by `helmholtz_filter`.
        min_width: The minimum width of a printed line (m), > 0, at most
            1.25 times the filter radius.
        min_spacing: The minimum width of a printed space (m), > 0, at most
            1.25 times the filter radius.
        eta: The threshold the pattern is printed at, in (0, 1), the same as
            the pattern's.

    Returns:
        The constraints, a fixed callable on the filtered density.

    Raises:
        NotImplementedError: ``blur`` filters a free-form ``(mesh, region)``
            design.
        TypeError: ``blur`` is not a filter, or a rule is not a real number.
        ValueError: A rule not finite or not > 0, a rule above 1.25 times
            the filter radius, a grid pitch above a third of the radius, or
            ``eta`` outside (0, 1).
    """
    # imported here: gfem.design imports this module
    from gfem.design import HelmholtzFilter

    where = "length_scale_constraints"
    if not isinstance(blur, HelmholtzFilter):
        raise TypeError(
            f"{where}: blur is the design's filter, "
            f"helmholtz_filter(grid, radius=...), got {type(blur).__name__}."
        )
    grid = _grid_of(blur)
    rules = {
        "min_width": _length(min_width, "min_width", where),
        "min_spacing": _length(min_spacing, "min_spacing", where),
    }
    eta = static_float(eta, f"{where} eta", "The threshold is fixed.")
    if not 0 < eta < 1:
        raise ValueError(f"{where}: the threshold eta must be in (0, 1), got {eta}.")
    radius = blur.radius
    needed = max(rules.values()) / _MAX_RULE_PER_RADIUS
    for name, value in rules.items():
        if value > _MAX_RULE_PER_RADIUS * radius * (1 + _SLACK):
            raise ValueError(
                f"{where}: {name}={value:g} m is more than "
                f"{_MAX_RULE_PER_RADIUS} times the filter radius {radius:g} m; "
                "there the constraints let features far narrower than the rule "
                f"pass. Raise the filter radius to at least {needed:g} m "
                f"(max(min_width, min_spacing) / {_MAX_RULE_PER_RADIUS})."
                f"{_metres_hint(value)}"
            )
    coarsest = radius / _PITCHES_PER_RADIUS
    if grid.pitch > coarsest * (1 + _SLACK):
        raise ValueError(
            f"{where}: the grid's pitch {grid.pitch:g} m is coarser than a third "
            f"of the filter radius (radius / 3 = {coarsest:g} m), where the "
            "constraints stop measuring the printed width. Lay a finer grid "
            f"(pitch <= {coarsest:g} m) or raise the filter radius to at least "
            f"{_PITCHES_PER_RADIUS * grid.pitch:g} m."
        )
    return LengthScaleConstraints(
        blur, min_width=rules["min_width"], min_spacing=rules["min_spacing"], eta=eta
    )


def _grid_of(blur: HelmholtzFilter) -> LayerGrid:
    """The pixel grid a filter acts on, or raise for a free-form design."""
    grid = blur._grid  # noqa: SLF001
    if grid is None:
        raise NotImplementedError(
            "length_scale_constraints: the constraints need the gradient of the "
            "filtered density on a planar pixel grid; a free-form (mesh, region) "
            "design has none. Lay a LayerGrid over the region and filter on it: "
            "helmholtz_filter(LayerGrid(mesh, region, pitch=...), radius=...)."
        )
    return grid


def radius_for_length_scale(length: float, *, delta: float, eta: float = 0.5) -> float:
    r"""The filter radius whose robust triple holds a minimum length.

    With `robust` at thresholds :math:`\eta \pm \delta`, a line whose eroded
    design survives prints at least :math:`\ell` wide in the nominal design,
    and a space whose dilated design stays open prints at least
    :math:`\ell`, when the filter radius is

    .. math::

        R = \frac{\sqrt3\,\ell}{\min\left(
        \operatorname{arccosh}\frac{1-\eta}{1-\eta-\delta},\
        \operatorname{arccosh}\frac{\eta}{\eta-\delta}\right)}

    (exact for straight features under `helmholtz_filter`). At
    :math:`\eta = 1/2`, :math:`\delta` = 0.1 gives :math:`R = 2.50\,\ell`
    and :math:`\delta` = 0.25 gives :math:`R = 1.32\,\ell`::

        radius = gfem.design.radius_for_length_scale(150e-9, delta=0.1)
        blur = gfem.design.helmholtz_filter(grid, radius=radius)
        eroded, nominal, dilated = gfem.design.robust(
            blur(rho), beta=beta, delta=0.1
        )

    Args:
        length: The minimum length :math:`\ell` of both phases (m), > 0.
        delta: The threshold shift of `robust`, > 0, with ``eta - delta``
            and ``eta + delta`` in (0, 1).
        eta: The nominal threshold.

    Returns:
        The filter radius (m).

    Raises:
        TypeError: ``length`` is not a real number.
        ValueError: ``length`` not finite or not > 0, ``delta <= 0``, or a
            threshold outside (0, 1).
    """
    where = "radius_for_length_scale"
    ell = _length(length, "length", where)
    remedy = "The robust thresholds are fixed."
    d = static_float(delta, f"{where} delta", remedy)
    e = static_float(eta, f"{where} eta", remedy)
    if not d > 0 or not 0 < e - d < e + d < 1:
        raise ValueError(
            f"{where}: delta must be > 0 with eta - delta and eta + delta in "
            f"(0, 1), got eta={e}, delta={d}."
        )
    solid = math.acosh((1 - e) / (1 - e - d))
    void = math.acosh(e / (e - d))
    return math.sqrt(3) * ell / min(solid, void)


def _length(value: Any, name: str, where: str) -> float:
    """A length rule as a finite float > 0 (m), or raise."""
    if not is_traced(value) and (
        isinstance(value, bool | np.bool_)
        or np.ndim(value) != 0
        or np.asarray(value).dtype.kind not in "iuf"
    ):
        raise TypeError(
            f"{where}: {name} is a length in metres, a real number, got {value!r}."
        )
    out = static_float(
        value, f"{where} {name}", "A length rule is fixed: build it from numbers."
    )
    if not out > 0:
        raise ValueError(f"{where}: {name} must be > 0 m, got {out:g} m.")
    return out


def _metres_hint(value: float) -> str:
    """A reminder of the unit when a length looks like micrometres."""
    return " The lengths are in metres." if value >= 1e-3 else ""
