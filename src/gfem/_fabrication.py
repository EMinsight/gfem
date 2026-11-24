r"""The fabrication model: the printed design a process makes of a drawn one."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from gfem import _spectral
from gfem._checks import SLACK, is_traced, metres_hint, static_float

if TYPE_CHECKING:
    from gfem.design import LayerGrid, LayerPattern, Robust

#: Largest edge bias per blur width: beyond it the edge sits in the blur's tail,
#: where the threshold (below 0.023) is narrower than the projection's transition.
_MAX_BIAS_PER_SIGMA = 2.0


class FabricationModel:
    r"""A lithography process: the printed design of a drawn design.

    The drawn design :math:`\hat\rho` is what the optimizer designs and the
    layer pattern exports, ``project(blur(rho), beta)``. The process blurs it
    by a Gaussian of standard deviation :math:`\sigma` and prints where the
    blurred image :math:`I = G_\sigma * \hat\rho` reaches a threshold, so the
    printed design is

    .. math::

        \hat\rho_b = P_{\beta,\,\eta(b)}(G_\sigma * \hat\rho), \qquad
        \eta(b) = \tfrac12\operatorname{erfc}\frac{b}{\sqrt2\,\sigma},

    with :math:`P` the tanh projection of `project` at the drawn design's
    sharpness. A straight edge of the drawn design prints moved outwards by
    the edge bias :math:`b` (:math:`b > 0` grows the solid), and corners
    round off over about :math:`2\sigma`. Lines narrower than about
    :math:`3\sigma + 2\max(-b, 0)` print narrower than :math:`w + 2b`, and
    vanish below about :math:`2\sigma`; spaces are the mirror case.

    Solve the printed designs, export the drawn one::

        grid = gfem.design.LayerGrid(mesh, "design", pitch=7.5e-9)
        blur = gfem.design.helmholtz_filter(grid, radius=120e-9)
        fab = gfem.design.FabricationModel(grid, sigma=30e-9, bias=-5e-9)

        drawn = gfem.design.project(blur(rho), beta)
        eroded, nominal, dilated = fab.robust(drawn, beta=beta, tolerance=10e-9)

    The blur is the discrete heat kernel of the grid's second differences,
    :math:`\exp(\tfrac12\sigma^2\Delta_h)`: non-negative, of unit mass and of
    variance exactly :math:`\sigma^2` per axis, applied as one gain per mode
    in the cosine basis of `helmholtz_filter` (Fourier along a periodic
    axis). Its edges are those of the filter: a wall reflects, so a guide
    meeting it at right angles prints as :math:`w + 2b` up to the wall (a
    feature meeting it at an angle is mirrored, not continued), and a
    periodic seam wraps.

    The bias is exact to the drawn design's pixel quantization: at
    ``beta`` 32 a printed edge sits within a few hundredths of :math:`\sigma`
    of the drawn edge moved by :math:`b` when the pitch is at most
    :math:`\sigma/4` (about :math:`0.1\sigma` at :math:`\sigma/2`). Keep
    :math:`|b|` within :math:`1.5\sigma`; beyond it the projection's own
    width shows. Fixed geometry outside the grid (feed guides) is not
    biased, so a printed variant steps by :math:`b` per side against it.
    On a 2D model the blur is in the model's xy plane: lithography when
    that plane is the wafer's.

    Registered static: a model closes over into `jax.jit`. ``bias`` may be
    traced per call, for a derivative in the bias.

    Attributes:
        grid: The layer grid it acts on.
        sigma: The blur's standard deviation (m).
        bias: The edge bias (m), > 0 grows the solid.
    """

    def __init__(self, grid: LayerGrid, *, sigma: float, bias: float = 0.0) -> None:
        r"""Describe a process by its blur and edge bias.

        Args:
            grid: The layer grid of the drawn design (a volume layer, a
                horizontal sheet or a 2D model's area).
            sigma: The blur's standard deviation (m), at least the grid's
                pitch; a pitch of :math:`\sigma/4` or less holds the bias to
                a few hundredths of :math:`\sigma`.
            bias: The edge bias (m): how far a straight edge of the drawn
                design moves outwards when printed (negative for an
                over-etch), within :math:`2\sigma`.

        Raises:
            TypeError: ``grid`` is not a `LayerGrid` (a mesh).
            ValueError: ``sigma`` not > 0 or below the pitch, :math:`|b|`
                above :math:`2\sigma`, or ``sigma`` or ``bias`` computed
                inside `jax.grad`, `jax.jit` or `jax.vmap`.
        """
        # imported here: gfem.design imports this module
        from gfem.design import LayerGrid

        where = "FabricationModel"
        if not isinstance(grid, LayerGrid):
            raise TypeError(
                f"{where} takes a LayerGrid (lithography patterns a planar "
                f"layer), got {type(grid).__name__}. Lay a grid over the design "
                "region: FabricationModel(LayerGrid(mesh, region, pitch=...), "
                "sigma=...)."
            )
        sigma = static_float(sigma, f"{where} sigma", "The blur is fixed.")
        if not sigma > 0:
            raise ValueError(f"{where}: sigma must be > 0 m, got {sigma:g} m.")
        if sigma < grid.pitch * (1 - SLACK):
            raise ValueError(
                f"{where}: sigma={sigma:g} m is below the grid's pitch "
                f"{grid.pitch:g} m, so the blur is not resolved (printed edges off "
                "by up to 0.4 sigma). Refine the pitch to sigma / 4 or less "
                f"({sigma / 4:g} m)."
            )
        self.grid = grid
        self.sigma = sigma
        self.bias = static_float(
            bias, f"{where} bias", "Pass a traced bias per call: fab(rho, bias=...)."
        )
        self._check_bias(self.bias, "bias", where)
        # The heat kernel exp(sigma² Δ_h / 2): non-negative with variance sigma²,
        # where the sampled continuous spectrum would ring at a binary edge.
        self._blur = _spectral.function_of_laplacian(
            grid.shape, grid.pitch, grid.periodic, lambda k: np.exp(-0.5 * sigma**2 * k)
        )

    def __call__(self, rho: jax.Array, *, beta: float, bias: Any = None) -> jax.Array:
        r"""The printed design of a drawn design.

        Args:
            rho: The drawn design, the projected filtered density, shape
                ``grid.shape`` (on a 2D model the pixel array, not the
                `LayerDensity`).
            beta: The projection sharpness, > 0: the drawn design's.
            bias: The edge bias (m), the model's if ``None``. It may be
                traced; a concrete one must be within :math:`2\sigma`.

        Returns:
            :math:`P_{\beta,\,\eta(b)}(G_\sigma * \hat\rho)`, shape
            ``grid.shape``, for `LayerGrid.to_elements` or
            `LayerGrid.to_faces`.

        Raises:
            TypeError: ``rho`` is a `LayerDensity` (pass ``density.pixels``).
            ValueError: ``rho`` has the wrong shape or is complex, ``beta``
                not > 0, or :math:`|b|` above :math:`2\sigma`.
        """
        # imported here: gfem.design imports this module
        from gfem.design import project

        return project(self.image(rho), beta, self._threshold(bias))

    def robust(
        self, rho: jax.Array, *, beta: float, tolerance: float, bias: Any = None
    ) -> Robust:
        r"""The printed designs at the edge bias and :math:`\pm` a tolerance.

        The physical form of `robust`: the eroded, nominal and dilated
        designs are printed at biases :math:`b - t`, :math:`b` and
        :math:`b + t`, so their straight edges sit :math:`t` either side of
        the nominal one. Optimize the worst of the three objectives::

            printed = fab.robust(drawn, beta=beta, tolerance=10e-9)
            losses = jax.vmap(loss)(jnp.stack(printed))
            worst = 0.01 * jax.nn.logsumexp(losses / 0.01)

        Args:
            rho: The drawn design, shape ``grid.shape``.
            beta: The projection sharpness, > 0: the drawn design's.
            tolerance: The bias spread :math:`t` (m), > 0, with
                :math:`|b \pm t|` within :math:`2\sigma`. It may be traced.
            bias: The edge bias :math:`b` (m), the model's if ``None``. It
                may be traced.

        Returns:
            ``Robust(eroded, nominal, dilated)``.

        Raises:
            TypeError: ``rho`` is a `LayerDensity` (pass ``density.pixels``).
            ValueError: ``tolerance`` not > 0, :math:`|b|` or
                :math:`|b \pm t|` above :math:`2\sigma`, ``rho`` of the wrong
                shape or complex, or ``beta`` not > 0.
        """
        # imported here: gfem.design imports this module
        from gfem.design import Robust, project

        where = "FabricationModel.robust"
        b = self.bias if bias is None else bias
        if not is_traced(b):
            b = static_float(b, f"{where} bias", "")
            self._check_bias(b, "bias", where)
        t = tolerance
        if not is_traced(t):
            t = static_float(t, f"{where} tolerance", "")
            if not t > 0:
                raise ValueError(f"{where}: tolerance must be > 0 m, got {t:g} m.")
            if not is_traced(b):
                self._check_bias(b - t, "bias - tolerance", where)
                self._check_bias(b + t, "bias + tolerance", where)
        image = self.image(rho)
        return Robust(
            eroded=project(image, beta, self._eta(b - t)),
            nominal=project(image, beta, self._eta(b)),
            dilated=project(image, beta, self._eta(b + t)),
        )

    def image(self, rho: jax.Array) -> jax.Array:
        r"""The blurred drawn design, the image :math:`G_\sigma * \hat\rho`.

        Linear and self-adjoint: its gradient is the same blur.

        Args:
            rho: The drawn design, shape ``grid.shape``.

        Returns:
            The image, shape ``grid.shape``.

        Raises:
            TypeError: ``rho`` is a `LayerDensity` (pass ``density.pixels``).
            ValueError: ``rho`` has the wrong shape or is complex.
        """
        rho = self.grid._pixels(rho, "FabricationModel", real=True)  # noqa: SLF001
        return self._blur(rho.astype(jnp.float64))

    def threshold(self, bias: Any = None) -> float | jax.Array:
        r"""The print threshold :math:`\eta(b)` that moves an edge by the bias.

        .. math::

            \eta(b) = \tfrac12\operatorname{erfc}\frac{b}{\sqrt2\,\sigma}:

        the blurred image of a straight edge is the normal distribution
        function across it, which crosses :math:`\eta(b)` at a distance
        :math:`b` outside the edge. Exact in 2D, since the Gaussian's line
        marginal is the 1D Gaussian.

        Args:
            bias: The edge bias (m), the model's if ``None``; traced or
                within :math:`2\sigma`.

        Returns:
            :math:`\eta(b)`, a float for a concrete bias.

        Raises:
            ValueError: :math:`|b|` above :math:`2\sigma`.
        """
        return self._threshold(bias)

    def _threshold(self, bias: Any) -> Any:
        """`threshold`, typed for `project`, which takes a traced threshold too."""
        b = self.bias if bias is None else bias
        if not is_traced(b):
            b = static_float(b, "FabricationModel bias", "")
            self._check_bias(b, "bias", "FabricationModel")
        return self._eta(b)

    def pattern(self, rho: Any, *, bias: Any = None) -> LayerPattern:
        r"""The printed pattern: the layer pattern of the image at :math:`\eta(b)`.

        ``grid.pattern(fab.image(rho), eta=fab.threshold(bias))``: the
        prediction of what the process prints, for review, feature checks
        (`LayerPattern.check`: ``min_width`` on the eroded pattern,
        ``min_spacing`` on the dilated one) and a re-solve. The pattern to
        export stays the drawn one, `LayerGrid.pattern` of the filtered
        density. It contours the image, not the printed design, whose
        saturated pixels lose the sub-pixel edge. Not differentiable: call it
        on the final design, outside `jax.grad`.

        Args:
            rho: The drawn design, shape ``grid.shape``.
            bias: The edge bias (m), the model's if ``None``.

        Returns:
            The printed pattern, with the grid's region, bounds and seams.

        Raises:
            TypeError: ``rho`` is a `LayerDensity` (pass ``density.pixels``).
            ValueError: ``rho`` has the wrong shape, is complex or holds NaN,
                :math:`|b|` above :math:`2\sigma`, or ``rho`` or ``bias`` is
                computed inside `jax.grad`, `jax.jit` or `jax.vmap`.
        """
        return self.grid.pattern(self.image(rho), eta=self._threshold(bias))

    def _eta(self, b: Any) -> Any:
        r""":math:`\eta(b)`: a float, or a traced array (`project` takes both)."""
        if is_traced(b):
            return 0.5 * erfc(b / (math.sqrt(2) * self.sigma))
        return 0.5 * math.erfc(float(b) / (math.sqrt(2) * self.sigma))

    def _check_bias(self, b: float, name: str, where: str) -> None:
        """Raise when a concrete bias sits in the blur's tail."""
        limit = _MAX_BIAS_PER_SIGMA * self.sigma
        if abs(b) > limit * (1 + SLACK):
            raise ValueError(
                f"{where}: {name}={b:g} m is beyond 2 sigma ({limit:g} m): the "
                "edge would sit in the blur's tail, threshold below 0.023, "
                "narrower than the projection's transition. Keep the bias "
                f"within 1.5 sigma ({1.5 * self.sigma:g} m).{metres_hint(b)}"
            )

    def __repr__(self) -> str:
        return (
            f"FabricationModel({self.grid!r}, sigma={self.sigma:g} m, "
            f"bias={self.bias:g} m)"
        )


jax.tree_util.register_static(FabricationModel)
