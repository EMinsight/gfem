r"""The static field and the mode weights of an electro-optic shift.

`StaticFieldKernel` reads the electrostatic field
:math:`E_\mathrm{s} = -\nabla V` of a potential at fixed physical points of
the optical model's tetrahedra (a quadrature rule of each), located in the
potential's mesh, and gives each tetrahedron's quadrature mean:

.. math::

    E_{\mathrm{s},t} = -\sum_q w_q \nabla V(x_q), \qquad \sum_q w_q = 1

It is linear in the potential's coefficients, ``E = G c``, and runs behind
gfem's gradient wrapper (`gfem._adjoint.measure`) with inputs
``coefficients`` and ``nodes``: the backward pass is ``G^T``. The points stay
where they are when the potential's mesh nodes move, so a node derivative is
the field's change at fixed points.

`ElectroOpticWeightsRule` is the port mode solve of the shift: the real
symmetric 3x3 weight per section triangle, its refusals naming the
electro-optic shift.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np

from gfem._engine import probe as _probe
from gfem._engine.thermo_optic import ShiftWeightsRule
from gfem.mesh import Mesh

__all__ = ["ElectroOpticWeightsRule", "GradientProbe", "StaticFieldKernel"]


class ElectroOpticWeightsRule(ShiftWeightsRule):
    """The port mode solve of an electro-optic shift, and its 3x3 section weights."""

    def unsupported_input(self, kind: tuple) -> str | None:
        """Every input the port mode problem reads: the mode is held constant."""
        del kind
        return (
            "an electro-optic shift holds the port mode constant (first order in "
            "the static field), and its derivative with respect to the optical "
            "model's permittivity, losses, permeability, node coordinates or "
            "boundary values would need the mode's own derivative, which this "
            "version does not have (a frozen mode gives a wrong value). "
            "Differentiate through the potential or the electro_optic_tensor"
        )

    def nondifferentiable_output(self, name: str) -> str:
        """The error message for a cotangent on output ``name``."""
        return (
            f"gfem cannot differentiate the electro-optic shift's mode output "
            f"{name!r} with respect to the optical model: the mode is a constant "
            "of the shift. Wrap the input in jax.lax.stop_gradient to use its "
            "value without its derivative."
        )


class GradientProbe(_probe.Probe):
    r"""The gradient :math:`\nabla V` of a scalar H1 layout at fixed points.

    `values` gives ``(m, n_points, 3)`` gradients in V/m and `transpose`
    its transpose, as `gfem._engine.probe.Probe` does for values.
    """

    def __init__(self, mesh: Mesh, layout: Any, located: Any, comm: Any = None) -> None:
        """Build the engine space and the points (the mesh's node positions)."""
        super().__init__(mesh, layout, located, comm)
        # d/dx of the engine's scaled coordinates
        self.factor = 1.0 / self._built.scale

    def _evaluate(self, fes: ngs.FESpace, values: np.ndarray) -> np.ndarray:
        """Gradients at this rank's points, ``(its points, 3)``."""
        if self._mips is None:
            return np.zeros((0, 3))
        gf = ngs.GridFunction(fes)
        gf.vec.FV().NumPy()[:] = values
        out = np.asarray(ngs.grad(gf)(self._mips))
        return out.reshape(self._own.size, -1)


class StaticFieldKernel:
    """The mean of ``-grad V`` over each optical tetrahedron: ``G c`` and ``G^T``.

    The points are fixed physical points; at given node coordinates of the
    potential's mesh they are located in it (or taken as given, ``located``,
    for the nodes they were located at).
    """

    name = "the electro-optic shift's static field"
    pieces = frozenset({"coefficients"})
    comm = None

    def __init__(
        self,
        topology: Any,
        layout: Any,
        points: np.ndarray,
        weights: np.ndarray,
        *,
        located: Any = None,
        nodes: np.ndarray | None = None,
    ) -> None:
        """The points and their rule.

        Args:
            topology: The potential's mesh topology.
            layout: The potential's `gfem._engine.probe.Layout`.
            points: ``(n * q, 3)`` points, m, element-major.
            weights: ``(q,)`` rule weights summing to 1.
            located: The points located at ``nodes``, or ``None``.
            nodes: The node coordinates ``located`` holds for.
        """
        self.topology = topology
        self.layout = layout
        self.points = np.asarray(points, float)
        self.weights = np.asarray(weights, float)
        self.n = len(self.points) // len(self.weights)
        self._located = located
        self._nodes = None if nodes is None else np.asarray(nodes)
        self._probes: dict[int, GradientProbe] = {}
        self.output_specs = {"field": jax.ShapeDtypeStruct((self.n, 3), jnp.float64)}

    def role(self, name: str) -> str:
        del name
        return "read"

    def unsupported(self, name: str) -> str | None:
        if name == "nodes":
            return (
                "the static field's derivative at the optical points with "
                "respect to the electrostatic mesh's node coordinates is not "
                "available in this version"
            )
        return None

    def describe(self, name: str) -> str:
        return {
            "nodes": "the electrostatic mesh node coordinates (potential.mesh.nodes)",
        }.get(name, f"the potential {name}")

    def _probe(self, nodes: np.ndarray) -> GradientProbe:
        key = _probe._digest(nodes)  # noqa: SLF001
        found = self._probes.get(key)
        if found is None:
            mesh = Mesh(nodes, self.topology)
            located = self._located
            held = self._nodes is not None and np.array_equal(nodes, self._nodes)
            if located is None or not held:
                located = _probe.locate(mesh, self.layout, self.points)
            found = self._probes[key] = GradientProbe(mesh, self.layout, located)
        return found

    def forward(self, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        grad = self._probe(values["nodes"]).values(values["coefficients"])[0]
        grad = grad.reshape(self.n, len(self.weights), 3)
        return {"field": -np.einsum("q,nqd->nd", self.weights, grad)}

    def backward(
        self,
        values: dict[str, np.ndarray],
        cotangents: dict[str, np.ndarray],
        wanted: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if wanted != ("coefficients",):  # the measure wrapper raised for others
            raise NotImplementedError(f"no static field gradient for {wanted}")
        p = self._probe(values["nodes"])
        ct = np.asarray(cotangents["field"]).reshape(self.n, 3)
        g = -np.einsum("q,nd->nqd", self.weights, ct).reshape(1, -1, 3)
        out = p.transpose(g)
        return {"coefficients": out.reshape(values["coefficients"].shape)}
