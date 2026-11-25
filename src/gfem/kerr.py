r"""The Kerr nonlinearity of a port mode and the coupling between its waves.

A material's ``nonlinear_index`` (:math:`n_2`, m²/W) and
``two_photon_absorption`` (:math:`\beta_\mathrm{TPA}`, m/W) act on the mode
of a wave port that `gfem.solve` found with `gfem.PortMode`. Two functions
read that result and the model's materials, on every row of its frequency
grid (reduced rows included), so a curve over frequency costs no extra solve:

- `kerr_nonlinearity`: the nonlinear coefficient :math:`\gamma_\mathrm{NL}`
  of each row, the number a pulse-propagation tool takes;
- `kerr_coupling`: the tensor :math:`\gamma_{abcd}` between the waves (the
  rows) of one or several results: self- and cross-phase modulation and
  four-wave mixing.

Both integrate the mode at 1 W on the port's cross-section. They run on
lossy and leaky sections, where they hold while :math:`\alpha \ll \beta`.
Under `jax.grad` they read the traced mode field, so they differentiate with
respect to the section's materials (not the node coordinates).
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem._checks import is_traced
from gfem._materials import Material, MaterialField, PermittivityLaw
from gfem._pml import layer_tetrahedra
from gfem._pytree import register_fields
from gfem.errors import NonDifferentiableError
from gfem.results import PortModeResult

if TYPE_CHECKING:
    from gfem.model import Model

__all__ = ["KerrCoupling", "KerrNonlinearity", "kerr_coupling", "kerr_nonlinearity"]

_C = scipy.constants.c
_EPS0 = scipy.constants.epsilon_0
_PROPERTIES = ("nonlinear_index", "two_photon_absorption")

_NODES = (
    "{owner} reads the mode at fixed quadrature points while gfem differentiates "
    "the node coordinates: the derivative of an overlap on a moving section is "
    "not available in this version. Differentiate with respect to the materials, "
    "or wrap the node coordinates in jax.lax.stop_gradient."
)


@dataclasses.dataclass(frozen=True, eq=False)
class KerrNonlinearity:
    r"""The nonlinear coefficient of a port mode over frequency.

    :math:`\gamma_\mathrm{NL} = \overline{\gamma_K}` with, for the mode
    :math:`E` at 1 W, :math:`k_0 = \omega/c` and :math:`n^2 =
    \mathrm{Re}\,\varepsilon_r` of each element,

    .. math::

        \gamma_K = \frac{k_0 c^2 \varepsilon_0^2}{12} \int_S n^2\,\tilde n_2
        \left[2 |E|^4 + |E\cdot E|^2\right] \mathrm{d}A,
        \qquad \tilde n_2 = n_2 - \frac{i \beta_\mathrm{TPA}}{2 k_0},

    over all three components of :math:`E` (the unconjugated product
    :math:`E\cdot E` too). The power-dependent changes of the propagation
    constant are therefore :math:`\delta\beta = \mathrm{Re}\,\gamma_\mathrm{NL}\,P`
    and :math:`\delta\alpha = \mathrm{Im}\,\gamma_\mathrm{NL}\,P \ge 0`, the
    convention of the generalized nonlinear Schrodinger equation.

    Attributes:
        port: The wave port's name.
        mode: Which mode (1 = the most propagating).
        frequency: ``(n_freq,)`` Hz.
        gamma_nl: :math:`\gamma_\mathrm{NL}` at each frequency, ``(n_freq,)``
            complex, 1/(W m); 1/W on a line port of a 2D mesh (the field is
            then 1 W/m and the integral runs over the line). The real part
            is the Kerr effect, the imaginary part the two-photon loss.
    """

    port: str
    mode: int
    frequency: np.ndarray
    gamma_nl: jax.Array

    def __repr__(self) -> str:
        n_f = np.shape(self.frequency)[-1]
        head = (
            f"KerrNonlinearity(port {self.port!r}, mode {self.mode}, {n_f} frequencies"
        )
        try:
            g = np.asarray(self.gamma_nl)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, gamma_nl Re {_range(g.real)}, Im {_range(g.imag)} 1/(W m))"


register_fields(KerrNonlinearity, ("gamma_nl",))


@dataclasses.dataclass(frozen=True, eq=False)
class KerrCoupling:
    r"""The Kerr coupling tensor between the waves of one port.

    A wave is one row of a `gfem.PortModeResult`: a mode at a frequency. In
    the equation of wave :math:`a`'s amplitude (:math:`|A|^2` a power,
    :math:`e^{-i\omega t}`, the convention of pulse-propagation tools),

    .. math::

        \frac{\mathrm{d}A_a}{\mathrm{d}z} \ni i \sum_{bcd}
        \gamma_{abcd}\, A_b A_c \overline{A_d},

    summed over every index tuple with no degeneracy factors, so the factor 2
    of cross-phase modulation and of four-wave mixing comes out of the sum.
    :math:`\gamma_{abcd} = \overline{T_{abcd}}`,

    .. math::

        T_{abcd} = \frac{\omega_a \varepsilon_0}{16} \int_S \chi_a\,
        S(E_b, E_c, \overline{E_a}, \overline{E_d})\,\mathrm{d}A,
        \quad
        \chi_a = \tfrac43 n_a^2 c \varepsilon_0
        \left(n_2 - \frac{i \beta_\mathrm{TPA} c}{2 \omega_a}\right),

    :math:`S(a,b,c,d) = (a\cdot b)(c\cdot d) + (a\cdot c)(b\cdot d) +
    (a\cdot d)(b\cdot c)` with unconjugated dot products, :math:`E` the modes
    at 1 W and :math:`n_a^2 = \mathrm{Re}\,\varepsilon_r` at wave :math:`a`'s
    frequency. :math:`\gamma_{aaaa}` is `KerrNonlinearity`'s
    :math:`\gamma_\mathrm{NL}`. Every tuple is computed; an equation uses
    only those that conserve energy (:math:`\omega_a + \omega_b = \omega_c +
    \omega_d` up to ordering).

    Attributes:
        port: The wave port's name.
        waves: ``(mode, frequency)`` of each wave, in order (Hz).
        gamma_nl: :math:`\gamma_{abcd}`, ``(K, K, K, K)`` complex, 1/(W m);
            1/W on a line port of a 2D mesh.
    """

    port: str
    waves: tuple[tuple[int, float], ...]
    gamma_nl: jax.Array

    def __repr__(self) -> str:
        head = f"KerrCoupling(port {self.port!r}, {len(self.waves)} waves"
        try:
            g = np.asarray(self.gamma_nl)
        except (TypeError, jax.errors.TracerArrayConversionError):
            return head + ")"
        return f"{head}, |gamma_nl| up to {np.abs(g).max():.4g} 1/(W m))"


register_fields(KerrCoupling, ("gamma_nl",))


def _range(a: np.ndarray) -> str:
    lo, hi = float(a.min()), float(a.max())
    return f"{lo:.4g}" if lo == hi else f"{lo:.4g}-{hi:.4g}"


# --- quadrature columns ----------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Columns:
    """The mode field at the points of a degree-4p rule, per element of the section.

    ``dofs (n_el, k)`` are the DOF numbers of each element's tangential
    (Nedelec) part, ``weights (n_el, n_q, n_c, k)`` real, so that component
    ``c`` at point ``q`` of element ``e`` is ``weights[e, q, c] @ x[dofs[e]]``
    (``n_c`` = 2 for ``(E_u, E_v)`` of a planar port, 1 for the scalar of a
    line port). A planar port's normal component ``E_n`` lives on the other
    DOFs of the element: ``normal_dofs (n_el, k_n)`` and ``normal_weights
    (n_el, n_q, k_n)``, ``None`` on a line. Both are jax arrays, put on the
    device once. ``measure (n_el, n_q)`` is the weight of each point in m² (m
    on a line). Elements are in the order of the section's triangles (segments).
    """

    dofs: jax.Array
    weights: jax.Array
    normal_dofs: jax.Array | None
    normal_weights: jax.Array | None
    measure: np.ndarray


def _coo(form: Any, shape: tuple[int, int]) -> sp.csr_matrix:
    form.Assemble()
    rows, cols, vals = form.mat.COO()
    return sp.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=shape
    )


def _point_maps(
    space: Any, raised: int
) -> tuple[list[sp.csr_matrix], np.ndarray, np.ndarray]:
    """The component maps of the DOF vector to the rule's points, and their weights.

    Returns the sparse ``(n_points, n_dof)`` map of each component, the weight
    of each point (local units squared) and ``point_dofs (n_el, n_q)``, the
    point numbers of each element. ``raised`` adds to the order of the
    integration rule space (a test's check that the rule is exact).
    """
    planar = len(space.section.normal) == 3
    q = 2 * space.order + raised
    mesh = space.mesh
    irs = ngs.comp.IntegrationRuleSpace(mesh, order=q, complex=True)
    cell = ngs.ET.TRIG if planar else ngs.ET.SEGM
    rule = ngs.IntegrationRule(cell, 2 * q)
    n_el, n_q = int(mesh.ne), len(rule.points)
    if irs.ndof != n_el * n_q:  # pragma: no cover - guards the NGSolve contract
        raise RuntimeError(
            f"the integration rule space holds {irs.ndof} points, not {n_q} per "
            f"element on {n_el} elements."
        )
    dx = ngs.dx(intrules={cell: rule})
    v = irs.TestFunction()
    if planar:
        et, en = space.space.TrialFunction()
        comps = (et[0], et[1], en)
    else:
        comps = (space.space.TrialFunction(),)
    mats = []
    for comp in comps:
        form = ngs.BilinearForm(space.space, irs)
        form += comp * v * dx
        mats.append(_coo(form, (irs.ndof, space.space.ndof)))
    mass = ngs.BilinearForm(irs)
    mass += irs.TrialFunction() * v * dx
    w = _coo(mass, (irs.ndof, irs.ndof)).diagonal()
    if np.abs(w.imag).max() > 1e-14 * np.abs(w.real).max():  # pragma: no cover
        raise RuntimeError("the quadrature weights of the mode space are complex.")
    point_dofs = np.array(
        [list(irs.GetDofNrs(ngs.ElementId(ngs.VOL, e))) for e in range(n_el)], int
    )
    return mats, w.real, point_dofs


def _columns(space: Any, raised: int = 0) -> _Columns:
    r"""Build the columns of ``space`` (a port mode space), cached on it.

    NGSolve evaluates the field at the points of an integration rule as a
    sparse map of the full DOF vector: an ``IntegrationRuleSpace`` holds one
    DOF per point, a mixed bilinear form per component maps the field into
    it, and the space's own mass (a diagonal, :math:`w_q |J|`) divides the
    weights out. The rule of degree :math:`4p` integrates the quartic
    integrand exactly for fields of degree :math:`p` and a material constant
    per element on straight elements. The rule is named on the form: a line
    mesh's ``GetIntegrationRules()`` has no segment key and the default rule
    gives wrong point values.
    """
    cache = space.__dict__.setdefault("_kerr_columns", {})
    if raised in cache:
        return cache[raised]
    section = space.section
    planar = len(section.normal) == 3
    mats, w, point_dofs = _point_maps(space, raised)
    n_el, n_q = point_dofs.shape
    n_dof = space.space.ndof
    owner = [
        list(space.space.GetDofNrs(ngs.ElementId(ngs.VOL, e))) for e in range(n_el)
    ]
    k = max(len(d) for d in owner)
    dofs = np.full((n_el, k), -1, np.int64)
    for e, d in enumerate(owner):
        dofs[e, : len(d)] = d
    row_el = np.empty(len(w), int)
    row_q = np.empty(len(w), int)
    row_el[point_dofs] = np.arange(n_el)[:, None]
    row_q[point_dofs] = np.arange(n_q)[None, :]
    # slot of a (element, DOF) pair among the element's DOFs
    keys = (np.arange(n_el)[:, None] * n_dof + dofs)[dofs >= 0]
    slots = np.broadcast_to(np.arange(k), dofs.shape)[dofs >= 0]
    by_key = np.argsort(keys)
    sorted_keys, sorted_slots = keys[by_key], slots[by_key]
    weights = np.zeros((n_el, n_q, len(mats), k))
    for c, mat in enumerate(mats):
        coo = mat.tocoo()
        el, pt = row_el[coo.row], row_q[coo.row]
        key = el * n_dof + coo.col
        at = np.minimum(np.searchsorted(sorted_keys, key), len(sorted_keys) - 1)
        hit = sorted_keys[at] == key
        value = coo.data / w[coo.row]
        if not hit.all() and np.abs(value[~hit]).max() > 0:  # pragma: no cover
            raise RuntimeError(
                "a quadrature point couples to a DOF of another element."
            )
        value = value[hit]
        if np.abs(value.imag).max(initial=0.0) > 1e-14 * np.abs(value.real).max(
            initial=0.0
        ):  # pragma: no cover - guards the NGSolve contract
            raise RuntimeError("the field's quadrature columns are complex.")
        np.add.at(weights, (el[hit], pt[hit], c, sorted_slots[at[hit]]), value.real)
    measure = np.zeros((n_el, n_q))
    measure[row_el, row_q] = w
    measure *= section.scale**2 if planar else section.scale
    dofs = np.maximum(dofs, 0)
    if not planar:
        # NGSolve elements to section segments
        order = np.empty(n_el, int)
        order[np.asarray(space.element_segment)] = np.arange(n_el)
        dofs, weights, measure = dofs[order], weights[order], measure[order]
    if planar:
        # the tangential and the normal component read disjoint slots of the
        # element: keep each its own, without the zeros of the other
        on_t = (weights[:, :, :2] != 0).any(axis=(0, 1, 2))
        on_n = (weights[:, :, 2] != 0).any(axis=(0, 1))
        cols = _Columns(
            jnp.asarray(dofs[:, on_t]),
            jnp.asarray(weights[:, :, :2][..., on_t]),
            jnp.asarray(dofs[:, on_n]),
            jnp.asarray(weights[:, :, 2][..., on_n]),
            measure,
        )
    else:
        cols = _Columns(jnp.asarray(dofs), jnp.asarray(weights), None, None, measure)
    cache[raised] = cols
    return cache[raised]


# --- materials -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Kerr:
    """The Kerr material values of the section's elements, in the columns' order.

    ``n2`` and ``beta`` ``(n_el,)`` (0 where an element has none), ``eps``
    ``(n_wave, n_el)`` the real part of the permittivity at each wave's
    frequency, ``keep`` ``(n_el,)`` False on a perfectly matched layer's
    triangles.
    """

    n2: jax.Array
    beta: jax.Array
    eps: jax.Array
    keep: np.ndarray


def _elements(model: Model, section: Any) -> tuple[np.ndarray, np.ndarray]:
    """The solve-domain element behind each section element, and its region tag."""
    mesh = model.mesh
    if len(section.normal) == 3:
        from gfem._engine.port_mode_sensitivity import owner_tetrahedra

        cells = owner_tetrahedra(model, section)
        return cells, np.asarray(mesh.tetrahedron_tags)[cells]
    cells = np.asarray(section.cells)
    return cells, np.asarray(mesh._topology.volume_tags)[cells]  # noqa: SLF001


def _anisotropic(material: Material | MaterialField) -> bool:
    if isinstance(material, Material):
        return not material.isotropic or material.axes is not None
    return material.axes is not None


def _per_element(value: Any, pos: np.ndarray) -> jax.Array:
    arr = jnp.asarray(value, dtype=jnp.float64)
    return jnp.broadcast_to(arr, pos.shape) if arr.ndim == 0 else arr[pos]


def _region_permittivity(
    material: Material | MaterialField,
    pos: np.ndarray,
    frequency: np.ndarray,
    *,
    port: str,
    region: str,
    owner: str,
) -> jax.Array:
    r""":math:`\mathrm{Re}\,\varepsilon_r` of a region's elements at each frequency."""
    if isinstance(material, Material):
        eps = material.eps_r
        if isinstance(eps, PermittivityLaw):
            return jnp.real(eps(jnp.asarray(frequency)))[:, None] * jnp.ones(len(pos))
        return jnp.full((len(frequency), len(pos)), float(np.real(eps)))
    if material.eps_r is None:
        raise NotImplementedError(
            f"{owner}: region {region!r} of wave port {port!r}'s section is a "
            "density blend with a permittivity-law end or a layer grid and carries "
            "a Kerr property: its permittivity is not a per-element value. Give "
            "the elements beside the port a region of their own with a constant "
            "gfem.Material."
        )
    eps = jnp.real(jnp.asarray(material.eps_r))
    return jnp.broadcast_to(_per_element(eps, pos), (len(frequency), len(pos)))


def _kerr_materials(
    model: Model, section: Any, frequency: np.ndarray, port: str, owner: str
) -> _Kerr:
    """Read the Kerr properties of the elements of ``section`` by owner element.

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            Kerr property.
        NotImplementedError: A region carrying one is anisotropic, a density
            blend without a per-element permittivity, or a layer-grid design
            blending a Kerr property.
    """
    mesh = model.mesh
    cells, tags = _elements(model, section)
    names = mesh._region_names()  # noqa: SLF001
    n_el = len(cells)
    n2 = jnp.zeros(n_el)
    beta = jnp.zeros(n_el)
    eps = jnp.zeros((len(frequency), n_el))
    carries = np.zeros(n_el, bool)
    for tag in np.unique(tags):
        region = names[int(tag)]
        material = model.materials[region]
        given = {n: getattr(material, n) for n in _PROPERTIES}
        ends = getattr(material, "blend_ends", None)
        if ends is not None and ends.pixels is not None:
            named = [n for n, _, _ in ends.others if n in _PROPERTIES]
            if named:
                raise NotImplementedError(
                    f"{owner}: region {region!r} of wave port {port!r}'s section is "
                    f"a layer-grid design that blends {named[0]!r}: a Kerr "
                    "property is not a per-element value there in this version. "
                    "Give the elements beside the port a region of their own with "
                    "a constant gfem.Material."
                )
        if all(v is None for v in given.values()):
            continue
        if _anisotropic(material):
            raise NotImplementedError(
                f"{owner}: region {region!r} of wave port {port!r}'s section is "
                "anisotropic and carries a Kerr property: the refractive index "
                "squared and the isotropic nonlinearity are not defined on a "
                "permittivity tensor in this version. Give the region an isotropic "
                "eps_r."
            )
        rows = np.flatnonzero(tags == tag)
        pos = np.searchsorted(mesh.elements(region), cells[rows])
        carries[rows] = True
        if given["nonlinear_index"] is not None:
            n2 = n2.at[rows].set(_per_element(given["nonlinear_index"], pos))
        if given["two_photon_absorption"] is not None:
            beta = beta.at[rows].set(_per_element(given["two_photon_absorption"], pos))
        eps = eps.at[:, rows].set(
            _region_permittivity(
                material, pos, frequency, port=port, region=region, owner=owner
            )
        )
    # from the mesh, as the section of a traced field has no layer to read
    layer = layer_tetrahedra(mesh, model.pml) if len(section.normal) == 3 else None
    keep = np.ones(n_el, bool) if layer is None else ~layer[cells]
    if not (carries & keep).any():
        regions = ", ".join(repr(names[int(t)]) for t in np.unique(tags))
        raise ValueError(
            f"{owner}: no element of wave port {port!r}'s section outside a "
            "perfectly matched layer (whose triangles are skipped) has a "
            "nonlinear_index or a two_photon_absorption (its regions are "
            f"{regions}), so the nonlinearity is zero. Set one on the core's "
            "material."
        )
    return _Kerr(n2, beta, eps, keep)


# --- fields ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class _Waves:
    """The waves of the results: DOF vectors, frequencies, columns, materials."""

    port: str
    modes: tuple[int, ...]
    frequency: np.ndarray
    vectors: jax.Array
    section: Any
    columns: _Columns
    kerr: _Kerr

    @property
    def weights(self) -> jax.Array:
        """``(n_el, n_q)`` quadrature weights, m² (m), zero on a layer."""
        return jnp.asarray(self.columns.measure * self.kerr.keep[:, None])

    def row(self, vector: jax.Array) -> jax.Array:
        """``(n_el, n_q, n_c)`` the mode of one DOF vector, in the global frame."""
        cols = self.columns
        e = jnp.einsum("eqck,ek->eqc", cols.weights, vector[cols.dofs])
        if cols.normal_weights is None:
            return e
        e_n = jnp.einsum("eqk,ek->eq", cols.normal_weights, vector[cols.normal_dofs])
        full = jnp.concatenate([e, e_n[..., None]], axis=-1)
        return full @ jnp.asarray(self.section.frame)

    def field(self) -> jax.Array:
        """``(n_wave, n_el, n_q, n_c)`` the mode at 1 W in the global frame."""
        return jax.lax.map(self.row, self.vectors)


def _vector(field: Any) -> jax.Array:
    """The full DOF vector of each frequency, ``(n_freq, n_dof)``."""
    if field._vector is not None:  # noqa: SLF001
        return jnp.asarray(field._vector)  # noqa: SLF001
    _, basis, coords = field._data()  # noqa: SLF001
    return jnp.asarray(np.asarray(coords) @ np.asarray(basis).T)


def _same_section(a: Any, b: Any) -> bool:
    if len(a.normal) != len(b.normal):
        return False
    names = ("points", "triangles") if len(a.normal) == 3 else ("points", "segments")
    return all(np.array_equal(getattr(a, n), getattr(b, n)) for n in names)


def _waves(model: Model, modes: Any, owner: str) -> _Waves:
    """Gather the waves of ``modes`` on one section.

    Raises:
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NotImplementedError: An Hz line port.
        ValueError: The results are of different ports or discretizations, or
            the model is not the one they were solved on.
        NonDifferentiableError: The node coordinates are traced.
    """
    from gfem.model import Model

    if not isinstance(model, Model):
        raise TypeError(f"{owner} needs a gfem.Model, got {type(model).__name__}.")
    results = [modes] if isinstance(modes, PortModeResult) else modes
    if isinstance(results, str) or not isinstance(results, Sequence) or not results:
        raise TypeError(
            f"{owner} needs a gfem.PortModeResult (from gfem.solve(model, "
            f"gfem.PortMode(...))) or a sequence of them, got {type(modes).__name__}."
        )
    for r in results:
        if not isinstance(r, PortModeResult):
            raise TypeError(
                f"{owner} needs gfem.PortModeResult entries, got {type(r).__name__}: "
                "solve the mode with gfem.solve(model, gfem.PortMode(...))."
            )
        if r.h_field is not None:
            raise NotImplementedError(
                f"{owner}: wave port {r.port!r} is an Hz line port, whose electric "
                "field needs the gradient of Hz; solve it with polarization='Ez'."
            )
        if r.e_field is None:
            raise TypeError(
                f"{owner}: the result of wave port {r.port!r} carries no electric "
                "field (a driven result's port modes do not): solve "
                "gfem.solve(model, gfem.PortMode(...)) and pass that result."
            )
    if is_traced(model.mesh.nodes):
        raise NonDifferentiableError(_NODES.format(owner=owner))
    spaces = []
    for r in results:
        try:
            spaces.append(r.e_field._space)  # noqa: SLF001
        except NonDifferentiableError as err:
            raise NonDifferentiableError(_NODES.format(owner=owner)) from err
    first = spaces[0]
    for r, space in zip(results, spaces, strict=True):
        if (
            r.port != results[0].port
            or space.order != first.order
            or not _same_section(space.section, first.section)
        ):
            raise ValueError(
                f"{owner}: the waves must share one port, element order and "
                f"section geometry; got port {results[0].port!r} (order "
                f"{first.order}) and port {r.port!r} (order {space.order}). Solve "
                "them with the same SolverOptions on the same model."
            )
    section = first.section
    frequency = np.concatenate(
        [np.atleast_1d(np.asarray(r.frequency, float)) for r in results]
    )
    counts = [len(np.atleast_1d(np.asarray(r.frequency))) for r in results]
    modes_of = tuple(
        int(r.mode) for r, n in zip(results, counts, strict=True) for _ in range(n)
    )
    vectors = jnp.concatenate([_vector(r.e_field) for r in results], axis=0)
    _check_model(model, section, results[0].port, owner)
    kerr = _kerr_materials(model, section, frequency, results[0].port, owner)
    columns = _cached_columns(model, section, results[0].port, first)
    return _Waves(results[0].port, modes_of, frequency, vectors, section, columns, kerr)


def _check_model(model: Model, section: Any, port: str, owner: str) -> None:
    """Refuse a model whose port is not the section the mode was solved on.

    The section is rebuilt from the model's mesh and port; a mode of another
    mesh (or of moved nodes) would be read at the wrong elements.
    """
    from gfem._line_section import line_section
    from gfem._port_section import port_geometry

    planar = len(section.normal) == 3
    try:
        if planar:
            other = port_geometry(model, section.port)
        else:
            other = line_section(model, section.port, geometry=True)
        same = (
            other.scale == section.scale
            and _same_section(other, section)
            and (
                not planar
                or np.array_equal(np.asarray(other.faces), np.asarray(section.faces))
            )
        )
    except (ValueError, KeyError, IndexError, AttributeError):
        same = False
    if not same:
        raise ValueError(
            f"{owner}: the model is not the one wave port {port!r}'s mode was "
            "solved on: its mesh gives another cross-section."
        )


#: Columns of the sections already read, per mesh topology then ``(port, order)``,
#: with the section they were built on: a traced result rebuilds its space at
#: every call, and the columns cost a few NGSolve assemblies.
_COLUMNS: weakref.WeakKeyDictionary[Any, dict[tuple, tuple[Any, _Columns]]] = (
    weakref.WeakKeyDictionary()
)


def _cached_columns(model: Model, section: Any, port: str, space: Any) -> _Columns:
    store = _COLUMNS.setdefault(model.mesh._topology, {})  # noqa: SLF001
    key = (port, space.order)
    hit = store.get(key)
    if (
        hit is not None
        and hit[0].scale == section.scale
        and _same_section(hit[0], section)
    ):
        return hit[1]
    columns = _columns(space)
    store[key] = (section, columns)
    return columns


# --- public functions ------------------------------------------------------------


def kerr_nonlinearity(model: Model, mode: PortModeResult) -> KerrNonlinearity:
    r"""The nonlinear coefficient :math:`\gamma_\mathrm{NL}` of a port mode.

    Reads the electric field of ``mode`` (the 1 W mode of a `gfem.PortMode`
    solve) on every row of its frequency grid, reduced rows included, and the
    Kerr properties of ``model``'s materials. :math:`n^2 =
    \mathrm{Re}\,\varepsilon_r` follows each region's permittivity (or law) at
    the row's frequency; ``nonlinear_index`` and ``two_photon_absorption``
    are constants over the grid, given at its centre, and an element without
    one counts as zero. See `KerrNonlinearity` for the formula and the sign
    convention: the real part is :math:`\delta\beta/P`, the imaginary part
    :math:`\delta\alpha/P \ge 0` is the two-photon loss.

    The integral runs over the cross-section outside any perfectly matched
    layer (its triangles are skipped), with a quadrature rule exact for the
    mode's polynomial degree. A lossy or leaky section is computed with the
    mode normalized to 1 W; the formula holds while :math:`\alpha \ll \beta`.
    An Ez line port of a 2D mesh gives the slab's value, in 1/W. For a
    single-material core, :math:`A_\mathrm{eff} = k_0 n_2 / \mathrm{Re}\,
    \gamma_\mathrm{NL}`.

    Exactly linear in ``nonlinear_index`` and ``two_photon_absorption``;
    under `jax.grad` the traced mode field makes it differentiable with
    respect to the section's ``eps_r``, laws, losses and density blends (one 2D
    adjoint per row), not to the node coordinates (width is co-designed by
    finite differences over a remesh). The inputs of the solve can still be
    differentiated through the exact :math:`\gamma` of the same rows.

    Args:
        model: The model the mode was solved on, with the Kerr properties.
        mode: The result of ``gfem.solve(model, gfem.PortMode(...))``,
            concrete or traced.

    Returns:
        The coefficient on every row of ``mode``'s grid.

    Raises:
        ValueError: No element outside a perfectly matched layer carries a
            Kerr property.
        NotImplementedError: A region carrying one is anisotropic, or a density
            blend without a per-element permittivity; an Hz line port.
        TypeError: ``mode`` is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`.
    """
    owner = "gfem.kerr_nonlinearity"
    if not isinstance(mode, PortModeResult):
        raise TypeError(
            f"{owner} takes one gfem.PortModeResult, got {type(mode).__name__}: "
            "solve gfem.solve(model, gfem.PortMode(...)) and pass that result "
            "(gfem.kerr_coupling takes several)."
        )
    waves = _waves(model, mode, owner)
    weights = waves.weights

    @jax.checkpoint
    def integral(vector: jax.Array) -> jax.Array:
        e = waves.row(vector)
        re, im = jnp.real(e), jnp.imag(e)
        e2 = jnp.sum(re * re + im * im, axis=-1)
        ee = jnp.sum(e * e, axis=-1)
        quartic = 2 * e2 * e2 + jnp.real(ee) ** 2 + jnp.imag(ee) ** 2
        return jnp.sum(weights * quartic, axis=-1)

    # one row at a time: the point values of a row are (n_el, n_q, 3) complex
    per_element = jax.lax.map(integral, waves.vectors)  # (n_wave, n_el)
    k0 = jnp.asarray(2 * np.pi * waves.frequency / _C)[:, None]
    kerr = waves.kerr
    # conjugate of n2 - i beta / (2 k0): the e^{-i omega t} convention
    n2 = kerr.n2[None] + 1j * kerr.beta[None] / (2 * k0)
    gamma = (k0[:, 0] * _C**2 * _EPS0**2 / 12) * jnp.sum(
        kerr.eps * n2 * per_element, axis=-1
    )
    return KerrNonlinearity(
        port=waves.port,
        mode=int(mode.mode),
        frequency=np.asarray(waves.frequency),
        gamma_nl=gamma,
    )


def kerr_coupling(
    model: Model, modes: PortModeResult | Sequence[PortModeResult]
) -> KerrCoupling:
    r"""The Kerr coupling tensor :math:`\gamma_{abcd}` between the waves of a port.

    Every row of every result in ``modes`` is one wave, in order: another mode
    number, another frequency, or both. Entry :math:`abcd` is the coefficient
    of :math:`A_b A_c \overline{A_d}` in wave :math:`a`'s amplitude equation,
    summed over all index tuples with no degeneracy factors; its special cases
    are self-phase (:math:`aaaa`), cross-phase and four-wave mixing. See
    `KerrCoupling` for the formula and the sign convention. The diagonal
    :math:`\gamma_{aaaa}` equals `kerr_nonlinearity`.

    Every tuple is computed, energy-conserving or not; an equation uses
    only those with :math:`\omega_a + \omega_b = \omega_c + \omega_d`. The
    phase mismatch of a four-wave mixing process, :math:`\Delta\beta = 2\beta_p
    - \beta_s - \beta_i`, is one line on the results' exact ``gamma``.

    All waves must share one port, element order and cross-section (solve them
    with one `gfem.SolverOptions` on one model). The cost is
    :math:`K^4` per quadrature point, so pass the few waves of a process, not
    a whole frequency grid (that is `kerr_nonlinearity`'s diagonal).
    Everything else is as `kerr_nonlinearity`: material values, the layer,
    lossy sections, and the gradient through the mode fields and materials.

    Args:
        model: The model the modes were solved on, with the Kerr properties.
        modes: One `gfem.PortModeResult` or a sequence of them.

    Returns:
        The ``(K, K, K, K)`` tensor for the :math:`K` waves.

    Raises:
        ValueError: The results are of different ports, element orders or
            section geometries; no element outside a perfectly matched layer
            carries a Kerr property.
        NotImplementedError: As `kerr_nonlinearity`.
        TypeError: An entry is not a `gfem.PortModeResult`, or has no electric
            field.
        NonDifferentiableError: The node coordinates are traced by `jax.grad`.
    """
    waves = _waves(model, modes, "gfem.kerr_coupling")
    e = waves.field()
    n_wave, n_el, n_q, n_c = e.shape
    points = e.reshape(n_wave, n_el * n_q, n_c)
    omega = jnp.asarray(2 * np.pi * waves.frequency)
    kerr = waves.kerr
    chi = (
        (4 / 3)
        * kerr.eps
        * _C
        * _EPS0
        * (kerr.n2[None] - 1j * kerr.beta[None] * _C / (2 * omega[:, None]))
    )
    x = jnp.repeat(chi, n_q, axis=1) * waves.weights.reshape(-1)[None]
    p = jnp.einsum("bnx,cnx->bcn", points, points)
    q = jnp.einsum("bnx,cnx->bcn", points, jnp.conj(points))
    t = (
        jnp.einsum("an,bcn,adn->abcd", x, p, jnp.conj(p))
        + jnp.einsum("an,ban,cdn->abcd", x, q, q)
        + jnp.einsum("an,bdn,can->abcd", x, q, q)
    )
    gamma = jnp.conj(omega[:, None, None, None] * _EPS0 / 16 * t)
    return KerrCoupling(
        port=waves.port,
        waves=tuple(
            (m, float(f)) for m, f in zip(waves.modes, waves.frequency, strict=True)
        ),
        gamma_nl=gamma,
    )
