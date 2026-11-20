"""The 2D port mode problem on NGSolve: mesh, spaces and the affine pencil.

Formulation (Lee, Sun and Cendes): a port mode is
``E = (E_t + n E_n) exp(-gamma s)``, ``s`` the distance along the inward
port normal ``n``. With ``e_t = gamma E_t`` and ``e_n = E_n`` in
first-kind Nédélec x H1 elements of the same order, it solves
``A x = lam B x`` with ``lam = gamma^2`` and

    A = [ (nu_nn curl e_t, curl v_t) - k0^2 (eps_tt e_t, v_t), 0 ; 0, 0 ]
    B = [ (nu_tt (e_t + grad e_n), v_t + grad v_n) - k0^2 (eps_nn e_n, v_n) ]

(``B`` holds the whole second row and column), where ``eps_tt`` and
``eps_nn`` include ``-i sigma_tt / (omega eps0)`` and ``-i sigma_nn /
(omega eps0)`` (the conductivity tensor split like the permittivity), and
an absorbing edge adds ``i k0 Y_tau (E_T, V_T)`` to the first diagonal
block and ``i k0 Y_n (E_T, V_T)`` to the second
(`gfem._engine.anisotropy.edge_admittance`). A `Conductor` or `Impedance`
edge adds its admittance to both, as the driven problem's face term
(`gfem._engine.terms`): ``i k0 eta0 / Z_s(omega)`` for a conductor, ``i k0
eta0 / R``, ``mu0 scale / L`` and ``-k0^2 C / (eps0 scale)`` for an
impedance. A region whose eps_r is a permittivity law holds eps_inf in
``eps_tt`` and ``eps_nn`` and adds its susceptibility as one more piece,
``-k0^2 chi(omega) ((e_t, v_t) + (e_n, v_n))`` on its triangles, and an
absorbing edge beside it ``i k0 sqrt(Re eps(omega) / Re mu)`` on both blocks.
Both matrices are affine: each is a sum of fixed matrices times
1, -k0^2, i k0, a conductor's own scalar or a law's, assembled once.
Coordinates are in the mesh file's unit, so ``k0`` is per local unit.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable

import netgen.meshing as ngm
import ngsolve as ngs
import numpy as np
import scipy.constants
import scipy.sparse as sp

from gfem._boundaries import Conductor
from gfem._engine import quadrature
from gfem._engine.dispersion import susceptibility
from gfem._engine.terms import surface_impedance
from gfem._linalg.base import free_mask
from gfem._port_section import PortSection

# Points this close to a midline of the phase-reference quadrant (relative to
# the port's diameter) count as inside it.
_QUADRANT_RTOL = 1e-10

Z0 = scipy.constants.mu_0 * scipy.constants.c  # vacuum wave impedance, ohm


@dataclasses.dataclass(frozen=True)
class Pencil:
    """``A(k0) = sum theta_j(k0) A_j`` and ``B(k0)`` likewise, on free DOFs.

    Attributes:
        a: Matrices multiplying 1, -k0^2, i k0, then each of ``extra``.
        b: Likewise for ``B``.
        extra: The further scalars of k0 (one per conductor edge group, then
            one per law and one per law with absorbing edges).
    """

    a: tuple[sp.csr_matrix, ...]
    b: tuple[sp.csr_matrix, ...]
    extra: tuple[Callable[[float], complex], ...] = ()

    def theta(self, k0: float) -> tuple[complex, ...]:
        """The scalar of each matrix at wavenumber ``k0`` (per local unit)."""
        return (1.0, -(k0**2), 1j * k0, *(f(k0) for f in self.extra))

    def at(self, k0: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """``A`` and ``B`` at wavenumber ``k0`` (per local length unit)."""
        theta = self.theta(k0)
        a = sum(t * m for t, m in zip(theta, self.a, strict=True))
        b = sum(t * m for t, m in zip(theta, self.b, strict=True))
        return sp.csr_matrix(a), sp.csr_matrix(b)


class PortModeSpace:
    """The discrete port mode problem of one `PortSection`.

    Attributes:
        section: The cross-section.
        order: Element order (Nédélec first kind and H1).
        quantity: The mode field's name, ``"E"``.
        unit: Its unit, ``"V/m"``.
        mesh: The 2D NGSolve mesh in local plane coordinates.
        space: The compound space ``HCurl x H1``.
        free: Indices of the free (non-Dirichlet) DOFs.
        n_t: Number of Nédélec DOFs (the first block of ``space``).
        pencil: The affine pieces of ``A`` and ``B``.
    """

    quantity = "E"
    unit = "V/m"

    def __init__(self, section: PortSection, order: int) -> None:
        """Build the mesh and spaces and assemble the pencil."""
        if not isinstance(section, PortSection):
            # a backstop: a 2D mesh's line ports never reach this space
            raise TypeError(
                "PortModeSpace solves the modes of a planar PortSection, got "
                f"{type(section).__name__}."
            )
        self.section = section
        self.order = order
        self.mesh = _build_mesh(section)
        dirichlet = "pec" if len(section.pec_edges) else ""
        nd = ngs.HCurl(
            self.mesh, order=order, type1=True, complex=True, dirichlet=dirichlet
        )
        h1 = ngs.H1(self.mesh, order=order, complex=True, dirichlet=dirichlet)
        self.space = ngs.FESpace([nd, h1])
        self.n_t = nd.ndof
        self.free = np.flatnonzero(free_mask(self.space.FreeDofs(), self.space.ndof))
        self._coef = _coefficients(self.mesh, section)
        self.pencil = self._assemble()

    def __reduce__(self) -> tuple[type[PortModeSpace], tuple[PortSection, int]]:
        """Pickle as the section and order, rebuilt on load.

        The NGSolve mesh and spaces do not pickle; the rebuild from the same
        section is deterministic, so a basis on the old space (the mode
        fields of a result sent across processes) stands on the new one.
        """
        return (PortModeSpace, (self.section, self.order))

    # --- assembly -------------------------------------------------------------

    def _assemble(self) -> Pencil:
        c = self._coef
        (et, en), (vt, vn) = self.space.TnT()
        q = quadrature.degree(self.order)
        dx = ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})
        ds_abc = ngs.ds(
            "absorbing.*", intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)}
        )
        sigma_scale = Z0 * self.section.scale
        wt = et + ngs.grad(en)
        wv = vt + ngs.grad(vn)
        has_abc = len(_admittance_groups(self.section)) > 0
        abc_t = c["Y_tau"] * et.Trace() * vt.Trace() * ds_abc if has_abc else None
        abc_n = c["Y_n"] * en.Trace() * vn.Trace() * ds_abc if has_abc else None
        forms_a = [
            [c["nu_nn"] * ngs.curl(et) * ngs.curl(vt) * dx],
            [(c["eps_tt"] * et) * vt * dx],
            [sigma_scale * (c["sigma_tt"] * et) * vt * dx, abc_t],
        ]
        forms_b = [
            [(c["nu_tt"] * wt) * wv * dx],
            [c["eps_nn"] * en * vn * dx],
            [sigma_scale * c["sigma_nn"] * en * vn * dx, abc_n],
        ]
        extra = []
        fill = _edge_fill(self.mesh, self.section)
        for g, (item, factor) in enumerate(self.section.impedance_terms):
            ds = ngs.ds(
                f"impedance{g}", intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)}
            )
            edge_t = fill * et.Trace() * vt.Trace() * ds
            edge_n = fill * en.Trace() * vn.Trace() * ds
            if isinstance(item, Conductor):
                forms_a.append([edge_t])
                forms_b.append([edge_n])
                extra.append(self._conductor_theta(item, factor))
                continue
            for slot, weight in self._impedance_weights(item):
                forms_a[slot].append(weight * edge_t)
                forms_b[slot].append(weight * edge_n)
        index = self.section.law_index
        absorbing_law = self.section.absorbing_law
        for g, (_, law) in enumerate(self.section.laws):
            inside = _p0(self.mesh, (np.asarray(index) == g).astype(float))
            forms_a.append([inside * et * vt * dx])
            forms_b.append([inside * en * vn * dx])
            extra.append(self._law_theta(law))
            if absorbing_law is not None and (absorbing_law == g).any():
                ds = ngs.ds(
                    f"law{g}", intrules={ngs.SEGM: ngs.IntegrationRule(ngs.SEGM, q)}
                )
                forms_a.append([et.Trace() * vt.Trace() * ds])
                forms_b.append([en.Trace() * vn.Trace() * ds])
                extra.append(self._law_edge_theta(law, g))
        a = tuple(self._matrix(f) for f in forms_a)
        b = tuple(self._matrix(f) for f in forms_b)
        return Pencil(a=a, b=b, extra=tuple(extra))

    def _conductor_theta(
        self, item: Conductor, factor: float
    ) -> Callable[[float], complex]:
        """``i k0 eta0 / Z_s(omega)`` of a conductor edge group."""
        scale = self.section.scale
        h = None if item.thickness is None else factor * item.thickness
        sigma = item.conductivity

        def theta(k0: float) -> complex:
            omega = k0 * scipy.constants.c / scale
            return 1j * k0 * Z0 / surface_impedance(sigma, h, omega)

        return theta

    def _omega(self, k0: float) -> float:
        """The angular frequency (rad/s) of a wavenumber per local unit."""
        return k0 * scipy.constants.c / self.section.scale

    def _law_theta(self, law) -> Callable[[float], complex]:
        """``-k0^2 chi(omega)`` of a law's susceptibility piece."""

        def theta(k0: float) -> complex:
            return -(k0**2) * susceptibility(law, self._omega(k0))

        return theta

    def _law_edge_theta(self, law, g: int) -> Callable[[float], complex]:
        """``i k0 sqrt(Re eps(omega) / Re mu)`` of a law's absorbing edges."""
        rows = np.asarray(self.section.law_index) == g
        mu = 1 / float(self.section.nu_nn[rows][0].real)
        eps_inf = float(law.eps_inf)

        def theta(k0: float) -> complex:
            eps = eps_inf + susceptibility(law, self._omega(k0)).real
            return 1j * k0 * np.sqrt(eps / mu)

        return theta

    def _impedance_weights(self, item) -> list[tuple[int, float]]:
        """(pencil slot, weight) of an `Impedance` edge's R, L and C."""
        scale = self.section.scale
        out = []
        if item.resistance is not None:
            out.append((2, Z0 / item.resistance))
        if item.inductance is not None:
            out.append((0, scipy.constants.mu_0 * scale / item.inductance))
        if item.capacitance is not None:
            out.append((1, item.capacitance / (scipy.constants.epsilon_0 * scale)))
        return out

    def _matrix(self, integrands) -> sp.csr_matrix:
        n = self.space.ndof
        form = ngs.BilinearForm(self.space)
        for term in integrands:
            if term is not None:
                form += term
        form.Assemble()
        rows, cols, vals = form.mat.COO()
        m = sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
        )
        return m[self.free][:, self.free].tocsr()

    def _vector(self, integrand) -> np.ndarray:
        form = ngs.LinearForm(self.space)
        form += integrand
        form.Assemble()
        return np.array(form.vec.FV().NumPy())[self.free]

    def _dx(self):
        q = quadrature.degree(self.order)
        return ngs.dx(intrules={ngs.TRIG: ngs.IntegrationRule(ngs.TRIG, q)})

    @functools.cached_property
    def mass(self) -> sp.csr_matrix:
        """Unweighted L2 mass of ``(e_t, e_n)`` on free DOFs (real SPD).

        The inner product that tracks a mode across frequencies.
        """
        (et, en), (vt, vn) = self.space.TnT()
        return self._matrix([(et * vt + en * vn) * self._dx()]).real.tocsr()

    @functools.cached_property
    def power_forms(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """The matrices of the power integral, on free DOFs.

        ``int E_t . conj(nu_tt (grad E_n + gamma E_t)) = c^T G conj(c) +
        conj(gamma) c^T T conj(c)``; rows belong to the unconjugated factor
        (see `power`).
        """
        (et, en), (vt, _vn) = self.space.TnT()
        nu = ngs.Conj(self._coef["nu_tt"])
        g = self._matrix([(nu * ngs.grad(en)) * vt * self._dx()])
        t = self._matrix([(nu * et) * vt * self._dx()])
        return g, t

    @functools.cached_property
    def coupling_forms(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """The matrices that couple a mode to the 3D field on the port face.

        ``M_G[i, j] = int (nu_tt grad psi_j) . phi_i`` and ``M_T[i, j] = int
        (nu_tt phi_j) . phi_i``: rows over every Nédélec DOF ``phi_i`` (the
        trace of the 3D field, Dirichlet DOFs included), columns over the free
        DOFs (the mode). ``nu_tt`` is the port's tangential inverse-permeability
        tensor ``A`` in the port frame, the tensor of the mode's n x H (as in
        `power`) and of the 3D Robin term (`gfem._engine.terms`).
        """
        (et, en), (vt, _vn) = self.space.TnT()
        nu = self._coef["nu_tt"]
        dx = self._dx()
        out = []
        for integrand in ((nu * ngs.grad(en)) * vt * dx, (nu * et) * vt * dx):
            form = ngs.BilinearForm(self.space)
            form += integrand
            form.Assemble()
            rows, cols, vals = form.mat.COO()
            n = self.space.ndof
            m = sp.csr_matrix(
                (np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n)
            )
            out.append(m[: self.n_t][:, self.free].tocsr())
        return out[0], out[1]

    @functools.cached_property
    def gauge_forms(self) -> tuple[np.ndarray, np.ndarray]:
        """``l_n`` and ``l_t`` with ``gauge(c, gamma) = -i (l_n + gamma l_t) . c``.

        Free-DOF vectors; see `gauge`.
        """
        _, (vt, vn) = self.space.TnT()
        weight = self._quadrant() * self._coef["nu_tt"]
        t = self._gauge_direction()
        dx = self._dx()
        ln = self._vector((t * (weight * ngs.grad(vn))) * dx)
        lt = self._vector((t * (weight * vt)) * dx)
        return ln, lt

    def _quadrant(self) -> ngs.CoefficientFunction:
        s = self.section
        glob = [
            s.origin[k] + ngs.x * s.frame[0, k] + ngs.y * s.frame[1, k]
            for k in (0, 1, 2)
        ]
        corners = s.to_global(s.points) / s.scale
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        da, db = np.argsort(hi - lo)[::-1][:2]
        # Closed (x >= c), with a margin: symmetric quadrature
        # rules put points on the medians of a structured mesh, which can lie
        # on a midline, and rounding would put them on either side.
        tol = _QUADRANT_RTOL * float((hi - lo).max())
        ca, cb = 0.5 * (lo + hi)[da] - tol, 0.5 * (lo + hi)[db] - tol
        return ngs.IfPos(glob[da] - ca, 1.0, 0.0) * ngs.IfPos(glob[db] - cb, 1.0, 0.0)

    def _gauge_direction(self) -> ngs.CoefficientFunction:
        s = self.section
        return ngs.CF((float(s.frame[0].sum()), float(s.frame[1].sum())))

    # --- the sweep's protocol -------------------------------------------------

    def reduce(self, v: np.ndarray) -> SectionReduction:
        """The sweep's fixed field basis on the reduced basis ``v``.

        The pencil vector is ``(gamma E_t, E_n)``: ``v`` splits into a
        tangential and a normal block, so the field is affine in its
        coordinates.
        """
        return SectionReduction(self, v)

    def port_columns(self, basis: np.ndarray) -> dict[str, np.ndarray]:
        """What the 2D forms give per column of the sweep's ``basis``.

        ``r``: the entries of the coupling's ``vectors`` on the free 2D DOFs;
        ``reaction``: hᵀ e on the port face (the mode vanishes on fixed
        DOFs).
        """
        g_form, t_form = self.coupling_forms
        w = basis[self.free]
        m = w.shape[1] // 2
        r = np.concatenate([t_form @ w[:, :m], g_form @ w[:, m:]], axis=1)
        tangential = basis[: self.n_t, :m]
        return {"r": r, "reaction": r.T @ tangential}

    def phi(self, coords: np.ndarray, gamma: np.ndarray, omega: np.ndarray):
        """``(n_freq, 2 m)``: the mode's n x H on the columns of ``r``.

        ``gamma c_t`` and ``c_n``, ``gamma`` per file unit.
        """
        del omega
        m = coords.shape[1] // 2
        return np.concatenate([gamma[:, None] * coords[:, :m], coords[:, m:]], 1)

    def scalar(self, trace: np.ndarray, gamma0: complex, omega: float) -> np.ndarray:
        """``gamma0 M E`` on the columns of ``r``, for traces ``(m, n)``."""
        del omega
        return np.concatenate([gamma0 * trace, np.zeros_like(trace)])

    def pairing_factor(self, omegas: np.ndarray) -> np.ndarray:
        """``factor hᵀ e`` is the reciprocity pairing ``(1/2) int (e x h) . n``, W."""
        return -0.5j * self.section.scale / (omegas[:, None] * scipy.constants.mu_0)

    # --- fields ---------------------------------------------------------------

    def full(self, x: np.ndarray) -> np.ndarray:
        """A full coefficient vector from free-DOF values (zeros elsewhere)."""
        out = np.zeros(self.space.ndof, complex)
        out[self.free] = x
        return out

    def _gridfunction(self, coeffs: np.ndarray) -> ngs.GridFunction:
        gf = ngs.GridFunction(self.space)
        gf.vec.FV().NumPy()[:] = coeffs
        return gf

    def evaluate(self, coeffs: np.ndarray, local: np.ndarray) -> np.ndarray:
        """``(E_u, E_v, E_n)`` at plane points; zero off the cross-section.

        Args:
            coeffs: Full coefficient vectors of ``(E_t, E_n)``, ``(m, ndof)``.
            local: Plane coordinates ``(n, 2)`` in local units.

        Returns:
            ``(m, n, 3)`` complex.
        """
        coeffs = np.atleast_2d(coeffs)
        mips = [self.mesh(float(x), float(y)) for x, y in np.asarray(local, float)]
        found = [i for i, mip in enumerate(mips) if mip.nr >= 0]
        out = np.zeros((len(coeffs), len(mips), 3), complex)
        gf = ngs.GridFunction(self.space)
        et, en = gf.components
        for j, c in enumerate(coeffs):
            gf.vec.FV().NumPy()[:] = c
            for i in found:
                out[j, i, :2] = et(mips[i])
                out[j, i, 2] = en(mips[i])
        return out

    def magnetic_parts(
        self, coeffs: np.ndarray, local: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""The two parts of the in-plane magnetic field at plane points.

        :math:`H_t = \frac{i}{\omega \mu_0 s} (h_n + \gamma h_t)` with
        :math:`h_n = R\, \nu_{tt} \nabla E_n` and :math:`h_t = R\, \nu_{tt}
        E_t` (Faraday under :math:`e^{+i\omega t}`, :math:`R a = a \times n`,
        :math:`s` the scale and :math:`\gamma` per local unit). Zero off the
        cross-section.

        Args:
            coeffs: Full coefficient vectors of ``(E_t, E_n)``, ``(m, ndof)``.
            local: Plane coordinates ``(n, 2)`` in local units.

        Returns:
            ``h_n`` and ``h_t``, each ``(m, n, 2)`` complex in ``(u, v)``.
        """
        coeffs = np.atleast_2d(coeffs)
        mips = [self.mesh(float(x), float(y)) for x, y in np.asarray(local, float)]
        found = [i for i, mip in enumerate(mips) if mip.nr >= 0]
        hn = np.zeros((len(coeffs), len(mips), 2), complex)
        ht = np.zeros_like(hn)
        gf = ngs.GridFunction(self.space)
        et, en = gf.components
        nu = self._coef["nu_tt"]
        cf_n, cf_t = nu * ngs.grad(en), nu * et
        for j, c in enumerate(coeffs):
            gf.vec.FV().NumPy()[:] = c
            for i in found:
                hn[j, i] = cf_n(mips[i])
                ht[j, i] = cf_t(mips[i])
        # R a = a x n = (a_v, -a_u) in the right-handed frame (u, v, n)
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
        return hn @ rot.T, ht @ rot.T

    def power(self, coeffs: np.ndarray, gamma: complex, omega: float) -> complex:
        """Complex power ``1/2 int (E x H*) . n dA`` through the port, W.

        Args:
            coeffs: Full coefficient vector of ``(E_t, E_n)`` (V/m).
            gamma: Propagation constant per local unit.
            omega: Angular frequency, rad/s.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        mu0 = scipy.constants.mu_0
        cf = et * ngs.Conj(self._coef["nu_tt"] * a)
        value = ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        return 0.5 * self.section.scale * 1j / (omega * mu0) * complex(value)

    def pairing(self, coeffs: np.ndarray, gamma: complex, omega: float) -> complex:
        """Reciprocity pairing ``1/2 int (E x H) . n dA``, unconjugated, W.

        `power` without the conjugate: 1 W for a lossless 1 W mode of real
        phase; the normalization of a wave port's modal waves.

        Args:
            coeffs: Full coefficient vector of ``(E_t, E_n)`` (V/m).
            gamma: Propagation constant per local unit.
            omega: Angular frequency, rad/s.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        mu0 = scipy.constants.mu_0
        cf = et * (self._coef["nu_tt"] * a)
        value = ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        return -0.5j * self.section.scale / (omega * mu0) * complex(value)

    def gauge(self, coeffs: np.ndarray, gamma: complex) -> complex:
        """Phase reference: ``int_Q t . (n_out x H) dA`` up to a factor.

        ``Q`` is the closed upper-right quadrant of the port's bounding box
        along its two largest global extents and ``t`` the in-plane part of
        (1, 1, 1); the factor is real and positive.
        """
        et, en = self._gridfunction(coeffs).components
        a = ngs.grad(en) + gamma * et
        inside = self._quadrant()
        t = self._gauge_direction()
        cf = inside * (t * (-1j * (self._coef["nu_tt"] * a)))
        return complex(
            ngs.Integrate(cf, self.mesh, order=quadrature.degree(self.order))
        )


@dataclasses.dataclass(eq=False)
class SectionReduction:
    """A port mode's field basis, power and gauge on a fixed reduced basis.

    Attributes:
        space: The port mode problem.
        v: ``(n_free, m)`` the reduced basis of pencil vectors.
    """

    space: PortModeSpace
    v: np.ndarray

    def __post_init__(self) -> None:
        space, v = self.space, self.v
        m = v.shape[1]
        tangential = space.free < space.n_t
        w = np.zeros((len(space.free), 2 * m))
        w[tangential, :m] = v[tangential]
        w[~tangential, m:] = v[~tangential]
        self.w = w
        g_form, t_form = space.power_forms
        self._g, self._t = w.T @ (g_form @ w), w.T @ (t_form @ w)
        ln, lt = space.gauge_forms
        self._ln, self._lt = ln @ w, lt @ w

    def coords(self, y: np.ndarray, gamma: complex) -> np.ndarray:
        """The field's coordinates ``(y / gamma, y)`` of a pencil vector's."""
        return np.concatenate([y / gamma, y])

    def power(self, z: np.ndarray, gamma: complex, omega: float) -> complex:
        """Complex power through the port, W (`PortModeSpace.power`)."""
        return (
            0.5
            * self.space.section.scale
            * 1j
            / (omega * scipy.constants.mu_0)
            * (z @ self._g @ z.conj() + gamma.conjugate() * (z @ self._t @ z.conj()))
        )

    def gauge(self, z: np.ndarray, gamma: complex, omega: float) -> complex:
        """Phase reference (`PortModeSpace.gauge`)."""
        del omega
        return -1j * (self._ln @ z + gamma * (self._lt @ z))


def _build_mesh(section: PortSection) -> ngs.Mesh:
    m = ngm.Mesh(dim=2)  # ty: ignore[unresolved-attribute]
    pts = np.zeros((len(section.points), 3))
    pts[:, :2] = section.points
    m.AddPoints(pts)
    m.Add(ngm.FaceDescriptor(surfnr=1, domin=1, domout=0, bc=1))  # ty: ignore[unresolved-attribute]
    m.SetMaterial(1, "port")
    m.AddElements(dim=2, index=1, data=section.triangles, base=0)
    m.SetBCName(0, "pec")
    for edge in section.pec_edges:
        m.Add(ngm.Element1D([int(edge[0]) + 1, int(edge[1]) + 1], index=1))  # ty: ignore[unresolved-attribute]
    groups = _admittance_groups(section)
    for g, (_, edges) in enumerate(groups):
        m.SetBCName(g + 1, f"absorbing{g}")
        for edge in edges:
            m.Add(ngm.Element1D([int(edge[0]) + 1, int(edge[1]) + 1], index=g + 2))  # ty: ignore[unresolved-attribute]
    first = len(groups) + 1
    for g in range(len(section.impedance_terms)):
        m.SetBCName(first + g, f"impedance{g}")
        mask = section.impedance_group == g
        for edge in section.impedance_edges[mask]:
            m.Add(
                ngm.Element1D(  # ty: ignore[unresolved-attribute]
                    [int(edge[0]) + 1, int(edge[1]) + 1], index=first + g + 1
                )
            )
    first += len(section.impedance_terms)
    for g in range(len(section.laws)):
        m.SetBCName(first + g, f"law{g}")
        if section.absorbing_law is None:
            continue
        for edge in section.absorbing_edges[section.absorbing_law == g]:
            m.Add(
                ngm.Element1D(  # ty: ignore[unresolved-attribute]
                    [int(edge[0]) + 1, int(edge[1]) + 1], index=first + g + 1
                )
            )
    return ngs.Mesh(m)


def _edge_fill(mesh: ngs.Mesh, section: PortSection) -> ngs.CoefficientFunction:
    """The sheet fields' fill per impedance edge (1 without a sheet field)."""
    if section.impedance_fill is None or (section.impedance_fill == 1.0).all():
        return ngs.CF(1.0)
    fill = {
        (int(min(a, b)), int(max(a, b))): float(w)
        for (a, b), w in zip(
            section.impedance_edges, section.impedance_fill, strict=True
        )
    }
    space = ngs.SurfaceL2(mesh, order=0)
    gf = ngs.GridFunction(space)
    values = gf.vec.FV().NumPy()
    values[:] = 1.0
    for el in mesh.Elements(ngs.BND):
        a, b = sorted(v.nr for v in el.vertices)
        if (a, b) in fill:
            values[space.GetDofNrs(el)[0]] = fill[a, b]
    return gf


def _admittance_groups(
    section: PortSection,
) -> list[tuple[tuple[float, float], np.ndarray]]:
    """The absorbing edges grouped by their ``(Y_tau, Y_n)``.

    Not those beside a law triangle: each law's own edge piece holds them.
    """
    keep = np.ones(len(section.absorbing_edges), bool)
    if section.absorbing_law is not None:
        keep = section.absorbing_law < 0
    values, inverse = np.unique(
        section.absorbing_admittance[keep].reshape(-1, 2), axis=0, return_inverse=True
    )
    edges = section.absorbing_edges[keep]
    return [
        ((float(v[0]), float(v[1])), edges[inverse.ravel() == g])
        for g, v in enumerate(values)
    ]


def _p0(mesh: ngs.Mesh, values: np.ndarray) -> ngs.CoefficientFunction:
    """A piecewise-constant coefficient, one value per element (mesh order)."""
    values = np.asarray(values)
    if np.abs(values - values[0]).max() <= 1e-14 * max(np.abs(values).max(), 1.0):
        return ngs.CF(complex(values[0]))
    gf = ngs.GridFunction(ngs.L2(mesh, order=0, complex=True))
    gf.vec.FV().NumPy()[:] = values
    return gf


def _coefficients(mesh: ngs.Mesh, s: PortSection) -> dict[str, ngs.CoefficientFunction]:
    def matrix(t: np.ndarray) -> ngs.CoefficientFunction:
        parts = [_p0(mesh, t[:, i, j]) for i in (0, 1) for j in (0, 1)]
        return ngs.CF(tuple(parts), dims=(2, 2))

    groups = _admittance_groups(s)
    y_tau, y_n = (
        mesh.BoundaryCF(
            {f"absorbing{g}": v[k] for g, (v, _) in enumerate(groups)},
            default=ngs.CF(0.0),
        )
        for k in (0, 1)
    )
    return {
        "eps_tt": matrix(s.eps_tt),
        "eps_nn": _p0(mesh, s.eps_nn),
        "nu_tt": matrix(s.nu_tt),
        "nu_nn": _p0(mesh, s.nu_nn),
        "sigma_tt": matrix(s.sigma_tt.astype(complex)),
        "sigma_nn": _p0(mesh, s.sigma_nn.astype(complex)),
        "Y_tau": y_tau,
        "Y_n": y_n,
    }
