"""Sign, unit and normalization conventions of gfem's results.

Every complex result and every docstring that mentions a phasor follows
the conventions below. Each section is self-contained; new physics adds a
section rather than changing an existing one, because changing a
convention silently changes users' numbers.

Units
-----
SI at the API: metres, seconds, hertz, volts, amperes, ohms, siemens,
farads, joules, watts, kelvin (reserved for thermal problems). Only the
mesh file declares its own length unit (``Mesh.from_gmsh(...,
length_unit=)``); gfem converts on import, and every field value is in SI
(an electric field in V/m, whatever the mesh unit).

Time dependence
---------------
Time-harmonic quantities are complex phasors under e^{+iωt}: the physical
field is ``Re(E exp(+iωt))``, so the operator is ``K + iωC - ω²M``. This
is the electrical engineering convention (Touchstone, scikit-rf, SPICE write
``j`` for ``i``). Consequences:

- a wave travelling along ``+s`` is ``exp(-gamma s)``, with
  ``gamma = alpha + i beta`` and attenuation ``alpha >= 0``;
- a lossy permittivity is ``ε_r (1 - i tan δ) - i sigma / (ε0 ω)``, per
  principal axis for an anisotropic material (negative imaginary part);
- an inductor's impedance is ``+iωL``, a capacitor's ``1 / (iωC)``;
- a decaying eigenmode has ``Im ω > 0``.

Phasors and field amplitude
---------------------------
Phasors are peak values, not RMS: ``|E|`` is the amplitude of the
oscillation. A driven field (``Driven(keep_fields=...)``) is the field
for 1 W time-averaged incident power at the excited port, with every other
port terminated in its reference impedance ``z_ref`` (a lumped port's L
and C, if any, belong to the network and load it in parallel). A wave
port is terminated by absorbing its mode (matched), including the other
modes of the excited port's own surface. It is the field behind one
column of S: the excited port's incident wave is
``a = 1`` (√W), every other port's is 0.
Eigenmode fields store 1 J (see Eigenmodes). A port mode field is
normalized to 1 W time-averaged power through the port.

Network parameters
------------------
S-parameters are power waves (Kurokawa) with a reference impedance per
port and per frequency, ``z_ref``:

    a_i = (V_i + z_i I_i) / (2 sqrt(Re z_i)),
    b_i = (V_i - conj(z_i) I_i) / (2 sqrt(Re z_i)),   b = S a,

with ``V``, ``I`` peak port voltage and current, so ``|a_i|^2`` is the
time-averaged incident power in W (``|a| = 1`` is 1 W). When ``z_ref`` is
real, power waves and pseudo-waves coincide.

- A `LumpedPort`'s ``z_ref`` is its resistance. A port without a
  resistance has no reference impedance: its ``z_ref`` and its row of S are
  NaN, and it cannot be excited.
- A `WavePort`'s S is normalized to unit modal power; its ``z_ref`` is the
  power-voltage impedance ``Z_PV`` along its ``voltage_path`` (NaN without
  one). A wave port's S is the projection of the field on the conjugated
  port mode, with the mode's magnetic field taken at its complex
  wavenumber, and the mode scaled so that this projection of the mode
  itself carries 1 W; for a lossless mode this is the modal
  power wave. The conjugated projection separates the requested modes of
  a shared port surface only for a lossless port line, and a lossy port
  line reflects its own mode at order ``alpha/beta`` (transmission is
  right to the square of it).
  ``WavePort(offset=d)`` refers S to a plane ``d`` inside the
  domain: ``S_ij`` times ``exp(gamma_i d_i) exp(gamma_j d_j)``.
  Without a ``voltage_path``, the sign of a wave port's row and column of
  S is fixed by a mesh-dependent rule; every ``|S|`` and power is
  unaffected.
- ``NetworkParameters.y()``, ``.z()`` and ``.renormalize(z_new)`` use the
  power-wave formulas; ``to_touchstone`` labels a file with ``z_ref`` only
  when it is one real value, and otherwise needs ``z0=`` to renormalize
  first. gfem never labels a port 50 Ω unless S was computed or
  renormalized for 50 Ω, or no port has a reference impedance (wave ports
  without ``voltage_path``): then an explicit ``z0=`` is a nominal label,
  S is written unchanged and the file says so in a comment. On a shared
  port surface each mode is its own port, so ``y()``, ``z()`` and
  ``renormalize()`` act on modal voltages and currents, not on the
  conductors' terminal ones.
- Under a Bloch wave vector (see Bloch periodicity), S is referenced to the
  field E. A solver that reports the periodic envelope
  ``u = E exp(+i k·r)`` gives
  ``S_ij exp(+i k·(r_i - r_j))``, with i the output port, j the excited
  port and r_p a point of port p, when k·r is constant over each port.

Time-averaged densities
-----------------------
For peak phasors, the time average of a product of two harmonic quantities
carries a factor 1/2:

- electric energy density ``w_e = Re(E · D*) / 4`` (1/2 for the time
  average of ``(E · D) / 2``), magnetic ``w_m = Re(B · H*) / 4``;
- Poynting vector ``S = Re(E x H*) / 2``;
- ohmic loss density ``p = Re(sigma_eff) |E|^2 / 2``;
- a lumped port's power ``|V|^2 / (2 R)``.

With 1 W incident, the power a driven field delivers into the network
through the excited port is ``1 - |S_ii|^2`` W, and the power arriving at
port j's termination is ``|S_ji|^2`` W.

Surface conductors
------------------
A `Conductor` face adds the boundary term ``iω/Z_s(ω) ∫ E_t · v_t``, with
``Z_s`` the surface impedance of a metal of conductivity sigma and
thickness t in the finite-thickness form of HFSS::

    Z_s = (1 / (sigma δ)) [(sinh nu + sin nu) + i (sinh nu - sin nu)]
          / (cosh nu - cos nu)

with skin depth ``δ = sqrt(2 / (μ0 sigma ω))`` and ``nu = t / δ``
(``thickness=None``: ``Z_s = (1 + i) / (sigma δ)``). This form describes ONE
face of a film whose other face is modelled separately, as the shell of an
excluded metal volume is: for a film much thinner than its skin depth,
``Z_s → 2 / (sigma t)``.

- **Interior surface** (solve domain on both sides, e.g. a zero-thickness
  sheet): one term for the whole sheet, so a thin film is a sheet
  admittance of about ``sigma t / 2`` (`INTERIOR_SHEET_ADMITTANCE` times
  sigma t), not the film's conductance sigma t. To model a zero-thickness
  film of thickness t with its full conductance, give ``thickness=2 * t``.
- **Exterior surface** (mesh boundary, one-sided): the thickness counts
  twice (`EXTERIOR_THICKNESS_FACTOR`), so a thin film is ``sigma t``.
- **Shell of an excluded volume** (dielectric on one side, the excluded
  metal on the other): an interior face of the mesh, one term per face with
  the given thickness; the top and bottom faces of a trace together carry
  ``sigma t``.

gsim layouts assume this convention face for face.

Dispersive permittivity
-----------------------
A `gfem.PermittivityLaw` is a relative permittivity that depends on
frequency, ``eps_r(f) = eps_inf + Σ term(f)``. Every law parameter that is
a frequency is an ordinary frequency in hertz (``ω = 2π f``), not rad/s or
eV (convert an energy as ``f = E / h``). Under e^{+iωt} the terms are:

- `gfem.Drude`: ``-ωp² / (ω (ω - i gamma))``, ``ωp = 2π plasma_frequency``,
  ``gamma = 2π damping``; no static value (it diverges as f -> 0);
- `gfem.Lorentz`: ``Δε ω0² / (ω0² - ω² + i gamma ω)``,
  ``ω0 = 2π resonance_frequency``;
- `gfem.Debye`: ``Δε / (1 + i f / f_r)``, ``f_r = relaxation_frequency``
  (a relaxation time τ is ``f_r = 1 / (2π τ)``);
- `gfem.DjordjevicSarkar`: ``Δε / ln(f2 / f1) · ln((f2 + i f) / (f1 + i f))``,
  the principal log of the ratio, corners ``f1 = low_frequency`` and
  ``f2 = high_frequency``.

``Δε`` is each term's ``delta_eps``. With every strength and damping >= 0
each term is passive alone, so a law has ``Im eps_r <= 0`` at every real
frequency; a negative strength is refused. A law evaluates at a complex
frequency as the analytic continuation (``Im f > 0`` decays), and
``law(0)`` is the static permittivity. Most optics fits (Drude, Lorentz)
are written under e^{-iωt}: there the same law is the complex conjugate,
so their ``+i gamma`` reads ``-i gamma`` here, with the same parameter values.
Circuit-board laws (Djordjevic-Sarkar) are usually written under e^{+jωt}
already.

Eigenmodes
----------
An eigenmode is a complex angular frequency ``omega`` and a field ``E``
with ``Re(E exp(+iωt))`` source-free; ``Im omega > 0`` decays, and
``Q = |omega| / (2 |Im omega|)``. The mode's
electric energy is::

    E_elec = 1/2 Re ∫ D* · E dV + Σ C_j |V_j|^2 / 2 + Σ 1/2 ∫ w C_s |E_t|^2 dS

over the solve domain, the capacitive lumped ports (capacitance ``C_j``,
voltage ``V_j``) and the capacitive `Impedance` sheets (capacitance per
square ``C_s``, tangential field ``E_t``, fill ``w``, 1 without a
`SheetField`) (peak phasors: the total time-averaged energy of a lossless
mode, electric plus magnetic).
In a region with a `gfem.PermittivityLaw`, ``Re(D* · E)`` reads the energy
permittivity ``Re[eps + (omega / 2) d eps / d omega]`` at the mode's real
frequency, so that ``E_elec`` stays the total energy of a lossless mode
(Brillouin); for a lossy law it is the weak-loss value. An eigenmode solve
raises where that weight is not positive (loss dominates the law there).
The phase below reads the law's ``eps_inf`` in ``D``.
Each mode field is scaled to ``E_elec`` =
`MODE_ENERGY` and phased so that ``∫ D · E dV`` (no conjugate) is real
and positive, which makes a lossless mode's field real; the remaining sign
puts the largest coefficient's real part positive (among coefficients equal
to the largest within a relative 1e-12, the lowest index). Participation ratios
and coupling Q are ratios to ``E_elec`` and do not depend on this scale.

Bloch periodicity
-----------------
A `gfem.Periodic` pair relates two surfaces by the translation a, from its
first surface to its second (`Model.periodic_translations`, m). With a Bloch
wave vector k (rad/m, ``wave_vector=`` on `gfem.Driven` and
`gfem.Eigenmode`), the fields obey `BLOCH_PHASE`::

    E(r + a) = exp(-i k·a) E(r)

so, under e^{+iωt}, the phase of a Bloch wave travels along +k. Without a
wave vector the fields are equal on both surfaces (plain periodicity).
Swapping a pair's surfaces flips a and gives the same field.

Under a Bloch wave vector, ``∫ D · E dV`` over the cell can vanish, so an
eigenmode field is phased to make its largest coefficient real and positive
instead (still scaled to `MODE_ENERGY`).

Floquet ports
-------------
A `gfem.FloquetPort` is one diffraction order in one polarization of a plane
wave on an open face of a periodic cell (a Floquet surface).

- Lattice frame: a1 and a2 are the translations of the face's two pairs, in
  ``boundaries`` order; ``ê1 = a1 / |a1|``, ``ê3 = a1 x a2 / |a1 x a2|``,
  ``ê2 = ê3 x ê1``. Order (m, n) has the in-plane wave vector
  ``κ = k_t + m b1 + n b2`` with ``b_i · a_j = 2π δ_ij`` and ``k_t`` the
  problem's ``wave_vector`` (the specular order's). Its field on the face
  is ``c ê exp(-i κ·r)``, with ``r`` the model coordinates: the in-plane
  phase is referred to the model origin, and translating the model in-plane
  by ``r0`` multiplies ``S_pq`` by ``exp(-i (κ_p - κ_q)·r0)``.
- References: TE ``ê = ê3 x κ̂`` (s), TM ``ê = κ̂`` (p), the same on both
  faces of a cell, with ``κ̂ = ê1`` at ``κ = 0`` (at normal incidence TE is
  E along ê2, TM along ê1).
- Normalization: ``S_pq = (c_p / c_q) sqrt(Z_q / Z_p)``, ``c`` the
  tangential-E coefficients at the face plane and ``Z`` the order's wave
  impedance in the homogeneous medium behind its own face, ``η / cos θ``
  (TE) or ``η cos θ`` (TM), ``η = η0 sqrt(mu_r / eps_r)`` and θ the order's
  angle from the face normal. The incident wave carries `INCIDENT_POWER` per
  cell, ``|S|²`` is the diffraction efficiency, and for a lossless cell the
  efficiencies of every propagating order on both faces, both
  polarizations, add up to 1. ``z_ref`` is ``Z``.
- Signs: tangential-E coefficients, so a flat interface reflects
  ``(Z2 - Z1) / (Z2 + Z1)``, ``(n1 - n2) / (n1 + n2)`` in both polarizations
  at normal incidence from medium 1 (the Fresnel ``r_p`` of many optics
  texts has the opposite sign), and an empty cell of thickness L transmits
  ``exp(-i beta L)``.
- Orders that do not propagate: where an order is evanescent or grazing
  (``gamma² >= 0``), its row of S is exactly 0 and its ``z_ref`` is NaN, and
  it cannot be excited. ``FloquetPort(offset=d)`` multiplies the row and
  column by ``exp(gamma d)`` where the order propagates, 1 elsewhere.
- Bloch reciprocity: ``S_pq(k) = sigma_p sigma_q S_{-q,-p}(-k)``, where
  ``-p`` is port p's face and polarization with order ``(-m, -n)`` and
  ``sigma = -1`` for an order with ``κ ≠ 0``, ``+1`` for ``κ = 0`` (the
  references are odd in κ̂). The plain transpose does not hold as it stands.
- Circular polarization: with the TE and TM ports of one order on a face
  whose outward normal is ``+ê3`` (incidence along ``-ê3``),
  ``(S[:, TM] + i S[:, TE]) / sqrt(2)`` is the response to IEEE
  right-handed circular incidence; on a face whose outward normal is
  ``-ê3`` the same combination is left-handed.

Quantum metrics
---------------
`gfem.quantum.dressed_modes` puts one junction's cosine on the solved
linear modes::

    H / hbar = Σ_m omega_m a_m† a_m - (E_J / hbar) [cos(phi) - 1 + phi² / 2]
    phi = Σ_m phi_m (a_m + a_m†)
    phi_m = sign(p_m) sqrt(|p_m| hbar omega_m / (2 E_J))
    E_J = (hbar / 2e)² / L_J

with ``omega_m`` the real part of the mode's angular frequency, ``p_m`` its
signed junction participation and ``L_J`` the junction's linear inductance.
With ``E`` its levels (``E(0)`` the ground state, ``E(1_m)`` one photon in
mode ``m``), every output is a signed angular frequency in rad/s and a
difference of transition frequencies, so its sign is an observable:

- dressed frequency ``E(1_m) - E(0)``; the Lamb shift is that minus
  ``omega_m``;
- anharmonicity ``alpha = E(2_m) - 2 E(1_m) + E(0)``, the second transition
  frequency minus the first;
- dispersive shift ``chi = E(1_m 1_n) - E(1_m) - E(1_n) + E(0)``: the full
  shift of mode ``n``'s frequency per photon in mode ``m``, not half of it.

For an ordinary transmon all three are negative. Other sources' conventions
convert as ``chi = -chi_Minev = chi_Nigg = 2 chi_Koch`` and ``alpha =
-alpha_Minev = alpha_Koch`` (Minev et al. and pyEPR report a downward shift
as positive; Koch et al. write half the shift).

Temperature and thermo-optic phase
----------------------------------
Temperatures are absolute, in kelvin. A thermal result's rise is the
temperature minus the source-free temperature (the temperature with every
`gfem.HeatSource` off, which boundary temperatures and fluxes still set),
so it is the heaters' response even where the boundaries differ.

A material's ``thermo_optic_coefficient`` dn/dT (1/K, any sign) acts on
the rise about the index ``n0 = sqrt(eps_r)`` of its ``eps_r``, the index
at the source-free temperature; no reference temperature is stored.
`gfem.Model.with_temperature_rise` gives each element
``eps_r = (n0 + dn/dT rise_e)^2``, ``rise_e`` the element's mean rise.

`gfem.thermo_optic_shift` is first order in the rise, about the port's
mode ``E0`` at 1 W on a lossless section::

    delta_n_eff = (c eps0 / 4) sum_e 2 n0_e (dn/dT)_e rise_e int_e |E0|^2 dA
    phase_per_length = k0 Re(delta_n_eff)      (rad/m)

``phase_per_length`` is the extra phase delay per unit length of guide:
positive for a positive coefficient and a positive rise. Under
``exp(+iωt)`` a guide of length L transmits ``exp(-i beta L)``, so a
transmission's argument moves by minus ``phase_per_length * L``. The
power that gives a pi shift is ``P_pi = pi * P' / phase_per_length`` for
a heater dissipating ``P'`` per unit length (``power_per_length``), which
equals ``lambda / (2 s)`` with ``s = delta_n_eff / P'``.
"""

from __future__ import annotations

__all__ = [
    "BLOCH_PHASE",
    "EXTERIOR_THICKNESS_FACTOR",
    "INCIDENT_POWER",
    "INTERIOR_SHEET_ADMITTANCE",
    "MODE_ENERGY",
    "PHASOR",
    "TIME_DEPENDENCE",
    "WAVES",
]

#: The time dependence of every complex phasor.
TIME_DEPENDENCE = "exp(+iωt)"
#: Phasors are peak values (not RMS).
PHASOR = "peak"
#: Time-averaged incident power at the excited port of a driven field, W.
INCIDENT_POWER = 1.0
#: The wave definition of S-parameters.
WAVES = "power"
#: Sheet admittance of one `Conductor` term on an interior surface for a film
#: much thinner than its skin depth, in units of sigma t (conductivity times
#: thickness).
INTERIOR_SHEET_ADMITTANCE = 0.5
#: Factor on a `Conductor`'s thickness on mesh-exterior faces, where the metal
#: is one-sided.
EXTERIOR_THICKNESS_FACTOR = 2.0
#: Electric energy of every eigenmode field (see Eigenmodes), J.
MODE_ENERGY = 1.0
#: The Bloch phase across a periodic pair of translation a (see Bloch
#: periodicity).
BLOCH_PHASE = "E(r + a) = exp(-i k·a) E(r)"
