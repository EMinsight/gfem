r"""Sign, unit and normalization conventions of gfem's results.

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
Time-harmonic quantities are complex phasors under :math:`e^{+i\omega t}`:
the physical field is :math:`\mathrm{Re}(E\, e^{+i\omega t})`, so the
operator is :math:`K + i\omega C - \omega^2 M`. This
is the electrical engineering convention (Touchstone, scikit-rf, SPICE write
:math:`j` for :math:`i`). Consequences:

- a wave travelling along :math:`+s` is :math:`e^{-\gamma s}`, with
  :math:`\gamma = \alpha + i\beta` and attenuation :math:`\alpha \ge 0`;
- a lossy permittivity is
  :math:`\varepsilon_r (1 - i \tan\delta) - i \sigma / (\varepsilon_0 \omega)`,
  per principal axis for an anisotropic material (negative imaginary part);
- an inductor's impedance is :math:`+i\omega L`, a capacitor's
  :math:`1 / (i\omega C)`;
- a decaying eigenmode has :math:`\mathrm{Im}\,\omega > 0`.

Phasors and field amplitude
---------------------------
Phasors are peak values, not RMS: :math:`|E|` is the amplitude of the
oscillation. A driven field (``Driven(keep_fields=...)``) is the field
for 1 W time-averaged incident power at the excited port, with every other
port terminated in its reference impedance ``z_ref`` (a lumped port's L
and C, if any, belong to the network and load it in parallel). A wave
port is terminated by absorbing its mode (matched), including the other
modes of the excited port's own surface. It is the field behind one
column of S: the excited port's incident wave is
:math:`a = 1` (√W), every other port's is 0.
Eigenmode fields store 1 J (see Eigenmodes). A port mode field is
normalized to 1 W time-averaged power through the port.

Network parameters
------------------
S-parameters are pseudo-waves (Marks and Williams) with a reference
impedance per port and per frequency, ``z_ref``:

.. math::

    a_i = \frac{\sqrt{\mathrm{Re}\, z_i}}{2 |z_i|} (V_i + z_i I_i), \qquad
    b_i = \frac{\sqrt{\mathrm{Re}\, z_i}}{2 |z_i|} (V_i - z_i I_i), \qquad
    b = S a,

with :math:`V`, :math:`I` peak port voltage and current, scaled so that
:math:`|a| = 1` is 1 W incident on a real reference. When ``z_ref`` is
real, pseudo-waves and power waves (Kurokawa) coincide; on a complex
reference, a line matched to its own characteristic impedance has
:math:`S_{11} = 0`, which power waves do not give.

- A `LumpedPort`'s ``z_ref`` is its resistance. A port without a
  resistance has no reference impedance: its ``z_ref`` and its row of S are
  NaN, and it cannot be excited.
- A `WavePort`'s S reads the field's modal amplitudes by the biorthogonal
  projection on the port surface's modes, :math:`c = R^{-1} H^T E` with
  :math:`R_{lk} = \int (n \times h_l) \cdot e_k` unconjugated, which by
  reciprocity separates the modes and matches a uniform line to its own
  mode, lossy or not. With a ``voltage_path`` its ``z_ref`` is the complex
  power-voltage impedance :math:`Z_{PV} = |V|^2 / (2 P^*)` of its mode, so
  the port's waves are the pseudo-waves of the mode's voltage and current;
  a lossless line has a real ``z_ref`` and unit-power modal waves. Without
  one, ``z_ref`` is NaN and the port's waves are its modal amplitudes
  normalized by reciprocity, :math:`\tfrac12 \int (e \times h) \cdot n = 1`
  unconjugated (1 W on a lossless line), which keeps S symmetric for a
  reciprocal device. S is symmetric when every port has the same kind of
  reference (all real, all wave ports with equal ``z_ref``, or all modal);
  a wave port with a ``voltage_path`` beside one without breaks the
  symmetry by the ratio of their normalizations on a lossy line.
  ``WavePort(offset=d)`` refers S to a plane :math:`d` inside the
  domain: :math:`S_{ij}` times :math:`e^{\gamma_i d_i} e^{\gamma_j d_j}`.
  Without a ``voltage_path``, the sign of a wave port's row and column of
  S is fixed by a mesh-dependent rule; every :math:`|S|` and power is
  unaffected.
- ``NetworkParameters.y()``, ``.z()`` and ``.renormalize(z_new)`` use the
  pseudo-wave formulas; ``to_touchstone`` labels a file with ``z_ref`` only
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
  :math:`u = E\, e^{+i k \cdot r}` gives
  :math:`S_{ij}\, e^{+i k \cdot (r_i - r_j)}`, with i the output port, j the
  excited port and :math:`r_p` a point of port p, when :math:`k \cdot r` is
  constant over each port.

Time-averaged densities
-----------------------
For peak phasors, the time average of a product of two harmonic quantities
carries a factor 1/2:

- electric energy density :math:`w_e = \mathrm{Re}(E \cdot D^*) / 4`
  (1/2 for the time average of :math:`(E \cdot D) / 2`), magnetic
  :math:`w_m = \mathrm{Re}(B \cdot H^*) / 4`;
- Poynting vector :math:`S = \mathrm{Re}(E \times H^*) / 2`;
- ohmic loss density :math:`p = \mathrm{Re}(\sigma_\mathrm{eff})\, |E|^2 / 2`;
- a lumped port's power :math:`|V|^2 / (2 R)`.

With 1 W incident, the power a driven field delivers into the network
through the excited port is :math:`1 - |S_{ii}|^2` W, and the power arriving
at port j's termination is :math:`|S_{ji}|^2` W.

Surface conductors
------------------
A `Conductor` face adds the boundary term
:math:`\frac{i\omega}{Z_s(\omega)} \int E_t \cdot v_t`, with
:math:`Z_s` the surface impedance of a metal of conductivity :math:`\sigma`
and thickness :math:`t` in the finite-thickness form of HFSS:

.. math::

    Z_s = \frac{1}{\sigma \delta}
          \frac{(\sinh\nu + \sin\nu) + i (\sinh\nu - \sin\nu)}
               {\cosh\nu - \cos\nu}

with skin depth :math:`\delta = \sqrt{2 / (\mu_0 \sigma \omega)}` and
:math:`\nu = t / \delta` (``thickness=None``:
:math:`Z_s = (1 + i) / (\sigma \delta)`). This form describes ONE
face of a film whose other face is modelled separately, as the shell of an
excluded metal volume is: for a film much thinner than its skin depth,
:math:`Z_s \to 2 / (\sigma t)`.

- **Interior surface** (solve domain on both sides, e.g. a zero-thickness
  sheet): one term for the whole sheet, so a thin film is a sheet
  admittance of about :math:`\sigma t / 2` (`INTERIOR_SHEET_ADMITTANCE`
  times :math:`\sigma t`), not the film's conductance :math:`\sigma t`. To
  model a zero-thickness film of thickness t with its full conductance,
  give ``thickness=2 * t``.
- **Exterior surface** (mesh boundary, one-sided): the thickness counts
  twice (`EXTERIOR_THICKNESS_FACTOR`), so a thin film is :math:`\sigma t`.
- **Shell of an excluded volume** (dielectric on one side, the excluded
  metal on the other): an interior face of the mesh, one term per face with
  the given thickness; the top and bottom faces of a trace together carry
  :math:`\sigma t`.

gsim layouts assume this convention face for face.

Dispersive permittivity
-----------------------
A `gfem.PermittivityLaw` is a relative permittivity that depends on
frequency,
:math:`\varepsilon_r(f) = \varepsilon_\infty + \sum \mathrm{term}(f)`.
Every law parameter that is a frequency is an ordinary frequency in hertz
(:math:`\omega = 2\pi f`), not rad/s or eV (convert an energy as
:math:`f = E / h`). Under :math:`e^{+i\omega t}` the terms are:

- `gfem.Drude`: :math:`-\omega_p^2 / (\omega (\omega - i\gamma))`,
  :math:`\omega_p = 2\pi \cdot` ``plasma_frequency``,
  :math:`\gamma = 2\pi \cdot` ``damping``; no static value (it diverges as
  :math:`f \to 0`);
- `gfem.Lorentz`:
  :math:`\Delta\varepsilon\, \omega_0^2 / (\omega_0^2 - \omega^2 + i\gamma\omega)`,
  :math:`\omega_0 = 2\pi \cdot` ``resonance_frequency``;
- `gfem.Debye`: :math:`\Delta\varepsilon / (1 + i f / f_r)`,
  :math:`f_r` = ``relaxation_frequency`` (a relaxation time :math:`\tau` is
  :math:`f_r = 1 / (2\pi\tau)`);
- `gfem.DjordjevicSarkar`:
  :math:`\frac{\Delta\varepsilon}{\ln(f_2 / f_1)} \ln\frac{f_2 + i f}{f_1 + i f}`,
  the principal log of the ratio, corners :math:`f_1` = ``low_frequency``
  and :math:`f_2` = ``high_frequency``.

:math:`\Delta\varepsilon` is each term's ``delta_eps``. With every strength
and damping :math:`\ge 0` each term is passive alone, so a law has
:math:`\mathrm{Im}\,\varepsilon_r \le 0` at every real frequency; a negative
strength is refused. A law evaluates at a complex frequency as the analytic
continuation (:math:`\mathrm{Im} f > 0` decays), and
``law(0)`` is the static permittivity. Most optics fits (Drude, Lorentz)
are written under :math:`e^{-i\omega t}`: there the same law is the complex
conjugate, so their :math:`+i\gamma` reads :math:`-i\gamma` here, with the
same parameter values. Circuit-board laws (Djordjevic-Sarkar) are usually
written under :math:`e^{+j\omega t}` already.

Eigenmodes
----------
An eigenmode is a complex angular frequency :math:`\omega` and a field
:math:`E` with :math:`\mathrm{Re}(E\, e^{+i\omega t})` source-free;
:math:`\mathrm{Im}\,\omega > 0` decays, and
:math:`Q = |\omega| / (2 |\mathrm{Im}\,\omega|)`. The mode's
electric energy is:

.. math::

    E_\mathrm{elec} = \frac{1}{2} \mathrm{Re} \int D^* \cdot E \, dV
        + \sum_j \frac{C_j |V_j|^2}{2}
        + \sum \frac{1}{2} \int w\, C_s |E_t|^2 \, dS

over the solve domain, the capacitive lumped ports (capacitance :math:`C_j`,
voltage :math:`V_j`) and the capacitive `Impedance` sheets (capacitance per
square :math:`C_s`, tangential field :math:`E_t`, fill :math:`w`, 1 without
a `SheetField`) (peak phasors: the total time-averaged energy of a lossless
mode, electric plus magnetic).
In a region with a `gfem.PermittivityLaw`, :math:`\mathrm{Re}(D^* \cdot E)`
reads the energy permittivity
:math:`\mathrm{Re}[\varepsilon + (\omega / 2)\, d\varepsilon / d\omega]` at
the mode's real frequency, so that :math:`E_\mathrm{elec}` stays the total
energy of a lossless mode (Brillouin); for a lossy law it is the weak-loss
value. An eigenmode solve raises where that weight is not positive (loss
dominates the law there). The phase below reads the law's ``eps_inf`` in
:math:`D`. Each mode field is scaled to :math:`E_\mathrm{elec}` =
`MODE_ENERGY` and phased so that :math:`\int D \cdot E \, dV` (no conjugate)
is real and positive, which makes a lossless mode's field real; the
remaining sign puts the largest coefficient's real part positive (among
coefficients equal to the largest within a relative 1e-12, the lowest
index). Participation ratios and coupling Q are ratios to
:math:`E_\mathrm{elec}` and do not depend on this scale.

Bloch periodicity
-----------------
A `gfem.Periodic` pair relates two surfaces by the translation :math:`a`,
from its first surface to its second (`Model.periodic_translations`, m).
With a Bloch wave vector :math:`k` (rad/m, ``wave_vector=`` on `gfem.Driven`
and `gfem.Eigenmode`), the fields obey `BLOCH_PHASE`:

.. math::

    E(r + a) = e^{-i k \cdot a} E(r)

so, under :math:`e^{+i\omega t}`, the phase of a Bloch wave travels along
:math:`+k`. Without a wave vector the fields are equal on both surfaces
(plain periodicity). Swapping a pair's surfaces flips :math:`a` and gives
the same field.

Under a Bloch wave vector, :math:`\int D \cdot E \, dV` over the cell can
vanish, so an eigenmode field is phased to make its largest coefficient real
and positive instead (still scaled to `MODE_ENERGY`).

Floquet ports
-------------
A `gfem.FloquetPort` is one diffraction order in one polarization of a plane
wave on an open face of a periodic cell (a Floquet surface).

- Lattice frame: :math:`a_1` and :math:`a_2` are the translations of the
  face's two pairs, in ``boundaries`` order;
  :math:`\hat e_1 = a_1 / |a_1|`,
  :math:`\hat e_3 = a_1 \times a_2 / |a_1 \times a_2|`,
  :math:`\hat e_2 = \hat e_3 \times \hat e_1`. Order :math:`(m, n)` has the
  in-plane wave vector :math:`\kappa = k_t + m b_1 + n b_2` with
  :math:`b_i \cdot a_j = 2\pi \delta_{ij}` and :math:`k_t` the problem's
  ``wave_vector`` (the specular order's). Its field on the face is
  :math:`c\, \hat e\, e^{-i \kappa \cdot r}`, with :math:`r` the model
  coordinates: the in-plane phase is referred to the model origin, and
  translating the model in-plane by :math:`r_0` multiplies :math:`S_{pq}`
  by :math:`e^{-i (\kappa_p - \kappa_q) \cdot r_0}`.
- References: TE :math:`\hat e = \hat e_3 \times \hat\kappa` (s), TM
  :math:`\hat e = \hat\kappa` (p), the same on both faces of a cell, with
  :math:`\hat\kappa = \hat e_1` at :math:`\kappa = 0` (at normal incidence
  TE is E along :math:`\hat e_2`, TM along :math:`\hat e_1`).
- Normalization: :math:`S_{pq} = (c_p / c_q) \sqrt{Z_q / Z_p}`, :math:`c`
  the tangential-E coefficients at the face plane and :math:`Z` the order's
  wave impedance in the homogeneous medium behind its own face,
  :math:`\eta / \cos\theta` (TE) or :math:`\eta \cos\theta` (TM),
  :math:`\eta = \eta_0 \sqrt{\mu_r / \varepsilon_r}` and :math:`\theta` the
  order's angle from the face normal. The incident wave carries
  `INCIDENT_POWER` per cell, :math:`|S|^2` is the diffraction efficiency,
  and for a lossless cell the efficiencies of every propagating order on
  both faces, both polarizations, add up to 1. ``z_ref`` is :math:`Z`.
- Signs: tangential-E coefficients, so a flat interface reflects
  :math:`(Z_2 - Z_1) / (Z_2 + Z_1)`, :math:`(n_1 - n_2) / (n_1 + n_2)` in
  both polarizations at normal incidence from medium 1 (the Fresnel
  :math:`r_p` of many optics texts has the opposite sign), and an empty cell
  of thickness :math:`L` transmits :math:`e^{-i\beta L}`.
- Orders that do not propagate: where an order is evanescent or grazing
  (:math:`\gamma^2 \ge 0`), its row of S is exactly 0 and its ``z_ref`` is
  NaN, and it cannot be excited. ``FloquetPort(offset=d)`` multiplies the
  row and column by :math:`e^{\gamma d}` where the order propagates, 1
  elsewhere.
- Bloch reciprocity:
  :math:`S_{pq}(k) = \sigma_p \sigma_q S_{-q,-p}(-k)`, where
  :math:`-p` is port p's face and polarization with order
  :math:`(-m, -n)` and :math:`\sigma = -1` for an order with
  :math:`\kappa \ne 0`, :math:`+1` for :math:`\kappa = 0` (the references
  are odd in :math:`\hat\kappa`). The plain transpose does not hold as it
  stands.
- Circular polarization: with the TE and TM ports of one order on a face whose
  outward normal is :math:`+\hat e_3` (incidence along :math:`-\hat e_3`),
  :math:`(S_{:,\mathrm{TM}} + i S_{:,\mathrm{TE}}) / \sqrt{2}` is the response
  to IEEE right-handed circular incidence; on a face whose outward normal is
  :math:`-\hat e_3` the same combination is left-handed.

2D models
---------
A model whose mesh has dimension 2 (``Mesh.dim == 2``, read from a
triangle-only gmsh file) lies in the :math:`(x, y)` plane: its nodes are
:math:`(x, y)`, :math:`z` is normal to the plane, the fields do not vary
along :math:`z` (:math:`\partial_z = 0`), and waves travel in the plane.
A driven solve computes one polarization, the field component normal to the
plane, chosen by ``Driven(polarization=...)``:

.. math::

    \text{Ez:}\quad & \nabla \cdot (\mu_r^{-1} \nabla E_z)
        + k_0^2 \varepsilon_r E_z = 0 \\
    \text{Hz:}\quad & \nabla \cdot (\varepsilon_r^{-1} \nabla H_z)
        + k_0^2 \mu_r H_z = 0

with :math:`k_0 = \omega / c` and :math:`\varepsilon_r` including a
conductivity as :math:`-i\sigma / (\omega \varepsilon_0)` under
:math:`e^{+i\omega t}`. Ez has :math:`E` normal to the plane and
:math:`H` in it; Hz the reverse. A `gfem.PEC` line sets
:math:`E_z = 0` in Ez and leaves :math:`H_z` free (zero normal derivative)
in Hz; a `gfem.PMC` line does the reverse.

Every extensive quantity of a 2D model is per metre along :math:`z`: each
excited port is fed 1 W/m time-averaged incident power, and a kept field
(``"Ez"`` in V/m, ``"Hz"`` in A/m) is the peak phasor for that 1 W/m. S is
dimensionless. Against a slab of thickness :math:`h` between plates
(``Mesh.from_gmsh_cross_section``) fed 1 W, which is :math:`1/h` W/m, S is
the same and the fields relate by

.. math::

    E^{\mathrm{2D}} = \sqrt{h}\, E^{\mathrm{slab}}.

Quantum metrics
---------------
`gfem.quantum.dressed_modes` puts one junction's cosine on the solved
linear modes:

.. math::

    H / \hbar &= \sum_m \omega_m a_m^\dagger a_m
        - (E_J / \hbar) \left[\cos\varphi - 1 + \varphi^2 / 2\right] \\
    \varphi &= \sum_m \varphi_m (a_m + a_m^\dagger) \\
    \varphi_m &= \mathrm{sign}(p_m) \sqrt{|p_m| \hbar \omega_m / (2 E_J)} \\
    E_J &= (\hbar / 2e)^2 / L_J

with :math:`\omega_m` the real part of the mode's angular frequency,
:math:`p_m` its signed junction participation and :math:`L_J` the junction's
linear inductance. With :math:`E` its levels (:math:`E(0)` the ground
state, :math:`E(1_m)` one photon in mode :math:`m`), every output is a
signed angular frequency in rad/s and a difference of transition
frequencies, so its sign is an observable:

- dressed frequency :math:`E(1_m) - E(0)`; the Lamb shift is that minus
  :math:`\omega_m`;
- anharmonicity :math:`\alpha = E(2_m) - 2 E(1_m) + E(0)`, the second
  transition frequency minus the first;
- dispersive shift :math:`\chi = E(1_m 1_n) - E(1_m) - E(1_n) + E(0)`: the
  full shift of mode :math:`n`'s frequency per photon in mode :math:`m`, not
  half of it.

For an ordinary transmon all three are negative. Other sources' conventions
convert as
:math:`\chi = -\chi_\mathrm{Minev} = \chi_\mathrm{Nigg} = 2 \chi_\mathrm{Koch}`
and :math:`\alpha = -\alpha_\mathrm{Minev} = \alpha_\mathrm{Koch}` (Minev et
al. and pyEPR report a downward shift as positive; Koch et al. write half
the shift).

Temperature and thermo-optic phase
----------------------------------
Temperatures are absolute, in kelvin. A thermal result's rise is the
temperature minus the source-free temperature (the temperature with every
`gfem.HeatSource` off, which boundary temperatures and fluxes still set),
so it is the heaters' response even where the boundaries differ.

A material's ``thermo_optic_coefficient`` :math:`dn/dT` (1/K, any sign) acts
on the rise about the index :math:`n_0 = \sqrt{\varepsilon_r}` of its
``eps_r``, the index at the source-free temperature; no reference
temperature is stored. `gfem.Model.with_temperature_rise` gives each element
:math:`\varepsilon_r = (n_0 + (dn/dT)\, \Delta T_e)^2`, :math:`\Delta T_e`
(``rise_e``) the element's mean rise.

`gfem.thermo_optic_shift` is first order in the rise, about the port's
mode :math:`E_0` at 1 W on a lossless section:

.. math::

    \delta n_\mathrm{eff} &= \frac{c \varepsilon_0}{4} \sum_e 2 n_{0,e}
        (dn/dT)_e\, \Delta T_e \int_e |E_0|^2 \, dA \\
    \text{phase per length} &= k_0\, \mathrm{Re}(\delta n_\mathrm{eff})
        \quad \text{(rad/m)}

``phase_per_length`` is the extra phase delay per unit length of guide:
positive for a positive coefficient and a positive rise. Under
:math:`e^{+i\omega t}` a guide of length :math:`L` transmits
:math:`e^{-i\beta L}`, so a transmission's argument moves by minus
``phase_per_length * L``. The power that gives a :math:`\pi` shift is
:math:`P_\pi = \pi P' /` ``phase_per_length`` for a heater dissipating
:math:`P'` per unit length (``power_per_length``), which equals
:math:`\lambda / (2 s)` with :math:`s = \delta n_\mathrm{eff} / P'`.

`gfem.route_phase` integrates the same first-order phase per length along
a route through a 3D rise (rad, a phase delay, positive for a positive
coefficient and rise; :math:`P_\pi = \pi P /` ``phase`` for a heater
dissipating :math:`P`). Its window is in route-local coordinates: the
section's x points to the left of travel (``up x tangent``), its y along
``up``, and the station sits at the section's origin.

Incident fields
---------------
A `gfem.PlaneWave` or `gfem.GaussianBeam` travels along

.. math::

    \hat k = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta),

the propagation direction in spherical angles from :math:`+z`, in rad:
:math:`\theta < \pi/2` travels up, :math:`\theta > \pi/2` down (30° off
the normal from above is :math:`\theta = \pi - 30°`).

- Polarization basis: TM (p) is :math:`\hat\theta = \partial_\theta \hat k
  = (\cos\theta\cos\phi, \cos\theta\sin\phi, -\sin\theta)`, TE (s) is
  :math:`\hat\phi = \hat z \times \hat\kappa = (-\sin\phi, \cos\phi, 0)`
  with :math:`\hat\kappa = (\cos\phi, \sin\phi, 0)`; :math:`(\hat\theta,
  \hat\phi, \hat k)` is right-handed. Along :math:`\pm z`, :math:`\phi`
  still fixes the plane of incidence: TM is :math:`(\cos\phi, \sin\phi, 0)`
  at :math:`\theta = 0` and :math:`-(\cos\phi, \sin\phi, 0)` at
  :math:`\theta = \pi`.
- Floquet references: TE equals a `gfem.FloquetPort`'s TE reference in
  every direction. TM's tangential part is :math:`\cos\theta\,\hat\kappa`,
  along the Floquet TM reference for a wave going up and opposite to it for
  a wave going down.
- A polarization is a Jones pair :math:`(a_\mathrm{TM}, a_\mathrm{TE})`,
  normalized: the field direction is :math:`(a_\mathrm{TM}\hat\theta +
  a_\mathrm{TE}\hat\phi) / \lVert a\rVert`. ``"TM"`` is :math:`(1, 0)`,
  ``"TE"`` :math:`(0, 1)`, ``"RCP"`` :math:`(1, -i)/\sqrt2` and ``"LCP"``
  :math:`(1, i)/\sqrt2`, IEEE handedness under :math:`e^{+i\omega t}`: with
  the thumb along :math:`\hat k`, the real field of RCP turns from
  :math:`\hat\theta` to :math:`\hat\phi` (right hand), as for the Floquet
  ports' circular combination.
- Plane wave amplitude: :math:`E_\mathrm{inc}(r) = E_0\, \hat e\,
  e^{-i k \hat k \cdot r}` in a homogeneous background of wavenumber
  :math:`k`, with :math:`E_0` (``amplitude``) the peak field in V/m in the
  layer the wave comes from, complex for a phase, and :math:`r` the model
  coordinates: the phase is referred to the model origin, as the Floquet
  in-plane phase is.
- Gaussian beam: 1 W (`INCIDENT_POWER`) time-averaged power through any
  plane normal to its axis, and the field at the focus real and positive
  along the polarization (zero phase at the focus); the waist is a length,
  the same at every frequency.
- Total field = incident field + scattered field. A kept field of a source
  column is the total field; the scattered field is outgoing at the
  truncation. In a perfectly matched layer the kept field of an incident
  column is the scattered field: the incident field is zero there. Its
  incident part (``DrivenResult.e_field(..., part="incident")``) is the
  incident field interpolated on the kept field's space outside the layer,
  and its scattered part the kept field minus it, so the three add exactly.
- Truncation and walls: where the incident field is dropped or must hold a
  wall, it must stay at most :math:`10^{-3}` of its peak over the physical
  region: where a scatterer (a region whose material differs from the
  background, a material field always, or a sheet, conductor or lumped
  face) meets the layer or an absorbing face, on a wave port face, and as
  :math:`|n\times E|` on an exterior PEC wall or :math:`\eta|n\times H|`
  on an exterior PMC wall (named, or by default).
- Port waves: in a source column every port is terminated in its reference
  impedance, and ``DrivenResult.port_waves`` is the outgoing pseudo-wave
  :math:`b` at each port, in √W as a column of S: on a real ``z_ref``,
  :math:`|b|^2` is the power in W the source sends into the port.

Point dipoles
-------------
A `gfem.ElectricDipole` of moment :math:`p` (C·m) at :math:`r_0` is the
current :math:`J = i\omega p\, \delta(r - r_0)` under :math:`e^{+i\omega t}`;
a `gfem.MagneticDipole` of moment :math:`m` (A·m²) is the current
:math:`J = \nabla \times (m\, \delta(r - r_0))`, i.e. the magnetic current
:math:`M = i\omega\mu_0 m\, \delta(r - r_0)`. Both moments are peak
phasors, the same at every frequency. An antenna current moment :math:`Il`
(A·m) is :math:`p = Il / (i\omega)`.

- Emitted power: the time-averaged power the dipole gives the field,

  .. math::

      P = -\frac{\omega}{2} \operatorname{Im}\left(p^* \cdot E(r_0)\right),

  positive for a dipole that radiates or drives a lossy medium; gfem pairs
  the dipole's load with the discrete solution, which is finite although the
  exact field at :math:`r_0` is not. Its real counterpart, the
  self-reaction, has no limit and is never reported.
- Bulk power: the emitted power of the same dipole in the unbounded,
  lossless material at :math:`r_0`, with :math:`n = \sqrt{\varepsilon_r
  \mu_r}` and :math:`c` the speed of light in vacuum,

  .. math::

      P_0^{(p)} = \frac{\mu_0 \mu_r n\, \omega^4 |p|^2}{12 \pi c}, \qquad
      P_0^{(m)} = \frac{\mu_0 \varepsilon_r n\, \omega^4 |m|^2}{12 \pi c^3}.

- Purcell factor: :math:`F = P / P_0`. In a non-magnetic medium the partial
  local density of states along :math:`p` is :math:`\rho_p = \rho_0 F`, with
  :math:`\rho_0 = \omega^2 n / (\pi^2 c^3)`.

Far fields
----------
`gfem.far_field` reads the field radiated into the exterior medium, the
model's background (one constant, isotropic, lossless `gfem.Material`), of
wavenumber :math:`k = n\omega/c` and wave impedance :math:`\eta =
\eta_0\sqrt{\mu_r/\varepsilon_r}`.

- Far-field amplitude: under :math:`e^{+i\omega t}` an outgoing wave is
  :math:`e^{-ikr}/r`, and

  .. math::

      F(\hat r) = \lim_{r\to\infty} r\,e^{+ikr} E(r\hat r),

  in V, with :math:`r` measured from the model origin, the phase reference.
  :math:`\hat r` is given by the same angles as an incident field's
  :math:`\hat k`, and :math:`F = F_\theta\hat\theta + F_\phi\hat\phi`
  in that basis. For an incident-field column it is the scattered field's.
- Circular components: :math:`F_R = (F_\theta + iF_\phi)/\sqrt2` and
  :math:`F_L = (F_\theta - iF_\phi)/\sqrt2`, the field's components along
  ``"RCP"`` and ``"LCP"`` (IEEE handedness); axial ratio
  :math:`(|F_R| + |F_L|)/\big||F_R| - |F_L|\big|`, 1 for circular and
  infinite for linear polarization.
- Radiation intensity :math:`U = |F|^2/(2\eta)` in W/sr (peak phasors);
  radiated power :math:`P_\mathrm{rad}`, the power flowing out through the
  far-field layer. The IEEE 145 ratios: directivity :math:`D = 4\pi
  U/P_\mathrm{rad}`, gain :math:`G = 4\pi U/P_\mathrm{acc}`, realized gain
  :math:`G_r = 4\pi U/P_\mathrm{inc}`, radiation efficiency
  :math:`P_\mathrm{rad}/P_\mathrm{acc}`. For a port column
  :math:`P_\mathrm{inc}` is `INCIDENT_POWER` and :math:`P_\mathrm{acc} = 1 -
  |S_{ii}|^2`, so :math:`G_r = 4\pi U`; for a point dipole both are its
  emitted power.
- Bistatic radar cross section of a `gfem.PlaneWave` column, in m²:
  :math:`\sigma(\hat r) = 4\pi|F(\hat r)|^2/|E_0|^2`.
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
#: The wave definition of S-parameters (Marks and Williams; power waves on a
#: real reference).
WAVES = "pseudo"
#: Sheet admittance of one `Conductor` term on an interior surface for a film
#: much thinner than its skin depth, in units of :math:`\sigma t` (conductivity
#: times thickness).
INTERIOR_SHEET_ADMITTANCE = 0.5
#: Factor on a `Conductor`'s thickness on mesh-exterior faces, where the metal
#: is one-sided.
EXTERIOR_THICKNESS_FACTOR = 2.0
#: Electric energy of every eigenmode field (see Eigenmodes), J.
MODE_ENERGY = 1.0
#: The Bloch phase across a periodic pair of translation a (see Bloch
#: periodicity).
BLOCH_PHASE = "E(r + a) = exp(-i k·a) E(r)"
