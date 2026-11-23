# How gfem is checked

Each claim on {doc}`capabilities` and in the recipes is backed by a test that
solves a case with a known answer. The answer is a closed form, an identity
the physics must obey, an independent method, or finite differences for a
gradient. This
page lists those cases by area, in physical terms, and links the test files
on GitHub. The tolerance each check asserts is in its test, not on this page.

Most checks run on every change to gfem. Cases too large for that run before
each release, and the PETSc and SLEPc checks run twice a week; each section
says which.

## Gradients

Every FD-verified mark of the gradient matrix on {doc}`capabilities` is a set
of cells, one per problem, input, linear solver and sweep. Each cell compares
the gradient with fourth-order central differences of the same objective,
with tight solver tolerances. It then checks, at the default tolerances, that
the Taylor remainder shrinks at second order as the step shrinks.

```{gfem-gradient-checks}
```

The inputs are the per-element permittivity and conductivity (isotropic, and
as principal values on tilted axes), the permeability and the loss tangent
(driven and eigenmode problems), the fill of a sheet, the mesh node
coordinates moved through `gfem.shape.boundary_morph`, the parameters of a
permittivity law (its high-frequency permittivity, strengths and
frequencies) and the density of a metal-dielectric blend. The problems are
capacitance and the electric energy of each region, driven S on lumped ports, on wave ports and between modes of a
shared port surface, driven S on periodic and Bloch cells,
eigenfrequency and Q, and, on the direct solver, the participation ratios,
port Q and mode field of an isolated mode and the dispersive shift and
anharmonicity of a junction's dressed modes.

An exact zero is checked to be zero. Each "raises" mark, and each row of
"Gradients that raise", is a call that must raise the named error. Both the
matrix and that list are built from these tests when the docs are built, so
they cannot drift from them.

The same cells run on `linear_solver="petsc"` and `eigen_solver="slepc"`
where the {term}`PETSc stack` is installed (see
[PETSc and SLEPc](#verification-petsc)), so the matrix does not list PETSc.
The cells between modes of a shared port surface are among the large cases
that run before each release; the other cells run on every change.

A driven optimization on `linear_solver="direct"` that solves designs on an
earlier design's factorization is checked against factorizing every design:
S agrees to a twentieth of `tolerance` and the gradient to `tolerance`.

Tests: [gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py),
[gradients that raise](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix_raise_table.py),
[reused factorizations](https://github.com/benvial/gfem/blob/main/tests/verification/test_factor_reuse_gradient.py).

## Low frequency

The {term}`tree-cotree gauge` is checked on a case whose S has a closed form
far below the low-frequency floor: a parallel-plate line between two 50 Ω
lumped ports, its S from the line's ABCD matrix. Its mesh's floor is about
95 kHz.

- **At 10 kHz** ($k_0 h_\mathrm{min} \approx 10^{-7}$), the gauged direct solve
  gives S within $10^{-13}$ of the closed form (measured:
  $4 \times 10^{-15}$), with $S_{12} = S_{21}$ to $10^{-14}$. The same solve
  with `gauge="none"` is off by more than $10^{-10}$ (measured:
  $2 \times 10^{-8}$).
- **At 1 MHz and 100 MHz**, the gauged S is within $10^{-13}$ of the closed
  form too.
- **The adjoint gradient** of $S_{11}$ with respect to the permittivity at
  10 kHz matches the derivative of the closed form to $10^{-10}$, relative.
- **An adaptive sweep** through the gauge keeps the same digits below the
  floor, and above it matches the uniform gauged sweep to the sweep
  tolerance.

A model with wave ports gives the same S gauged and ungauged above the
floor. Below it, the wave ports' mode problem is not gauged, and the solve warns.

Tests: [gauged solves](https://github.com/benvial/gfem/blob/main/tests/unit/test_gauge_solve.py),
[adaptive sweep](https://github.com/benvial/gfem/blob/main/tests/unit/test_gauge_adaptive.py),
[gauge basis](https://github.com/benvial/gfem/blob/main/tests/unit/test_gauge_basis.py).

## Wave ports and shared port surfaces

- A matched parallel-plate line transmits its TEM mode, and its first higher
  mode, with their analytic propagation.
- An open dielectric guide between two wave ports transmits its fundamental
  mode with its slab's effective index, almost without reflection, and S is
  passive and reciprocal.
- A rectangular guide with a dielectric slab against one wall converts TE10
  into TE20. S, with both modes on each end's shared port surface, matches an
  independent mode-matching solution. It is unitary and reciprocal, and the
  power through each lead equals the sum of the requested modes' |S|².
- Before each release: TE0 and TM0 of a 500 × 220 nm silicon strip, requested
  on one port surface at each end, are each transmitted with their own phase,
  without reflection or conversion.

Tests: [shared port surface](https://github.com/benvial/gfem/blob/main/tests/verification/test_shared_port_surface.py),
[wave ports](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_wave_ports.py),
[silicon strip](https://github.com/benvial/gfem/blob/main/tests/reference/test_shared_port_surface.py).

## 2D models

Each case solves a {term}`2D model` and compares it with a closed form, with
the same plane posed as a {term}`slab <Slab model>`, or with finite
differences:

- **A loaded line, both polarizations, both kinds of wall.** The topology
  tutorial's line in top view, a section of $\varepsilon_r = 10$ between air feeds,
  transmits and reflects as the closed-form three-part transmission line. Its
  TEM mode is checked in Ez between magnetic walls and in Hz between conducting
  walls; the mode that varies as $\sin(\pi y / w)$ across the width, above its cutoff,
  in Ez between conducting walls and in Hz between magnetic walls. A straight
  guide passes its mode with no reflection, and a conducting wall reflects the
  electric field with $-1$ in both polarizations.
- **Guide modes.** The {term}`line port <Line port>` across the bend tutorial's
  guide finds the effective index of the symmetric slab guide's closed form,
  even mode with E along z (Ez) and with H along z (Hz), at 1.50, 1.55 and
  1.60 µm.
- **Agreement with the slab.** The bend, with its design square all cladding or
  all core, gives the same S as a slab one cell thick between PEC plates in Ez.
  In Hz the bend solves exactly the plain $H_z$ equation with the line port's
  boundary term; the slab, between magnetic plates, solves it with its own port
  term, and agrees with that problem instead. The topology line's S agrees with
  the slab's lumped ports matched to the line and with the closed form, empty
  and with a uniform ceramic section. The field of the 2D model is
  $\sqrt{h}$ times the slab's, for a slab $h$ thick fed 1 W.
- **Gradients.** Gradients of S by per-element material fields (an index blend
  with a Drude end in Ez, constant permittivities in Hz, a loss tangent and a
  permeability) and by the pixels of a layer grid at element orders 2 and 3
  match central differences, with a Taylor remainder of slope 2. The pixels'
  permittivity, read inside each triangle, is their bilinear interpolation. A
  design area that owns a second-order absorbing line is differentiated too.
  The {doc}`tutorials/metal-bend` bend's band objective, through gold's law and
  the index blend, matches a difference on both the 2D model and the slab.
- Before each release, on that bend's 45° gold mirror: the tutorial mesh at
  element order 3 gives the S of the same 2D model on cells eight times
  smaller at order 2, a value computed once and frozen; and the band
  objective's gradient with respect to the pixels points the same way on the
  2D model and on the slab, though the two read the pixels at different
  points.

Tests: [closed forms and the slab](https://github.com/benvial/gfem/blob/main/tests/verification/test_2d_models.py),
[guide modes](https://github.com/benvial/gfem/blob/main/tests/verification/test_2d_line_port_modes.py),
[material-field gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_2d_gradients.py),
[layer-grid gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_2d_layer_gradients.py),
[metal bend](https://github.com/benvial/gfem/blob/main/tests/verification/test_metal_bend.py),
[metal bend, fine mesh and slab gradient](https://github.com/benvial/gfem/blob/main/tests/reference/test_metal_bend_fine.py).

## Periodic cells

Each case compares a periodic solve with a closed form:

- **A Bloch two-port line.** A lossy parallel-plate line periodic along x,
  with two 50 Ω lumped ports, gives the S of the closed-form Bloch impedance
  matrix at +k, −k and 0, on the direct and BDDC + GMRES solvers. S at +k is
  far from the closed form at −k, which fixes the sign of the Bloch phase.
- **The same line, loaded.** Loaded by its two ports, the line is a damped
  Bloch eigenmode problem. Its complex resonance is the root of
  det(Z(f, k) + R) = 0.
- **An empty parallel-plate cell periodic along x and y.** Its four lowest
  bands are those of the empty lattice and are real. The lowest band's field
  advances by exp(−i k·Δr) between two points.
- **A layered (Bragg) line.** Its bands are the transfer-matrix roots,
  lossless and lossy, at both edges of the gap at k = π/a, at π/2a, and at
  k = 0 with plain periodicity.
- **An empty cube periodic along x, y and z.** Its lowest band is the
  light-line doublet c|k|/2π.

Further checks: a wave vector of (0, 0, 0) gives the solution with no wave
vector, plain periodic walls give the same S as PMC walls on a field uniform
across the pair, and S(k)ᵀ = S(−k) ({term}`Bloch reciprocity`). On PETSc,
the plain line's S matches the direct solver's, SLEPc finds the Bragg line's
bands, and a Bloch operator is refused.

Tests: [driven](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_driven.py),
[eigenmode](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_eigen.py),
[identities, driven](https://github.com/benvial/gfem/blob/main/tests/unit/test_periodic_driven.py),
[identities, eigenmode](https://github.com/benvial/gfem/blob/main/tests/unit/test_periodic_eigen.py),
[PETSc](https://github.com/benvial/gfem/blob/main/tests/verification/test_periodic_petsc.py).

## Floquet ports

Two cells with Floquet ports on both open faces, solved against references
that do not use gfem:

- **A dielectric slab on a substrate, under air.** Its reflection and
  transmission, complex, match the tangential-E transfer-matrix (Airy) form
  for TE and TM: at normal incidence, at conical incidence (40° from the
  normal, azimuth 30°) under a Bloch wave vector, and at a fixed angle over a
  band of three frequencies. The cross-polarized entries vanish, and each
  column's efficiencies add up to 1.
- **The same slab between air on both sides, at Brewster's angle.** Its TM
  reflection vanishes.
- **A lamellar grating.** The diffraction efficiency of every propagating
  order, reflected and transmitted, in both polarizations, matches a
  rigorous coupled-wave (RCWA) solution computed with many harmonics: at
  normal incidence, at 20°, at conical incidence (20°, azimuth 30°) and with
  a lossy ridge, whose missing power matches the RCWA absorption. In the
  lossless cases the efficiencies add up to 1, and the power flowing into
  the substrate, measured on the field, equals the transmitted efficiencies.
- **Reference planes.** Moving the open faces away from the slab, with the
  ports' offsets grown by the same distance, leaves S unchanged.

Identities checked on coarse cells: a flat glass-air interface reflects
(n − 1)/(n + 1) in both polarizations; a cell closed by a PEC wall reflects
all power; the full S of the grating obeys {term}`Bloch reciprocity` in its
Floquet form, with the order relabelling and the signs σ, at normal and
conical incidence, and fails it without the signs; a zero wave vector gives
the plain periodic S; a fixed-angle sweep gives the S of solves at each
frequency's own wave vector; S is continuous through a
{term}`Rayleigh anomaly`; the combination (S[:, TM] + i S[:, TE]) / √2
turns right-handed for incidence from the top face and left-handed from the
bottom one; and the BDDC + GMRES solver and frequency workers agree with the
serial direct solve. Gradients of the efficiencies through Floquet ports,
plain and Bloch, with respect to the permittivity and the node coordinates,
are in the gradient matrix ({doc}`capabilities`). All of these run on every
change to gfem.

Tests: [Floquet ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_floquet_driven.py),
[identities](https://github.com/benvial/gfem/blob/main/tests/unit/test_floquet_driven.py),
[orders](https://github.com/benvial/gfem/blob/main/tests/unit/test_floquet_orders.py).

## Perfectly matched layers

Each case ends a guide or a cell with a layer backed by a wall and compares
the reflection read at a port with the layer's closed form,
$|R| = R_0^{\,n f \cos\theta / f_\text{ref}}$, at order 3 on meshes layered
along the stretch:

- **A TEM line ended by a layer.** Its reflection is the target reflection
  at the reference frequency; with two frequencies and no reference
  frequency given, the lowest one is the reference and the other reflects
  less; a pinned reference frequency above the solve frequency reflects
  more, as the formula says; a background of index $n$ reflects $R_0^{\,n}$.
- **A TE10 guide ended by a layer.** Its mode is two plane waves at
  $\cos\theta = \beta/k_0$, and its reflection follows the formula, near
  cutoff too, where it rises; a reference frequency where $k_0 = \beta$
  brings it back below the mesh's floor.
- **A periodic cell over a layer, read at a Floquet port.** Its specular
  reflection follows the formula at normal, oblique and conical incidence,
  TE and TM, in vacuum and in a dielectric filling, with no cross-polarized
  reflection and no warning of an oblique absorber.
- **A leaky dielectric guide radiating into a layer.** Its complex
  resonance matches the root of its dispersion relation, plain and under a
  Bloch wave vector, and every returned mode stays put when the layer's
  target reflection changes, so no mode of the layer is returned.
- **A strip guide through side layers.** Between two wave ports cut by the
  layers, the guide transmits its own mode's propagation and reflects
  nothing, S is symmetric and passive, and the effective index matches a
  wide closed section's to the clearance rule.

Identities checked on coarse models: the layer absorbs the power the port
does not get back; a model without a layer is bitwise unchanged;
electrostatic and DC conduction solves see the layer's background material;
changing the target reflection changes the operator and its cache keys;
energies and participations count the physical regions; a port section cut
by side layers returns the guided mode; a solve across MPI ranks equals one
process's. Each refusal of the layer is a test of its own. Gradients next
to a layer (the permittivity of a slab beside it, nodes in the air, on the
layer's inner face and inside it, a boundary morph, a probe inside the
layer and an eigenfrequency) match finite differences. All of these run on
every change to gfem.

Tests: [closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_pml.py),
[identities and refusals](https://github.com/benvial/gfem/blob/main/tests/unit/test_pml.py),
[gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_pml_gradient.py).

## Anisotropic materials

- An absorbing boundary ending a biaxial, magnetic slab reflects as the
  half-space of transmission-line theory, for both polarizations and at two
  electrical slab thicknesses.
- A WR-28 guide filled with anisotropic ε and μ transmits its TE10 mode with
  the closed-form propagation constant, without reflecting. With its tensors
  turned about the guide axis, it does not reflect either.
- The port mode of that turned tensor propagates, with a propagation constant
  between those of the two guides filled with the tensor's principal values
  across the guide.
- The loss along each principal axis from {term}`principal participation`
  matches $1/Q$ from the complex eigenfrequency, on a material whose field lies
  off every principal axis.
- A model and its rigidly rotated copy, with the material given as rotated
  global matrices, give the same S, eigenfrequency and capacitance. A
  wave-port guide rotated to an oblique port normal gives the same S.
- A uniform anisotropic material field on tilted axes solves as the
  anisotropic material it holds.

Tests: [wave ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_anisotropic_wave_ports.py),
[rotation](https://github.com/benvial/gfem/blob/main/tests/unit/test_anisotropic_rotation.py),
[absorbing boundary](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_boundaries.py),
[port mode](https://github.com/benvial/gfem/blob/main/tests/unit/test_port_mode.py),
[principal participation](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode.py),
[material field](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven.py).

## Dispersive materials

Each check below uses a region whose permittivity follows a
{term}`permittivity law`:

- **Equivalence.** A dispersive region solves exactly as a constant material
  set to the law's permittivity at each frequency. This holds for every term
  kind, with a conductivity, and beside first- and second-order absorbing
  boundaries. It also holds for a metal (Re ε < 0), against the same
  permittivity set directly. In the electrostatic problem the law gives its
  {term}`static permittivity`.
- **Filled cavities.** A closed box filled with a Lorentz or Drude medium
  resonates at the closed-form polariton and plasma branches of its cutoffs.
  A Drude medium returns no field at its plasma frequency. A lossless Drude
  metal solved as a polynomial problem gives the same modes.
- **Resonances.** A target or mode where a law region's wavelength is
  shorter than two of its longest edges raises before solving. Below that
  limit the modes are the exact ones of the discretization, at orders 1
  and 3.
- **Mode energy.** The {term}`energy permittivity` makes the mode energy
  equal to the eigenproblem's own normalization. Beside a dispersive region,
  a junction's participation obeys −2 d ln ω / d ln L, and the surrounding
  dielectric's share
  obeys −2 d ln ω / d ln ε. Where the energy permittivity is not positive,
  the solve refuses.
- **Wave ports.** A guide filled with a Debye or Lorentz medium transmits its
  fundamental mode with the closed-form propagation constant at every
  frequency of a sweep. A metal in a port's cross-section is refused before
  any solve.
- **Adaptive sweep.** A lossless resonance inside the band is sampled on both
  sides, and the reduced model matches the uniform sweep. So does the sweep
  of a Djordjevic-Sarkar substrate and of a law region behind an absorbing
  boundary.
- **Gradients.** Law parameters are checked by finite differences on driven
  S (uniform and adaptive sweeps, wave ports, a periodic cell, behind
  absorbing boundaries), on eigenfrequency and Q, and on capacitance.
  Existing inputs are checked beside a law, including participations and
  mode fields beside a lossy Lorentz region. The density of a
  metal-dielectric blend is checked under both the index and the linear
  interpolation. The gradient of {doc}`tutorials/metal-bend` through gold
  is checked by finite differences.
- **Solvers and ranks.** The driven and eigenmode law cells also run on
  `linear_solver="petsc"`. On two MPI ranks, S and its gradient by a law
  agree with one process.
- **Mesh refinement.** A line with a Drude metal block is one of the field
  estimator's cases (see [Mesh refinement error
  estimates](#verification-mesh-refinement)).

Not checked: port modes of plasmonic guides (refused), absorbed power in a
dispersive region, agreement with measured metal data, eigenmodes inside a
resonance's crowded band, and convergence of the tutorial's unresolved skin
depth.

All of these run on every change. The PETSc cells run twice a week, as in
[PETSc and SLEPc](#verification-petsc).

Tests: [driven](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_dispersive.py),
[law gradients](https://github.com/benvial/gfem/blob/main/tests/unit/test_driven_law_gradient.py),
[cavities and bands](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_dispersive.py),
[mode energy](https://github.com/benvial/gfem/blob/main/tests/unit/test_mode_energy_dispersive.py),
[electrostatic](https://github.com/benvial/gfem/blob/main/tests/unit/test_electrostatic_dispersive.py),
[index blend](https://github.com/benvial/gfem/blob/main/tests/unit/test_index_blend.py),
[wave ports](https://github.com/benvial/gfem/blob/main/tests/verification/test_dispersive_wave_ports.py),
[adaptive sweep](https://github.com/benvial/gfem/blob/main/tests/verification/test_dispersive_adaptive.py),
[metal bend](https://github.com/benvial/gfem/blob/main/tests/verification/test_metal_bend.py),
[field estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_field_estimate.py),
[MPI](https://github.com/benvial/gfem/blob/main/tests/mpi/test_dispersive.py), and the law and blend cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py) and the
[gradients that raise](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix_raise_table.py).

## Patterned sheets

- On a parallel-plate line, a sheet of 2 µm of copper or of a 1 pH kinetic
  inductance, solid or with a void / solid pattern, takes BDDC + GMRES the
  iterations of an empty sheet, and its S matches the direct solver's.
- A copper pixel design graded across the default contrast passes the
  finite-difference check on both solvers.
- The whole design chain (pixels on a layer grid, filter, projection,
  exponential fill) differentiates end to end, and so do node coordinates
  that move the filled sheet.
- In a cavity with a resistive film of random fill, the eigenfrequency
  matches an independent iteration that refactorizes the operator at every
  step.

Tests: [sheets](https://github.com/benvial/gfem/blob/main/tests/verification/test_sheet_gradient.py).

## Quantum metrics

- **Participations.** A junction's participation equals
  −2 d ln ω / d ln L_J from two solves, with and without a capacitance on the
  port and beside a capacitive sheet. A port with an inductance and a
  capacitance at the end of a line resonates at the closed-form frequency of
  the loaded line.
- **The dressed modes.** As the junction's phase shrinks they tend to the
  first-order closed forms. They do not move when more Fock levels are
  added, a transmon below its resonator gets negative shifts, and a
  hybridized pair raises. They reproduce two stored outputs of the pyEPR
  tutorials, for one and two modes.
- **Gradients.** The gradient of the dressed modes matches fourth-order
  differences and differentiating the eigenvalue problem directly. The
  dispersive shift's gradient through a solve is checked by finite
  differences on a small two-mode model on every change, and before each
  release on the tutorial's transmon, along its pad gap.
- **Interface layers.** `"MA"` on a film with vacuum on both faces equals the
  sum of its two faces read one at a time.

Not checked: agreement with a measured device, and convergence of a sheet's
or an interface's participation at a film edge.

Tests: [dressed modes](https://github.com/benvial/gfem/blob/main/tests/unit/test_quantum.py),
[participations](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode.py),
[transmon](https://github.com/benvial/gfem/blob/main/tests/verification/test_transmon_chi.py),
[interface layers](https://github.com/benvial/gfem/blob/main/tests/verification/test_interface_participation.py),
and the dispersive-shift cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Thermal

Thermal and DC conduction solves are checked against closed forms, against
each other, and against two independent codes, gsim and femwell, on the same
heater. Each check runs on every change to gfem.

- **Exact in the discrete space.** Layered slabs, in series, under
  convection and with a convection-only bottom, give their hand temperatures,
  on a cross-section and on unstructured 3D tetrahedra. A bar gives
  R = L/(σA), a two-conductor slab the hand formula for its resistance per
  length, heat and power, and the power a conductor dissipates equals the
  power its terminals deliver. Switching a source on and off equals two
  superposed steps, and twice the power gives twice the rise.
- **Closed forms.** A strip heater on a substrate follows the ln coth profile
  of its temperature, and its far field decays over 2H/π. A slab's slowest
  thermal mode gives its time constant. The late decay of a step response
  matches it, with TR-BDF2 and with the θ schemes, and the 10–90 % and 1/e
  times match the analytic trace normalised over the same window. After ten
  time constants a run reaches the steady temperature. The resistance of an
  annular bend matches its closed form, and the Joule heat of each element is
  the exact mean of σ|∇V|² over it.
- **The thermo-optic shift.** The first-order shift agrees with a mode
  re-solved on the heated permittivity at the drive that gives π. An optical
  window around the core gives the same shift as the whole section with the
  other regions excluded. A route through a z-invariant 3D rise gives the
  section's phase per length times its length, and the first order agrees
  with a re-solved mode at a station in a 3D heater.
- **3D against the cross-section.** A heater section extruded in layers of
  tetrahedra gives the slab's resistance per length and its core rise per
  watt per metre.
- **Two independent codes.** On the femwell TiN heater section, meshed by
  gsim and read by gfem from gsim's own file, gfem's P_π matches gsim's
  (22.39 mW) and femwell's published 22.6 mW, and the heater's resistance
  matches gsim's. Driven at P_π with the backward Euler scheme on gsim's time
  grid, the rise time and the time constant match gsim's.
- **Gradients.** Every thermal gradient of the matrix on
  {doc}`capabilities`, and one chain from the heater's conductivity, every
  element's thermal conductivity, the core's thermo-optic coefficient and
  the drive current through DC conduction, Joule heating, the steady and the
  transient temperature to the phase, matches fourth-order central
  differences, on the direct and BDDC + GMRES solvers.

Two comparisons are observations, not checks, and live on the tutorials. The
TiN heater of Jacques et al. (2019): gfem's P_π falls inside the published
bands at both heater widths ({doc}`tutorials/heater`). The 3D spiral heater:
its P_π, resistance and per-pass phases on three meshes
({doc}`tutorials/heater3d`).

Tests: [closed forms](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermal_closed_forms.py),
[gsim and femwell](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermal_parity.py),
[extrusion](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermal_extrusion.py),
[DC conduction](https://github.com/benvial/gfem/blob/main/tests/verification/test_dc_conduction.py),
[thermo-optic shift](https://github.com/benvial/gfem/blob/main/tests/verification/test_thermo_optic.py),
[route phase](https://github.com/benvial/gfem/blob/main/tests/verification/test_route_phase.py),
[steady](https://github.com/benvial/gfem/blob/main/tests/unit/test_steady_thermal.py),
[transient](https://github.com/benvial/gfem/blob/main/tests/unit/test_transient_thermal.py),
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Electro-optics

The electro-optic shift is checked against closed forms, against modes
re-solved on the perturbed permittivity, and against itself on rotated and
remeshed sections. Each check runs on every change to gfem.

- **The electro-optic shift.** A lithium niobate film between two electrodes,
  closed by electric side walls, gives the slab's closed-form shift, for an
  x-cut film through $r_{33}$ and for a film whose field reaches $r_{22}$
  alone. The first order agrees with a mode re-solved on the perturbed
  permittivity at $\pm 10$ V: on a ridge, for the diagonal part of the change,
  and on an L-shaped core that breaks every mirror symmetry, for $r_{51}$
  alone, the off-diagonal part. An x-cut section and the same section turned
  by 90° into z-cut, crystal axes and electrodes with it, give the same
  shift; $r_{22}$ adds nothing where the mode and the field leave it nothing
  to act on. The shift is linear in the tensor and in the drive, a uniform
  static field is read exactly, and a tensor of the same value on every
  diagonal entry gives the thermo-optic shift of the matching coefficient.
- **Across meshes.** The static field is read at fixed points of the optical
  model, wherever they fall in the static mesh: an optical model on a
  second mesh of the same section gives the same shift to
  $5.5 	imes 10^{-4}$.
- **Capacitance per length.** Parallel plates on a section give
  $\varepsilon_0 \varepsilon_r H / d$ per metre, whatever the slab's thickness.
- **Gradients.** The gradients with respect to the tensor and to the static
  permittivity match fourth-order central differences, and so does the
  gradient with respect to the static mesh's nodes: alone, through an
  electrode gap moved by a shape morph with the optical model on the same
  mesh, and through an optical window on its own mesh, the ridge held. The
  optical model's permittivity and nodes raise.

Three comparisons are observations, not checks, and live on the pages. The
modulator of Wang et al. (2018): gfem's push-pull $V_\pi L$ is 1.61 V·cm
against the measured 1.8 V·cm, at assumed values of the dimensions the paper
withholds ({doc}`tutorials/modulator`). An optical window and the whole
section with the electrodes excluded give shifts within a few percent, and
an order-1 electrostatic solve biases the shift by about 1 %
({doc}`recipes/electro-optic-phase`).

Tests: [unit checks](https://github.com/benvial/gfem/blob/main/tests/unit/test_electro_optic.py),
[closed forms and re-solves](https://github.com/benvial/gfem/blob/main/tests/verification/test_electro_optic.py),
[node gradients](https://github.com/benvial/gfem/blob/main/tests/verification/test_electro_optic_gradient.py).

(verification-mesh-refinement)=
## Mesh refinement error estimates

The {term}`error estimate` of {doc}`recipes/mesh-refinement` is checked by
its effectivity: the estimated error divided by the error against a known
answer, which must stay within a range set in the test for every mesh of
every case. Each kind of estimate (`ErrorEstimate.kind`) has its own range,
generated from its cases:

```{gfem-effectivity-table}
```

The quantity estimator's cases, all at `order=2`:

- **A square coaxial line** (a square conductor in a square shield), whose
  inner corners make the field singular: its capacitance over a mesh
  refinement run, against a constant extrapolated from fine uniform meshes
  and checked on a graded one.
- **A circular coaxial line**, faceted: its capacitance on meshes of 1 to 8
  times the base density, against a higher-order solve on the same facets.
  Against the exact formula for the circle the estimate stays below the
  true error, since it does not see the faceting.
- **A stepped parallel-plate line**, whose top plate steps down to half its
  height, with a singular field at the step's edge: $S_{21}$ at 3 and 8 GHz on
  uniform meshes and over a mesh refinement run with lumped ports, and at
  3 GHz with wave ports, against constants from fine uniform and graded
  meshes.
- **A parallel-plate line** between ports of 50 and 30 Ω: $S_{11}$, $S_{21}$ and $S_{22}$
  from 3 to 25 GHz on three meshes, against the line's closed form.
- **An L-shaped cavity**, whose re-entrant edge makes the field singular:
  its first resonance over a mesh refinement run, against a published
  high-accuracy value.
- **A rectangular cavity** with metal walls: four modes on two meshes,
  against the closed form.

The quantity estimator is also checked on cases that exercise one feature
each: a capacitance between two bars with a grounded screen between them,
and one between two conductors on either side of an interior metal sheet;
lines and cavities filled with anisotropic and magnetic materials, and a
coaxial line with an anisotropic fill; a stepped line with its wave ports
de-embedded, and a waveguide with two modes on each of two shared wave port
surfaces; the resonances of cavities with lossy fills and lossy walls
(impedance, absorbing and conducting); and a line closed by resistive ports
as a resonator. With loss in the volume: S of the parallel-plate line with
a lossy fill (dielectric loss and conductivity) between lumped ports,
against its closed form; and S between wave ports of a lossy waveguide:
uniformly filled, half filled by a lossy dielectric slab (whose port mode
is hybrid, so the estimate solves the ports' {term}`adjoint dual`), and
with that slab widening halfway along, each with and without
de-embedding, against a higher-order solve. A waveguide holding a
dielectric strip along its length, whose wave ports carry a hybrid mode,
is checked de-embedded over half its length against its closed form.

The S estimate is also checked on a line between conducting plates of
finite conductivity; on a line through a metal described by a permittivity
law deep below zero ($\mathrm{Re}\,\varepsilon$ about $-30$), at a wedge, on a flat face and
through a metal block; on wave ports whose face crosses a permeability
contrast and lumped ports beside a loss contrast; and on lines filled with
lossy magnetic and lossy anisotropic materials. The resonance estimate
covers permittivity laws with $\mathrm{Re}\,\varepsilon > 0$ (a dispersive fill and layered
cavities), second-order absorbing side walls, and lossy magnetic fills in
one and two layers.

The other kinds of quantity:

- **Q** of a mode: a line closed by resistive ports as a resonator, cavities
  with resistive, absorbing and conducting walls, and lossy fills whose Q
  is known exactly.
- **The mean frequency of a degenerate cluster**: pairs and triples of
  modes in symmetric cavities, a pair split slightly by the geometry, a
  pair in a cavity with conducting walls, and a pair with a third mode
  close by.
- **Objectives of S**: lossy and lossless lines (matched and nearly
  matched), the stepped line, waveguides with shared and lossy wave ports,
  and a line through a deep metal. On a lossless model with lumped ports
  the estimate goes through the {term}`lossless rewrite`.
- **Objectives of capacitance and electrostatic energy, and of potential
  means**: the two bars, with the space around them split into regions,
  and the coaxial line.
- **Objectives of frequency and Q**: the L-shaped cavity, the line closed
  by resistive ports, and a cavity with two modes.
- **Absorbed power and field means**: a waveguide with a lossy dielectric
  block, a waveguide with regions across half its width, and stepped lines
  with a lossy layer and with a small region at the step's edge.
- **Participations**: a cavity with a dielectric slab and one with a
  dielectric block.

A few references are extrapolated from finer meshes, with their
uncertainty kept below a fifth of every error checked.

Over the capacitance, S and resonance cases, the quantity estimate was
between {gfem-effectivity}`quantity low` and
{gfem-effectivity}`quantity high` times the true error; the higher ratios
came from fields with no sharp metal edge, such as the parallel-plate line
and the rectangular cavity, and from layered cavities. Some objectives are
estimated far more pessimistically near their target; the recipe's Limits
quote those ratios.

The field estimator is checked the same way, against the field's error in
the energy norm: on the square and circular coaxial lines, the coaxial
line with an anisotropic fill, the parallel-plate line, and the L-shaped
and rectangular cavities; and with loss, on the parallel-plate line with a
lossy fill (dielectric loss and conductivity) and with a Drude metal block
($\mathrm{Re}\,\varepsilon < 0$) against their closed forms, and against a higher-order solve on
the same mesh on a cavity in two layers, the upper one lossy, and on a
cavity whose lossy wall (impedance or absorbing) gives its resonance a
complex frequency. It is also checked on the line between conducting
plates, at a deep metal's wedge and flat face, in lossy magnetic and lossy
anisotropic fills, and in a cavity with second-order absorbing side walls.
Its estimate was between {gfem-effectivity}`field low` and
{gfem-effectivity}`field high` times the true error, the circular line
measured against a higher-order solve on its own facets; against the exact
circle it stays below the true error. These cases run at `order=2`, and
the original cases also at `order=3`.

Four configurations lie below that range and are not claimed; each test
holds the estimate to its measured band:

- on a line in two layers, the upper one lossy, with lumped ports spanning
  both: {gfem-effectivity}`lumped port across a loss jump low` to
  {gfem-effectivity}`lumped port across a loss jump high` times the true
  error on coarse meshes;
- next to impedance walls that carry the field's current:
  {gfem-effectivity}`impedance surface low` to
  {gfem-effectivity}`impedance surface high`;
- next to a lumped port spanning a permeability contrast:
  {gfem-effectivity}`lumped port across a permeability contrast low` to
  {gfem-effectivity}`lumped port across a permeability contrast high`;
- at a metal wedge with a permittivity law at $\mathrm{Re}\,\varepsilon = -1.5$, against a
  higher-order solve the field itself does not converge to:
  {gfem-effectivity}`plasmonic window low` to
  {gfem-effectivity}`plasmonic window high`.

The S estimate has one such configuration: a lumped port whose face
crosses a permittivity interface, on cells stretched 10:1 along the line,
measured
{gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast low`
to
{gfem-effectivity}`stretched cells at a lumped port across a permittivity contrast high`
times the S error.

On the stepped line, whose field error has
no known value, the estimate is checked to keep a steady ratio to the $S_{21}$
error as the mesh is refined. On the three singular benchmark cases a mesh refinement run also
ends with a smaller error than uniform splits at the same number of
unknowns. Every range is read from the tests when the docs are built, and
every set of checks runs on every change.

Tests: [quantity estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_quantity_estimate.py),
[objective estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_objective_estimate.py),
[field estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_field_estimate.py).

## The p-multigrid solver

`linear_solver="pmg_gmres"` ({doc}`recipes/large-driven-model`) is checked
against the direct solver on models that need each part of its
{term}`direct set`:

- **Against the direct solver.** S of a strip line at orders 2 and 3, with
  each factorization of the direct set. A two-frequency sweep builds the
  solver once and matches fresh builds; the adjoint solve runs on the
  forward's solver. At 50 MHz on a metal line, where `"auto"` picks it once
  the direct solver does not fit, S matches the direct solver.
- **A guide between perfectly matched layers.** A strip guide cut by side
  layers solves in a few iterations, and its S matches the direct solver's.
  With the layer left to the smoother instead, the solve stalls.
- **A thin film.** A silicon guide with a film one nanometre thick along its
  outer faces, meshed one cell across, takes as many iterations as the guide
  without the film: the film's {term}`poor cells <poor cell>` go on the direct
  set.
- **Size in wavelengths.** A silicon guide one wavelength across at order 2,
  and half a wavelength across at order 3, run on every change. Before each
  release, the guide two and four wavelengths across takes about the same
  number of iterations; at two wavelengths S matches the direct solver, and at
  four it is reciprocal and passive.
- **Metal lines.** The coplanar lines of the reference cases, before each
  release, at the frequencies of their reference S, with fewer iterations than
  `"bddc_gmres"`.
- **Gradients.** S gradients on lumped and wave ports, the design's initial
  guess, and a gradient next to a layer are checked against finite
  differences on `"pmg_gmres"` too. The gradient matrix does not list it.
- **`"auto"` and refusals.** The memory estimate counts the same direct set
  as the solver builds. `"auto"` picks the solver only where the direct solver
  does not fit and the solver's own estimate does. Each refusal (order, problem
  type, Bloch, periodic and Floquet models, MPI ranks, 2D) is a test of its
  own.

### At scale

Before the solver was merged, silicon guides of 2.9 and 6.6 million unknowns
and a half model with a perfectly matched layer of 2.1 million unknowns were
checked against the direct solver. A bend of 17.2 million unknowns was checked
against the direct solver's outgoing waves from an earlier run on a
large-memory node. Times and memory are on {doc}`project/scaling`. This run is
not part of the release checks.

Tests: [p-multigrid](https://github.com/benvial/gfem/blob/main/tests/unit/test_pmg.py),
[mechanisms](https://github.com/benvial/gfem/blob/main/tests/verification/test_pmg.py),
[guides and metal lines](https://github.com/benvial/gfem/blob/main/tests/reference/test_driven.py),
[`"auto"`](https://github.com/benvial/gfem/blob/main/tests/unit/test_auto_solver.py).

(verification-petsc)=
## PETSc and SLEPc

- With no `petsc_options`, `linear_solver="petsc"` is an exact factorization:
  its results agree with `"direct"`.
- SLEPc's modes are refined as ARPACK's are: modes, Q, fields and gradients
  agree with ARPACK's.

These checks need the {term}`PETSc stack`. They run twice a week and on
every change to gfem's PETSc code.

Tests: [PETSc](https://github.com/benvial/gfem/blob/main/tests/unit/test_linalg_petsc.py),
[SLEPc](https://github.com/benvial/gfem/blob/main/tests/unit/test_eigenmode_slepc.py),
and the PETSc cells of the
[gradient matrix](https://github.com/benvial/gfem/blob/main/tests/verification/test_gradient_matrix.py).

## Several MPI ranks

Each thing a {term}`distributed solve` runs (see
{doc}`recipes/distributed-solve`) has a test launched on several ranks. It
compares the result with one process, on `"direct"` and on `"bddc_gmres"`,
and checks that every rank holds the same result, bit for bit. A driven solve
on `"bddc_gmres"` must also take the same number of iterations. An eigenmode
solve is compared with one process on SLEPc; its frequencies, Q and fields
must match, not its iteration count. Periodic driven cells are checked on
partitions that keep the pairs as given and on partitions that swap them. Some
things have no such test yet: a lumped port terminated by an inductance or a
capacitance, {py:obj}`gfem.characteristic_impedance`, `"petsc"` on an
electrostatic, eigenmode or periodic solve, and a periodic eigenmode solve on
`"bddc_gmres"`. They run the same code and do not raise.

The launched tests on `"bddc_gmres"` run on every change. Those on
`"direct"`, and every eigenmode test, need the {term}`MPI PETSc stack` and
run twice a week.

Tests: [launched tests](https://github.com/benvial/gfem/tree/main/tests/mpi).

### On many nodes

The runs of {doc}`project/scaling` were checked three ways:

- **Every rank count against one process.** On a 10,248,856 DOF coplanar
  line, from 1 to 32 nodes, each solver against its own one-process run. S
  agreed to rounding error on both solvers, and `bddc_gmres` took the same
  number of iterations.
- **The two solvers against each other.** On the largest mesh both solvers
  finished, 47,277,960 DOF on 16 nodes, `bddc_gmres` against `direct`. S
  agreed to well within the GMRES tolerance of that run. `direct` did not
  finish at 108,570,046 DOF, so this check did not run there.
- **The largest mesh against the 10,248,856 DOF line.** The meshes differ, so
  S differs by the change of mesh. This is an observation, not a check.

In every run, every rank held the same S, bit for bit.
