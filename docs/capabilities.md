# Capabilities

What gfem solves, which results it differentiates, and what it refuses, at
v0.1.

## What you can solve

- **Problems**: {py:obj}`gfem.Driven` (the response of a model to its ports over a
  frequency grid, uniform or adaptive sweep), {py:obj}`gfem.Eigenmode` (resonant
  modes nearest a target frequency, with their Q; lumped ports only: a wave
  port raises; with a permittivity law, outside the law's
  {term}`unresolved band`), {py:obj}`gfem.Electrostatic` (the
  Maxwell capacitance matrix between named terminals) and {py:obj}`gfem.PortMode`
  (the mode of a wave port's cross-section; a permittivity law in it where
  $\mathrm{Re}\,\varepsilon > 0$). Eigenmode and uniform driven problems take a
  Bloch wave vector (`wave_vector=`, rad/m) on a periodic model; a uniform
  driven sweep may keep the incidence angle fixed (`wave_vector_frequency`).
- **2D models**: a plane of triangles read with {py:meth}`gfem.Mesh.from_gmsh`,
  where neither the structure nor the field varies along z. A uniform driven
  sweep solves one polarization, `Driven(polarization="Ez")` or `"Hz"`, with
  {term}`line ports <Line port>`, PEC, PMC and first- or second-order absorbing
  lines, materials (permittivity laws included), material fields and
  layer-grid designs, with their gradients. Powers and fields are per metre
  along z, and S equals the slab's ({doc}`recipes/2d-models`).
- **Ports**: {py:obj}`gfem.LumpedPort` on one or several surfaces driven together,
  and {py:obj}`gfem.WavePort` on an exterior surface, fed by its port mode;
  several wave ports on one surface each request a different mode of it,
  each its own row and column of S ({doc}`recipes/shared-port-surface`);
  {py:obj}`gfem.FloquetPort` on the open face of a periodic cell: one
  diffraction order in one polarization (TE or TM) of a plane wave, each its
  own row and column of S, $|S|^2$ the diffraction efficiency
  ({doc}`recipes/floquet-ports`).
- **Boundaries**: {py:obj}`gfem.PEC`, {py:obj}`gfem.PMC` (symmetry planes), {py:obj}`gfem.Absorbing`
  (first or second order), {py:obj}`gfem.Conductor` (a finite-conductivity metal, by
  its surface impedance), {py:obj}`gfem.Impedance` (per-square R, L and C),
  and {py:obj}`gfem.Periodic` (a periodic pair of surfaces, with plain
  periodicity or a Bloch wave vector set on the problem;
  {doc}`recipes/periodic-cell`).
- **Open space**: {py:obj}`gfem.PML` on meshed layer regions around the air
  box: absorbs outgoing waves at any angle, for driven and eigenmode solves,
  wave ports and periodic cells ({doc}`recipes/perfectly-matched-layer`).
- **Materials and designs**: {py:obj}`gfem.Material` per region (isotropic, or
  anisotropic: real symmetric tensors on shared principal axes,
  {doc}`recipes/anisotropic-materials`), {py:obj}`gfem.MaterialField` per
  element (isotropic, or principal values on fixed axes), and {py:obj}`gfem.SheetField` for the fill of a patterned metal sheet
  ({doc}`recipes/sheet-design`). A `Material`'s permittivity may follow a
  {py:obj}`gfem.PermittivityLaw` of Drude, Lorentz, Debye and
  Djordjevic-Sarkar terms, metals included
  ({doc}`recipes/dispersive-materials`).
- **Measurements**: {py:obj}`gfem.characteristic_impedance`,
  {py:obj}`gfem.interface_participation`, {py:obj}`gfem.power_flow` and
  {py:obj}`gfem.electric_energy` ({doc}`recipes/measurements`).
- **Quantum metrics**: {py:obj}`gfem.quantum.dressed_modes` (the dressed
  frequencies, Lamb shifts, anharmonicities and dispersive shifts of one
  junction on the solved modes) and `EigenmodeResult.sheet_participation`
  (the kinetic inductance fraction of a film), with their gradients
  ({doc}`recipes/quantum-metrics`). A mode's loss splits into a $1/Q$ budget
  and $T_1$ limits ({doc}`recipes/loss-budget`).
- **Thermal problems**: {py:obj}`gfem.SteadyThermal` (the temperature under
  heat sources), {py:obj}`gfem.TransientThermal` (the temperature over time
  under switched sources, with traces recorded at every step) and
  {py:obj}`gfem.DCConduction` (the current through conductors between
  terminals: their conductance and Joule heat), on 3D meshes and on 2D
  cross-sections read with {py:meth}`gfem.Mesh.from_gmsh_cross_section`.
  Boundaries {py:obj}`gfem.FixedTemperature`, {py:obj}`gfem.Convection` and
  {py:obj}`gfem.HeatFlux`; a surface no thermal boundary names is insulated.
  Materials take `thermal_conductivity`, `mass_density`, `specific_heat`,
  an electrical `conductivity` and a `thermo_optic_coefficient`, and
  `excluded=` leaves regions out of one physics at a time
  ({doc}`recipes/heater-temperature`).
- **Heat and phase**: {py:obj}`gfem.joule_heating` and
  {py:meth}`gfem.HeatSource.uniform` (heat sources),
  {py:obj}`gfem.thermo_optic_shift` (a guided mode's phase per length under
  a rise, on a cross-section) and {py:obj}`gfem.route_phase` (the phase
  along routes through a 3D rise) ({doc}`recipes/thermo-optic-phase`);
  {py:obj}`gfem.rise_time` and {py:obj}`gfem.time_constant` of a step
  response ({doc}`recipes/thermal-step-response`); {py:obj}`gfem.element_mean`.
- **Electro-optic shift**: {py:obj}`gfem.electro_optic_shift` (a guided
  mode's index and phase per volt from an electrostatic potential, on a
  cross-section, through any 6 × 3 electro-optic tensor; $V_\pi L$ per arm
  and push-pull) and `ElectrostaticResult.capacitance_per_length`. Materials
  take an `electro_optic_tensor` on explicit `axes`
  ({doc}`recipes/electro-optic-phase`).

## Solvers

- **`linear_solver="direct"`**: pick it for a model whose factorization fits
  in one machine's memory, for eigenmode problems, and for the repeated solves
  of an optimization loop, where a design can solve on the previous design's
  factorization ({doc}`recipes/s-parameter-gradient`). It factorizes once per frequency
  and reuses the factorization for every port. The default (`"auto"`) picks it
  for a driven problem on one machine while its factorization fits in half the
  memory. Install the `mkl` extra for its fastest factorization.
- **`linear_solver="pmg_gmres"`**: pick it for a driven model several
  wavelengths across that the direct solver cannot hold, such as a photonic
  device in dielectric cladding, with absorbing faces or a perfectly matched
  layer, or a metal circuit on a fine mesh. Its iterations stay near 10 as the
  model grows, it needs about a third of the direct solver's memory at order 2,
  and it is faster than the direct solver from a few million unknowns on.
  Driven problems at order 2 or 3 on one CPU process, with gradients, and
  without Bloch, periodic or Floquet models. `"auto"` picks it where the direct
  solver does not fit and it does ({doc}`recipes/large-driven-model`).
- **`linear_solver="bddc_gmres"`**: pick it across MPI ranks, for periodic and
  Bloch models, and for electrostatic problems, when the direct factorization
  does not fit; `"auto"` picks it for a driven problem that neither the
  direct solver nor `"pmg_gmres"` takes, and for electrostatic problems. A model with a perfectly matched layer refuses it:
  use `"pmg_gmres"`.
- **`linear_solver="petsc"`** and **`eigen_solver="slepc"`**: pick them for
  MUMPS and its options, your own PETSc Krylov method and preconditioner, or
  SLEPc's own controls. With no options, `"petsc"` gives the same results as
  `"direct"`, and SLEPc the same modes and Q as ARPACK, the default eigen
  solver on one process (across MPI ranks the default is SLEPc). You build
  them from source ({doc}`recipes/petsc-slepc`).
- **`device="cuda"`**: the direct solver on one NVIDIA GPU through cuDSS,
  experimental ({doc}`recipes/gpu-direct-solver`).
- **Thermal and DC conduction problems** solve on `"direct"` by default and
  accept `"bddc_gmres"` and `"petsc"`. One factorization serves a whole
  transient run. They do not run on the GPU solver or across MPI ranks: both
  raise.
- **Across MPI ranks**: driven uniform sweeps, electrostatic and eigenmode
  solves, and periodic cells on one axis run split across the ranks of one
  job, with their gradients (except gradients through a mode's field), on
  Open MPI and Linux x86_64. Everything else raises
  {py:obj}`gfem.DistributedUnsupportedError` there; the full table is in
  {doc}`recipes/distributed-solve`.

How large a model each solver has finished, and in what time:
{doc}`project/scaling`.

## What you can differentiate

```{gfem-gradient-matrix}
```

- **FD-verified**: supported; the gradient is checked against finite
  differences.
- **Exact zero**: the result does not depend on this input; `jax.grad`
  returns zero, correctly.
- **Raises**: asking for this gradient raises
  {py:obj}`gfem.NonDifferentiableError` (or the error named) instead of
  returning a wrong value.

The thermal rows differentiate the temperature, the traces, the conductance
and the phase with respect to per-element thermal and electrical
conductivity, the thermo-optic coefficient, a transient's mass density and
specific heat, the heat sources and the drive
({doc}`recipes/thermal-gradient`).

The electro-optic shift differentiates with respect to its tensor, the
static permittivity of every element, the drive and the electrostatic
mesh's nodes (an electrode gap through a shape morph), each checked against
finite differences; with respect to the optical model's permittivity or
nodes it raises, because the mode is held constant
({doc}`recipes/electro-optic-gradient`).

How each mark is checked: {doc}`verification`.

### Gradients that raise

Off the matrix, asking for these gradients raises the named error instead of
returning a wrong value:

```{gfem-raises}
```

## Limits

- **Low frequency**: a direct driven solve on the CPU in one process keeps its
  digits below the low-frequency floor, $k_0 h_\mathrm{min} = 10^{-6}$
  ($f h_\mathrm{min} \approx$ 48 Hz·m, with $h_\mathrm{min}$ the shortest
  edge: 48 MHz with 1 µm elements, 5 GHz with 10 nm elements), through the
  {term}`tree-cotree gauge`, on by default (`SolverOptions.gauge`). The wave
  ports' mode problem is not gauged and keeps its own floor: below it, a
  model with wave ports warns with {py:obj}`gfem.LowFrequencyWarning`, and
  lumped ports keep their digits. Other driven solves (`"bddc_gmres"`,
  `"pmg_gmres"`, `"petsc"`, the GPU, several MPI ranks, periodic and Floquet models, or
  `gauge="none"`) are ungauged: below the floor their fields, gradients and,
  with an exact factorization, the capacitive part of S lose digits, and they
  warn. For capacitance at low frequency, solve {py:obj}`gfem.Electrostatic`;
  the details are under {py:obj}`gfem.Driven`. An eigenmode solve whose target
  or lowest mode is below the floor warns too: its frequencies, fields and
  participations lose the same digits, and the reported `error` does not
  show it.
- **Gyrotropic materials**: non-symmetric (non-reciprocal) tensors, such as a
  magnetized ferrite, are not supported. Neither is loss whose principal axes
  differ from the permittivity's. Both raise `ValueError`.
- **Anisotropic materials at wave ports**: a port's cross-section may contain
  anisotropic materials when every tensor has a principal axis along the port
  normal. Elsewhere in the model the principal axes may point in any
  direction: a model with tilted principal axes and its rigidly rotated copy
  give the same S, eigenfrequency and capacitance.
- **Second-order absorbing boundaries** raise next to an anisotropic
  material; use order 1, which is exact at normal incidence for any
  anisotropic material.
- **Periodic models**: a Bloch wave vector needs the direct solver for
  eigenmode problems, and the direct or BDDC + GMRES solver for driven
  problems. PETSc and cuDSS refuse it, as does an adaptive sweep.
  Electrostatic problems refuse any periodic pair, and a wave port may not
  touch one. An absorbing boundary reflects waves that meet it obliquely,
  and a driven solve warns ({py:obj}`gfem.ObliqueAbsorberWarning`); a
  Floquet port absorbs them, or a perfectly matched layer below the cell.
  Details: {doc}`recipes/periodic-cell`.
- **Floquet ports**: the half-space behind the face is homogeneous,
  isotropic and lossless, and the face needs two periodic pairs. Only the
  uniform driven sweep on the direct or BDDC + GMRES solver; every other
  path raises. Only orders that propagate over the whole sweep can be
  excited. No gradient with respect to the incidence, the half-space or the
  nodes of a Floquet surface. In the gradient matrix, a model with Floquet ports takes the
  rows of the periodic driven problems (plain and Bloch), with S read from
  its Floquet ports. Details: {doc}`recipes/floquet-ports`.
- **Perfectly matched layers**: no error estimates or mesh refinement for
  driven and eigenmode problems (electrostatic refinement runs). Box shells
  around the interior box only, one thickness per side. Each layer region's
  background is a constant `Material`, isotropic or on axis-aligned
  principal axes. No `Absorbing` face, `Impedance` or `Conductor` sheet in or
  on the layer. Lumped ports and Floquet surfaces keep clear of it, and a
  wave port may cut a side layer but not one stretched along its normal. No
  stretch along a periodic axis. No `"bddc_gmres"`; `"pmg_gmres"` solves the layer exactly
  inside its iterations. The fields inside the
  layer are not physical; energies, participations and port power count the
  physical regions. Details: {doc}`recipes/perfectly-matched-layer`.
- **Unrequested port modes**: a propagating mode that no wave port on its
  surface requests is partly reflected at the port, not converted; request
  it to see where its power goes ({py:obj}`gfem.SkippedPortModeWarning`
  flags a mode left out below a requested one).
- **Lossy port lines**: wave-port S reads each mode by the biorthogonal
  projection, so a lossy port line is matched to its own mode and the modes
  of a lossy shared port surface separate. With a `voltage_path` its
  reference impedance is the mode's complex $Z_{PV}$ (pseudo-waves); give
  every wave port a `voltage_path`, or none, for a symmetric S.

- **2D models**: normal incidence and one polarization only. Eigenmode,
  electrostatic and thermal problems, lumped and Floquet ports, periodic
  pairs, conductor and impedance surfaces, sheet designs, anisotropic
  materials, PMLs, the adaptive sweep, the iterative and GPU solvers, several
  processes, measurements other than `Field.at`, mesh refinement and node
  gradients raise `NotImplementedError` and name the alternative, most often
  the {term}`slab model`. In Hz a design blend takes constant permittivities
  only. Details: {doc}`recipes/2d-models`.
- **Dispersive materials**: a permittivity law is isotropic, and
  `mu_r` does not disperse; give an anisotropic or magnetic material constant
  values per band. A metal in a wave port's cross-section raises: keep it off
  the port faces or use lumped ports. A blend with a law end solves on the
  driven uniform sweep only. An eigenmode target or mode in a law region's
  {term}`unresolved band` raises; move the target or refine that region.
  Details: {doc}`recipes/dispersive-materials`.
- **Mode-field gradients**: participations, port Q and the mode field
  differentiate for isolated modes of non-periodic models, on the direct
  solver, on the CPU, in one process ({doc}`recipes/eigenmode-gradient`).
  Elsewhere only the eigenfrequencies differentiate.
- **Quantum metrics**: one junction per model; the junction's cosine on the
  solved linear modes, in the transmon regime. They are checked against
  identities, finite differences and published outputs, not against a
  measured device.
- **Thermal problems**: linear heat conduction with constant, isotropic
  properties: no temperature-dependent conductivity, no radiation, no
  thermal contact resistance, and the heat does not change the current. A
  transient runs on a uniform time grid with sources switched at grid
  times. A cross-section is a slice of an infinitely long device. The
  thermo-optic shift and the route phase are first order in the rise, with
  the mode held constant, and refuse a lossy optical section. Mesh
  refinement and error estimates do not run on thermal problems, and the
  rise time and time constant have no gradient. Details:
  {doc}`recipes/heater-temperature`, {doc}`recipes/thermo-optic-phase`,
  {doc}`recipes/thermal-step-response`, {doc}`recipes/thermal-gradient`.
- **Electro-optic shift**: a cross-section at DC, first order in the static
  field with the mode held constant. It refuses a lossy optical section, a 2D
  model and a potential split across ranks. The optical model does not see
  the electrodes, so their absorption is absent. No gradient goes through the
  mode: the ridge's shape or width takes a difference over remeshes. Details:
  {doc}`recipes/electro-optic-phase`, {doc}`recipes/electro-optic-gradient`.

## Platforms

Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.
The cuDSS GPU solver installs on Linux x86_64 and Windows x86_64. Importing
gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve
requires. See {doc}`get-started`.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain
GPL-2.0+ SuiteSparse code; see {doc}`project/licence`.
