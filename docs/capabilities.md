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
  (the 2D mode of a wave port; a permittivity law in its cross-section where
  Re ε > 0). Eigenmode and uniform driven problems take a
  Bloch wave vector (`wave_vector=`, rad/m) on a periodic model; a uniform
  driven sweep may keep the incidence angle fixed (`wave_vector_frequency`).
- **Ports**: {py:obj}`gfem.LumpedPort` on one or several surfaces driven together,
  and {py:obj}`gfem.WavePort` on an exterior surface, fed by its port mode;
  several wave ports on one surface each request a different mode of it,
  each its own row and column of S ({doc}`recipes/shared-port-surface`);
  {py:obj}`gfem.FloquetPort` on the open face of a periodic cell: one
  diffraction order in one polarization (TE or TM) of a plane wave, each its
  own row and column of S, |S|² the diffraction efficiency
  ({doc}`recipes/floquet-ports`).
- **Boundaries**: {py:obj}`gfem.PEC`, {py:obj}`gfem.PMC` (symmetry planes), {py:obj}`gfem.Absorbing`
  (first or second order), {py:obj}`gfem.Conductor` (a finite-conductivity metal, by
  its surface impedance), {py:obj}`gfem.Impedance` (per-square R, L and C),
  and {py:obj}`gfem.Periodic` (a periodic pair of surfaces, with plain
  periodicity or a Bloch wave vector set on the problem;
  {doc}`recipes/periodic-cell`).
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
  ({doc}`recipes/quantum-metrics`). A mode's loss splits into a 1/Q budget
  and T1 limits ({doc}`recipes/loss-budget`).
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

## Solvers

- **`linear_solver="direct"`**: pick it for a model whose factorization fits
  in one machine's memory, for eigenmode problems, and for the repeated solves
  of an optimization loop, where a design can solve on the previous design's
  factorization ({doc}`recipes/s-parameter-gradient`). It factorizes once per frequency
  and reuses the factorization for every port. The default (`"auto"`) picks it
  for a driven problem on one machine while its factorization fits in half the
  memory. Install the `mkl` extra for its fastest factorization.
- **`linear_solver="bddc_gmres"`**: pick it for a large model, or when the
  direct factorization does not fit in memory; `"auto"` picks it for such a
  driven problem and for electrostatic problems.
- **`linear_solver="petsc"`** and **`eigen_solver="slepc"`**: pick them for
  MUMPS and its options, your own PETSc Krylov method and preconditioner, or
  SLEPc's own controls. With no options, `"petsc"` gives the same results as
  `"direct"`, and SLEPc the same modes and Q as ARPACK, the default eigen
  solver. You build them from
  source ({doc}`recipes/petsc-slepc`).
- **`device="cuda"`**: the direct solver on one NVIDIA GPU through cuDSS,
  experimental ({doc}`recipes/gpu-direct-solver`).
- **Thermal and DC conduction problems** solve on `"direct"` by default and
  accept `"bddc_gmres"` and `"petsc"`. One factorization serves a whole
  transient run. They do not run on the GPU solver or across MPI ranks: both
  raise.
- **Across MPI ranks**: a driven uniform sweep and its gradients run split
  across the ranks of one job, on Open MPI and Linux x86_64. Everything else
  raises {py:obj}`gfem.DistributedUnsupportedError` there; the full table is
  in {doc}`recipes/distributed-solve`.

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

How each mark is checked: {doc}`verification`.

### Gradients that raise

Off the matrix, asking for these gradients raises the named error instead of
returning a wrong value:

```{gfem-raises}
```

## Limits

- **Low frequency**: below the low-frequency floor, k0·h_min = 1e-6 (f·h_min ≈
  48 Hz·m, with h_min the shortest edge: 48 MHz with 1 µm elements, 5 GHz with
  10 nm elements), a driven solve's fields, gradients and, with the direct
  solver or PETSc's default MUMPS factorization, the capacitive part of S
  lose digits, and it warns with {py:obj}`gfem.LowFrequencyWarning`. For
  capacitance at low frequency, solve {py:obj}`gfem.Electrostatic`; the
  details are under {py:obj}`gfem.Driven`. An eigenmode solve whose target or
  lowest mode is below the floor warns too: its frequencies, fields and
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
  Floquet port absorbs them. Details: {doc}`recipes/periodic-cell`.
- **Floquet ports**: the half-space behind the face is homogeneous,
  isotropic and lossless, and the face needs two periodic pairs. Only the
  uniform driven sweep on the direct or BDDC + GMRES solver; every other
  path raises. Only orders that propagate over the whole sweep can be
  excited. No gradient with respect to the incidence, the half-space or the
  nodes of a Floquet surface. In the gradient matrix, a model with Floquet ports takes the
  rows of the periodic driven problems (plain and Bloch), with S read from
  its Floquet ports. Details: {doc}`recipes/floquet-ports`.
- **Unrequested port modes**: a propagating mode that no wave port on its
  surface requests is partly reflected at the port, not converted; request
  it to see where its power goes ({py:obj}`gfem.SkippedPortModeWarning`
  flags a mode left out below a requested one).
- **Lossy port lines**: wave-port S is the conjugated modal projection, exact
  for a lossless port line. On a shared port surface with loss in the
  cross-section the modes do not fully separate, and a lossy port line
  reflects its own mode at the order of its attenuation per radian,
  alpha/beta; {py:obj}`gfem.PortModeProjectionWarning` states either error.

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

## Platforms

Linux x86_64 (glibc >= 2.28), macOS arm64 and Windows x86_64, Python >= 3.12.
The cuDSS GPU solver installs on Linux x86_64 and Windows x86_64. Importing
gfem turns on JAX double precision (`jax_enable_x64`), which every gfem solve
requires. See {doc}`get-started`.

## Licence

MIT. gfem depends on NGSolve (LGPL-2.1), whose PyPI wheels also contain
GPL-2.0+ SuiteSparse code; see {doc}`project/licence`.
