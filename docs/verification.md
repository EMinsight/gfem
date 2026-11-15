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
  matches 1/Q from the complex eigenfrequency, on a material whose field lies
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
  height, with a singular field at the step's edge: S21 at 3 and 8 GHz on
  uniform meshes and over a mesh refinement run with lumped ports, and at
  3 GHz with wave ports, against constants from fine uniform and graded
  meshes.
- **A parallel-plate line** between ports of 50 and 30 Ω: S11, S21 and S22
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
de-embedding, against a higher-order solve.

Over all cases the quantity estimate was between
{gfem-effectivity}`quantity low` and {gfem-effectivity}`quantity high`
times the true error; the higher ratios came from fields with no sharp
metal edge, such as the parallel-plate line and the rectangular cavity.

The field estimator is checked the same way, against the field's error in
the energy norm: on the square and circular coaxial lines, the coaxial
line with an anisotropic fill, the parallel-plate line, and the L-shaped
and rectangular cavities; and with loss, on the parallel-plate line with a
lossy fill (dielectric loss and conductivity) and with a Drude metal block
(Re ε < 0) against their closed forms, and against a higher-order solve on the same mesh on a cavity in two
layers, the upper one lossy, and on a cavity whose lossy wall (impedance
or absorbing) gives its resonance a complex frequency. Its estimate was
between {gfem-effectivity}`field low` and {gfem-effectivity}`field high`
times the true error, the circular line measured against a higher-order
solve on its own facets; against the exact circle it stays below the true
error. These cases run at `order=2`, and all but the anisotropic fill also
at `order=3`. One configuration lies below that range and is not claimed:
on a line in two layers, the upper one lossy, with lumped ports spanning
both, the estimate was
{gfem-effectivity}`lumped port across a loss jump low` to
{gfem-effectivity}`lumped port across a loss jump high` times the true
error on coarse meshes; its test holds it to that band. On the stepped line, whose field error has
no known value, the estimate is checked to keep a steady ratio to the S21
error as the mesh is refined. On the three singular benchmark cases a mesh refinement run also
ends with a smaller error than uniform splits at the same number of
unknowns. Both ranges are read from the tests when the docs are built, and
both sets of checks run on every change.

Tests: [quantity estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_quantity_estimate.py),
[field estimate](https://github.com/benvial/gfem/blob/main/tests/verification/test_field_estimate.py).

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
compares the result with one process, on `"direct"` and on `"bddc_gmres"`
(where the iteration count must also match), and checks that every rank
holds the same result, bit for bit. Two things have no such test yet: a
lumped port terminated by an inductance or a capacitance, and
{py:obj}`gfem.characteristic_impedance`. They run the same code and do not
raise.

The launched tests on `"bddc_gmres"` run on every change. Those on
`"direct"` need the {term}`MPI PETSc stack` and run twice a week.

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
