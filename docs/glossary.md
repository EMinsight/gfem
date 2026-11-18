# Glossary

The words these docs use, with the meaning gfem gives them.

## Problems

```{glossary}
Problem type
  One of the physics analyses gfem solves: a {term}`driven <Driven problem>`,
  {term}`eigenmode <Eigenmode problem>`, electrostatic or
  {term}`port mode <Port mode problem>` problem.

Driven problem
  A frequency-domain full-wave analysis excited through {term}`ports <Port>`,
  producing network parameters and fields at given frequencies.

Adaptive sweep
  A driven frequency sweep that solves the full problem at a few
  {term}`sample frequencies <Sample frequency>` and evaluates a
  {term}`reduced model` over the requested grid.

Frequency worker
  A separate process that solves one contiguous block of a uniform
  {term}`driven <Driven problem>` sweep's frequencies with its share of the
  solve's threads (`SolverOptions.frequency_workers`); the blocks join back in
  frequency order.

Solve threads
  The CPU threads a solve runs on: `SolverOptions.threads` when given, else
  `NGS_NUM_THREADS`, else the physical cores the process may run on (its affinity
  mask), not every hardware thread of the machine. Under MPI ranks the cores are
  shared between the ranks holding that mask.

Sample frequency
  An output frequency at which an {term}`adaptive sweep` solved the full
  problem.

Reduced model
  The projection of a {term}`driven problem` onto a basis of its sampled
  solutions, used to evaluate network parameters and fields between
  {term}`sample frequencies <Sample frequency>`.

Eigenmode problem
  A source-free analysis finding resonant frequencies, quality factors and mode
  fields of a structure.

Nonlinear eigenmode problem
  An {term}`eigenmode problem` whose operator depends on frequency in a way that
  is not polynomial of degree at most 2 (conductor surface impedance,
  second-order {term}`absorbing boundary`, wave port), so eigenfrequencies are
  roots of a frequency-dependent system.

Port mode problem
  A 2D cross-section eigenmode analysis that supplies the field profile and
  propagation constant of a wave port.

Port mode sweep
  The {term}`port mode problem` solved over a driven problem's output
  frequencies before the 3D solve, giving each wave port's propagation constant
  and mode profile at every output frequency. The wave ports of a
  {term}`shared port surface` share one sweep.

Unresolved band
  The frequencies at which a region with a {term}`permittivity law` carries a
  wavelength shorter than two of its longest element edges,
  $k_0\, \mathrm{Re} \sqrt{\varepsilon(f)}\, h_\mathrm{max} > \pi$. An eigenmode target or mode there raises
  `ValueError`: near a Lorentz resonance the modes crowd so densely that which
  ones are nearest depends on the mesh.

Modal overlap
  The number of a lossy region's cavity modes within one linewidth of a
  frequency, about $8\pi (V / \lambda_m^3) |\mathrm{Im}\,\varepsilon| / \mathrm{Re}\,\varepsilon$, with $V$ the region's volume and
  $\lambda_m$ the wavelength in it. Above about 10 the modes nearest a target cannot
  be told apart, and an {term}`eigenmode problem` there may stop with
  `gfem.EigenmodeConvergenceError`.
```

## Models

```{glossary}
Model
  Everything physical about a device independent of problem type: the mesh, a
  {term}`material` per {term}`region`, boundary conditions and
  {term}`ports <Port>`. One model feeds any problem type.

Region
  A named volume or surface of the mesh (a gmsh physical group), the only way
  materials, boundaries and ports refer to geometry.

Excluded region
  A volume {term}`region` present in the mesh but removed from the
  electromagnetic solve domain, its bounding surfaces acting as boundaries;
  typically a meshed conductor. Named either by region name or, when the mesh
  tags the conductor with the surrounding dielectric's name (gsim), by
  `gfem.enclosed_by(*surfaces)`, which selects every volume entity whose whole
  boundary lies on those surfaces.

Absorbing boundary
  A boundary condition on a truncating surface that approximates an open,
  non-reflecting exterior, of first or second order. It reflects a wave that
  meets it obliquely; on a periodic cell a driven solve warns
  (`gfem.ObliqueAbsorberWarning`) when that reflection reaches 1 %.

Transverse-anisotropic
  An {term}`anisotropic material` with one of its {term}`principal axes`
  along the surface normal at a port or boundary, so each tensor is
  block-diagonal in the surface frame (a 2x2 tangential block and a normal
  value). A wave port's cross-section may contain only transverse-anisotropic
  materials; tilted axes are allowed elsewhere.

Absorbing admittance
  The relative wave admittance a first-order {term}`absorbing boundary`
  applies to the tangential electric field on each face, set by the material
  behind the face and the face normal: a scalar for an isotropic material, a
  tangential tensor for an {term}`anisotropic <Anisotropic material>` one.

Air box
  The truncated exterior region between the device and the
  {term}`absorbing boundary`.

Slab model
  A {term}`model` one element thick between PEC plates, whose field is normal to
  the plates and uniform through the thickness, so it solves a 2D TM problem;
  how gfem poses top-view (effective-index) photonic and parallel-plate problems.

Mirror half model
  A {term}`model` cut on a symmetry plane of the stack and design, with a PMC or
  PEC on the plane chosen by the parity of the wanted mode; it solves half the
  volume for the same S-matrix.

TE-like mode
TM-like mode
  A guided mode of a strip waveguide whose transverse electric field lies mostly
  parallel to the wafer (TE-like) or normal to it (TM-like).

Port
  A named excitation and measurement surface, lumped or wave, whose order in the
  {term}`model` fixes the order of the network parameters.

Lumped port
  A {term}`port` on a surface that spans from one conductor to another, or on
  several such surfaces driven together: a resistor, with an optional
  inductance and capacitance in parallel, driven along a stated direction. Its
  resistance is its {term}`reference impedance`.

Lumped value
  The resistance, inductance or capacitance of a {term}`lumped port` or of an
  `Impedance` {term}`sheet`. Those that enter only the operator (a port's
  inductance and capacitance, a sheet's three values) are design inputs; a
  lumped port's resistance is not, because it is also its
  {term}`reference impedance`.

Wave port
  A {term}`port` on an exterior surface, fed by one mode of the surface's
  cross-section, which a {term}`port mode problem` supplies. It absorbs that
  mode without reflection.

Shared port surface
  An exterior surface that carries several wave ports, each requesting a
  different mode of one {term}`port mode problem`; together they form one
  boundary.

Requested mode
  The port mode, by rank at the lowest frequency, that a wave port excites and
  reports, with its own row and column of the network parameters. Wave ports
  that share a surface request different modes.

Skipped port mode
  A propagating mode of a surface's {term}`port mode problem`, ranked below the
  highest mode requested on that surface, that no wave port on the surface
  requests.

Floquet port
  A {term}`port` on a {term}`Floquet surface` that is one
  {term}`diffraction order` in one polarization (TE or TM), with its own row
  and column of the S-parameters (`gfem.FloquetPort`).

Floquet surface
  An open face of a periodic cell, closed by
  {term}`Floquet ports <Floquet port>`: a planar exterior face whose rim lies on the seams of two
  {term}`periodic pairs <Periodic pair>` spanning it, with one homogeneous,
  lossless, isotropic medium behind it. Its ports together absorb every
  diffraction order at any angle.

Diffraction order
  One of the plane waves a periodic cell sends into the medium behind a face:
  order $(m, n)$ has the in-plane wave vector $\boldsymbol{\kappa} = \mathbf{k}_t + m \mathbf{b}_1 + n \mathbf{b}_2$, with $\mathbf{b}_1$, $\mathbf{b}_2$
  the reciprocal vectors of the face's two pairs and $\mathbf{k}_t$ the
  {term}`Bloch wave vector`. It propagates when $|\boldsymbol{\kappa}|$ is below the medium's
  wavenumber, and decays away from the face otherwise.

Diffraction efficiency
  The fraction of the incident power per cell that leaves in one
  {term}`diffraction order` and polarization: $|S|^2$ of its
  {term}`Floquet port`'s row.

Rayleigh anomaly
  The frequency or angle at which a {term}`diffraction order` grazes its face
  ($|\boldsymbol{\kappa}|$ equal to the medium's wavenumber), changing from evanescent to
  propagating. Efficiencies change steeply there.

Dispersive cross-section
  A wave port's cross-section that contains a region with a
  {term}`permittivity law`. It is supported where $\mathrm{Re}\,\varepsilon(f) > 0$ throughout, at
  every frequency of the sweep; a metal in it raises.

Terminal
  A conductor, given as one or more regions (volumes or surfaces), whose
  potential is set in an electrostatic problem; terminals index the capacitance
  matrix.

Solver options
  The numerical knobs of a solve ({term}`element order`, linear solver,
  {term}`eigen solver`, tolerances, {term}`device`), kept apart from the
  {term}`model` and problem type.

Element order
  The polynomial order of the finite elements, `SolverOptions.order`, 2 by
  default and at most 12 (the order of the Nédélec elements that carry the
  electric field). A higher order is more accurate on the same mesh and costs
  more time and memory.

Device
  The hardware a solve's linear algebra runs on, the CPU or one CUDA GPU; part
  of the {term}`solver options`, never of the {term}`model`.

Eigen solver
  The iteration that finds an eigenmode problem's modes near the target,
  ARPACK or SLEPc; gfem refines and differentiates the modes the same way for
  both.

PETSc stack
  Complex PETSc and SLEPc with their Python bindings, built as
  {doc}`recipes/petsc-slepc` describes: what `linear_solver="petsc"` and
  `eigen_solver="slepc"` need. It comes in two builds, the
  {term}`no-MPI PETSc stack` and the {term}`MPI PETSc stack`.

No-MPI PETSc stack
  The {term}`PETSc stack` for one process: what `linear_solver="petsc"` and
  `eigen_solver="slepc"` need on a workstation or as {term}`rank copies`. It
  needs no MPI library and cannot run a {term}`distributed solve`.

MPI PETSc stack
  The {term}`PETSc stack` built against your Open MPI: what a
  {term}`distributed solve` with `"direct"` or `"petsc"` needs. It runs
  one-process solves too, and only under the Open MPI it was built on.

Periodic pair
  Two exterior surfaces of a {term}`model` related by a translation, whose
  fields are identified; one to three pairs make a 1D, 2D or 3D periodic cell
  (`gfem.Periodic`).

Plain periodicity
  A {term}`periodic pair` whose fields are equal on both surfaces: a problem
  with no {term}`Bloch wave vector`.

Bloch wave vector
  The wave vector k (rad/m) set on an eigenmode or driven problem
  (`wave_vector=`) that gives each {term}`periodic pair` its
  {term}`Bloch phase`.

Bloch phase
  The factor $e^{-i \mathbf{k} \cdot \mathbf{a}}$ between the fields of a {term}`periodic pair`,
  $\mathbf{E}(\mathbf{r} + \mathbf{a}) = e^{-i \mathbf{k} \cdot \mathbf{a}} \mathbf{E}(\mathbf{r})$ under $e^{+i\omega t}$, with $\mathbf{a}$ the translation from the
  pair's first surface to its second.

Job
  A self-contained run, {term}`model` plus problem type plus
  {term}`solver options`, and the unit that is serialized.
```

## Results

```{glossary}
Field
  A solution quantity (electric field, potential) on the model's mesh, carried
  by a result with its quantity, unit, space, order and support.

Partial result
  The result of a uniform sweep restricted to the frequencies it solved before
  the linear solver stopped converging, carried by the error that reports it
  (`gfem.LinearSolverConvergenceError`).

Result bundle
  The files a {term}`job`'s run writes for its result, `result.json` and
  `result.npz`, read back by `gfem.load_result` as the same result the
  in-process solve returns. It exists only for a run that completed.

Partial bundle
  The files a {term}`job`'s run writes in place of its result files when a
  uniform sweep stopped at an {term}`unsolved frequency`: `partial.json` and
  `partial.npz`, holding the {term}`partial result`, the unsolved frequencies
  and the error's message. Read it by name,
  `gfem.load_result("DIR/partial.json")`; `gfem.load_result("DIR")` reads only
  a complete result.

Unsolved frequency
  A frequency of a uniform sweep that has no entry in the
  {term}`partial result`: the linear solve did not converge there, or the sweep
  (or its {term}`frequency worker`'s block) stopped at an earlier frequency that
  did not converge.

S-parameters
  The network parameters of a {term}`driven problem`: the entry from port j to
  port i is the wave leaving port i for a unit wave sent into port j, every
  other port terminated in its {term}`reference impedance`. gfem's are
  {term}`pseudo-waves`, one matrix per frequency.

Power waves
  The incident and outgoing waves $a$ and $b$ of each {term}`port` (Kurokawa),
  defined from its peak voltage and current and its
  {term}`reference impedance` so that $|a|^2$ is the time-averaged incident
  power in W; $b = S a$. On a real reference they equal the
  {term}`pseudo-waves`.

Pseudo-waves
  The waves gfem's S relates (Marks and Williams):
  $a = \sqrt{\mathrm{Re}\, z} / (2|z|)\,(V + zI)$ and
  $b = \sqrt{\mathrm{Re}\, z} / (2|z|)\,(V - zI)$ for a
  {term}`reference impedance` $z$. On a real $z$ they are the
  {term}`power waves`; on a line's complex characteristic impedance the line
  is matched to its own mode.

Reference impedance
  The impedance per {term}`port` that S-parameters are normalized to: the port
  resistance for a lumped port, the power-voltage impedance for a wave port with
  a voltage path, and for a {term}`Floquet port` the wave impedance of its
  order in the medium behind the face ($\eta / \cos\theta$ for TE, $\eta \cos\theta$ for TM; none
  where the order does not propagate).

Participation ratio
  The fraction of a mode's electric or inductive energy stored in one region,
  port, sheet or interface. The participation ratio of a lumped port with an
  inductance is the junction participation when that port stands for a
  Josephson junction.

Principal participation
  The fraction of a mode's electric energy stored in one
  {term}`anisotropic <Anisotropic material>` region by the field component
  along one of the region's {term}`principal axes`; the three sum to the
  region's {term}`participation ratio`.

Sheet participation
  The {term}`participation ratio` of an inductive {term}`sheet`: the fraction
  of a mode's energy stored in the sheet's inductance. For a lossless mode on
  a superconducting film it equals the kinetic inductance fraction.

Dressed mode
  A mode of the structure after the junction's cosine potential is added to
  the solved linear modes: its transition frequency, {term}`Lamb shift`,
  {term}`anharmonicity` and {term}`dispersive shifts <Dispersive shift>`.

Dispersive shift
  The change of one {term}`dressed mode`'s transition frequency per photon in
  another, a signed angular frequency; the full shift, not half of it.

Anharmonicity
  The second transition frequency of a {term}`dressed mode` minus its first, a
  signed angular frequency; negative for a transmon.

Lamb shift
  A {term}`dressed mode`'s transition frequency minus the frequency of the
  solved linear mode.

Loss budget
  The split of a mode's $1/Q$ into the contributions of its loss channels: the
  loss in the model (materials, walls, sheets, resistive ports, absorbing
  boundaries), which the eigenfrequency already carries, plus each channel
  outside the model, which adds its {term}`participation ratio` times its
  loss tangent.

Measurement
  A post-solve function of a result's {term}`fields <Field>` that needs extra
  physics input (a surface, a thickness, a point, a path).

Power flow
  The net time-averaged power crossing the boundary between a chosen set of
  elements and the rest of the solve domain, positive when it leaves the set.
  Measured by `gfem.power_flow` from the residual of the discrete equations, so
  exact for the solved field. Poynting flux is kept for the pointwise integral
  of $\tfrac{1}{2}\mathrm{Re}(\mathbf{E} \times \mathbf{H}^*)$.

Bloch reciprocity
  $S(\mathbf{k})^\mathsf{T} = S(-\mathbf{k})$: a Bloch cell's S matrix is not symmetric, but swapping two
  ports is the same as reversing the {term}`Bloch wave vector`. With
  {term}`Floquet ports <Floquet port>` the orders relabel and take a sign:
  $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$, with $-p$ the port of the opposite order
  and $\sigma = -1$ for an order with a nonzero in-plane wave vector, $+1$ otherwise.
```

## Materials and optimization

```{glossary}
Material
  A linear medium with relative permittivity, permeability, loss tangent and
  electrical conductivity (S/m), assigned to a whole {term}`region`; each
  property may be one value or, for an {term}`anisotropic material`, a tensor,
  and the permittivity may follow a {term}`permittivity law`.

Loss tangent
  The ratio $\tan\delta$ of the imaginary to the real part of a {term}`material`'s
  permittivity, $\varepsilon = \varepsilon' (1 - i \tan\delta)$ under $e^{+i\omega t}$: the dielectric loss
  (`Material(tan_delta=...)`). A region holding the share $p$ of a mode's
  electric energy adds $p \tan\delta$ to the mode's $1/Q$.

Dispersive material
  A {term}`material` whose permittivity depends on frequency through a
  {term}`permittivity law`.

Permittivity law
  A relative permittivity that depends on frequency: a high-frequency
  permittivity $\varepsilon_\infty$ plus a sum of {term}`dispersion terms <dispersion term>`
  (`gfem.PermittivityLaw`), given as a `Material`'s `eps_r`.

Dispersion term
  One Drude, Lorentz, Debye or Djordjevic-Sarkar contribution to a
  {term}`permittivity law`, passive by its own parameter bounds.

Static permittivity
  The zero-frequency value of a {term}`permittivity law`: $\varepsilon_\infty$ plus the
  strengths of its Lorentz, Debye and Djordjevic-Sarkar terms. A Drude term
  has none, because a Drude metal conducts at zero frequency. The
  electrostatic problem reads it.

Energy permittivity
  The weight $\mathrm{Re}[\varepsilon + (\omega/2)\, d\varepsilon/d\omega]$ of $|E|^2$ in a mode's energy at its frequency,
  for a region with a {term}`permittivity law`: $\mathrm{Re}\,\varepsilon_r$ for a constant
  material, and $\varepsilon_\infty$ for a lossless Drude law.

Anisotropic material
  A {term}`material` whose permittivity, permeability, loss tangent or
  conductivity depends on the field direction: real symmetric tensors, given
  as {term}`principal values` along the material's {term}`principal axes` or
  as symmetric 3x3 matrices, all sharing one set of principal axes.

Principal axes
  The three orthonormal directions (`axes`) shared by every tensor property
  of one {term}`anisotropic material`; along each one a property takes its
  {term}`principal value <Principal values>`.

Principal values
  The three values a tensor property takes along a material's
  {term}`principal axes`; equal principal values mean the property is
  isotropic.

Gyrotropic material
  A material with a non-symmetric (for example Hermitian) permittivity or
  permeability tensor, such as a magnetized ferrite or a magneto-optic
  crystal; it is non-reciprocal and not supported.

Material field
  A spatially varying assignment of material properties (permittivity,
  permeability, conductivity) over the mesh, possibly derived from
  {term}`design parameters`; an {term}`anisotropic material field` holds
  principal values on fixed axes.

Anisotropic material field
  A {term}`material field` whose tensor properties are
  {term}`principal values` per element along one fixed set of
  {term}`principal axes` shared by the whole field; the principal values are
  design inputs, the axes are not.

Index interpolation
  A density blend in which the complex refractive index $n - i\kappa$ is linear in
  the density $\rho$, $\varepsilon = ((1 - \rho) N_a + \rho N_b)^2$ (`scheme="index"`); the blend to
  use between a metal and a dielectric.

Sheet
  A zero-thickness conducting layer on an interior surface, described by its
  admittance per square.

Sheet field
  A spatially varying fill over the faces of a named interior surface, scaling a
  {term}`sheet`'s admittance from none (0) to full (1); the surface counterpart
  of a {term}`material field`.

Layer grid
  A planar pixel grid of {term}`design parameters` over a named volume's
  footprint, each value held constant through the volume's thickness and mapped
  onto the volume's elements.

Design parameters
  The vector of quantities an optimization adjusts, such as per-element
  densities or mesh node positions, upstream of the {term}`material field` or
  geometry.

Topology optimization
  Optimization where {term}`design parameters` are densities interpolated into
  a {term}`material field` on a fixed mesh.

Shape optimization
  Optimization where {term}`design parameters` move mesh node positions, so
  material regions keep sharp interfaces.

Boundary morph
  The fixed linear map from normal displacements of a
  {term}`moving surface`'s nodes to moved mesh coordinates.

Moving surface
  The selected surface whose nodes carry the shape {term}`design parameters`.

Sliding surface
  A surface whose nodes may move only within their own facet plane during a
  {term}`boundary morph`. On a {term}`periodic pair`, a node and its partner
  move together.

Mesh extension
  The elastic or harmonic solve that carries boundary motion into the mesh
  interior.

Element quality
  The scaled Jacobian of an element; zero or below means inverted.

Remesh and restart
  Regenerating the mesh at the current {term}`design parameters` and restarting
  the optimizer's history.

Adjoint solve
  The extra linear solve, on the system a forward solve already assembled,
  that gives the gradient of an {term}`objective` with respect to all
  {term}`design parameters` at once. `jax.grad` runs it; you never write it.

Objective
  A scalar function of a solution (S-parameter, eigenfrequency, Q, field
  overlap) whose gradient with respect to {term}`design parameters` is
  requested.

Cluster-symmetric objective
  An {term}`objective` that depends on a degenerate group of modes only through
  quantities unchanged when the modes are relabelled, such as their sum, so its
  gradient exists where individual eigenvalue gradients do not.

Non-differentiable quantity
  A result value or model input that gfem refuses to differentiate in a given
  version, raising an error instead of returning a zero gradient.
```

## Mesh refinement

```{glossary}
Mesh refinement
  Repeated {term}`refinement cycles <Refinement cycle>` that subdivide the
  tetrahedra where an {term}`error estimator` reports the most error, until
  its {term}`error estimate` meets a tolerance or a cap stops them
  (`gfem.refine_mesh`). It gives a new model whose mesh keeps the physical
  groups and the imported facets.

Refinement cycle
  One pass of {term}`mesh refinement`: solve, estimate the error, mark the
  tetrahedra holding most of it, split them at the midpoint of their longest
  edge (and their neighbours, so no hanging node remains), and carry the
  design to the new tetrahedra. Cycle 0 is the solve on the input mesh.

Error estimator
  A rule that computes, from a result's fields, each tetrahedron's share of
  the error and one {term}`error estimate`: the {term}`field estimator` or
  the {term}`quantity estimator`.

Field estimator
  The {term}`error estimator` of the field's error in the energy norm,
  relative to the field's energy norm (`gfem.FieldEstimator`). It compares
  the solved electric flux with a smoother flux recovered from it
  ({term}`flux recovery`), so it needs no extra solve.

Flux recovery
  Projecting the solved electric flux (`eps E`, or `eps grad V` in
  electrostatics) and the curl of `E` onto smoother fields, one polynomial
  degree below the solve's order. On the solve domain's natural surfaces,
  the recovered flux's normal component and the recovered curl's
  tangential component are held at zero. In electrostatics these are every
  exterior surface not held at a potential, ports and impedance surfaces
  included; in wave problems, the PMC surfaces and the exterior surfaces
  given no condition. The difference between the solved and the recovered
  fields, weighted by the field's energy, is the field's estimated error.

Quantity estimator
  The {term}`error estimator` of one quantity: a capacitance entry, an S
  entry, one mode's `omega` or Q, the mean `omega` of a degenerate cluster,
  or an {term}`objective` of the result (`gfem.QuantityEstimator`). It
  weights each tetrahedron's field error by its effect on that quantity,
  using fields already solved, except for an S entry or objective whose
  dual is not a solved field, which takes one {term}`adjoint dual` each.

Objective estimate
  The {term}`error estimate` of an optimization {term}`objective`, a
  function of one result, in the objective's units
  (`gfem.QuantityEstimator(objective=...)`).

Estimate kind
  What an {term}`error estimate` is of: the field, an entry, Q, a
  {term}`cluster mean`, or one kind of objective (`ErrorEstimate.kind`).
  Each kind has its own measured range ({doc}`verification`).

Cluster mean
  The mean complex `omega` of the modes of a degenerate cluster
  (`gfem.QuantityEstimator(omega=(i, j))`). It does not depend on which
  basis of the cluster the solver returns, so it can be followed and
  estimated where each member cannot.

Region mean
  A field's or a potential's mean over named volume regions
  (`gfem.region_mean`): the quantity an objective reads in place of a point
  value, which has no estimate.

Lossless rewrite
  On a lossless model with lumped ports, estimating an objective of S
  together with the power each port's column must carry, which makes its
  estimate tighter (`ErrorEstimate.kind` names it).

Adjoint dual
  The field solved with a port's S read-out, or an objective's
  measurement, as the source instead of the excitation, at the same order.
  The {term}`quantity estimator` weights field errors with it when no
  solved field can stand in: an S entry whose read port is not excited, or
  is a wave port with a hybrid mode in a lossy cross-section, and an
  objective of a field measurement, of electrostatic energy or a potential
  mean, or of a participation. Otherwise the dual is a port's own field,
  scaled, and needs no solve.

Error estimate
  The number an {term}`error estimator` gives for a result, compared with
  the tolerance of {term}`mesh refinement`. It has no sign or phase and is
  an estimate, not a bound. For a {term}`quantity estimator` it is relative
  to a capacitance entry, `omega`, a cluster's mean `omega` or Q; absolute
  for an S entry; and in the objective's units for an objective. gfem
  never adds it to the quantity. On gfem's verification cases, against
  errors on the mesh's own facets, the field estimator's estimate was
  between {gfem-effectivity}`field low` and {gfem-effectivity}`field high`
  times the true error (orders 2 and 3) and the quantity estimator's on
  entries between {gfem-effectivity}`quantity low` and
  {gfem-effectivity}`quantity high` times (order 2); each other
  {term}`estimate kind` has its own range ({doc}`verification`). Neither
  sees the faceting of curved surfaces.
```

## Thermal

```{glossary}
Slab cross-section
  A 2D cross-section read as a mesh one element thick along z
  ({py:meth}`gfem.Mesh.from_gmsh_cross_section`), with insulated
  {term}`end faces`, so nothing varies along the extrusion. Its extensive
  quantities are stated per metre of extrusion ({term}`per-length quantity`).

End faces
  The two surfaces `"front"` and `"back"` of a {term}`slab cross-section`. They
  carry the wave port of a mode solve and the terminals of a DC conduction
  solve along the guide.

Thermal domain
  The volumes a thermal problem solves on: every volume of the mesh except
  those excluded from thermal problems (`excluded={"thermal": [...]}`).
  Volumes excluded only from electromagnetic problems stay in it.

Modelled depth
  The depth of substrate under the stack that the mesh includes, with the
  {term}`heat sink` at its bottom: a choice of the model, not the wafer's
  thickness.

Heat sink
  A surface that removes heat: a {py:obj}`gfem.FixedTemperature` or a
  {py:obj}`gfem.Convection` boundary. Every connected piece of the
  {term}`thermal domain` needs one in a steady solve.

Insulated surface
  An exterior surface of the {term}`thermal domain` that no thermal boundary
  names: no heat crosses it (`Model.default_insulated` lists the named ones).

Temperature
  Absolute temperature, in kelvin: the unknown of a thermal problem.

Temperature rise
  The temperature minus the {term}`source-free temperature`, in kelvin: the
  part of the temperature linear in the heat sources. When every boundary
  shares one temperature, it is the temperature above that one.

Source-free temperature
  The steady temperature a thermal model reaches with every heat source off,
  set by its boundary temperatures, ambients and heat fluxes alone; where a
  transient starts by default.

Thermo-optic coefficient
  The rate of change of a material's refractive index with temperature,
  dn/dT in 1/K, constant, about the index of its `eps_r`. It may be negative.

DC conduction
  The steady current through the conductors of a model between named
  terminals ({py:obj}`gfem.DCConduction`), giving their
  {term}`conductance matrix` and the heat they dissipate.

Conductance matrix
  The currents into a DC conduction solve's terminals per volt on each, in
  siemens; its inverse for one terminal and a ground is the resistance.

Joule heating
  The heat a current dissipates in a conductor; {py:obj}`gfem.joule_heating`
  turns a DC conduction result and a drive (voltage, current or power) into a
  heat source with no new solve.

Heater terminal
  A surface where a DC conduction solve holds a potential: a pad, a contact
  face, or an {term}`end face <End faces>` of a slab cross-section.

Thermo-optic shift
  The change of a guided mode's effective index, and of its phase per length,
  caused by a {term}`temperature rise`, to first order in the rise
  ({py:obj}`gfem.thermo_optic_shift`).

Thermo-optic phase
  The phase a temperature rise adds to guided light, $k_0\, \mathrm{Re}\, \Delta n_\mathrm{eff}$ per metre: a
  delay, positive for a positive {term}`thermo-optic coefficient`. A
  transmission's argument moves by minus it.

Thermal crosstalk
  The {term}`thermo-optic phase` a waveguide, or a {term}`pass`, that is not
  under the heater gains, as a signed fraction of the heated one's phase at
  the same drive.

Optical window
  A model of the optical neighbourhood of one waveguide core, on which a
  thermo-optic shift or a route phase solves the mode: the thermal model with
  the other regions excluded from electromagnetic solves, or a mesh of its
  own.

Route
  The reference line of a waveguide core in a 3D model, a polyline in metres,
  along which an {term}`optical window` is swept ({py:obj}`gfem.route_phase`).

Station
  A point on a {term}`route`, evenly spaced in arc length, where the optical
  window is placed and the rise is read.

Pass
  One straight stretch of a waveguide route under or beside a heater, whose
  phase is read on its own.

Route phase
  The {term}`thermo-optic phase` a {term}`route` gains from a 3D temperature
  rise, to first order, summed over its {term}`stations <Station>`.

Waveform
  The piecewise-constant scale of a transient problem's heat sources, as
  (start time, scale) pairs on its {term}`time grid`; off before the first.

Time grid
  The uniform times $t_n = n\, dt$, $dt = \mathrm{duration} / \mathrm{steps}$, on which a transient
  thermal solve steps, records its traces and keeps fields.

Trace
  A linear functional of the temperature (a point value, a region's mean, a
  thermo-optic phase) recorded at every time of a transient solve's
  {term}`time grid` ({py:obj}`gfem.Trace`).

Settled trace
  A trace whose change still to come after its last sample is below 1 % of its
  swing; the rise time and the time constant are meaningful only on one, and
  warn ({py:obj}`gfem.UnsettledTraceWarning`) on any other.

Rise time
  The 10–90 % time of a step response, each part of the trace normalised to
  its first and last samples ({py:func}`gfem.rise_time`).

Time constant
  The time a step response takes to reach 1 − 1/e = 63.2 % of its swing, with
  the same normalisation as the rise time ({py:func}`gfem.time_constant`).

Per-length quantity
  On a {term}`slab cross-section`, an extensive quantity (a power, a
  resistance) per metre of the extrusion: the only form gfem takes for
  extensive inputs on a cross-section (`power_per_length=`).

Mass density
  A material's mass per volume, in kg/m³ (`mass_density`), read by transient
  thermal problems.
```

## Distributed runs

```{glossary}
Distributed solve
  A solve whose mesh and linear system are split across the ranks of a communicator
  with more than one rank; every rank calls it with the same values and gets the
  same result.

Rank copies
  Independent complete solves, one per launched process, each on its own single-rank
  communicator, asked for by name.
```
