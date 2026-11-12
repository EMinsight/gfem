# Release notes

## Unreleased

- **Rerun eigenmode results on models with lossy walls.** `gfem.Eigenmode` now
  returns the modes nearest the target on a model with a `Conductor`, `Impedance`
  or `Absorbing` wall. Its starting vectors were wrong on such walls, so a solve
  could skip modes without a word, or return a different set from run to run.
  Eigenmode results on models with `Conductor`, `Impedance` or `Absorbing` walls
  from earlier versions may have skipped modes and should be rerun.

Eigenmode solves:

- When a pair nearer the target than the farthest mode found stops within 1e-6 of
  an eigenpair without converging, `gfem.Eigenmode` now raises
  `gfem.EigenmodeConvergenceError` instead of returning a set of modes without it.
- A heavily inductive `Impedance` wall can now return its sheet modes near the
  target, where cavity modes farther away were returned before.
- `gfem.Eigenmode` warns with `gfem.LowFrequencyWarning` when its target or its
  lowest mode is below the low-frequency floor (k0·h_min = 1e-6). The reported
  `error` does not show the digits lost there.
- A linear solve that does not converge inside an eigenmode solve now names
  `SolverOptions.eigen_tolerance` as the remedy, with the inner tolerance in use.
- With `eigen_solver="slepc"`, a shift solve that fails while seeding no longer
  leaks SLEPc's eigen solver; the solve's own error is raised as before. Bloch
  eigenmode problems with `eigen_solver="slepc"` are supported and tested.
- `gfem.interface_participation` decided whether a surface belongs to a periodic
  pair from a cache kept per mesh. A wall of a non-periodic model was refused once
  a `gfem.Periodic` model had been built on the same mesh, and a paired surface was
  measured without a word after it had been measured on another model or on a
  result read with `gfem.load_result`. It now reads the pairs the result records,
  in the new `EigenmodeResult.periodic_surfaces`, which is saved with the result.
  Results saved by earlier versions read as having no pair.

Optimization and gradients:

- **Optimization loops redo less work per design.** Within one process, a driven
  solve of a new design with the same mesh, ports and frequencies takes up the
  wave ports' couplings instead of rebuilding them. With `linear_solver="direct"`
  on the CPU in one process, it also takes up the last design's solver plan and
  the assembled parts that do not depend on the design. With
  `linear_solver="direct"` and the `mkl` extra, a large model may solve the next
  design on the previous design's factorization, with GMRES to a twentieth of
  `tolerance`, while that takes less time than factorizing again; small models
  factorize every design as before. gfem decides from the times it measures, so
  two runs of the same script can solve different designs on an earlier
  factorization, and their S and gradients can differ within `tolerance`. The
  default `linear_solver="auto"` picks `"bddc_gmres"` for a driven problem, which
  sets up again at every design. See {doc}`../recipes/s-parameter-gradient`.
  Per step on one MN5 node (Intel Xeon Platinum 8480+), main at 129c4ad against
  this release's code (measured at 135fbfd), 8 threads: the 2D bend tutorial
  0.77 → 0.48 s, the harmonic filter tutorial's three variants 0.54 → 0.30 s with
  peak memory 11.2 → 1.4 GiB, a 3D bend of 381 134 unknowns 11.1 → 9.1 s with peak
  memory 18.1 → 7.5 GiB; the 3D bend tutorial's model, 2.49 M unknowns, 16 threads,
  `retain_memory` at 64 GiB: 100 → 92 s.
  - **Behaviour change.** `retain_memory` now bounds everything gfem keeps between
    solves, these kept parts and couplings included; 0 keeps nothing and
    factorizes every design. A direct solve on an earlier design's factorization
    is accurate to `tolerance`, not exact.
- **Lumped values are design inputs.** A `LumpedPort`'s `inductance` and
  `capacitance`, and an `Impedance`'s `resistance`, `inductance` and
  `capacitance`, now differentiate: driven S-parameters on uniform sweeps and on
  adaptive sweeps with frozen samples (plain and Bloch periodic models included),
  kept fields, and eigenfrequencies. Electrostatic results get exact zeros, and the
  port mode problem raises.
  - A model with an `Impedance`, or with a lumped port's L or C, gains scalar
    leaves in `jax.tree.leaves(model)` and in a model's cotangent. Models with
    neither keep their leaf count. Every `Impedance` adds leaves, resistance-only
    ones included.
  - An `Impedance` whose surface shares an edge with a wave port face raises
    `gfem.PortGeometryError` when its values are differentiated.
  - A lumped port's resistance stays fixed, because it is the port's reference
    impedance: a traced one raises `gfem.NonDifferentiableError`, and the message
    says which values are differentiable.
  - A lumped value built under `jax.grad`, `jax.jit` or `jax.vmap` is checked when
    the solve runs: a non-finite or non-positive one raises `ValueError`.
- **`gfem.power_flow` is differentiable** through its field: under `jax.grad`,
  `jax.jit` and `jax.vmap`, the power absorbed, radiated or transmitted through a
  surface is an objective, on lumped-port, wave-port and periodic models. A
  material field or sheet fill in the element layer just outside the side (move
  the surface one element away), the node coordinates, the field's `omega`, and a
  field of an adaptive sweep (use `sweep="uniform"`) raise
  `gfem.NonDifferentiableError` when differentiated; every other model input gets
  an exact zero from the measurement. A side given by a predicate needs fixed node
  coordinates: under a `jax.jit` over the model, name the side's regions instead.
- **The electrostatic energy is differentiable.** `result.energy` of an
  electrostatic solve differentiates with respect to `eps_r` (isotropic and
  principal values) and the node coordinates, on both linear solvers, for one
  adjoint solve for all terminals together. No electrostatic output raises under
  `jax.grad` any more.
- **Kept fields with wave ports under a transformation.** `Driven(keep_fields=...)`
  on a model with wave ports now runs under `jax.jit`, `jax.grad` and `jax.vmap`,
  and the kept fields are differentiable there. The wave ports' mode fields are
  `None` inside a transformation.
- Fixed: the gradient with respect to a `MaterialField` conductivity was exactly
  zero where the conductivity is zero everywhere (driven and eigenmode); it is now
  the one-sided derivative.
- Fixed: in an adaptive sweep with frozen samples, the gradient of an S entry
  whose port is not excited was off at the samples, by up to about 1e-2. Each
  sample now also solves one right-hand side per lumped port left out of
  `excite`. Adaptive results with `excite=` a strict subset of the lumped ports
  move by less than the sweep tolerance, and `excite=` no longer saves solves in an
  adaptive sweep.
- An objective reading the S row of a lumped port without a resistance gets a NaN
  gradient in an adaptive sweep, as in a uniform one, where it raised an NGSolve
  exception.
- **Behaviour change.** `MaterialField` values built under `jax.grad`, `jax.jit` or
  `jax.vmap` are checked when the solve runs. A non-finite or non-passive value
  raises the constructor's error with the region named, where it gave NaN, a
  solver convergence error or a finite result.
- `gfem.design.project` returns values in [0, 1] exactly; round-off could leave a
  few ulp outside, which `SheetField` refused. The gradient at densities 0 and 1
  is unchanged.
- The {doc}`../capabilities` page's gradient checks gain the permeability and the
  loss tangent (driven, isotropic and principal values; eigenmode) and the
  electrostatic region energies.
- **Fixed: periodic gradients of a `MaterialField` with `axes`.** Recompute them if
  your periodic model's fields touch a first-order `Absorbing` surface. There, the
  gradients with respect to `eps_r` and `mu_r` principal values were wrong: off by 0.5 %
  to 10 % in the cases measured, more for eigenmodes. They now agree with finite
  differences. Periodic models without such a surface got the right gradients and now
  get them faster. Non-periodic models were not affected.

Wave ports, S-parameters and measurements:

- **`Field.at` reads the tetrahedron that holds the point.** A point near a face, within
  1e-4 of an element's size, could get the neighbouring element's value, off by the
  field's full jump there, and the value could change with the rank count. It now gets
  the tetrahedron it lies in, on any number of ranks. Within 1e-9 of an element's size
  from a face, a point counts as on the face. A point on or just above an excluded
  conductor no longer raises a false "lies in volume region"; a point just inside one,
  which used to read the air's value, now raises. See {doc}`../recipes/measurements`.
- A wave port whose port line is lossy now warns with
  `gfem.PortModeProjectionWarning` when α/β exceeds 1e-3, lone ports included: its
  reflection entries of S are off by about α/β, and transmission is right to the
  square of it. A shared surface still warns once, for its cross-projection when
  that is above the floor. The warning is no longer hidden by an unsolved
  frequency of a partial sweep, and a mode exactly at cutoff no longer gives a
  spurious warning.
- `characteristic_impedance` works on any eager driven result with wave ports,
  with or without `keep_fields`. Inside a transformation it raises `ValueError`
  pointing to `gfem.PortMode`.
- A uniform sweep with wave ports runs with `frequency_workers > 1`. The main
  process solves the port modes once over the whole grid before the workers
  start. S and the reference impedances match the one-worker sweep, bit for bit on
  one thread and to rounding otherwise.
- `gfem.PortMode` ignores `SolverOptions.device`, `linear_solver` and
  `eigen_solver` instead of raising, so the options of a driven solve can be
  reused. Its degenerate-mode `PortModeError` now suggests a mirror half model (a
  symmetry plane through the guide's axis) before a `pec_edges` wall.
- `DrivenResult.with_offsets({port: d})` moves wave-port reference planes after
  the solve, from the stored propagation constants. Offsets are absolute, and the
  result equals a solve with `WavePort(offset=d)`.
- `gfem.electric_energy(field, model=model)` gives the time-averaged electric
  energy, in J, that a kept driven field or a mode field stores in each volume
  region. Dielectric loss in a region with one loss tangent absorbs
  `2 * omega * tan_delta * W_e`. It is forward only. The measurements recipe adds
  that the power of several ports fed at once is `power_flow` of the superposed
  kept fields.
- `NetworkParameters.to_touchstone` writes rows in increasing frequency, as the
  format requires, whatever the order of the result. `to_touchstone(path, z0=R)`
  writes a network in which no port has a reference impedance, such as wave ports
  without `voltage_path`: S is written unchanged with R as a nominal label, and a
  comment line says so. Without `z0` it raises with that remedy.
- **Behaviour change.** `Driven.frequency` and `PortMode.frequency` refuse a
  frequency given twice (`ValueError`). A repeat was solved in full twice, crashed
  `keep_fields=True` and wrote an invalid Touchstone file. Drop repeats with
  `numpy.unique(frequency)`. The order stays free, and results keep the order
  given.
- A `LumpedPort` direction accepts an axis without a sign (`"x"`, `"Z"`): it means
  the positive axis.
- `Material`'s message for a complex `eps_r` or `mu_r` names the remedy for each:
  `tan_delta` or `conductivity` for dielectric loss, a complex value through
  `gfem.MaterialField` for either.
- Frequencies in error messages, warnings and result reprs use Hz, kHz, MHz, GHz
  or THz, so a 193 THz cutoff no longer reads "193414 GHz"; text between 1 GHz and
  1 THz is unchanged. The low-frequency warnings of driven and eigenmode solves and
  the mode-following errors of `refine_mesh` use the same units, and a
  one-frequency band in port mode messages reads "60 GHz", not "60-60 GHz".
- `power_flow`'s wrong-quantity error now reads "needs an electric field".

Mesh refinement:

- **Adaptive mesh refinement.** `gfem.refine_mesh` solves, estimates the error in
  each tetrahedron, subdivides where most of it is and solves again, until the
  {term}`error estimate` meets a tolerance. It returns the refined model, its
  result and the history of the run. See {doc}`../recipes/mesh-refinement`.
- `gfem.refine_mesh` frees the solve entries a gradient has already read before
  its first solve, so the last optimizer step's factorization no longer stays in
  memory through the refinement cycles.
- With `QuantityEstimator(omega=...)`, `gfem.refine_mesh` raises
  `gfem.DegenerateModeError` when the followed mode is not clearly the nearest on a
  refined mesh (a second mode less than twice as far), or when the nearest mode
  moved further than the previous error estimate allows: ask for more modes or
  start from a finer mesh.
- `gfem.error_estimate` raises `gfem.DistributedUnsupportedError` (feature
  `error_estimate`) for a result solved across ranks, instead of a NumPy broadcast
  error or a wrong value. `refine_mesh` across ranks raises with feature
  `mesh_refinement`.

Solvers, memory and ranks:

- **Wrong S from a PETSc stack built by 0.1.0's script: rebuild it.** A stack built by
  0.1.0's [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
  can return wrong S, with no error, on a large direct solve with
  `linear_solver="petsc"`. A 10 M DOF mesh solved in one process stayed below the size
  where this happens. Rebuild the stack with the current script, as
  {doc}`../recipes/petsc-slepc` says. The cause was an integer overflow in an OpenBLAS
  routine that MUMPS calls; the script now builds MUMPS without it.
  - **Behaviour change.** gfem refuses a stack built without this fix, at any size, with
    `gfem.SolverUnavailableError`. `linear_solver="bddc_gmres"` above one rank still
    runs on such a stack, without using it for its coarse problem.
- **Behaviour change: default thread count.** With `threads=None`, NGSolve now runs on
  `NGS_NUM_THREADS`, else on the physical cores the process may run on, instead of every
  hardware thread of the machine. Without an affinity mask (macOS, Windows) the count is
  the logical CPUs. Your own `ngs.SetNumThreads` no longer carries into such a solve:
  set `SolverOptions.threads` instead. Inside your own `ngs.TaskManager()`, a solve
  runs on that TaskManager's threads, whatever `threads` says, on one process and
  under ranks; gfem no longer calls `SetNumThreads` there, so NGSolve's "can't
  change number of threads" line is gone. Under ranks, `threads=None` shares each
  physical core between the ranks whose affinity mask holds it, so partly
  overlapping masks no longer oversubscribe; identical and disjoint masks give
  the same counts as before.
- **Driven solves across MPI ranks.** `mpirun -n N python script.py` or
  `srun --mpi=pmix` now runs a uniform driven sweep and its gradients across the
  ranks, with `comm=` on `gfem.solve`, `Job.run`, `Mesh.from_gmsh`, `Job.from_json`
  and `load_result`. Every other problem and some options raise
  {py:obj}`gfem.DistributedUnsupportedError`. See {doc}`../recipes/distributed-solve`.
  - The `mpi` extra is mpi4py; `import gfem` needs neither it nor MPI. Only Open MPI
    is accepted above one rank.
  - [`tools/petsc_wheelhouse.sh`](https://github.com/benvial/gfem/blob/main/tools/petsc_wheelhouse.sh)
    gains `--mpi`, `--sources` and `--fetch-sources`, for the {term}`MPI PETSc stack`,
    which sits beside the unchanged no-MPI stack.
  - **Behaviour change.** A launched solve used to run N complete one-process solves
    with no warning. It now runs distributed or raises, and a launched install
    without mpi4py raises `gfem.SolverUnavailableError`.
    `comm=MPI.COMM_SELF` asks for the copies by name.
- **`gfem.release_memory()`** frees everything gfem keeps between solves in a
  process: solve entries, pooled solver plans and factorizations, and cached
  discretizations. Results are unchanged; a released gradient is recomputed.
- A band sweep no longer grows memory with every wave vector: a discretization
  keeps the spaces of at most 8 Bloch phase sets.
- Freeing a direct solver handle from another thread during a solve now always
  gives "This solver handle was freed." or waits for the solve, never a native
  exception or an answer from a freed handle. `refactorize` on a direct, PETSc or
  cuDSS handle with another free-DOF set raises `ValueError` instead of solving on
  the old free DOFs. The `SolverOptions.threads` docs say that a process runs one
  solve at a time.
- **Behaviour change.** With `linear_solver="petsc"`, `ksp_type preonly` (the
  default) is refused with a `pc_type` other than `lu` or `cholesky`, and with a
  `pc_factor_shift_type` other than `none`: these returned a wrong S with no error.
  Use a Krylov `ksp_type` such as `gmres` with any other preconditioner. A PETSc
  Krylov solve that ends on its iteration count without measuring a residual
  (`ksp_convergence_test skip`) raises `ValueError`. `petsc_options["ksp_atol"]` is
  reserved, like `ksp_rtol`: gfem stops on the relative residual set from
  `tolerance`. The `SolverOptions.tolerance` docstring no longer says S is accurate
  to the tolerance.
- Above one rank, `linear_solver="petsc"` with `pc_type` `ilu` or `icc`, or `lu`
  or `cholesky` on a package other than MUMPS, raises
  `gfem.DistributedUnsupportedError` (feature `petsc_preconditioner`) before any
  mesh is read. The `power_flow` and `electric_energy` refusals across ranks name
  the route that works: `Job.run(out=)`, then `load_result` on one process.
- `linear_solver="auto"` (the default) runs the direct solver on a GPU:
  `SolverOptions(device="cuda")` and `python -m gfem run --device cuda` on a
  default job are accepted. `"bddc_gmres"` and `"petsc"` on a GPU stay refused,
  with a message naming both accepted values. An `"auto"` job written for a GPU and
  run with `--device cpu` runs the problem type's CPU default.
- `SolverOptions.order` above 12 raises `ValueError`; higher orders failed inside
  NGSolve or hung.
- New warning classes `gfem.SweepRankDeficientWarning` (an adaptive sweep near the
  accuracy of its full solves) and `gfem.UnsampledPixelsWarning` (`LayerGrid`
  pixels that no centroid samples). Both subclass `UserWarning`; their messages
  are unchanged.

Jobs and files:

- A uniform sweep that stops at an unsolved frequency under `Job.run(out)` or
  `python -m gfem run` now leaves a {term}`partial bundle`, `partial.json` and
  `partial.npz`, never `result.json`. The error carries a note saying where it is;
  the command line prints it as a second stderr line `gfem: note: ...` and still
  exits with 1. Read it with `gfem.load_result("DIR/partial.json")`: its `run`
  entry lists `unsolved` and `error`. A complete run removes a stale partial
  bundle.
- **Behaviour change.** An unknown key in a job or result file is refused by name,
  with a `ValueError` giving the file, the class, the keys and both gfem versions.
  `Job` and `Mesh` dropped such keys without a word, and other classes raised a
  bare `TypeError`.

Documentation:

- New user pages: {doc}`../recipes/solve-a-device`, {doc}`../recipes/prepare-a-mesh`,
  {doc}`../recipes/save-and-export` and {doc}`../conventions`. Every number they
  print comes from a solve and is checked against a hand formula or a file read
  back. The glossary gains S-parameters, power waves, lumped port, wave port, loss
  tangent, element order and adjoint solve.
- Every user page's code is checked against the API without solving: names
  resolve and call keywords bind.
- The {doc}`../recipes/gsim` recipe reads gsim's one-face lumped ports
  (microstrip), which raised `KeyError: 'Elements'`.
- The periodic-cell recipe and `gfem.Absorbing` state the reflection of an
  absorbing boundary under oblique incidence in a Bloch cell, (1 − cos θ)/(1 + cos θ)
  at order 1, and that order 2 lowers it for one polarization only. Bloch S is now
  checked against its reference to 5e-6, and periodic pairing and Bloch bands are
  tested on an unstructured gmsh mesh.
- The PETSc recipe states the stack's build time (about 22 minutes, 26 with
  `--mpi`) and what a Krylov method on the MUMPS factorization does below the
  low-frequency floor.
- The get-started page links the 3D bend tutorial, the recipes page lists mesh
  refinement, and the package description matches the README.

Anisotropic materials and shared port surfaces:

- **Anisotropic materials.** Driven, eigenmode, electrostatic and port mode
  solves, wave ports, first-order absorbing boundaries, `MaterialField` and
  its gradients, `EigenmodeResult.principal_participation`,
  `interface_participation` and job files now take an anisotropic
  `Material`. The cases listed below as unsupported raise. See
  {doc}`../recipes/anisotropic-materials`.
  - `eps_r`, `mu_r`, `tan_delta` and now `conductivity` each take one value,
    three principal values along `axes`, or a real symmetric 3×3 matrix in
    the global frame, stored as principal values plus `axes`. Driven and
    eigenmode solves read a conductivity given as principal values along its
    principal axes.
  - Wave ports accept anisotropic `mu_r`, and anisotropic materials in a
    {term}`transverse-anisotropic` cross-section: every tensor has a
    principal axis along the port normal, at any orientation in the port
    plane.
  - First-order absorbing boundaries use the material's exact admittance
    (see the behaviour change below).
  - A `MaterialField` with `axes` holds three principal values along fixed axes:
    shape `(n_elements, 3)` per element, or `(3,)` for the whole region.
    `MaterialField.uniform` takes an anisotropic `Material`.
    `MaterialField.interpolate` blends two materials per principal value when
    their tensors share principal axes (air and an anisotropic material, for
    example); materials with different axes raise. Driven, eigenmode,
    electrostatic and port mode solves read the field's tensor on each element.
    `jax.grad` gives one derivative per principal value, checked against finite
    differences. The axes are fixed: the orientation is not a design input.
  - `EigenmodeResult.principal_participation` splits each anisotropic
    region's participation along its principal axes. It gives the exact
    per-region loss for `tan_delta` and `conductivity` given as principal
    values.
  - Errors: passivity errors name the failing principal value and its axis.
    The wave-port error names the property (`eps_r`, `mu_r`, `conductivity`)
    with no principal axis along the port normal, and gives the workarounds.
    Non-symmetric (gyrotropic) tensors and tensors with different principal
    axes raise `ValueError` at `Material`.

  Job files and saved results from 0.1.0 read unchanged.
- **Behaviour change: equal principal values are one value.**
  `Material(eps_r=(2, 2, 2))` is now stored as `eps_r=2.0`: it equals
  `Material(eps_r=2)`, `isotropic` is `True`, and its repr changes. Results are
  unchanged. Old job files holding such a tuple read back as the scalar.
- **Behaviour change: absorbing boundaries next to anisotropic materials.**
  - S, fields and Q change for a model whose first-order `Absorbing` surface
    touches an anisotropic material. Each such face now uses that material's
    normal-incidence admittance. The mean of the tensor diagonals, used before,
    reflected up to about 13 % at normal incidence next to a uniaxial dielectric.
    Faces next to isotropic materials are bit for bit unchanged.
  - The port mode problem's absorbing edges change the same way: each edge
    next to an anisotropic material takes its exact admittance for the two
    polarizations.
  - A second-order surface touching an anisotropic material now raises
    `NotImplementedError`.
  - Node-coordinate gradients of a model whose absorbing face has an
    anisotropic material behind it now raise `NonDifferentiableError`; 0.1.0
    differentiated the approximate coefficient.
- **Behaviour change: port modes of rotated transverse tensors.** A port
  cross-section holding a tensor rotated about the port normal no longer drops
  its propagating modes (or raises `PortModeError`). The effective-index bound
  now uses the tensor's largest principal value, not its largest diagonal
  entry. Models without such a tensor are unchanged.
- **Behaviour change: `interface_participation`'s vacuum test.** A side is
  vacuum only when no wave in it is slower than in vacuum (largest principal
  `eps_r` × largest `mu_r` equal to 1). A side such as `eps_r = (1, 1, 4)` was
  taken as vacuum, and `"MA"`/`"SA"` read its normal field with ε0 (a 4×
  error); it no longer is. Isotropic sides are unchanged.
- **Several modes on one wave port surface.** Several
  {py:obj}`gfem.WavePort`s on one surface, one `mode` each, form a
  {term}`shared port surface`. Each mode is its own port, with its own row
  and column of S, so S holds the coupling between modes (for example TE0 to
  TM0 conversion in a silicon strip). Gradients and adaptive sweeps work as
  for one mode per port. {py:obj}`gfem.PortMode` on such a port gives the mode
  the driven solve uses. See {doc}`../recipes/shared-port-surface`.
- **New warnings.** {py:obj}`gfem.SkippedPortModeWarning`: a surface leaves
  out a port mode ranked below a requested one (a lone `mode=2`, or modes 1
  and 3), which the port then partly reflects. A lone `mode=1` never warns.
  {py:obj}`gfem.PortModeProjectionWarning`: on a shared port surface with
  loss in the cross-section, one port's S reads part of another port's mode
  above 1e-3.
- **New errors at `gfem.Model`.** Two wave ports on one surface that request
  the same `mode`, or give different `pec_edges`, raise `ValueError`. A lumped
  port or a boundary condition beside a wave port on one surface still does.
- **Behaviour change for a lone wave port.** A port mode whose reaction or
  Gram matrix on the port face is singular now raises
  {py:obj}`gfem.PortModeError` before the 3D solve, where 0.1.0 went on
  without warning. Lone-port S is otherwise unchanged.
- **Behaviour change: lossy modes below cutoff.** A requested port mode whose
  attenuation is at least its phase constant (Re β ≤ |Im β|, for example an
  RC-like line at low frequency) now counts as below cutoff, where 0.1.0
  accepted it as a propagating mode. The solve raises the port mode sweep's
  cutoff {py:obj}`gfem.PortModeError`, naming the port (and its surface when
  shared), lone wave ports included.
- **`gfem.Eigenmode`'s message** for a wave port on a shared port surface
  names the surface and its wave ports and points to a `gfem.Driven` sweep.

Periodic cells: `gfem.Periodic` pairs two surfaces of a model related by a
translation (one to three pairs). Eigenmode and uniform driven problems take a
Bloch wave vector, `wave_vector=(kx, ky, kz)` in rad/m, with
E(r + a) = exp(−i k·a) E(r); without one the fields are plain periodic. See
{doc}`../recipes/periodic-cell`.

- **Gradients.** Permittivity, conductivity, sheet fill and node coordinates
  differentiate on periodic models, plain and Bloch, and are checked against
  finite differences (four new rows on {doc}`../capabilities`). A boundary
  morph moves the two nodes of a pair together. `wave_vector` is not
  differentiable.
- **Solvers under a Bloch wave vector.** Eigenmode problems need the direct
  solver. PETSc, cuDSS and the adaptive sweep refuse a wave vector. The direct
  solver factors the problem with MKL PARDISO (with the `mkl` extra) or
  UMFPACK, and BDDC factors its coarse problem likewise. Models without a
  periodic pair are unchanged.
- **Results.** Fields of a periodic model hold values on both surfaces of
  every pair. `gfem.power_flow` works on sides that cross a pair.
  `gfem.interface_participation` refuses a paired surface.
- **Job files** carry `Periodic` and `wave_vector`; older job files read
  unchanged.

## 0.1.0 (2026-11-01)

First release: driven problems (uniform and adaptive sweeps), eigenmode problems
(including nonlinear ones), electrostatic and port mode problems, with lumped
and wave ports. Gradients with respect to permittivity, conductivity, sheet
fill and mesh node coordinates are checked against finite differences.

Changes since the release candidate of 2026-10-21:

The direct and `bddc_gmres` solvers now keep getting faster past 16 CPU
threads on large problems. Results change only to the solver's precision.

- **`bddc_gmres` runs its GMRES on all its threads.** A driven or eigenmode solve
  with `bddc_gmres` now uses `SolverOptions.threads` threads for the step where
  NGSolve's GMRES used one. Iteration counts and S are unchanged, to rounding.
  While such a solve runs, other SciPy BLAS calls in the same process run on one
  thread; SciPy's thread count is restored when the solve ends.
- **With the `mkl` extra, BDDC's setup is faster.** BDDC factors its coarse
  problem with MKL PARDISO instead of NGSolve's sparse Cholesky. Without the
  extra, nothing changes. Results with and without the extra agree to the
  `tolerance` of the solve, not bit for bit.
- **On 16 or more threads, a BDDC setup that serves many solves uses sparse
  Cholesky.** Once one setup serves enough solves, sparse Cholesky is faster
  overall than MKL PARDISO there, and uses less memory. This applies when one
  BDDC setup expects 16 or more right-hand sides on 16 or more threads, or 6 or
  more on 56 or more threads. A uniform driven sweep counts its excited ports
  times its frequencies; a single-frequency solve or an {term}`adaptive sweep`
  counts its excited ports only, and an electrostatic solve its terminals. A
  {term}`frequency worker` counts its own share of the threads and its own
  frequencies. Below 16 threads, or without the `mkl` extra, nothing changes.
  The choice depends on these counts only, never on timings, so a run repeats
  exactly.
- **MKL PARDISO frees its cores as soon as each call returns.** Every PARDISO
  call gfem makes runs with Intel OpenMP's block time at 0, so the rest of the
  solve gets the cores back. The setting is restored after each call, so a
  `KMP_BLOCKTIME` you set applies everywhere else.
- **A reused PARDISO factorization keeps its thread count.** With the `mkl`
  extra, a factorization reused under a different thread count, or outside the
  `TaskManager` it was made in, used to solve on one thread: about 5 times
  slower on a 10-million-unknown problem. A gradient that reused a cached
  adjoint factorization under `threads=None` could hit this. It now solves on
  the threads it was factored with. A factorization made outside any
  `TaskManager`, such as `gfem.design.helmholtz_filter`'s, likewise keeps
  solving on one thread. Results do not change.
- **Single-frequency direct solves start factorizing sooner.**
  `linear_solver="direct"` no longer copies the assembled matrix when every
  unknown is free. It builds the index it needs to refactorize with new values
  only when a second frequency needs it.
