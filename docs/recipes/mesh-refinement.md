# Refine the mesh where the error is

Get a converged capacitance, S-parameter or resonance without meshing the
whole device finely. {py:obj}`gfem.refine_mesh` solves, estimates the error
of the solution in each tetrahedron, subdivides the tetrahedra holding most
of it, and solves again, until the {term}`error estimate` meets a tolerance.
It returns a new model on the refined mesh, its result, and the history of
the run. The mesh you import keeps its physical groups and its facets: gfem
only subdivides them.

## Refine a capacitance solve

The mesh `stack.msh` has an oxide and a nitride layer, terminals on its top
face and on the oxide-nitride interface, and the ground on its bottom face.

```python
import gfem

mesh = gfem.Mesh.from_gmsh("stack.msh")
model = gfem.Model(
    mesh,
    materials={"oxide": gfem.Material(eps_r=3.9), "nitride": gfem.Material(eps_r=7.5)},
)
problem = gfem.Electrostatic(
    terminals={"top": "top", "mid": "oxide__nitride"}, ground="bottom"
)

ref = gfem.refine_mesh(model, problem, tolerance=1e-2, max_tetrahedra=200_000)
ref.stop  # "tolerance": the estimate met 1e-2
ref.result.capacitance  # F, solved on ref.model.mesh
for cycle in ref.cycles:  # cycle 0 is the imported mesh
    print(cycle.tetrahedra, cycle.dofs, cycle.estimate)
```

- Each {term}`refinement cycle` solves, estimates the error, stops if the
  estimate is at or below `tolerance`, marks the fewest tetrahedra holding
  half the squared error (`error_fraction=0.5`), splits them at the
  midpoint of their longest edge, splits their neighbours until no hanging
  node remains, and carries the model's design to the new tetrahedra.
- The default estimator, {py:obj}`gfem.FieldEstimator`, estimates the
  field's error in the energy norm, relative to the field's energy norm:
  `tolerance=1e-2` asks for a field about 1 % off its value on an infinitely
  fine mesh of the same geometry. No extra solve is needed.
- `ref.model` is a new model: use it, not `model`, for any later solve.
  `ref.result` is its result, and `ref.estimate` that result's estimate.
- When the loop stops on `max_cycles` or `max_tetrahedra` with the estimate
  above `tolerance`, it warns with
  {py:obj}`gfem.MeshRefinementNotConvergedWarning`, `ref.converged` is
  `False`, and `ref.stop` says which cap stopped it. A solve that fails
  after the first mesh stops the loop with `stop="solve failed"` and keeps
  the last solved mesh.
- A driven problem on a uniform sweep (lumped or wave ports) and an
  eigenmode problem refine the same way. The estimate of a sweep is its
  worst frequency's, of an eigenmode solve its worst mode's.

## Read the estimate

`ref.estimate.value` is the number compared with `tolerance`, the largest
of `ref.estimate.values`: one value per frequency of a driven sweep, per mode
of an eigenmode solve, and one otherwise. `ref.estimate.indicators` holds one
value per tetrahedron of `ref.model.mesh`, the share of the error that
marking reads. {py:obj}`gfem.error_estimate` gives the same estimate for a
result you solved yourself, without refining:

```python
result = gfem.solve(model, problem)
estimate = gfem.error_estimate(result, gfem.FieldEstimator(), model=model)
estimate.value, estimate.indicators  # relative energy error, per tetrahedron
```

A driven result must keep its fields for this: solve it with
`gfem.Driven(..., keep_fields=True)`. `refine_mesh` keeps them itself.

## Refine for one quantity

When one number matters, a capacitance entry, an S entry or one mode's
frequency, refine for it with a {py:obj}`gfem.QuantityEstimator`. It weights
each tetrahedron's field error by its effect on that quantity, from the
fields already solved, so the mesh grows where the quantity is sensitive
and the estimate is that quantity's error. Its keyword is the result
attribute that holds the quantity, indexed the same way:

```python
problem = gfem.Driven([3e9, 8e9])  # every port excited
estimator = gfem.QuantityEstimator(s=("p2", "p1"))  # S21: result.s["p2", "p1"]
ref = gfem.refine_mesh(
    model, problem, estimator=estimator, tolerance=1e-3, max_tetrahedra=200_000
)
s21 = ref.result.s["p2", "p1"]
for f, s, e in zip(ref.result.frequency, s21, ref.estimate.values, strict=True):
    print(f"{f / 1e9:.0f} GHz: S21 = {complex(s):.4f} ± {e:.1e}")
```

`ref.estimate.values[i]` is the estimated error of the quantity at
`ref.result.frequency[i]`. For an S entry it is absolute, so
`S21 = {s:.4f} ± {e:.1e}`. For a capacitance or a frequency it is
relative: multiply by `abs(C)` or `abs(f)`. It has no sign or phase: it
tells you how far the quantity may be from the converged answer, not in
which direction. gfem does not add it to the quantity.

The other two quantities:

```python
gfem.QuantityEstimator(capacitance=("top", "top"))  # result.capacitance[0, 0]
gfem.QuantityEstimator(omega=0)  # result.omega[0], an eigenmode problem
```

- **Capacitance**: the entry `capacitance[i, j]`, named by its two
  terminals, or by one terminal twice for a diagonal entry. The estimate is
  relative to that entry. A diagonal entry never lies below its value on
  the meshed geometry, `C_h >= C`: the difference is the energy of the
  potential's error, which is never negative, so mesh refinement lowers it.
  An off-diagonal entry may lie on either side.
- **S entry**: both ports must be excited (`gfem.Driven(excite=...)`), since
  the estimate uses the second port's field as the weight. The estimate is
  absolute.
- **Mode**: `omega=0` names the first mode of the input mesh's solve; on
  each refined mesh the estimate follows the mode whose `omega` is nearest
  the previous one. A mode with another within `SolverOptions.cluster_rtol`
  of it raises {py:obj}`gfem.DegenerateModeError`. The estimate is relative
  to `omega`. On a model with loss it covers the complex `omega`; Q's
  relative error is not estimated.
- `RefinementCycle.value` holds the quantity on each mesh (a tuple per
  frequency for an S entry), so the quantity against `cycle.dofs` plots
  without another solve.

## Refine between optimization stages

`refine_mesh` is not differentiable: it builds a new mesh, which `jax.grad`
cannot follow. Refine between stages of an optimization, on the current
design, then differentiate {py:obj}`gfem.solve` on the refined model. A
density design already has stage boundaries, the steps of its projection
sharpness `beta`; a shape design has the points where you would otherwise
remesh ({doc}`remesh-and-restart`).

```python
import jax
import jax.numpy as jnp

from gfem import design

# material_of, make_loss, make_optimizer and run_stage are your own code.
for beta in (2, 4, 8, 16):
    # 1. The model holding the current physical design, with concrete values.
    current = model.with_material("design", material_of(rho, beta))
    # 2. Refine it, with a cap.
    ref = gfem.refine_mesh(
        current, problem, estimator=estimator, max_cycles=3, max_tetrahedra=200_000
    )
    model = ref.model
    # 3. Rebuild every helper on the refined mesh.
    grid = design.LayerGrid(model.mesh, "design", pitch=0.5e-3)
    blur = design.helmholtz_filter(grid, radius=1.5e-3)
    # 4. Rebuild the objective and its jitted gradient on the refined model.
    loss = make_loss(model, grid, blur, beta)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    # 5. Restart the optimizer from the current parameters, its state reset.
    optimizer = make_optimizer(x0=rho)
    rho = run_stage(optimizer, value_and_grad, STEPS_PER_BETA)
```

- **Rebuild the jitted objective.** A function jitted as a closure over a
  model keeps the model it captured when it was traced. Rebinding the name
  `model` leaves the old mesh in use, with no error.
- **Reset the optimizer.** The discrete objective jumps a little at a new
  mesh, so L-BFGS curvature pairs and MMA asymptotes from the old mesh no
  longer apply. Compare objective values only within one mesh.
- **Carry your design parameters.** A `LayerGrid` pixel array keeps its
  shape: bisection only adds midpoints, so the rebuilt grid has the same
  pixels. A density with one value per element goes through
  `ref.transfer(values, "design")`, which gives each refined element the
  value of the input element it lies in; rebuild
  `design.helmholtz_filter(ref.model.mesh, "design", radius=r)` on it. The
  model's own `MaterialField` values and sheet fills are carried for you.
- **Shape designs.** Rebuild `gfem.shape.boundary_morph(ref.model, surface,
  where=...)` with the same selection, and map your parameters relative to
  the current ones `q_k`: the refined mesh already has the geometry of
  `q_k`. A helper left on the old mesh gives arrays of the old length,
  which the model rejects with `ValueError`. A mesh from an old morph passed
  to `ref.model.replace(mesh=...)` is not caught: always build meshes from
  the morph rebuilt on `ref.model`.
- **Never call `refine_mesh` inside the objective.** Under `jax.jit` it
  raises {py:obj}`gfem.NonDifferentiableError`, and so does any traced
  value in its model or problem.
- **Refine once more on the final design** before you report its
  quantity, so the reported value comes with an error estimate on the
  design you deliver.
- The {py:obj}`gfem.QuantityEstimator` on the objective's quantity puts the
  elements where the objective needs them. Where it raises
  {py:obj}`gfem.MeshRefinementUnsupportedError` (see Limits), use the
  {py:obj}`gfem.FieldEstimator`.

## Options

- **`max_tetrahedra`**: set it on every run. Each cycle grows the mesh by
  about 1.6 times on the first cycle and 1.25 times on later ones, so the
  default 10 cycles can grow a mesh 10 to 30 times. A refined mesh above the
  cap is discarded and the loop stops on the last solved one.
- **`max_cycles`** (default 10): how many times the mesh may be refined; 0
  solves and estimates only.
- **`error_fraction`** (default 0.5): the share of the summed squared
  indicators the marked tetrahedra hold. A larger fraction refines more per
  cycle, in fewer cycles.
- **`tolerance`** must be at least 10 times `SolverOptions.tolerance`
  (`eigen_tolerance` for an eigenmode problem), or `refine_mesh` raises
  `ValueError`.
- **Memory.** Every cycle of a driven problem keeps the field of every
  excited port at every frequency, and `ref.result` keeps those of the last
  mesh: memory grows with ports × frequencies × DOF. Refine on the few
  frequencies that matter, then solve the sweep on `ref.model`.

## Check your results

- The quantity on each cycle, `ref.cycles[k].value`, should settle as the
  estimate falls, with each change of the order of the estimate or below.
- Solve `ref.model` again with `SolverOptions(order=3)`: the change in the
  quantity should be about the estimate or smaller.
- On a lossless device, S on the refined model should stay passive and
  reciprocal, `SᴴS ≈ I` and `S ≈ Sᵀ`.

How gfem itself checks the estimates against known answers:
{ref}`verification-mesh-refinement`.

## Limits

- The error estimate is an estimate, not a bound. On gfem's verification
  cases the field estimator's was between {gfem-effectivity}`field low` and
  {gfem-effectivity}`field high` times the true error, at `order=2` and
  `order=3`, and the quantity estimator's between
  {gfem-effectivity}`quantity low` and {gfem-effectivity}`quantity high` at
  `order=2`. The quantity estimator's higher ratios came from fields with no
  sharp metal edge (a uniform line, an empty cavity), where its curl and
  mass terms are never allowed to cancel. At `order=3` the quantity
  estimate is not checked against its range; at `order=1` both estimators
  refuse.
- A lumped port spanning materials of different loss (a lossy substrate
  and air, say) is not covered by that range: there the field estimate
  measured {gfem-effectivity}`lumped port across a loss jump low` to
  {gfem-effectivity}`lumped port across a loss jump high` times the true
  error on coarse meshes.
- It measures the error against your mesh's own facets. Mesh refinement
  subdivides the facets you imported and never moves them onto a curved
  surface, so the faceting error is neither estimated nor reduced. On a
  circular coaxial line meshed with flat facets, both estimates were below
  the error against the exact circle. Mesh curved surfaces finer in your
  mesher.
- A wave port's own mode is solved on the port's 2D mesh, and that error
  is not estimated. It grows with a long de-embedding offset on a coarse
  port mesh. A wave port in a lossy medium also leaves the attenuation
  along the port out of its boundary condition, so even a matched lossy
  port reflects a little; the estimate does not see that reflection.
- An S entry whose first port is a wave port in a lossy cross-section with
  a hybrid mode (a lossy dielectric slab, say) needs one more solve, of
  that port's {term}`adjoint dual`. `refine_mesh` adds it to each of its
  own solves, with your solver options. `error_estimate` on a finished
  result factorizes the system once more, with the direct solver at the
  result's order, which on a large model may not fit in memory: estimate
  such a model through `refine_mesh` (`max_cycles=0` estimates without
  refining).
- It does not include the linear solver's error. `refine_mesh` requires
  `tolerance` to be at least 10 times `SolverOptions.tolerance`
  (`eigen_tolerance` for an eigenmode problem).
- `refine_mesh` raises {py:obj}`gfem.MeshRefinementUnsupportedError`, before
  any solve, on an adaptive sweep, a port mode problem, a periodic model, a
  Bloch wave vector and `SolverOptions(order=1)` (the `"fast"` preset); both
  estimators refuse order 1. A `QuantityEstimator` also raises it on a
  mode of a model with a second-order absorbing boundary, a complex
  `mu_r` or a permittivity law: refine those models with the
  `FieldEstimator`. On a driven or
  eigenmode model with loss inside an anisotropic material, both the
  `FieldEstimator` and a `QuantityEstimator` on a mode or an S entry raise
  it: refine a
  copy of the model with that material lossless, then solve the lossy model
  on the refined mesh.
- Mesh refinement runs on one process. Above one rank it raises
  {py:obj}`gfem.DistributedUnsupportedError`: refine on one process, then
  solve `ref.model` on ranks ({doc}`distributed-solve`).

## Next

{py:obj}`gfem.refine_mesh`, {py:obj}`gfem.error_estimate`,
{py:obj}`gfem.FieldEstimator`, {py:obj}`gfem.QuantityEstimator`,
{py:obj}`gfem.MeshRefinement`, {doc}`remesh-and-restart`,
{doc}`robust-optimization`, {doc}`frequency-sweep`.
