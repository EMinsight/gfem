# Solve a periodic cell

Get the band diagram of a structure that repeats along one, two or three
directions, or the S-parameters of one phased unit cell, by solving a single
cell. You mesh one cell and pair its opposite faces with {py:obj}`gfem.Periodic`.
With no wave vector the fields repeat unchanged from cell to cell
({term}`plain periodicity`). With `wave_vector=k` they pick up a
{term}`Bloch phase` from cell to cell. To light the cell with a plane wave and
read its diffraction orders, see {doc}`floquet-ports`.

## Band frequencies of a layered cell

The cell below is a line of $\varepsilon = 2$ and $\varepsilon = 9$ layers that repeats every 10 mm
along x. Its mesh needs two outer faces that match node for node under a
translation; gmsh's `setPeriodic` makes such a pair.

```python
import numpy as np

import gfem

mesh = gfem.Mesh.from_gmsh("bragg_cell.msh", length_unit="mm")
cell = gfem.Model(
    mesh,
    materials={"low": gfem.Material(eps_r=2.0), "high": gfem.Material(eps_r=9.0)},
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.PEC("z_min"),
        gfem.PEC("z_max"),
    ],
)
(a,) = cell.periodic_translations  # m, from x_min to x_max: (0.01, 0.0, 0.0)
a = np.asarray(a)
options = gfem.SolverOptions(order=3)

bands = []
for phase in np.linspace(0.1 * np.pi, np.pi, 10):  # phase per cell, rad
    k = tuple(phase * a / (a @ a))  # rad/m
    # aim below every band: the light line of the densest layer, n = 3
    target = 0.9 * 299_792_458 * np.linalg.norm(k) / (2 * np.pi * 3)
    problem = gfem.Eigenmode(target=target, n_modes=2, wave_vector=k)
    modes = gfem.solve(cell, problem, options=options)
    bands.append(modes.frequency)  # Hz, the two lowest bands at this k
```

- `bands` holds the two lowest band frequencies, in Hz, at 10 wave vectors
  from $0.1\pi/a$ to the zone edge $\pi/a$. `modes.frequency` is $\operatorname{Re} \omega / 2\pi$ and
  `modes.q` is the Q. For a lossless cell `modes.omega` is exactly real.
- The wave vector is Cartesian, in rad/m. For a phase $\varphi$ per cell along a
  translation $\mathbf{a}$, $\mathbf{k} = \varphi \mathbf{a} / |\mathbf{a}|^2$, as in the loop. Under $e^{+i\omega t}$ the fields
  obey `E(r + a) = exp(-i k·a) E(r)` ({py:obj}`gfem.conventions`, Bloch
  periodicity).
- Give $\mathbf{k}$ only along the translations. Any other component raises `ValueError`
  at the solve, and the message gives the projection to use instead.
- Only the phase per cell enters the solve, so a $\mathbf{k}$ outside the first Brillouin
  zone gives the folded band. Keep your $\mathbf{k}$ path inside the first zone.
- At $\mathbf{k} = 0$ the cell also has a field at zero frequency: aim the target above
  it.
- Each mode field covers the whole cell, so `Field.at` reads it at any point.
  Under a Bloch wave vector, gfem sets a mode's overall phase from its largest
  coefficient, not from the physics: compare fields through ratios, not raw
  phases.

## Pair the faces

`gfem.Periodic` takes the names of the two faces. gfem measures the
translation between them from the mesh, so you give none:
`Model.periodic_translations` holds one per pair, in metres, in `boundaries`
order. The order of the two faces in a pair does not change the field.

- A rotated pair, an element touching both faces, more than three pairs, or
  two parallel translations raise `ValueError` when you build the model. Remesh
  with matching faces under a translation, at most one pair per direction.
- A paired face takes no other boundary condition and no port; the default PMC
  skips it. A wave port that touches a paired face raises
  `NotImplementedError`: place the port's face away from the pair; for
  plane-wave incidence on an open face of the cell, use
  {py:obj}`gfem.FloquetPort`.

## Drive a cell at a Bloch wave vector

A two-port line cell, $\varepsilon = 4$ with $\tan\delta = 0.02$, 10 mm long, with lumped ports
on two interior sheets:

```python
mesh = gfem.Mesh.from_gmsh("line_cell.msh", length_unit="mm")
line = gfem.Model(
    mesh,
    materials={"cell": gfem.Material(eps_r=4.0, tan_delta=0.02)},
    boundaries=[
        gfem.Periodic(("x_min", "x_max")),
        gfem.PEC("z_min"),
        gfem.PEC("z_max"),
    ],
    ports=[
        gfem.LumpedPort("p1", "sheet1", "+z", impedance=50.0),
        gfem.LumpedPort("p2", "sheet2", "+z", impedance=50.0),
    ],
)
k = (0.3 * np.pi / 0.01, 0.0, 0.0)  # rad/m: a phase of 0.3π per 10 mm cell
s = gfem.solve(line, gfem.Driven(5e9, wave_vector=k)).s.values[0]
```

- S is referenced to the field E, not to a periodic envelope;
  {py:obj}`gfem.conventions` gives the factor between the two. S is not
  symmetric under a wave vector: $S(\mathbf{k})^\mathsf{T} = S(-\mathbf{k})$ ({term}`Bloch reciprocity`).
- By default the same $\mathbf{k}$ holds at every frequency;
  `wave_vector_frequency=f_ref` scales it with frequency, which keeps an
  incidence angle fixed ({doc}`floquet-ports`).
- Only the uniform sweep takes a wave vector. `sweep="adaptive"` with a wave
  vector raises `NotImplementedError`: use `sweep="uniform"`. Plain
  periodicity takes both sweeps.

## Differentiate a band

A periodic model differentiates like any other. Here the lowest band at the
zone edge, with respect to the permittivity of each element of the $\varepsilon = 9$
layer:

```python
import jax
import jax.numpy as jnp

edge = (np.pi / 0.01, 0.0, 0.0)  # rad/m, k = π/a
problem = gfem.Eigenmode(target=4e9, n_modes=2, wave_vector=edge)


def lowest_band(eps):
    m = cell.with_material("high", gfem.MaterialField(eps_r=eps))
    return gfem.solve(m, problem, options=options).frequency[0]


n = len(cell.mesh.elements("high"))
grad = jax.grad(lowest_band)(jnp.full(n, 9.0))  # Hz per unit eps_r
```

- {py:obj}`gfem.shape.boundary_morph` moves a node and its partner on the other
  face of a pair together. A paired face cannot be the moving surface
  (`ValueError`): move an interior surface instead.
- `wave_vector` is not differentiable: a value inside `jax.grad` raises
  `gfem.NonDifferentiableError`. For a group velocity, difference two band
  solves at nearby k.
- A pixel design on a cell takes {py:obj}`gfem.design.LayerGrid` with
  `periodic=(True, False)`, one bool per grid axis (x, y): along a periodic
  axis the pitch must divide the period, the filter smooths across the seam
  and the element map wraps around it.
- {doc}`../capabilities` lists which periodic gradients each solver supports.

## Choose a solver

| | Plain periodic | Bloch wave vector |
|---|---|---|
| `direct` | driven and eigenmode | driven and eigenmode (MKL PARDISO with the `mkl` extra, UMFPACK otherwise) |
| `bddc_gmres` | driven and eigenmode | driven only; eigenmode raises `ValueError`: use `direct` |
| `petsc` | driven | raises `ValueError`: use `direct` |
| `device="cuda"` | untested: compare with `direct` once on your cell | raises `NotImplementedError`: use `direct` on the CPU |
| several MPI ranks | one pair: driven on `direct` and `bddc_gmres`, eigenmode on `direct`; two or three pairs raise | one pair: driven on `direct` and `bddc_gmres`; eigenmode raises |
| eigen solver | ARPACK or SLEPc | ARPACK or SLEPc |
| Floquet ports | `direct` and `bddc_gmres`, driven uniform sweep only; every other entry raises | the same |

## Check your results

Four checks you can run on your own cell:

- **Transpose.** Solve the driven cell at $\mathbf{k}$ and at $-\mathbf{k}$. With lumped and wave
  ports, `S(k).T` should equal `S(−k)` to the solver tolerance. With Floquet
  ports, the orders relabel and take a sign:
  $S_{p\leftarrow q}(\mathbf{k}) = \sigma_p \sigma_q S_{-q\leftarrow -p}(-\mathbf{k})$, see {doc}`floquet-ports`.
- **Zero wave vector.** `wave_vector=(0, 0, 0)` should give the same result as
  no wave vector: a cheap check of the pairing and the ports. It is a check
  only: the zero wave vector still takes the general Bloch solve, so run plain
  periodicity with `wave_vector=None`, which is cheaper.
- **Folding.** With the same target, $\mathbf{k}$ and $\mathbf{k} + 2\pi \mathbf{a} / |\mathbf{a}|^2$ should give the
  same band frequencies.
- **Lossless cell.** With no loss in the cell, `modes.omega` should be real,
  and the driven cell's S with resistive lumped ports should give `SᴴS ≈ I`:
  form the product and look at its largest departure from the identity.

How gfem itself is checked on periodic cells: {doc}`../verification`.

## Limits

- A {py:obj}`gfem.Absorbing` face reflects a wave that meets it obliquely.
  With a wave vector that has a component $\mathbf{k}_t$ along the face, the cell
  radiates at the angle $\theta$ from the face normal given by $\sin\theta = |\mathbf{k}_t| / (k_0 n)$, with $n$ the
  refractive index behind the face. The first-order boundary then reflects
  $(1 - \cos\theta) / (1 + \cos\theta)$ in amplitude: 1.7 % at 15°, 7 % at 30°, 17 % at
  45°. `order=2` lowers this about tenfold up to 30°, but only for the
  polarization with E parallel to the face and normal to $\mathbf{k}_t$; for the other
  polarization it changes nothing. A driven solve warns with
  {py:obj}`gfem.ObliqueAbsorberWarning` when some propagating wave meets the
  face with a reflection of at least 1 %; a {py:obj}`gfem.FloquetPort` on
  that face absorbs every order at any angle.
- A Bloch eigenmode solve with `gfem.Absorbing` faces meets the same oblique
  reflection, and gives no warning: the Q of a mode that radiates at the
  angle $\theta$ includes it. Floquet ports do not take eigenmode solves; for the
  resonances and Q of a radiating cell, sweep a driven problem with Floquet
  ports and read them off S.
- Translations only, one to three pairs. For a rotational symmetry, mesh the
  whole structure.
- Across MPI ranks, one pair only, and no Bloch eigenmode solve
  ({doc}`distributed-solve`).
- `gfem.Electrostatic` on a periodic model raises `NotImplementedError`. Mesh
  enough cells and solve without pairs.
- `gfem.interface_participation` on a paired face raises `ValueError`, since
  that face is inside the repeated structure. `gfem.power_flow` works on
  planes that cross or touch a pair.
- A gradient with respect to `mesh.nodes` is given per node, not summed over
  a node and its partner on the other surface of a pair. A move that breaks a
  pair raises `ValueError`; move nodes through
  {py:obj}`gfem.shape.boundary_morph`, which keeps the pairs.
- No k-path argument: loop over k, as in the first example.
- A gradient through an eigenmode's field raises, as on every model. Build the
  objective from `frequency` and `q`.

## Next

{doc}`floquet-ports`, {py:obj}`gfem.Periodic`, {py:obj}`gfem.conventions`,
{doc}`eigenmode-gradient`,
{doc}`shape-gradient`, {doc}`measurements`, {doc}`../capabilities`.
