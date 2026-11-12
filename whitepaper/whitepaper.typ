#import "gfem.typ": *

#let front = json("data/front.json")
#let caps = json("data/caps.json")
#let P = front.params
#let R = front.results

// Number formatting: fixed decimals, and thousands separated by commas.
#let fixed(x, d) = {
  let s = str(calc.round(x, digits: d))
  let parts = s.split(".")
  let frac = if parts.len() > 1 { parts.at(1) } else { "" }
  if d == 0 { parts.at(0) } else { parts.at(0) + "." + frac + "0" * (d - frac.len()) }
}
#let grouped(n) = {
  let s = str(n)
  let out = ""
  for (i, ch) in s.clusters().enumerate() {
    if i > 0 and calc.rem(s.len() - i, 3) == 0 { out += "," }
    out += ch
  }
  out
}

#show: paper.with(
  title: [Electromagnetics you can differentiate.],
  short-title: [Differentiable finite-element electromagnetics],
  kicker: "Whitepaper",
  version: "0.1.0",
  date: "November 2026",
  lead: [Gradients of full-wave finite-element results with respect to geometry and materials, in Python. Driven, eigenmode and electrostatic problems, from microwave circuits to photonic waveguides, on NGSolve and JAX.],
  hero: {
    readout(
      ("Iteration", "final"),
      ("β", "32"),
      ([|S#sub[21]|#super[2]], fixed(R.t_final, 4)),
      ("Wavelength", str(P.wavelength_um) + " µm"),
      ("Mesh", grouped(P.dof) + " DOF"),
    )
    block(width: 100%, stroke: 0.5pt + c.hair, clip: true, image("data/bend3d-final.png", width: 100%))
    v(4pt)
    text(size: 7.6pt, fill: c.muted)[A 90° bend between two #P.core_width_um µm guides of #(int(P.core_thickness_um * 1000)) nm silicon in oxide, optimized in 3D at #P.wavelength_um µm by #P.iterations gradient steps. Grey: silicon in the #P.square_um µm design square, shaded by density. Red: the surface where |E| is 0.35 of its peak in the input guide.]
  },
  meta: (
    ("Version", "0.1.0"),
    ("Licence", "MIT"),
    ("Python", "≥ 3.12"),
    ("Docs", "bvial.info/gfem"),
  ),
)

= Summary

gfem is a finite-element solver for Maxwell's equations whose results are differentiable. `gfem.solve` is a JAX function of a model's materials and mesh: an objective built from S-parameters, eigenfrequencies, Q factors or a capacitance matrix has a gradient, from `jax.grad`, with respect to per-element permittivity and conductivity, the fill of a patterned metal sheet and the mesh node coordinates. A gradient costs at most one extra linear solve per frequency, and none when the objective reads only the excited ports' S, whatever the number of design variables.

Three properties set the terms of this paper.

- *The gradient is verified, not assumed.* Every supported pair of result and input is checked against finite differences in the test suite: #caps.counts.cells cells, on both linear solvers. A gradient gfem cannot compute raises an error instead of returning zero.
- *It is a full-wave frequency-domain solver.* Driven problems with lumped and wave ports, eigenmode problems with their Q, electrostatic capacitance, periodic and Bloch cells, and anisotropic materials, on high-order tetrahedral meshes.
- *It is plain Python.* A design pipeline (filters, projections, symmetry, robustness) is ordinary JAX code around `gfem.solve`, and any optimizer that takes a gradient drives it.

This paper describes the formulation (@formulation), the gradient and how it is checked (@gradients), a three-dimensional photonic design run end to end (@case), how far one driven solve scales on a cluster (@scale), and the limits of version 0.1 (@limits). Every number is an observation from a named run; none is a promise for another model or machine.

= Why gradients of full-wave results <motivation>

Inverse design asks which geometry or material distribution best achieves a response: a filter's passband, a resonator's frequency and loss, a waveguide bend's transmission, a qubit's participation ratios. With thousands to millions of design variables, only gradient-based optimization is practical, and the gradient must come from the same full-wave model that predicts the response.

Finite differences need one or two extra solves per variable. The adjoint method needs at most one extra linear solve per objective and frequency, whatever the number of variables. But hand-writing an adjoint for each objective, port type and boundary condition is error-prone, and an adjoint that is silently wrong is worse than none: the optimizer follows it.

gfem does three things about that. It derives the adjoint once, inside `gfem.solve`, and exposes it through JAX's automatic differentiation, so any objective written in JAX from the solve's results is differentiated without user-written adjoints. It checks each supported gradient against finite differences in its test suite. And it raises a named error for each gradient it does not support.

The finite-element method suits the devices this targets. Tetrahedral meshes follow curved and thin features (superconducting films, metal traces a few micrometres wide on millimetre substrates, sub-micrometre silicon guides) without staircasing. A frequency-domain solve gives S-parameters, resonances and Q directly, at the frequencies asked for.

= Formulation <formulation>

== The forward problem

gfem solves the time-harmonic Maxwell equations for the electric field $bold(E)$ under the $e^(+i omega t)$ convention:

$ nabla times (mu_r^(-1) nabla times bold(E)) - k_0^2 tilde(epsilon)_r bold(E) = -i omega mu_0 bold(J), quad tilde(epsilon)_r = epsilon_r (1 - i tan delta) - (i sigma) / (epsilon_0 omega), $ <maxwell>

on a tetrahedral mesh, with $k_0 = omega slash c$. Material tensors may be anisotropic (real symmetric, sharing principal axes); gyrotropic, non-reciprocal materials are not supported. $bold(E)$ is discretized with Nédélec #box[$H("curl")$] elements of order $p$ (order 2 in the examples below). Boundaries are perfect electric or magnetic walls, first- or second-order absorbing boundaries, finite-conductivity surfaces, R-L-C impedance sheets, and periodic pairs with an optional Bloch wave vector. Ports are lumped (a resistor across one or several faces) or wave ports fed by the 2D mode of their cross-section.

Assembly gives one sparse complex linear system per frequency,

$ A(omega, bold(p)) thin bold(x) = bold(b), quad A = K + i omega C - omega^2 M, $ <system>

where $bold(p)$ collects the design inputs and $bold(b)$ the port excitations. S-parameters are power waves normalized to each port's reference impedance; with the excited port's incident wave at $a = 1 thin sqrt("W")$, a driven field carries 1 W of incident power.

== Problems

#figure(
  table(
    columns: (auto, 1fr),
    [Problem], [What it returns],
    [`gfem.Driven`], [S-parameters of the ports over a frequency grid, uniform or by adaptive sweep; optional fields.],
    [`gfem.Eigenmode`], [Resonant modes nearest a target frequency, with Q; lumped ports load the modes.],
    [`gfem.Electrostatic`], [The Maxwell capacitance matrix between named terminals.],
    [`gfem.PortMode`], [The 2D mode of a wave port's cross-section and its propagation constant.],
  ),
  caption: [Problem types in gfem 0.1. Driven and eigenmode problems accept a Bloch wave vector on a periodic model.],
)

== Linear solvers

The default for driven problems is a domain-decomposition preconditioner (BDDC) with GMRES, for large models and for models whose factorization does not fit in memory. A sparse direct factorization is the choice for eigenmode problems, for repeated solves of an optimization loop on a model that fits one machine's memory, and when many ports share one frequency: it factorizes once per frequency and reuses the factors for every port and for the adjoint. Optional backends add Intel MKL PARDISO, NVIDIA cuDSS on one GPU, and PETSc and SLEPc for MUMPS and user-chosen Krylov methods. A driven uniform sweep and its gradients run split across the ranks of an MPI job.

= The gradient <gradients>

== One adjoint solve per frequency

Let $J(bold(x), bold(p))$ be a real objective of the complex solution. Differentiating @system gives $d bold(x) slash d p = -A^(-1) (partial A slash partial p) bold(x)$, so

$ (d J) / (d p) = (partial J) / (partial p) - 2 op("Re")[ bold(lambda)^T (partial A) / (partial p) bold(x) ], quad A^T bold(lambda) = ((partial J) / (partial bold(x)))^T, $ <adjoint>

with $partial J slash partial bold(x)$ the Wirtinger derivative. One adjoint solve for $bold(lambda)$ serves every design variable. For the reciprocal media gfem supports, $A$ is complex symmetric, $A^T = A$, and the adjoint of an S-parameter is the forward solution with that port excited: when the objective reads only excited ports, the gradient needs no extra solve, and the backward pass is one element-wise integral per frequency. Reading a port left out of the excitation, or a kept field, costs one adjoint solve per frequency.

An eigenfrequency gradient needs no solve either: for a mode $bold(x)$ of the operator $T(k)$, $d k slash d p = -bold(x)^T (partial T slash partial p) bold(x) slash bold(x)^T T'(k) bold(x)$, where $T'(k)$ includes the frequency dependence of conductor surface impedance and absorbing boundaries. An adaptive sweep differentiates through its reduced model with one small reduced adjoint. Mesh node coordinates enter through the element Jacobians, and `gfem.shape.boundary_morph` maps a few normal displacements of a boundary to all node coordinates by a fixed linear map built once on the imported mesh.

== Inside JAX

`gfem.solve` is registered with JAX as a function with its own vector–Jacobian product. Everything around it stays ordinary JAX: the map from design pixels to elements, a Helmholtz filter for a length scale, a projection towards two materials, a symmetry, eroded and dilated variants for robustness, and the objective itself. `jax.value_and_grad` composes them:

#code(hl: (14,))[
```python
import jax
import gfem

model = gfem.Model(mesh, materials=..., boundaries=..., ports=...)
at_5ghz = gfem.Driven(frequency=5e9, excite="p1")

def transmission(ceramic):
    """|S21|^2 at 5 GHz with this ceramic in the middle section."""
    result = gfem.solve(model.with_material("design", ceramic), at_5ghz)
    s21 = result.s["p2", "p1"]
    return abs(s21[0]) ** 2

ceramic = gfem.MaterialField.uniform("design", gfem.Material(eps_r=10.0))
power, grad = jax.value_and_grad(transmission)(ceramic)
print(power, grad.eps_r)  # 0.8123998572975817 -0.30017734440224814
```
]

gfem provides differentiable solves and the design helpers (`gfem.design`: layer grids, filters, projections, robust variants); it does not provide an optimizer. The examples use optax's Adam; MMA and nlopt work the same way.

== What differentiates

@matrix is built from the test suite when the documentation builds; it cannot drift from the tests.

#let status-of(cell) = {
  let words = if cell.status == "verified" { "FD-verified" } else if cell.status == "zero" { "exact zero" } else { "raises" }
  mark(cell.status, word: words)
  if cell.note != "" { linebreak(); data(cell.note, fill: c.muted, size: 6.4pt) }
  if cell.status == "raises" and cell.error != "" and cell.error != "NonDifferentiableError" {
    linebreak(); data(cell.error, fill: c.muted, size: 6.4pt)
  }
}

#wide[
#figure(
  table(
    columns: (1.5fr, 1fr, 1fr, 1fr, 1fr),
    inset: (x: 5pt, y: 6pt),
    [Problem], ..caps.columns.map(col => col.label),
    ..caps.rows.map(row => (
      [#text(weight: 600, fill: c.ink, row.label) \ #text(size: 7pt, fill: c.muted, row.detail)],
      ..caps.columns.map(col => status-of(row.cells.find(cell => cell.column == col.key))),
    )).flatten(),
  ),
  caption: [What `jax.grad` differentiates, by problem and input. #caps.provenance. An exact zero means the result does not depend on that input; "raises" means asking for that gradient raises `gfem.NonDifferentiableError` (or the error named) instead of returning a wrong value.],
) <matrix>
]

== How each mark is checked

Each FD-verified mark is a set of cells, one per problem, input, linear solver and sweep. A cell draws three random directions $bold(v)$ in the input space and compares $nabla J dot bold(v)$ with a fourth-order central difference of the same objective,

$ nabla J dot bold(v) approx (-J(bold(p) + 2h bold(v)) + 8 J(bold(p) + h bold(v)) - 8 J(bold(p) - h bold(v)) + J(bold(p) - 2h bold(v))) / (12 h), $

at tight solver tolerances. It then checks, at the default tolerances, that the Taylor remainder $|J(bold(p) + h bold(v)) - J(bold(p)) - h nabla J dot bold(v)|$ shrinks at second order as $h$ shrinks. Exact zeros are checked to be zero, and each "raises" mark is a call that must raise the named error. #caps.raises.len() further gradients off the matrix raise as well, listed in the documentation.

Beyond gradients, forward results are checked against closed forms, identities the physics must obey and independent methods: the analytic propagation of a parallel-plate line's TEM and first higher mode; a mode-matching solution for TE#sub[10] to TE#sub[20] conversion in a slab-loaded guide; transfer-matrix band edges of a Bragg line; the light-line doublet of an empty periodic cube; Bloch reciprocity $S(bold(k))^T = S(-bold(k))$; and the invariance of S, eigenfrequency and capacitance under a rigid rotation of a model with anisotropic materials.

= Case study: a 3D silicon waveguide bend <case>

#aside[Run on a #P.cores\-core AMD EPYC 7V73X cloud instance, gfem 0.1.0.dev0, commit `006314c`. The tutorial reads the run record; it is not rerun when the docs build.]

*Question.* Two silicon waveguides #P.core_width_um µm wide and #(int(P.core_thickness_um * 1000)) nm thick, in oxide, meet at a right angle. In the #P.square_um µm × #P.square_um µm square where they meet, each point may be silicon or oxide. Which pattern carries the most #P.wavelength_um µm light from the fundamental mode of one guide into the fundamental mode of the other?

*Setup.* The design is a #P.pixels.at(0) × #P.pixels.at(1) image of the square, #fixed(P.pitch_um, 2) µm per pixel, extruded through the core layer. Silicon has $n = #fixed(P.n_si, 4)$ and oxide $n = #fixed(P.n_sio2, 3)$. Wave ports feed and terminate both guides. The mesh holds #grouped(P.tets) tetrahedra, #grouped(P.dof) unknowns at order #P.order.

#block(above: 14pt, below: 14pt, breakable: false, {
  set align(center)
  let arrow = text(fill: c.muted, size: 8pt)[ → ]
  (
    key[ρ], key[mirror about $y = x$], key[filter], key[project], key[permittivity],
    key(fill: c.panel, stroke: c.ink)[gfem.solve], key[$1 - |S_(21)|^2$],
  ).join(arrow)
  v(1pt)
  block(width: 92%, height: 7pt, stroke: (left: 0.8pt + c.accent, right: 0.8pt + c.accent, bottom: 0.8pt + c.accent))
  v(1pt)
  text(font: mono, size: 7pt, fill: c.accent)[← jax.value_and_grad back to ρ]
})

The design is mirrored about the diagonal, smoothed by a Helmholtz filter of radius #P.filter_radius_um µm, and projected towards silicon or oxide with a sharpness β that doubles every #P.steps_per_beta iterations, from #int(P.beta0) to 32. #P.iterations Adam steps start from ρ = #P.start everywhere. Each step is one driven solve and its gradient; the median step took #fixed(P.median_s, 0) s on #P.cores cores.

#wide[
#figure(
  {
    readout(("Optimizer", str(P.iterations) + " Adam steps"), ("Objective", [1 − |S#sub[21]|#super[2]]), ("One step", fixed(P.median_s, 0) + " s"), ("Final", [|S#sub[21]|#super[2]] + " " + fixed(R.t_final, 4)))
    v(4pt)
    history(front.history.t, front.history.beta)
    v(8pt)
    grid(
      columns: (1fr,) * 4,
      column-gutter: 10pt,
      ..front.snapshots.map(s => {
        block(width: 100%, {
          line(length: 100%, stroke: 0.6pt + c.ink)
          v(-3pt)
          [#legend("Iteration") #h(2pt) #data(str(s.iteration), fill: c.ink) #h(1fr) #data([|S#sub[21]|#super[2]] + " " + fixed(s.t, 4), fill: c.ink)]
          v(1pt)
          snapshot("data/" + s.file, square: 2 * front.design_um / (front.extent_um.at(1) - front.extent_um.at(0)))
        })
      }),
    )
  },
  caption: [Top: |S#sub[21]|#super[2] per iteration; ticks mark where β doubles. Bottom: the design over the whole #(front.extent_um.at(1) * 2) µm box at four iterations, input guide from the left and output guide downwards; dashed, the design square. Ink density is the silicon density.],
) <history>
]

*Result.* The thresholded design carries #fixed(R.t_final, 4) of the power into the output guide's fundamental mode, a loss of #fixed(R.db_final, 3) dB. The media are lossless, so the run checks reciprocity, $|S_(12) - S_(21)| = #fixed(R.reciprocity * 1e14, 1) times 10^(-14)$, and power balance, $|S_(11)|^2 + |S_(21)|^2 = #fixed(R.balance, 3) <= 1$; the rest leaves as the output guide's second mode or as radiation. The design was optimized at #P.wavelength_um µm only: over #R.band_um.at(0) to #R.band_um.at(1) µm, |S#sub[21]|#super[2] falls to #fixed(R.band_min.t, 3) at #R.band_min.um µm.

The same pipeline runs, in the tutorials, on a band-stop parallel-plate line (topology of a dielectric layer, robust to over- and under-etching), on a matched 3 dB attenuator (topology of a resistive film on a zero-thickness sheet), on a coplanar line matched to 50 Ω by moving its conductor walls (shape gradient through `boundary_morph`), and on a 2D slab model of this bend.

= Scale <scale>

How large a driven model one `gfem.solve` has finished across MPI ranks, and in what time, on MareNostrum 5's general-purpose partition: two Intel Xeon Platinum 8480+ per node, 112 cores, run as 28 ranks of 4 threads per node. The meshes are one family of coplanar lines at order 2, one frequency of 50 GHz, both ports excited, BDDC + GMRES at a relative tolerance of $10^(-8)$, and the direct solver as PETSc's MUMPS.

#figure(
  table(
    columns: (auto, 1fr, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    [Nodes], [DOF], [BDDC, s], [Iterations], [Direct, s], [Direct, largest node],
    [1], [3,061,188], [36], [67 / 66], [21], [71 GiB],
    [2], [6,238,560], [40], [66 / 67], [29], [70 GiB],
    [4], [14,217,194], [76], [61 / 62], [54], [87 GiB],
    [8], [23,148,138], [128], [71 / 69], [161], [237 GiB],
    [16], [47,277,960], [342], [68 / 68], [832], [401 GiB],
    [32], [108,570,046], [540], [63 / 64], [out of memory], [–],
  ),
  caption: [Weak scaling, about 3 M DOF per node. Times are the median `gfem.solve` of three runs, the slowest rank's whole call; iterations are for ports 1 / 2. Mesh read and partition (about 200 s at 108 M DOF) come before the solve.],
) <weak>

Observations from these runs:

- *Largest model.* BDDC + GMRES solved 108,570,046 DOF on 4 to 32 nodes; its fastest solve was 423 s, on 8 nodes. The iteration count stayed between 61 and 71 across a 35-fold change in model size.
- *Which solver.* Up to 14 M DOF the direct solver was faster. From 23 M DOF, BDDC + GMRES was faster at every node count tried, and on one node at 23,148,138 DOF it needed 175 GiB on the largest node against the direct solver's 716 GiB.
- *How many nodes.* Past a point, more nodes made BDDC + GMRES slower: its coarse problem's apply is 42 % to 53 % of the solve at 108 M DOF, and grows from 8 to 32 nodes.
- *Agreement.* Every rank held the same S, bit for bit. On a 10,248,856 DOF line, each rank count from 1 to 32 nodes agreed with its one-process run to rounding error, with the same iteration count; at 47,277,960 DOF the two solvers agreed to well within the GMRES tolerance.

These are one machine, one gfem version (the development version of 5 November 2026, after 0.1.0), one family of meshes and one frequency. The run records, logs, job scripts and meshes are archived with the commit they name.

= Limits of version 0.1 <limits>

gfem states what it does not do.

- *Low frequency.* Below $k_0 h_"min" = 10^(-6)$ (about 48 MHz with 1 µm elements), a driven solve's fields, gradients and the capacitive part of S lose digits, and gfem warns. Capacitance at low frequency comes from the electrostatic problem.
- *Materials.* Non-reciprocal (gyrotropic) tensors are not supported, nor loss whose principal axes differ from the permittivity's. A wave port's cross-section may hold anisotropic materials only with a principal axis along the port normal.
- *Periodic models.* There are no Floquet ports. A Bloch wave vector needs the direct solver for eigenmode problems; PETSc, cuDSS and the adaptive sweep refuse it.
- *Ports.* Wave-port S is the conjugated modal projection, exact for a lossless port line. A propagating mode that no port requests is partly reflected at the port, and gfem warns when one is skipped below a requested mode.
- *Meshing.* gfem reads gmsh `.msh` files; it does not mesh. A shape optimization that moves nodes too far must remesh and restart.
- *Distributed solves.* Only the driven uniform sweep and its gradients run across MPI ranks; everything else raises a named error there.
- *API.* 0.1.0 is the first release; the API may change before 1.0.

= Availability

gfem is MIT-licensed. It depends on NGSolve (LGPL-2.1), whose wheels contain GPL-2.0+ SuiteSparse code, and on JAX. It runs on Linux x86_64, macOS arm64 and Windows x86_64 with Python 3.12 or later. It is not on PyPI yet; pip installs it from the repository:

```bash
pip install "gfem @ git+https://github.com/benvial/gfem"
```

Extras add MKL PARDISO (`mkl`), the cuDSS GPU solver (`cuda12`, `cuda13`), xarray and PyVista outputs (`xarray`, `viz`) and distributed solves (`mpi`). Documentation, tutorials and the full gradient matrix: #link("https://bvial.info/gfem/")[bvial.info/gfem]. Source: #link("https://github.com/benvial/gfem")[github.com/benvial/gfem].

#v(1fr)
#note(kind: "note", title: "Sources")[
Every number in this paper comes from a data file or test in the gfem repository, read when the paper compiles or copied from the documentation: the 3D bend from `docs/_data/front-page.json` (built by `tools/front_page_data.py` from the tutorial's run record), the gradient matrix from the verification tests through `docs/_ext/gfem_capabilities.py`, and the scaling runs from the tables of the documentation's performance page, generated by `tools/scale_tables.py` from the run records of commit `9a57680`. `just whitepaper` refreshes the data files and compiles this paper.
]
