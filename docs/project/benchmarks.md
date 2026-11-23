(benchmarks)=
# Benchmarks

Measured in November 2026 with gfem `771da66` and Palace 0.19.0; gfem has changed
since. The numbers are observations of one machine.

What was compared: gfem (main `771da66`, run from the benchmark harness commit
`e7d58a9`) and [Palace](https://github.com/awslabs/palace) 0.19.0, on MareNostrum 5
general-purpose nodes, one node per run. Five problems, from 10,620 to 10,248,856
DOF, each solved by both codes on the same mesh with the same number of unknowns,
at 8, 16, 32, 56 and 112 cores. Every number on this page
comes from one campaign report, `20261120-37c723a`. These are observations from one
set of runs on one machine, not limits or promises for your model.
For gfem alone across nodes, up to 108,570,046 DOF, see {doc}`scaling`.

## What we observed

"Ratio" below is Palace 0.19.0 over gfem `771da66`, as the report prints it: work
time, or the sum of per-rank peak memory. A ratio above 1 means the Palace run took longer or used more
memory. Each solver is at its fastest layout of ranks and threads for that core
count, unless a line says otherwise.

- **Direct solvers.** On the three driven problems, `palace:mumps` took longer than
  `gfem:direct` at every core count. The ratio fell as cores were added: 14 at 8
  cores to 4.64 at 112 on microstrip, 5.38 to 2.66 on r10M_void, 20.8 to 5.93 on
  microstrip_p3 ([table](bench-direct)). gfem's own MUMPS kind,
  `gfem:petsc-mumps`, also took less time than `palace:mumps` on the driven
  problems (ratios 1.95 to 11.1). On the lcavity eigenproblem it did not: there
  `palace:mumps` took less time than `gfem:petsc-mumps` from 16 cores on, down to
  a ratio of 0.19 at 112 cores ([table](bench-lcavity)).
- **Iterative solvers, at equal achieved error.** At the same solver error,
  1.00e-08, `palace:pmg` took less time than `gfem:bddc` from 16 cores on, on
  microstrip and microstrip_p3 (ratio 0.268 and 0.259 at 112 cores), and at 112
  cores on r10M_void (0.685). At 8 cores it took longer on all three (1.63, 1.13,
  2.71) ([table](bench-iterative)). On r10M_void at 56 cores the answer depends on
  the error level: 1.03 at 1.00e-08, 0.954 at 1.42e-09. The equal-error ladder ran
  gfem at one process of $c$ threads, which is not its fastest layout on r10M_void
  from 32 cores on: at its best layout, `gfem:bddc` took less time than
  `palace:pmg` at every core count at the nominal tolerance $10^{-8}$ (ratio 1.38
  to 2.22, [table](bench-r10m)), a comparison at equal tolerance, not at equal
  achieved error.
- **Memory.** `palace:mumps` used more memory than `gfem:direct` at every core count
  on every problem except bragg at 32 cores (0.97); on r10M_void at 112 cores, 298
  against 103 GiB ([table](bench-memory)). Between the iterative solvers it
  depends on the problem: `palace:pmg` used 1.56 to 2.45 times the memory of
  `gfem:bddc` on r10M_void, and about half of it on microstrip_p3 at 8 and 16 cores
  (0.524 and 0.531).
- **Adding cores.** From 8 to 112 cores, `palace:pmg` became 5.71, 5.65, 4.29 and
  9 times faster on lcavity, microstrip, r10M_void and microstrip_p3;
  `gfem:direct` 1.33, 1.79, 1.53 and 2.7 times ([table](bench-scaling)).
  gfem's default layout (one process, $c$ threads) stayed nearly flat on the large
  problems; its iterative and MUMPS kinds scaled through several ranks of 4
  threads: on r10M_void at 112 cores, `gfem:bddc` took 60.5 s at 28 ranks of 4
  threads against 154 s at its default.
- **Accuracy.** On the driven problems both codes solve the same discrete problem:
  Palace's direct solve differs from gfem's by 8.86e-11 on microstrip and
  microstrip_p3, and by 1.06e-12 on r10M_void.
  So every run of an order sits at the same distance from the converged reference,
  and only the solver error separates the iterative runs ([table](bench-cases)). On
  bragg, where the reference is closed form, `gfem:direct` reached 2.92e-11 and
  `palace:mumps` 4.78e-09.

(bench-cases)=
## Cases

| case | problem | DOF (order) | reference | nodes |
|---|---|---|---|---|
| lcavity | eigenmode, PEC L-shaped cavity, 1 mode above 100 GHz | 246,416 (2) | closed form `lcavity-20261117-058b2f5`, bound 1.00e-12 | standard |
| microstrip | driven, 2 lumped 50 $\Omega$ ports, first-order absorbing boundary, 5 frequencies from 1 to 100 GHz, both ports excited | 315,870 (2) | Richardson over uniform splits `microstrip-20261120-7a078ba`, bound 1.07e-02 | standard |
| bragg | eigenmode, Bloch-periodic cell, $k = \pi/a$ along $x$, 2 modes | 10,620 (3) | closed form `bragg-20261118-a0ff134`, bound 1.00e-12 | standard |
| r10M_void | driven, coplanar line, 50 GHz, port 1 excited | 10,248,856 (2) | none (no converged reference fits the budget) | high-memory |
| microstrip_p3 | microstrip's problem and mesh at order 3 | 907,791 (3) | as microstrip | high-memory |

Errors are $\max |S_{ij} - S_{\mathrm{ref},ij}|$ over entries and frequencies for the
driven problems, and the largest relative eigenfrequency error for the eigenproblems.
An error within ten times the reference bound is printed as value ± bound,
"unresolved".

| case | error of every run, both codes | note |
|---|---|---|
| lcavity | 3.89e-04 | all five solver kinds at the order-2 floor |
| microstrip | 2.90e-02 ± 1.07e-02 (unresolved) | all runs at the order-2 floor |
| microstrip_p3 | 1.44e-02 ± 1.07e-02 (unresolved) | all runs at the order-3 floor |
| bragg | `gfem:direct` 2.92e-11; `palace:mumps`, `palace:pmg` 4.78e-09 | both resolved |
| r10M_void | no reference | solver error and difference from gfem's direct solve only |

## Method

The suite's `benchmarks/README.md` at commit `e7d58a9`, section
[Method and fairness rules](https://github.com/benvial/gfem/blob/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks/README.md#method-and-fairness-rules),
holds the rules. In short:

- **Same problem.** One case file writes the inputs of both codes (gfem `771da66`,
  Palace 0.19.0); the mesh is pinned by
  its hash and both codes must report the same DOF count, or the run is refused.
- **Each code at its own defaults**, except a short list of pinned settings (solver
  kind, tolerances, no field output, Palace's mesh refinement off and its error
  estimator lightened). Structural differences are recorded and never removed:
  Palace sets up and solves once per excited port, gfem solves all excited ports on
  one factorization; Palace's error estimator always runs; NGSolve builds the BDDC
  preconditioner inside assembly.
- **Five solver kinds.** `gfem:direct` (PARDISO, one process), `gfem:petsc-mumps`
  and `palace:mumps` (MUMPS), `gfem:bddc` (GMRES with BDDC) and `palace:pmg`
  (GMRES with p-multigrid, Palace's default iterative solver).
- **Time** is work time: wall time minus process startup. **Memory** is the sum of
  each rank's peak resident memory, an upper bound since shared pages count once
  per rank.
- **Repeats.** Three per cell; the page prints the median with the min-max range.
  A cell with fewer than 3 passing repeats shows its $n$.
- **Layouts.** At each core count $c$ every layout of $R$ ranks by $T$ threads with
  $R \times T = c$ was run on the small problems, and the code's default plus
  $(c/4) \times 4$ on the large ones. "Best" is the fastest layout with three
  passing repeats.
- **Iterative solvers at equal achieved error.** The two codes stop on different
  residuals, so equal nominal tolerances mean different things. Both iterative
  kinds ran a tolerance ladder $\{10^{-6}, 10^{-8}, 10^{-10}\}$ at their default
  layout; their costs are compared at the same achieved solver error (the
  difference from the same code's direct solve), interpolated log-log between
  rungs and never extrapolated.
- **Scaling.** Speedup $S(c) = T(c_0)/T(c)$ and parallel efficiency
  $E(c) = c_0\,T(c_0)/(c\,T(c))$ on median best-layout work time $T$, against the
  smallest core count $c_0$.
- Each run had the whole node's memory; the core count limits cores only.

(bench-direct)=
## Direct solvers

Ratios of work time, with memory in brackets, best layouts, every pair from the same
Slurm job:

| problem | ratio | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|---|
| microstrip | `palace:mumps / gfem:direct` | 14 (2.25) | 12.9 (2.58) | 9.84 (2.91) | 6.85 (4.88) | 4.64 (8.11) |
| microstrip | `palace:mumps / gfem:petsc-mumps` | 11.1 (1) | 6.96 (1.18) | 5.51 (1.5) | 4.56 (2.27) | 3.77 (4.71) |
| r10M_void | `palace:mumps / gfem:direct` | 5.38 (2.13) | 4.22 (2.2) | 3.84 (2.32) | 3.34 (2.5) | 2.66 (2.9) |
| r10M_void | `palace:mumps / gfem:petsc-mumps` | 1.95 (1.73) | 1.99 (1.73) | 2.47 (1.76) | 2.66 (1.83) | 3.07 (2) |
| microstrip_p3 | `palace:mumps / gfem:direct` | 20.8 (1.95) | 19.5 (2.17) | 15.5 (2.51) | 11.5 (2.95) | 5.93 (3.59) |
| microstrip_p3 | `palace:mumps / gfem:petsc-mumps` | 9.53 (1.65) | 10.1 (1.71) | 9.32 (1.72) | 9.5 (1.82) | 6.36 (1.84) |
| lcavity | `palace:mumps / gfem:direct` | 4.7 (2.41) | 3.32 (2.75) | 2.71 (2.97) | 2.08 (4.28) | 1.92 (3.56) |
| lcavity | `palace:mumps / gfem:petsc-mumps` | 1.6 (2.5) | 0.814 (3.01) | 0.427 (3.56) | 0.236 (5.67) | 0.19 (5.48) |
| bragg | `palace:mumps / gfem:direct` | 1.29 (2.2) | 0.91 (1.17) | 0.904 (0.97) | 0.503 (1.09) | 1.62 (4.23) |

The microstrip_p3 point at 8 cores rests on one `palace:mumps` run ($n = 1$): two of
its three repeats ran out of their 1,800 s limit. bragg runs take one to two seconds,
close to the noise.

(bench-iterative)=
## Iterative solvers at equal achieved error

Work time in seconds and memory in GiB at the same solver error, 1.00e-08, on the
ladder layouts (gfem one process of $c$ threads, Palace 0.19.0 $c$ ranks of one
thread):

| problem | cores | `gfem:bddc` s | `gfem:bddc` GiB | `palace:pmg` s | `palace:pmg` GiB | `palace:pmg / gfem:bddc` |
|---|---|---|---|---|---|---|
| microstrip | 8 | 73.5 | 3.23 | 120 | 3.34 | 1.63 |
| microstrip | 16 | 69 | 3.54 | 63.5 | 3.88 | 0.921 |
| microstrip | 32 | 70.1 | 4.14 | 36.4 | 5.89 | 0.519 |
| microstrip | 56 | 54.4 | 3.89 | 26.1 | 8.98 | 0.48 |
| microstrip | 112 | 74 | 4.75 | 19.8 | 20.4 | 0.268 |
| r10M_void | 8 | 166 | 40.3 | 449 | 65.1 | 2.71 |
| r10M_void | 16 | 144 | 41.4 | 263 | 67.5 | 1.83 |
| r10M_void | 32 | 134 | 43.7 | 173 | 89.4 | 1.28 |
| r10M_void | 56 | 134 | 45.3 | 138 | 102 | 1.03 |
| r10M_void | 112 | 143 | 49.1 | 98.1 | 157 | 0.685 |
| microstrip_p3 | 8 | 405 | 11.4 | 459 | 6.31 | 1.13 |
| microstrip_p3 | 16 | 372 | 11.8 | 251 | 6.73 | 0.676 |
| microstrip_p3 | 32 | 361 | 12.6 | 152 | 8.81 | 0.421 |
| microstrip_p3 | 56 | 147 | 11.3 | 84.9 | 11.9 | 0.579 |
| microstrip_p3 | 112 | 187 | 12.4 | 48.5 | 23.5 | 0.259 |

The ratio at other error levels, same layouts:

| problem | error | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|---|
| microstrip | 1.00e-09 | 1.56 | 0.868 | 0.483 | 0.461 | 0.247 |
| microstrip | 1.00e-07 | 1.7 | 0.977 | 0.559 | 0.5 | 0.291 |
| microstrip | 1.00e-06 | 1.77 | 1.04 | 0.601 | 0.522 | 0.317 |
| r10M_void | 1.00e-07 | 2.86 | 1.97 | 1.41 | 1.14 | 0.767 |
| microstrip_p3 | 1.00e-07 | 1.12 | 0.711 | 0.442 | 0.56 | 0.255 |
| microstrip_p3 | 1.00e-06 | 1.1 | 0.734 | 0.462 | 0.537 | 0.249 |

These tables use the ladder layouts, not the best ones. Where gfem's best layout is
several ranks of 4 threads (r10M_void from 32 cores on, microstrip_p3 at 16 and 32
cores), `gfem:bddc` took less time than on the ladder: on r10M_void at 112 cores
60.5 s (28 ranks of 4 threads) against 154 s. The ladder did not run those layouts,
so there is no equal-error comparison at them. At best layouts and the nominal
tolerance $10^{-8}$, the `palace:pmg / gfem:bddc` ratio on r10M_void was 1.38 to
2.22 across the five core counts (gfem one process at 8 and 16 cores, ranks of 4
threads from 32 cores on). At the nominal tolerance $10^{-8}$
the achieved solver errors differed: on r10M_void at 112 cores, 3.60e-09 for
`gfem:bddc` and 1.33e-07 for `palace:pmg`.

On the two eigenproblems there is no ladder. On lcavity both iterative kinds reached
the same error against the closed form, 3.89e-04, the order-2 floor; their times are
in the [lcavity table](bench-lcavity). On bragg gfem has no iterative eigensolver
for Bloch-periodic problems, so there is no iterative pair; one `palace:pmg` cell
was run at 16 cores as an observation.

## Results per problem

Work time in seconds at each solver's best layout: median (min-max) and the layout as
ranks x threads. "default" gives the code's default layout where it differs.

(bench-lcavity)=
### lcavity

| solver | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|
| `gfem:bddc` | 23.6 (23.4-25.2) 1x8 | 23.5 (23.2-23.7) 1x16 | 22.9 (22.9-23.3) 1x32 | 25.1 (25.1-25.2) 1x56 | 30.3 (29.9-30.9) 1x112 |
| `gfem:direct` | 9.74 (9.43-10) 1x8 | 7.78 (7.69-8.89) 1x16 | 7.2 (6.91-8.16) 1x32 | 7.25 (6.93-7.78) 1x56 | 7.31 (7.31-7.37) 1x112 |
| `gfem:petsc-mumps` | 28.6 (28.4-30.7) 1x8 | 31.7 (31.5-32) 1x16 | 45.7 (44.6-48.7) 1x32 | 64.1 (63.8-67.8) 1x56 | 74 (71.9-77.8) 1x112 |
| `palace:mumps` | 45.7 (40-50.3) 8x1 | 25.8 (24.2-25.9) 16x1 | 19.5 (15.4-27.4) 32x1 | 15.1 (13.3-15.7) 56x1 | 14 (13.9-14.2) 56x2 |
| `palace:pmg` | 46.3 (45.9-49) 8x1 | 26 (25.9-26) 16x1 | 15.2 (15.2-15.2) 32x1 | 10.7 (10.7-10.7) 28x2 | 8.1 (8.07-8.15) 28x4 |

gfem's eigensolver kinds run one process here, so `gfem:petsc-mumps` has no
multi-rank layout on lcavity. The iterative kinds ran at the nominal tolerance
$10^{-8}$ only; their errors against the closed form are equal (3.89e-04).

```{figure} _benchmarks/fig-lcavity-error-cost-c16-37c723a.svg
:alt: lcavity at 16 cores, error against work time per solver.

lcavity at 16 cores: error against work time.
```

```{figure} _benchmarks/fig-lcavity-error-cost-c112-37c723a.svg
:alt: lcavity at 112 cores, error against work time per solver.

lcavity at 112 cores: error against work time.
```

(bench-microstrip)=
### microstrip

| solver | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|
| `gfem:bddc` | 74.2 (74.1-74.4) 1x8 | 69.8 (69.5-70.2) 1x16 | 70.3 (70.2-70.4) 1x32 | 54 (54-54.2) 1x56 | 74.7 (74.4-75.2) 1x112 |
| `gfem:direct` | 12.7 (11.9-12.9) 1x8 | 8.32 (8.29-8.59) 1x16 | 6.86 (6.81-7.46) 1x32 | 6.58 (6.53-7.11) 1x56 | 7.11 (6.94-7.53) 1x112 |
| `gfem:petsc-mumps` | 16 (15.5-16.4) 8x1 | 15.4 (13.8-19.7) 8x2 | 12.2 (11.2-17) 8x4 | 9.9 (9.86-9.95) 14x4 | 8.75 (8.46-16.9) 14x8 |
| `palace:mumps` | 178 (157-180) 8x1 | 107 (98.3-107) 16x1 | 67.5 (63.3-71.3) 32x1 | 45.1 (44.4-46) 56x1 | 33 (31.7-35.8) 112x1 |
| `palace:pmg` | 94.4 (93.7-94.5) 8x1 | 50.6 (50.4-50.7) 16x1 | 29.2 (29.1-29.3) 32x1 | 21.4 (21.3-21.4) 56x1 | 16.7 (16.6-16.7) 56x2 |

Iterative times here are at the nominal tolerance $10^{-8}$; compare the iterative
kinds in the [equal-error table](bench-iterative). At 16 cores, `palace:mumps` spent
98.2 s of its 107 s in setup, mostly the MUMPS factorization.

```{figure} _benchmarks/fig-microstrip-error-cost-c16-37c723a.svg
:alt: microstrip at 16 cores, error and solver error against work time per solver.

microstrip at 16 cores: error against work time.
```

```{figure} _benchmarks/fig-microstrip-error-cost-c112-37c723a.svg
:alt: microstrip at 112 cores, error and solver error against work time per solver.

microstrip at 112 cores: error against work time.
```

(bench-bragg)=
### bragg

| solver | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|
| `gfem:direct` | 1.01 (0.934-1.08) 1x8 | 1.21 (0.864-1.41) 1x16 | 1.13 (0.919-1.62) 1x32 | 2.09 (1.76-2.32) 1x56 | 1.2 (1.11-1.22) 1x112 |
| `palace:mumps` | 1.3 (1.3-1.31) 8x1 | 1.1 (1.09-1.11) 4x4; default 1.39 (16x1) | 1.02 (1.02-1.02) 4x8 | 1.05 (1.05-1.06) 7x8 | 1.94 (1.92-1.94) 28x4 |
| `palace:pmg` | | 323 (323-323) 16x1 | | | |

The `palace:pmg` cell took 2.85e+04 GMRES iterations over its 26 solves. At 56 and
112 cores, `palace:mumps` crashed after the eigensolve at layouts with many threads
per rank, so its layout grid there is narrower (see [caveats](bench-caveats)).

```{figure} _benchmarks/fig-bragg-error-cost-c16-37c723a.svg
:alt: bragg at 16 cores, error against work time per solver.

bragg at 16 cores: error against work time.
```

```{figure} _benchmarks/fig-bragg-error-cost-c112-37c723a.svg
:alt: bragg at 112 cores, error against work time per solver.

bragg at 112 cores: error against work time.
```

(bench-r10m)=
### r10M_void

| solver | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|
| `gfem:bddc` | 178 (178-179) 1x8 | 155 (155-155) 1x16 | 101 (99.8-102) 8x4; default 146 | 91.8 (89.7-92.6) 14x4; default 145 | 60.5 (59-73.8) 28x4; default 154 |
| `gfem:direct` | 155 (154-155) 1x8 | 120 (120-121) 1x16 | 104 (103-104) 1x32 | 102 (102-103) 1x56 | 101 (100-101) 1x112 |
| `gfem:petsc-mumps` | 427 (426-429) 2x4; default 455 | 256 (253-257) 4x4; default 432 | 162 (158-162) 8x4; default 430 | 128 (127-136) 14x4; default 428 | 87.5 (81.8-107) 28x4; default 462 |
| `palace:mumps` | 832 (755-873) 8x1 | 508 (489-532) 16x1 | 399 (399-399) 32x1 | 341 (339-351) 56x1 | 269 (268-275) 112x1 |
| `palace:pmg` | 395 (394-397) 8x1 | 235 (234-237) 16x1 | 156 (156-156) 32x1 | 126 (126-129) 56x1 | 92.1 (91.7-92.6) 112x1 |

gfem's default layout is one process of $c$ threads. Palace's layouts with one rank
of $c$ threads were not run on the large problems.

```{figure} _benchmarks/fig-r10M_void-error-cost-c16-37c723a.svg
:alt: r10M_void at 16 cores, solver error against work time per solver.

r10M_void at 16 cores: solver error against work time (no converged reference).
```

```{figure} _benchmarks/fig-r10M_void-error-cost-c112-37c723a.svg
:alt: r10M_void at 112 cores, solver error against work time per solver.

r10M_void at 112 cores: solver error against work time (no converged reference).
```

(bench-p3)=
### microstrip_p3

| solver | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|
| `gfem:bddc` | 389 (388-391) 1x8 | 278 (277-279) 4x4; default 356 | 250 (250-251) 8x4; default 346 | 142 (141-143) 1x56 | 181 (178-183) 1x112 |
| `gfem:direct` | 74.7 (74.1-74.8) 1x8 | 45.6 (45.5-45.9) 1x16 | 32.7 (32.6-32.8) 1x32 | 29.5 (29.3-29.8) 1x56 | 27.7 (27.3-27.9) 1x112 |
| `gfem:petsc-mumps` | 163 (162-164) 2x4; default 211 | 88.3 (85.9-88.6) 4x4; default 191 | 54.4 (53.8-54.4) 8x4; default 187 | 35.8 (34-38.8) 14x4; default 185 | 25.8 (23.1-26) 28x4; default 208 |
| `palace:mumps` | 1.56e+03 8x1, $n = 1$ | 889 (829-1.01e+03) 16x1 | 507 (464-522) 32x1 | 339 (321-341) 56x1 | 164 (162-174) 112x1 |
| `palace:pmg` | 340 (340-355) 8x1 | 201 (198-211) 16x1 | 120 (101-122) 32x1 | 64.5 (63.9-65) 56x1 | 37.8 (37.8-38.4) 112x1 |

```{figure} _benchmarks/fig-microstrip_p3-error-cost-c16-37c723a.svg
:alt: microstrip_p3 at 16 cores, error and solver error against work time per solver.

microstrip_p3 at 16 cores: error against work time.
```

```{figure} _benchmarks/fig-microstrip_p3-error-cost-c112-37c723a.svg
:alt: microstrip_p3 at 112 cores, error and solver error against work time per solver.

microstrip_p3 at 112 cores: error against work time.
```

(bench-memory)=
## Memory

Sum of per-rank peak resident memory in GiB, best layouts. The sum counts shared
pages once per rank, so it is an upper bound; the proportional share (PSS) is
printed beside it in the report.

| problem | solver | 8 | 16 | 32 | 56 | 112 |
|---|---|---|---|---|---|---|
| r10M_void | `gfem:bddc` | 40.9 | 42.1 | 49.1 | 53.2 | 64 |
| r10M_void | `gfem:direct` | 84.7 | 85.6 | 90.4 | 93.9 | 103 |
| r10M_void | `gfem:petsc-mumps` | 104 | 109 | 119 | 128 | 149 |
| r10M_void | `palace:mumps` | 180 | 189 | 210 | 235 | 298 |
| r10M_void | `palace:pmg` | 63.8 | 67.1 | 89.4 | 102 | 157 |
| microstrip_p3 | `gfem:bddc` | 11.4 | 12.1 | 14 | 11.2 | 12.3 |
| microstrip_p3 | `gfem:direct` | 17.7 | 18 | 18.6 | 19.5 | 21.6 |
| microstrip_p3 | `gfem:petsc-mumps` | 20.9 | 22.8 | 27.2 | 31.8 | 42.1 |
| microstrip_p3 | `palace:mumps` | 34.5 | 39.1 | 46.8 | 57.7 | 77.6 |
| microstrip_p3 | `palace:pmg` | 5.96 | 6.41 | 8.47 | 11.5 | 23.1 |

On microstrip at 112 cores, `palace:mumps` at 112 ranks shows 55 GiB as the sum
of peaks but 18.5 GiB as PSS: with many ranks the sum overstates the memory the
node really used.

(bench-scaling)=
## Scaling

Speedup $S$ and parallel efficiency $E$ at 112 cores against 8 cores, best layouts,
one node:

| solver | lcavity $S$ / $E$ | microstrip $S$ / $E$ | r10M_void $S$ / $E$ | microstrip_p3 $S$ / $E$ |
|---|---|---|---|---|
| `gfem:bddc` | 0.78 / 0.0557 | 0.993 / 0.071 | 2.95 / 0.21 | 2.15 / 0.153 |
| `gfem:direct` | 1.33 / 0.0951 | 1.79 / 0.128 | 1.53 / 0.109 | 2.7 / 0.193 |
| `gfem:petsc-mumps` | 0.387 / 0.0276 | 1.83 / 0.131 | 4.88 / 0.349 | 6.32 / 0.452 |
| `palace:mumps` | 3.26 / 0.233 | 5.41 / 0.386 | 3.09 / 0.221 | 9.46 / 0.676 |
| `palace:pmg` | 5.71 / 0.408 | 5.65 / 0.404 | 4.29 / 0.306 | 9 / 0.643 |

The `palace:mumps` microstrip_p3 line starts from one run at 8 cores ($n = 1$).
`gfem:direct` (PARDISO, one process) gained little past 32 cores: on r10M_void it
took 104, 102 and 101 s at 32, 56 and 112 cores. Every point is one node; no run
spans several nodes.

```{figure} _benchmarks/fig-microstrip-scaling-37c723a.svg
:alt: microstrip, work time and memory against core count per solver.

microstrip: work time and memory against cores.
```

```{figure} _benchmarks/fig-r10M_void-scaling-37c723a.svg
:alt: r10M_void, work time and memory against core count per solver.

r10M_void: work time and memory against cores.
```

```{figure} _benchmarks/fig-microstrip_p3-scaling-37c723a.svg
:alt: microstrip_p3, work time and memory against core count per solver.

microstrip_p3: work time and memory against cores.
```

```{figure} _benchmarks/fig-lcavity-scaling-37c723a.svg
:alt: lcavity, work time and memory against core count per solver.

lcavity: work time and memory against cores.
```

```{figure} _benchmarks/fig-bragg-scaling-37c723a.svg
:alt: bragg, work time and memory against core count per solver.

bragg: work time and memory against cores.
```

(bench-caveats)=
## Caveats

- **One machine, one version of each code, one set of runs.** MareNostrum 5
  general-purpose nodes only. Nothing here says how the codes compare on other
  hardware, other builds or other versions.
- **The Palace build is the palace-solver wheel, packaged by the gfem author.**
  Palace ran from the palace-solver 0.19.0 wheel on PyPI, which the author of gfem
  packages. It was chosen because, among the Palace builds measured on this machine
  before the campaign (the wheel and two builds from source), it was the fastest or
  within 1 % on the test problems. A reader may still see a conflict of interest.
- **Different MPI and libraries.** Both codes used MUMPS 5.9.1, from different
  builds: gfem's through the PETSc 3.26.0 wheel on the cluster's Open MPI, Palace's
  bundled in its wheel with MPICH 4.3.2. The wheel's MPICH has no InfiniBand
  transport (TCP between nodes); this does not affect the single-node runs here.
- **Single node only.** Runs over several nodes were tried outside the suite and
  left out: the suite's memory sampler and placement check see the first node only.
- **Nominal tolerances differ.** The codes stop on different residuals, so the
  iterative kinds are compared only at equal achieved solver error, and only where
  the tolerance ladder ran (the default layouts). On r10M_void from 32 cores on the
  default layout is not gfem's fastest, so `gfem:bddc` took longer in the
  equal-error tables there than at its best layout.
- **Structural differences are kept.** Palace sets up and solves once per excited
  port and always runs its error estimator (lightened here); gfem solves all excited
  ports at once. `palace:pmg` has a matrix-free fine level; gfem assembles.
- **The microstrip reference cannot separate the codes.** Every run sits at the
  discretization floor of its order and the reference bound (1.07e-02 at 100 GHz)
  is about a third of that floor. r10M_void has no converged reference.
- **Bragg and the iterative solvers.** On this Bloch-periodic eigenproblem
  `palace:pmg` took 2.85e+04 GMRES iterations at the suite's inner tolerance;
  Palace has options that reduce the count, but the suite runs one configuration
  per solver kind on every problem. gfem has no iterative eigensolver for this
  problem.
- **Failed, refused and timed-out runs.** `palace:mumps` crashed after the
  eigensolve on bragg at 56 and 112 cores, at layouts with many threads per rank
  (24 runs). `palace:mumps` ran out of its time limit at some thread-heavy layouts
  on microstrip, microstrip_p3 and r10M_void (8 runs; 12 later repeats not started).
  gfem refused by design: BDDC and PETSc MUMPS on Bloch eigenproblems, and PARDISO
  on more than one rank. These are recorded outcomes, not rerun, and are excluded
  from every table.
- **Cells with fewer than 3 runs**: microstrip_p3 `palace:mumps` at 8 cores
  ($n = 1$); bragg `palace:mumps` at 56 ranks of 2 threads at 112 cores ($n = 2$).
- **Noise.** Rows whose min-max spread exceeds 10 % are flagged in the report (24
  rows), mostly the second-long bragg runs.
- **Known gfem limits seen here.** gfem reads and prepares the mesh in one process,
  and its default layout (one process of $c$ threads) was up to 2.55 times slower
  than its best layout for `gfem:bddc` on r10M_void. gfem's driven solvers set up
  once per frequency on microstrip, with no reuse across frequencies.
- **Memory** is a sum of per-rank peaks, an upper bound, and each run had a whole
  node's memory available.

## Reproduce and raw data

Everything below is at fixed commits.

- **Suite and method:** [`benchmarks/`](https://github.com/benvial/gfem/tree/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks)
  at commit `e7d58a9`, its
  [README](https://github.com/benvial/gfem/blob/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks/README.md)
  and the campaign files under `benchmarks/campaigns/` (`full-c8` to `full-c112`,
  `full-bragg-c8` to `full-bragg-c112`, `full-bragg-pmg-c16`, `full-scale-c8` to
  `full-scale-c112`).
- **Raw data:** the `bench-data` branch at commit
  [`434e6e6`](https://github.com/benvial/gfem/tree/434e6e6d0adc538e0cc1d9661d1f90bfa24d1dc4):
  one record per run under `campaigns/20261120-e7d58a9-<file>/`, the converged
  references under `references/`, and this page's report under
  `reports/20261120-37c723a/` (`index.md`, `report.json`, every figure).
- **Check the report** from a checkout of `e7d58a9`, with the records branch as the
  store:

  ```bash
  python -m benchmarks report --check gfem-bench-data/reports/20261120-37c723a
  ```

- **Run a campaign** on a Slurm cluster: `python -m benchmarks ship`, `plan`,
  `submit` and `close`, then `report`, as the README's
  [Running a campaign](https://github.com/benvial/gfem/blob/e7d58a9a9db93af772fb234c29c2ecc238f7d279/benchmarks/README.md#running-a-campaign)
  section shows.

## Versions and date

- **gfem**: the development revision `771da66` of main, run from the harness commit
  `e7d58a9` (2026-11-20). NGSolve 6.2.2608, PETSc 3.26.0 and petsc4py 3.26.0 wheels,
  MKL 2026.2.0, mpi4py 4.1.2, on the cluster's Open MPI.
- **Palace**: 0.19.0, from the palace-solver 0.19.0 wheel (PETSc 3.26.0, MUMPS
  5.9.1, MPICH 4.3.2, all bundled).
- **Machine**: MareNostrum 5 general-purpose partition, two Intel Xeon Platinum
  8480+ per node, 112 cores; exclusive nodes, one node per run; standard nodes for
  lcavity, microstrip and bragg, high-memory nodes for r10M_void and microstrip_p3.
- **Measured**: 2026-11-20, 16 campaign files at one commit, 3 repeats per cell.
- **Report**: `20261120-37c723a`.

## Acknowledgements

[Palace](https://github.com/awslabs/palace) is an open-source 3D finite-element
solver for computational electromagnetics, developed by AWS Labs and distributed
under the Apache License 2.0. This page used Palace 0.19.0. Thanks to its
developers for making it openly available.

We acknowledge EuroHPC Joint Undertaking for awarding us access to MareNostrum5 at BSC, Spain.
