# Performance data

## What to expect

How large a driven model gfem has solved across MPI ranks, and how fast, on one
cluster. These are observations from two sets of runs, each of one gfem version;
each table says which. They are not limits or promises for your model. Every
number in this summary comes from the tables below.

For gfem and another solver on the same meshes, on one node from 8 to 112 cores,
see {doc}`benchmarks`.

- **Largest model.** `linear_solver="bddc_gmres"` solved 108,570,046 DOF on 4 to 32
  nodes. Its fastest {py:obj}`gfem.solve` was 423 s, on 8 nodes, in the earlier
  version ([table](scaling-strong-108m)); 291 s on 32 nodes in the later one
  ([table](scaling-coarse)).
- **Largest model with `direct`.** `linear_solver="direct"` solved 47,277,960 DOF on
  16 nodes in 832 s ([table](scaling-weak)). At 108,570,046 DOF on 32 nodes it ran
  out of memory ([details](scaling-direct-108m)).
- **Which solver.** Up to 14,217,194 DOF on 1 to 4 nodes, `direct` was faster: 21 s
  against 36 s at 3,061,188 DOF on one node, 54 s against 76 s at 14,217,194 DOF on
  4 nodes ([table](scaling-weak)). From 23,148,138 DOF up, `bddc_gmres` was faster
  at every node count tried: 199 s against 410 s on one node, 128 s against 161 s
  on 8 nodes ([table](scaling-strong-23m)). It also needed less memory: 175 GiB
  against 716 GiB on the largest node at 23,148,138 DOF on one node.
- **How many nodes.** In the earlier version, past a point more nodes made
  `bddc_gmres` slower: at 108,570,046 DOF its time was lowest on 8 nodes, at
  23,148,138 DOF on 4 to 8 nodes. On 16 and 32 nodes that version factored
  BDDC's coarse problem on 64 ranks. In the later version, on a PETSc stack
  built by its build script, a single solve took 440, 323 and 291 s on 8, 16
  and 32 nodes at 108,570,046 DOF, and 129, 111 and 76 s at 23,148,138 DOF
  ([table](scaling-coarse)). Each of these times is one run. On a stack built
  by an earlier script, the later version factors the coarse problem another
  way, and these times do not apply to it.
- **Repeated solves.** When one script solves the same mesh again, as a loop
  over designs does, gfem keeps the ordering and symbolic factorization of
  BDDC's coarse problem between the solves, and GMRES starts from the previous
  design's solution. In one loop of 3 designs at 108,570,046 DOF in the later
  version, the second and third designs took 301 and 300 s on 8 nodes, 187 s
  each on 16 and 153 and 152 s on 32, against 440, 323 and 291 s for the first
  ([table](scaling-design-loop)). The 3 designs together took 1042, 697 and
  596 s.
- **Before the solve.** Reading and partitioning the mesh come before
  `gfem.solve` and are not in its time: about 200 s at 108,570,046 DOF
  ([table](scaling-phases)).

All of this is one machine, two gfem versions (each table says which), one family
of meshes (a coplanar line), order 2 and one frequency per solve. A time here is an
observation, not a promise for another mesh, layout or machine: time a small run of
your own model before you size a large job. {doc}`../recipes/distributed-solve`
shows how to run both solvers and how to choose ranks and threads.

The S of these runs were checked against one process and against the other solver:
see {doc}`../verification`. The later version's one-frequency S differ from the
earlier version's, on the same mesh and node count, by at most
$2.4 \times 10^{-12}$.

## The machine

MareNostrum 5 general-purpose nodes, two Intel Xeon Platinum 8480+ per node, 112
cores, run as 28 ranks of 4 threads per node. What the hardware gives apart from any
solver: MKL's dense multiply and a STREAM triad on one node against thread count,
then a ping-pong and `Allreduce` between nodes.

```{include} _scaling/machine.md
```

## The meshes

One family: a coplanar line on a TopMetal2 layer with its conductors removed from the
volume ("voided"), at three refinements, and the uniform split of each, which replaces
every tetrahedron by eight.

```{include} _scaling/meshes.md
```

(scaling-strong-108m)=
## Strong scaling, `bddc_gmres`, 108,570,046 DOF

The largest run:

```{include} _scaling/ms3.md
```

From 4 to 32 nodes. The `gfem.solve` median is lowest on 8 nodes and higher on 16
and 32. `direct` on 32 nodes is a row of its own.

```{include} _scaling/strong-108m.md
```

These runs are of the earlier version. On 16 and 32 nodes it factored the coarse
problem on 64 ranks; for the later version see {ref}`scaling-coarse`.

(scaling-strong-23m)=
## Strong scaling, both solvers, 23,148,138 DOF

From 1 to 32 nodes. Above 8 nodes a rank holds fewer than about 90 k DOF, under the
layout the recipe recommends ({ref}`distributed-layout`); the rows are there to show it.
`direct` on 16 nodes ran out of its time limit, so 32 nodes was not run.

```{include} _scaling/strong-23m.md
```

These runs are of the earlier version. Its `bddc_gmres` rows on 16 and 32 nodes
factored the coarse problem on 64 ranks; for the later version see
{ref}`scaling-coarse`.

(scaling-weak)=
## Weak scaling, about 3 M DOF per node

From 1 to 32 nodes, each rung its own mesh of the family; the DOF per node column gives
each rung's size. The 4-node rung holds more than the others.

```{include} _scaling/weak.md
```

These runs are of the earlier version. Its `bddc_gmres` rows on 16 and 32 nodes
factored the coarse problem on 64 ranks. The later version was not run on this
ladder; its 32-node solve of the largest mesh is in {ref}`scaling-coarse`.

(scaling-direct-108m)=
## `direct` at 108,570,046 DOF

One attempt on 32 nodes. The factorization reported too little memory, and gfem's
retry with a larger memory allowance was then stopped by the cluster for running out
of memory. No S came out of it. The estimates below are what MUMPS printed before the
factorization.

```{include} _scaling/direct-108m.md
```

(scaling-phases)=
## Where the time goes

The phases of the largest `bddc_gmres` rungs, and GMRES's own share: its vector work
(the basis's orthogonalization, norms, scaling and updates) and the time its ranks
wait in reductions.

```{include} _scaling/phases.md
```

These runs are of the earlier version. On 16 and 32 nodes it factored the coarse
problem on 64 ranks; for the later version see {ref}`scaling-coarse`.

At 108,570,046 DOF in the earlier version, the coarse apply is 42 % to 53 % of
`gfem.solve` at every node count, and the coarse setup grows from 8 to 32 nodes.
At 32 nodes the reduction wait is 0.8 % of `gfem.solve`.

(scaling-coarse)=
## The later version: one solve and repeated solves

These runs need the PETSc stack that the later version's
`petsc_wheelhouse.sh` builds ({doc}`../recipes/petsc-slepc`): its MUMPS orders
BDDC's coarse problem with PT-Scotch on at most 32 processes. On that stack
gfem factors the coarse problem on every fourth rank, 7 per node at 28 ranks
per node, in every set-up. Solves repeated on one mesh in one process keep the
coarse problem's ordering and symbolic factorization. On a stack built by an
earlier script, without that limit, gfem takes another route for the coarse
problem, and the tables of this section do not describe it.

One solve at one frequency, from 8 to 32 nodes, at 108,570,046 and 23,148,138
DOF, one run per row. Efficiency is against 8 nodes, so it does not compare
with the earlier tables' column, which is against 4 nodes at 108,570,046 DOF
and 1 node at 23,148,138 DOF.

```{include} _scaling/coarse-strong-108m.md
```

(scaling-design-loop)=
### A loop of 3 designs

Three designs solved one after the other in one process, on the 108,570,046 DOF
mesh, one run per node count. Each design changes one volume's permittivity and
keeps the mesh, and each starts GMRES from the previous design's solution.
GMRES took 63/64 iterations in every design. Design 0 is the 108,570,046 DOF
row above.

```{include} _scaling/coarse-design-loop.md
```

Designs 1 and 2 keep design 0's coarse analysis and only refactor: the coarse
set-up goes from 98, 77 and 71 s in design 0 to 53, 36 and 30 s on 8, 16 and
32 nodes. The coarse apply per call is the same in every design of a run.

The phases of the one-frequency runs, as in {ref}`scaling-phases`:

```{include} _scaling/coarse-phases.md
```

## Ranks and threads, 10,248,856 DOF

These are separate runs, not part of the tables above: one run per layout of a
prototype script on the 10,248,856 DOF coplanar line, on the same machine with an
earlier gfem version. The advice in {ref}`distributed-layout` rests on them.

| Nodes | Solver | Ranks x threads per node | Time, s |
|---|---|---|---|
| 4 | `direct` | 28 x 4 | 54 |
| 4 | `direct` | 16 x 7 | 61 |
| 4 | `direct` | 8 x 14 | 79 |
| 4 | `bddc_gmres`, with PETSc | 28 x 4 | 57 |
| 4 | `bddc_gmres`, with PETSc | 16 x 7 | 54 |
| 1 | `bddc_gmres`, without PETSc | 8 x 14 | 118 |
| 1 | `bddc_gmres`, without PETSc | 28 x 4 | 145 |

At 28 x 4 on 4 nodes a rank holds about 92 k DOF. S did not depend on the layout.

The one-node rows come from an earlier gfem and a prototype script, and time the
whole run. A later measurement on one node, with the benchmark suite and gfem
`771da66`, timed the work after startup and found 28 x 4 fastest for
`bddc_gmres`: 60.5 s against 154 s for one process of 112 threads
({ref}`bench-r10m`).
Layouts were compared up to 4 nodes and 10,248,856 DOF; past that the advice is an
extrapolation.

## How these were measured

- **gfem**: the earlier version is the development version of 5 November 2026,
  between the preview and 0.1.0, with Open MPI 4.1.6, PETSc 3.26.0 and NGSolve
  6.2.2608. The later version is the development version of 21 November 2026,
  with the same Open MPI, PETSc and NGSolve, on a PETSc stack built by that
  version's `petsc_wheelhouse.sh`, whose MUMPS caps PT-Scotch's parallel ordering
  at 32 processes.
- **Machine**: MareNostrum 5 general-purpose partition, `highmem` nodes, 28 ranks of 4
  threads per node.
- **Runs**: order 2, one frequency of 50 GHz unless a row says otherwise, both ports
  excited. `bddc_gmres` ran at the {py:obj}`gfem.SolverOptions` `tolerance` in each row's `tol`
  column; gfem's own default is `1e-6`. `direct` is PETSc's MUMPS with PT-Scotch
  ordering. The repeats of a row ran back to back on the same nodes, so a row's range
  is run-to-run noise, not placement.
- **Design loop**: design $i$, from 0, scales the permittivity of the
  highest-permittivity volume by $1 + 10^{-3} i$ and is one `gfem.solve` at one
  frequency, all in one process on one mesh.

The scaling record (Zenodo, DOI pending) holds both sets, each with the commit it
names: the run records, logs, job scripts, meshes and the `tools/` scripts at that
commit. `scale_env.sh` makes the
environment, `scale_steps.py` writes each job's steps, `scale_job.sbatch` runs one job
per rung per solver and `scale_run.py` makes each run and its record. `scale_bench.py`
is the machine benchmark. The voided meshes are in the record, by md5; `split_mesh.py`
makes the splits. `scale_mesh.py` rebuilds a voided mesh's source with `gsim==0.7.0`,
`ihp-gdsfactory==0.2.8` and `gmsh==4.15.2` only: a current gsim writes a different
conductivity, so S would move for a reason unrelated to the mesh. `scale_void.py` then
removes the conductors.

The tables are not computed when these docs build. To redo them from the record, with
`COMMIT` the commit the record names:

```bash
python tools/scale_tables.py RECORDS docs/project/_scaling --commit COMMIT \
  --logs LOGS --mesh-md5 meshes/md5.txt --mesh-md5 meshes/splits.md5
python tools/scale_tables.py RECORDS2 docs/project/_scaling --set coarse \
  --commit COMMIT2 --logs LOGS2
```

`RECORDS`, `LOGS` and `COMMIT` are the earlier version's; `RECORDS2`, `LOGS2` and
`COMMIT2` the later one's.
