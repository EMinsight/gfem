# Benchmark suite

`benchmarks/` runs a fixed set of suite cases with gfem and Palace on a Slurm
cluster under the fairness rules below, stores one suite record per run, and writes
a campaign report from those records. It is a developer tool. It is not part of the
gfem wheel or sdist, and no page of the docs build reads, links or includes this
directory, a record or a report (ADRs 0017, 0080, 0189). This file and the code
under `benchmarks/` may name Palace; user-facing docs may not.

Vocabulary (GLOSSARY "Benchmark suite"): suite case, case file, solver adapter,
solver kind (`code:kind`), pinned mesh, mesh store, suite record, campaign, campaign
file, campaign manifest, converged reference, reference bound, software stack, stack
description, site file, core budget, layout grid, default and best layout, work
time, campaign report. Bare "benchmark" is the test tier. A converged reference is
not a Baseline: no test asserts against one.

A campaign holds one gfem commit and one version per solver. Timings and memory are
observations. Only correctness checks gate.

## Contents

```
benchmarks/
  __main__.py        CLI (python -m benchmarks SUBCOMMAND)
  job.sbatch         one exclusive Slurm job of a planned run directory
  site.example.toml  template of the site file (the real one is never committed)
  harness/           config, case files, staging, plan, launch and measure,
                     records, scoring, budget, ship (standard library only)
  adapters/          solver adapters: gfem.py, gfem_worker.py, palace.py
  stacks/            stack descriptions (*.toml) and kind modules (wheels, native)
  cases/<name>/      case.toml, mesh.msh.xz (if small), reference.json, source/
  campaigns/         campaign files: smoke, smoke-r10m, release-gfem, reference-*
  references/        closed forms and the microstrip ladder (rung case files)
  report/            load, views, markdown, figures
```

Everything except `adapters/gfem_worker.py` (runs inside the stack's gfem env) and
`report/figures.py` (matplotlib, imported lazily) imports only the standard library
at module top level (ADR 0400). There is no requirements file. The CLI runs with any
Python 3.12 from the repository root or a shipped tree; the report's figures need
`uv run --group docs`. Harness tests are `tests/unit/test_benchmarks_*.py`; no test
tier runs Palace.

## Method and fairness rules

ADRs 0410, 0411 and 0412 hold the decisions.

### By construction

- One case file renders both codes' inputs. Lengths are in the mesh's unit,
  everything else is SI. A case holds no tolerance, solver, layout or thread count.
- The mesh is pinned by the sha256 of the uncompressed `.msh` and re-verified before
  every run. A mismatch is never repaired: the file is quarantined and the run is
  `refused`.
- `order` is one integer passed to both codes.
- One launcher rule for both codes at every rank count (below).
- One process per run: each run is its own srun step in a fresh work directory, so
  nothing (gfem's `retain_memory`, for example) carries over between repeats. Reuse
  inside one run (gfem factor reuse across frequencies, Palace `ReorderingReuse`)
  counts as part of each code's design.

### Pinned settings

Each code runs at its own defaults, except for the settings listed here. A default
that is not on this list is never changed to match the other code.

| code | setting | reason |
|---|---|---|
| Palace | `Solver.Linear.Type` always written (`MUMPS` for `palace:mumps`, `SuperLU` for `palace:pmg`) | the kind must not depend on build-time defaults; `SuperLU` is what the wheel resolves with `Type` omitted, so `config_resolved.json` matches the default |
| Palace | `palace:mumps`: `MGMaxLevels 1`, `ComplexCoarseSolve true` | makes the direct kind a complex MUMPS factorization (GMRES around it converges in 1-2 iterations) |
| Palace | `Tol` = run tolerance (iterative), 1e-10 (direct), 1e-12 for the inner linear solve of eigen runs | same nominal tolerance as gfem; the direct kind's GMRES must not stop early; the eigen inner rule is gfem's own, $\min(\mathrm{tol}, 0.01\,\mathrm{tol}_{\mathrm{eigen}})$ |
| Palace | `MaxIts 2000` | the default (100) would bind; iteration caps must never bind |
| Palace | `EstimatorTol 1e-2`, `EstimatorMG true` | the estimator always runs; this makes it cheaper without changing S |
| Palace | eigen `Tol 1e-10` | parity with gfem's `eigen_tolerance` |
| Palace | `Absorbing.Order 1` | the order-2 ABC sign differs between 0.18 and 0.19 |
| Palace | `Refinement` off (`UniformLevels 0`, `MaxIts 0`) | same mesh as gfem |
| Palace | 0.18: `RefineCrackElements false`; 0.19: no crack key | same DOF as gfem on the voided meshes (a no-op there on 0.18; 0.19 does not crack by default) |
| Palace | no field output (`SaveStep 0`, no ParaView) | output IO is not part of the case |
| Palace | never `AbsTol`, never `NSamples` | 0.19-only key, default 0; key removed in 0.19 |
| gfem | `linear_solver` per kind (`direct`, `petsc`, `bddc_gmres`) | defines the kind |
| gfem | `tolerance` (iterative kind only; direct runs record `null`) | the run tolerance |
| gfem | `eigen_tolerance 1e-10` | parity with Palace |
| gfem | no field output | as for Palace |

Wheels Palace always runs `palace-real`, never the `palace` wrapper: the 0.18 wrapper
forces `OPENBLAS_NUM_THREADS = -nt` and the 0.19 wrapper changed its thread policy.

### Structural differences

Each record's `structural` block lists these, and the report prints them beside
every cross-code view. They are never removed or normalised:

- Palace sets up and solves once per excitation; gfem solves all excited ports on one
  factorization. Counted as is; `setups` and `solves` are recorded.
- Palace's error estimator always runs (its own `estimate_s` bucket, inside work
  time).
- `palace:pmg` at $p \ge 2$ has a matrix-free fine level and a real-valued SuperLU
  coarse solve; gfem assembles.
- Direct arms: gfem PARDISO (one process) or PETSc MUMPS; Palace GMRES around complex
  MUMPS.
- NGSolve builds the BDDC preconditioner while it assembles, so the setup cost of
  `gfem:bddc` shows up in `assemble`.

### Tolerances

- Iterative kinds run a tolerance ladder {1e-6, 1e-8, 1e-10} with anchor 1e-8. The
  layout grid runs at the anchor. Off-anchor rungs (`[[ladder]]` in a campaign file)
  run at the default layout or the `RxT` the ladder names, never `best`, which is
  known only after the report.
- The two codes stop on different residuals (gfem: relative preconditioned residual;
  Palace: left-preconditioned GMRES residual), so equal nominal tolerances mean
  different things. Codes are compared only at equal achieved error, read off each
  code's error-against-cost curve.
- Eigen runs use 1e-10 in both codes with inner solves at 1e-12. The two codes define
  the eigen tolerance differently; 1e-10 is what parity needs.
- A run that stops above tolerance or at the cap fails `converged` and its status is
  `failed`. The rule is not loosened for one code.
- `plan` adds one direct run per (code, case, order) that has an iterative run and no
  direct one (`added_for: "same_code_error"`, default layout, 1 repeat). The
  same-code error is solver error, not accuracy.

### Layouts

- Default layout: gfem $1 \times c$ on one node (eigen and periodic runs $1 \times c$
  always; above one node $(c/4) \times 4$); Palace $c \times 1$. A default the adapter refuses is a
  `refused` cell, not replaced by another layout.
- `best` expands at plan time to the layout grid: every $R \times T = c$, whole cores
  per rank, no hyperthreads, $T$ divides 56 when $c > 56$ (no rank across a socket),
  $1 \times c$ always included, refused layouts (gfem $R > 1$ on eigen or periodic
  cases, `gfem:direct` $R > 1$) left out. A kind refused at every grid layout (gfem
  `petsc-mumps` and `bddc` on a Bloch-periodic eigenproblem, which gfem solves with
  `direct` only) is planned once at its default layout as `refused`. Records store
  `layout.asked = "grid"`. The report
  picks the best layout afterwards: the lowest median work time among cells passing
  every gate with all repeats; ties within the spread are shown as ties.
- Large cases run only the layouts the campaign lists.

### Launch, threads and affinity

Every run, both codes, every rank count:

```
srun --exact -n R -c T --cpu-bind=cores --distribution=block:block \
     --hint=nomultithread --ear=off --mpi=<plugin> <argv>
```

Plugins: gfem `none` at R = 1 and `pmix` above; wheels Palace `pmi2`; native stacks
per stack file. If a stack sets `launch_r1 = "taskset"` (decided by the stack check),
one-rank runs are `taskset -c <cores>` plus direct exec, recorded
`run.launch = "taskset"`. `run --local` uses Hydra `-launcher fork -bind-to core`
under taskset; its records are `machine: local` and never compared with cluster
records.

- The env script of each run unsets `OMP_PLACES OMP_PROC_BIND GOMP_CPU_AFFINITY
  KMP_AFFINITY` and every thread variable, then exports `OMP_PLACES=cores
  OMP_PROC_BIND=close`. Stack, site and campaign `env` tables may not name the
  affinity variables. Nothing from the caller's shell reaches a run.
- Every thread variable a code's libraries read is set to T per rank, except build
  facts written in the stack file: wheels Palace `OPENBLAS_NUM_THREADS = 1` (the
  wheel's OpenBLAS serialises MUMPS's OpenMP threads), OpenBLAS capped at 64 for
  gfem. gfem gets `NGS_`, `MKL_`, `OPENBLAS_` and `OMP_NUM_THREADS` = T.
- Jobs are exclusive (`--exclusive`, `--ear=off`). Cross-code pairs come from one
  job; pairs across jobs are labelled "different nodes". The campaign's `node_class`
  (`standard`, or `highmem`, which adds `--constraint=highmem`) is requested; the
  observed class (from `MemTotal`) is recorded, not gated.

### Time and memory

- Buckets per record: `startup_s`, `mesh_read_s`, `discretize_s`, `assemble_s`,
  `setup_s`, `solve_s`, `estimate_s`, `postpro_s`, optional `rom_s`, `other_s`. A
  phase a code does not report is `null`, never 0. gfem's come from the `gfem.timing`
  logger (ADR 0190); Palace's from `palace.json` through an explicit key table (ADR
  0404; an unknown key fails the parse).
- Headline time is work time, `work_s = wall_s - startup_s`. Startup (gfem: step wall
  minus the worker's span; Palace: step wall minus `Total`) is shown apart: it
  measures imports and file-system cache state. The estimator and postprocessing stay
  inside work time because Palace users cannot switch them off. Derived:
  `setup_total_s`, `solve_total_s`, `per_sample_s`.
- Buckets are the maximum over ranks, with rank 0's values in `timings.rank0`. For
  Palace with R > 1 they are the sum of per-row maxima of the stdout timing table
  (`rank_reduction: "max-of-rows"`, an upper bound).
- Headline memory is `peak_rss_sum_gib`, the sum of per-rank peak RSS (gfem worker
  `VmHWM`; Palace `PeakNodeMemoryMegabytes.Total`). Shared pages count once per rank,
  so it is an upper bound on the simultaneous footprint. Beside it,
  `peak_rss_rank_max_gib` (largest rank; Palace `PeakMemoryMegabytes.Max`) and
  `peak_rss_largest_node_gib` (largest per-node sum; Palace
  `PeakNodeMemoryMegabytes.Max`). A node sampler sums `Pss` over the step's pids
  every second, every 0.1 s over the first 5 s (`pss_peak_sum_gib`); sacct `MaxRSS`
  is kept for runs over 60 s; the codes' self-reports are kept beside. None of them gates.
- 3 repeats in rounds, round $r$ rotated by $r \lceil \text{cells} / \text{repeats} \rceil$, so repeats of a
  cell never run back to back. Medians per figure over passing repeats; a spread over
  10 % is flagged "noisy".
- Skip rules inside a job: repeat 1 ending `timeout` makes later repeats
  `not_started` (`earlier_repeat_timeout`); a default-layout iterative cell failing
  `converged` in round 1 makes the other cells of that (case, kind) `not_started`
  (`kind_not_converging`).

### Gating checks and observations

A record that fails a gate is kept and listed in the report's failures table, and
left out of medians, best-layout choice and comparisons.

| check | level | fails when |
|---|---|---|
| `provenance` | record, campaign | env, versions or pins differ from the manifest |
| `mesh_hash` | record | staged mesh differs from the pin (run refused before launch) |
| `inputs_match` | record | Palace `config_resolved.json`, read back through the importer, or the output shape (frequencies at rel 1e-12, ports, pairs, mode count) differs from the case |
| `dof` | record | DOF count differs from the pin in `reference.json` (`meshes.<sha>.dof.<order>`) |
| `layout` | record | observed ranks or threads differ from the asked layout |
| `placement` | record | a rank's mask is not T physical cores, or masks overlap |
| `converged` | record | an iterative solve stopped above tolerance or at the cap |
| `complete` | record, campaign | outputs missing, Palace `ElapsedTime` missing, a planned row without a final status |
| `reference_matches` | record | the reference file does not match the case, the mesh, or the file the run named |
| `accuracy_plausible` | record | error above the plausibility ceiling of the run's order |
| `dof_equal` | campaign | any two DOF counts observed on a case differ, `dof_mismatch` runs included (gates the "same-mesh comparison" label) |

A run counts as passing in the report only when `layout` and `placement` passed: a
run whose ranks or masks the sampler never saw, or that ended before the sampler saw
$T$ threads in a rank whose masks hold too few cores (both `skipped`), is labelled
`unverified` and left out of the medians and the best-layout choice. The sampler
looks every 0.1 s over a run's first 5 s, then every second, so short runs are seen.

The DOF guard stops a run on a mismatch: Palace from its `Level k (p = n): N
unknowns` stdout line, one-process gfem from its `discretize` record, multi-rank gfem
after the solve from the summed count. Status `failed`, reason `dof_mismatch: <code>
<version> N != pinned M`.

Observed, never gated: node class, same-job pairing, every time and bucket, every
memory figure, iteration counts, setups, error values, same-code error, cross-code
difference, repeat spread, busy threads per core (from a `ps -eLo` snapshot every
10 s), energy.

`status` says only whether a run executed: `ok`, `failed`, `timeout`,
`memory_watchdog`, `killed`, `not_started`, `started`, `refused`. Failure reasons
carry a class, `solver` or `harness`.

### Caveats

The report prints these with every campaign: times and memory are observations of
one machine, stack and campaign; work time excludes startup; memory counts shared
pages per rank; Palace phases with several ranks are an upper bound; nominal Krylov
tolerances differ between codes; Palace errors carry the definition offset δ of its
lumped ports; each code runs at its defaults except the pins above.

## Cases

| case | problem | DOF (order) | mesh | reference |
|---|---|---|---|---|
| `microstrip` | driven, 2 lumped 50 ohm ports, ABC order 1, {1, 25, 50, 75, 100} GHz, both ports excited, 2x2 S | 315 870 (p2) | committed, um, voided | gfem Richardson over a split ladder (not committed yet) |
| `lcavity` | eigen, PEC L-shaped cavity, target 100 GHz, 1 mode | 246 416 (p2) | committed, mm | closed form, f1 = 148 140 269 652.022 Hz |
| `bragg` | eigen, Bloch periodic, $k = \pi/a$ along $x$, 2 modes; a physics case | 10 620 (p3) | committed, mm, R1 mesh split once | transfer-matrix roots, ceiling 1e-5 rel |
| `r10M_void` | driven, 50 GHz, port 1 only, S11 and S21 | 10 248 856 (p2) | external, um | none (`accuracy: null`, `no_reference`); DOF pinned |

Meshes whose xz file is at most 1 MiB are committed as `cases/<case>/mesh.msh.xz`;
larger ones live in mesh stores as `<sha256>.msh.xz`. A case's problem does not
change across Palace versions. Wall caps (in the campaign files): bragg 120 s,
lcavity and microstrip 600 s, r10M_void 1 800 s. The `est_s` hints of a case file
feed only the plan's estimate and are left out of the case's definition hash.

## Converged references

Each case directory holds one `reference.json` (`suite-reference/1`): `id`
(`<case>-<YYYYMMDD>-<sha7>`), `physics_sha256` (what the scorer matches), `meshes`
(per mesh sha256 its DOF per order: the DOF guard's pins), `method`, `quantities`
with their `bound`, `ceiling` and `same_mesh_error` per order, `cross_checks`,
provenance. A reference is replaced only by a file with a new id and `computed_utc`;
the old one stays archived in the records store under `references/<id>/`.

- `python -m benchmarks reference lcavity` (also `bragg`, `r10M_void`) writes the
  file from the closed forms (40 digits, `decimal`). Rerunning it leaves an
  unchanged file as it is.
- microstrip is a ladder of sibling case files under `references/microstrip/` (L0 the
  pinned mesh, L1 and L2 its uniform splits, external meshes). Plan a stage, run it as
  a campaign, then extrapolate the fetched records:

  ```sh
  python -m benchmarks reference microstrip --plan pilot --site SITE --out RUNDIR
  python -m benchmarks reference microstrip --plan l2 --site SITE --out RUNDIR
  python -m benchmarks reference microstrip --plan palace-0.19.0 --site SITE --out RUNDIR
  python -m benchmarks reference microstrip --from RUNDIR [RUNDIR ...]
  ```

  The pilot's rate estimate decides whether `l2` is worth running (rate $\ge 1$). The
  Palace stage measures the definition offset δ, stored in `cross_checks` and never
  averaged in. Without L2 records, `--from` prints the pilot's rate and writes
  nothing.

Scoring (`close`, once per `ok` record): S is $\max |S_{ij} - S_{\mathrm{ref},ij}|$
over entries and frequencies; eigenfrequencies
$\max_k |\mathrm{Re}\, f_k - f_{\mathrm{ref},k}| / f_{\mathrm{ref},k}$ with modes paired
in ascending real part at or above the target. An error below $10\times$ the bound is
"unresolved" and printed as such. A mismatched reference gives `accuracy: null` and
`reference_matches: fail`.

## Records and the store

### Suite records

One `suite-record/1` JSON per planned run, at
`records/<case>/<code>-<kind>__<stack>__<layout>[__tol<t>]__rep<n>.json` (`layout`
is `default` or the resolved `RxT`; `__tol<t>` marks an off-anchor rung). Blocks:
`schema`, `campaign`, `name`, `run`, `status`, `reason`, `failure_class`,
`slurm_state`, `returncode`, `case` (definition and mesh sha256, order, ndof,
reference id and sha256), `outputs`, `accuracy`, `same_code_error`, `timings`,
`memory`, `checks`, `structural`, `stack`, `provenance`. Every planned row leaves a
record, refused and killed runs included. A rerun moves the earlier file aside as
`NAME.superseded-N.json`; superseded records never reach the store or a report.

Records are scrubbed when written: site roots become placeholders (`{cluster.root}`,
`{stacks.root}`, `{data.dir}`, `{output.dir}`, ...), and any account, user name or
remaining home or `/gpfs` path fails the scrub as a `harness` failure. Logs keep the
unscrubbed text and stay in the run directory.

### Campaign manifest

`campaign.json` (`suite-campaign/1`) is written by `plan` and completed by `close`.
Its id is `YYYYMMDD-<sha7>-<name>`, with `-2`, `-3` when the run directory's parent
already holds that id. Pins: gfem commit and its UTC date, harness commit, one
version per solver, per stack the description and env-manifest sha256. Also the
campaign file and its sha256, machine, core budget, `planned_utc`; after `close`,
`jobs` (Slurm ids, QOS, elapsed, billed core-h), `records` (path to sha256) and the
campaign `checks`. `plan` and `submit` refuse to add to a run directory at another
commit or solver version; `report` refuses a mixed campaign unless `--allow-mixed`.

### Schema versions

Majors bump only on an incompatible change (rename, removal, changed meaning); adding
an optional key keeps the major. One loader per major; an unknown major is refused
with its path; stored records are never rewritten; unknown keys of a known major are
kept. The typed contract is the dataclasses in `harness/records.py`
(`to_json`/`from_json`); there is no JSON Schema file. Examples:
`tests/unit/data/bench/record.example.json` and `campaign.example.json`.

### Records store

Suite records and campaign manifests are kept on `bench-data`, an orphan branch of
the gfem repository, never in main history (ADR 0402). Create its worktree once,
next to the gfem checkout:

```sh
git worktree add --orphan -b bench-data ../gfem-bench-data
```

Layout: `campaigns/<id>/{campaign.json, records/<case>/*.json,
jobs/<jobid>/{node.txt, pip-freeze.txt, machine.json}}`, `references/<id>/` (a copy
of each committed `reference.json`, by id) and `reports/<report-id>/`. There is no
index file: the trend axis is the gfem commit, ordered by its commit date.

After a campaign is closed on the cluster, copy its store set into the worktree:

```sh
python -m benchmarks fetch --from HOST:ROOT/campaigns/<id> \
    --to ../gfem-bench-data/campaigns/<id> --records-only
```

`--records-only` writes exactly `campaign.json`, `records/<case>/*.json` (superseded
records left out) and `jobs/<jobid>/{node.txt, pip-freeze.txt, machine.json}`; logs
stay in the run directory and in the full local copy (`fetch` without
`--records-only`). It exits 1 if a fetched file still names a home, scratch or
`/gpfs` path. Commit by hand in the worktree; the CLI never commits or pushes, and
pushing `bench-data` is a separate decision.

## Software stacks

One TOML per stack in `stacks/`: `schema = 1`, `name`, `kind`, `closed`,
`[versions]`, and per code `[gfem]` and `[palace]` with `modules`, `env`, `mpi`,
`[<code>.threads]` (`"threads"` = T, or a literal) and `caps`; Palace also
`launcher` and `entry`; optional `launch_r1`. A stack file never holds a path; the
site file says where its parts live.

- `wheels`: pip wheels installed offline. gfem's env is a hard-link copy of the site's
  base venv with the shipped commit's wheel, the MPI PETSc wheels and mpi4py, at
  `{cluster.root}/envs/bench-<sha7>-<stack>-<fp8>/`; Palace's is one venv per version
  at `{cluster.root}/envs/palace-<version>/`, from `palace_solver-<version>-*.whl`.
  The `ngsolve` and `petsc` versions are checked against `pip freeze`, not chosen.
  Palace defaults to 0.19.0; a campaign may override it
  (`[stacks.wheels.versions] palace = "0.18.1.post5"`). Building the gfem env takes
  about 12-15 minutes per commit on the login node.
- `native-oss`, `native-intel`: prefixes built by `tools/stacks/` (ADR 0191). They
  validate and refuse to launch while a prefix lacks `provenance.json` or the
  toolchain hashes differ (`plan --stack native-oss` exits 2 today).

`<fp8>` is the first 8 hex digits of the sha256 over the stack file, resolved versions
and resolved site inputs. The report flags a stack with two fingerprints in one
campaign. The harness never writes `LD_LIBRARY_PATH`.

A stack check (one job, logs only, no records) settles a stack's one-rank launch path
and its OpenBLAS thread rule before the first campaign on a new stack or Palace
version. Its outcome is a commit to the stack file, so the stack's hash pins it.

## Site file

The site file says where and as whom campaigns run. Copy `site.example.toml` outside
the repository, fill it in, and pass `--site FILE` or set `$GFEM_BENCH_SITE`. Never
commit a real one. `plan` copies it to `RUNDIR/site.toml`, which later subcommands
read by default. Keys: `name` (machine name in records); `[slurm]` `account`,
`constraint`, `[slurm.qos]` `debug` (jobs up to 2 h) and `long`; `[cluster] root`;
`[stacks] root`; `[stacks.wheels]` `base_venv`, `wheel_dirs`, `petsc_wheelhouse`,
`palace_wheelhouse`, `python`; `[data] dir` (staged meshes) and `stores` (mesh
stores, searched in order); `[ship] host` and `mesh_store`; `[output] dir`;
`[budget] ceiling_core_h` and `prior_core_h`; optional `[env]`. `{root}` is replaced
by `[cluster] root`.

## Running a campaign

Budget unit: billed core-hours, sacct $\mathrm{CPUTimeRAW} / 2 / 3600$ (one GPP node-hour =
112 core-h).

1. **Ship** a committed revision from the workstation:

   ```sh
   python -m benchmarks ship REV --to HOST:ROOT --site SITE [--store DIR] [--dry-run]
   ```

   `git archive REV` into `ROOT/refs/<sha7>/` with `COMMIT`, `COMMIT_UTC` and the
   gfem wheel (`uv build --wheel`); skipped when `ROOT/refs/<sha7>/dist/gfem-*.whl`
   already exists. External meshes are verified from the
   local mesh stores (`--store`, repeatable) and copied to `[ship] mesh_store`. Only
   committed content ships.

2. **Plan** on the login node, inside the shipped tree:

   ```sh
   python -m benchmarks plan benchmarks/campaigns/smoke.toml --site SITE \
       --out OUT/campaigns/<id>
   ```

   This writes `RUNDIR/{campaign.toml, COMMIT, COMMIT_UTC, plan.tsv,
   jobs/<job>.steps, campaign.json, site.toml}` and prints the jobs, runs, estimate,
   time limits and QOS. `--case`, `--solver`, `--stack`, `--layout` and `--repeats`
   replace an axis; `--list` prints the axis values. By default one job is one case x
   stack on one exclusive node; `[jobs] group = "stack"` makes one job per stack. The
   time limit is 1.5x the estimate; QOS `debug` up to 2 h, else `long`. Planned
   today: `smoke` 3 jobs / 149 runs / ~469 core-h, `smoke-r10m` 1 job / 4 runs / ~50,
   `release-gfem` 1 job / 23 runs / ~46 (gfem petsc-mumps and bddc on bragg, a
   Bloch-periodic eigenproblem, are planned once as `refused`).

3. **Submit** from the same tree:

   ```sh
   python -m benchmarks submit RUNDIR [--job J] [--qos Q] [--time H:MM:SS] \
       [--ceiling C] [--dry-run]
   ```

   `submit` refuses when `prior_core_h` + billed core-h of every `submitted.tsv`
   under `[output] dir` + this estimate exceeds `ceiling_core_h` (or `--ceiling`).
   A job still pending or running counts at its $\text{time limit} \times \text{nodes} \times 112$ until it
   finishes.
   It then prepares the stacks, stages the meshes, `sbatch`es
   `benchmarks/job.sbatch RUNDIR JOB` per job and appends the ids to
   `RUNDIR/submitted.tsv`. `--dry-run` prints the budget line and the sbatch lines
   only. `--time H:MM:SS` replaces the planned time limit of every job submitted,
   for a rerun after a job ran out of time (its remaining lines `not_started` with
   `job_time_left`): the estimate becomes the new limit / 1.5 and, without `--qos`,
   the QOS follows the new limit (`debug` up to 2 h, else `long`). `submitted.tsv`
   records the limit and estimate used; `plan.tsv` keeps the planned ones. Each job
   writes its machine facts once (`jobs/<jobid>/`, including a
   `tools/scale_bench.py node` machine benchmark), then runs
   `python -m benchmarks step RUNDIR JOB LINE` for each steps line.

4. **Status and close:**

   ```sh
   python -m benchmarks status RUNDIR
   python -m benchmarks close RUNDIR
   ```

   `status` prints per job its Slurm state and its records by status. `close` turns
   `started` stubs into `killed` and missing rows of ended jobs into `not_started`,
   scores `ok` records once against their case's `reference.json`, fills the
   manifest (jobs with billed core-h, record hashes, campaign checks), prints the
   number of `harness` failures, and exits 1 only when `complete`, `provenance` or
   `dof_equal` fails.

5. **Fetch** to the workstation: `fetch --from HOST:RUNDIR --to DIR` gives the full
   copy with logs, and `--records-only` into the store worktree gives the store set
   (see "Records store"). Commit there by hand.

6. **Report** (next section).

Without Slurm, `python -m benchmarks run RUNDIR --local [--job J]` runs the planned
lines on the current machine (gfem at one rank under taskset, Palace under Hydra;
gfem with several ranks is refused). Its records are `machine: local`.

**Release checklist:** on the release commit, ship it and run
`benchmarks/campaigns/release-gfem.toml` (plan, submit, close, fetch
`--records-only`, report); `close` must exit 0, and timing regressions are read off
the report's trend, never gated.

## Report

```sh
uv run --group docs python -m benchmarks report CAMPAIGN [CAMPAIGN ...] --out DIR \
    [--allow-mixed] [--rescore CASE=ID] [--no-figures]
uv run --group docs python -m benchmarks report --check DIR
```

The report writes `index.md` (first line: an HTML comment with the inputs digest),
`fig-*.svg`, `report.json` (`suite-report/1`) and `inputs.sha256` (`sha256  relpath`
per input, sorted). The report id is `<newest campaign date>-<digest7>` (`trend-`
prefix across several gfem commits) and is printed. `DIR` is any directory; by
convention `../gfem-bench-data/reports/<report-id>`.

- **Inputs.** Only the campaign manifests, the records they list (each sha256-checked)
  with the per-job `machine.json`, and the archived references of the store. No logs,
  no clock, no gfem import.
- **Campaign ids.** A `CAMPAIGN` argument is a directory holding `campaign.json`, or
  an id looked up as `campaigns/<id>` first in the store two levels above `--out`
  (`<store>/reports/<x>` gives `<store>`), then in `../gfem-bench-data` beside the
  repository.
- **Store root.** Every input path is relative to the store root: `<root>` when a
  campaign directory is `<root>/campaigns/<id>`, else the campaign directory's
  parent. References are read from `<root>/references/<id>/reference.json`. All
  campaigns of one report must share a store root. `report.json` keeps the store root
  relative to `DIR`, so a report directory can be renamed within `reports/`.
- **Refusals** (exit 1): a record whose sha256 differs from its manifest, an unknown
  schema major, a mixed campaign (unless `--allow-mixed`, then stamped MIXED), and
  `run --local` records mixed with cluster records. A local-only report is titled
  "local, not comparable".
- **`--rescore CASE=ID`** scores CASE again against archived reference ID, beside the
  stored scores; records are never rewritten.
- **`--no-figures`** writes tables and JSON with the standard library only.
- **`--check DIR`** regenerates `DIR`'s report into a temporary directory, from the
  inputs and flags its `report.json` names, and compares bytes. It prints
  `DIR: unchanged` and exits 0, or names the files that differ and exits 1. Figures
  made with another matplotlib version, or with none importable, are regenerated but
  not compared ("figures regenerated with matplotlib X ... not compared"); tables and
  JSON must still match.

Sections: gates and failures, then per case the structural block, error against work
time, phase bars, memory, accuracy, layouts and a cross-solver snapshot; a trend
section with at least two gfem commits; caveats. A job whose machine benchmark is
more than 10 % below the campaign median is flagged "slow node".

## Extending the suite

**A case.** Write `cases/<name>/case.toml` by hand, or import a gsim Palace config
once:

```sh
python -m benchmarks case import CONFIG.json MESH --name NAME [--frequencies-ghz F1,F2] [--site SITE]
python -m benchmarks case pin NAME MESH --unit U [--site SITE]
```

`case pin` hashes the mesh, writes `[mesh]` and `[regions]`, and stores the bytes by
the size rule (beside the case, or in the first mesh store of `--site`). It refuses a
case that already has a pin. Keep the original inputs in `source/`. Then add
`[est_s]` hints, a wall cap in each campaign file that lists the case, and a
`reference.json` (at least `method: "none"` with the DOF pin per order, or the DOF
guard has nothing to check). Mesh recipes are provenance only; generators live in
`tools/bench_meshes.py` and the harness never re-runs them.

**A stack.** Add `stacks/<name>.toml` of an existing kind and its name to `STACKS`
in `harness/config.py`. A new kind is a module in `stacks/` with `inputs`,
`activation`, `refusal` and `prepare`, registered in `stacks.KINDS`. A thread value
other than `"threads"` needs the build fact cited beside it.

**A solver adapter.** Write `adapters/<code>.py` implementing the `Adapter` protocol
of `adapters/__init__.py` and add one line for it to `ADAPTERS`. The protocol holds
what the harness needs of a code: `code`, `kinds`, `direct_kind` (what its same-code
error is measured against), `pins` (settings pinned away from its defaults, listed
in the report), `timing_source` and `memory_metric` (where its timings and own peak
memory come from), `stack_keys` (what its stack section must set), `supports`,
`default_layout`, `version`, `prepare` and `collect`; it never starts a process. The
`code:kind` names, the stack sections a stack file may hold, the versions a stack
may pin and the solver versions a campaign records all derive from `ADAPTERS`. Then
add a `[<code>]` section to each stack file that runs it (a stack without one
refuses at plan; a version under `[versions]` is optional). On a `wheels` stack the
kind must also learn to build the code's env. List its pinned settings in this
README.
