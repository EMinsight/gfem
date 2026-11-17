# Native software stacks

Developer tool, not for gfem users. These scripts build Palace, NGSolve (with
netgen), PETSc/SLEPc, OpenBLAS and gfem's stack venv from source on a Slurm
cluster (MareNostrum 5 GPP), in two native software stacks:

- `native-oss`: GCC 12.3, Open MPI 4.1.6 + UCX 1.16, self-built OpenBLAS 0.3.34;
- `native-intel`: GCC 12.3, MKL 2025.2 (`mkl_rt`, GNU layers), Intel MPI 2021.16.

The `wheels` software stack (the PyPI wheels) needs no build. Each package goes
into its own prefix with a provenance file, so a benchmark record can say
exactly what it ran on.

## Commands

```
export STACKS_ROOT=/path/to/stacks         # required, no default

stack.sh fetch  PKG [VERSION]              # login node: fill $STACKS_ROOT/src (network only)
stack.sh submit [--rebuild] [--allow-dirty] STACK PKG [VERSION] [-- SBATCH_ARGS...]
                                           # login node: sbatch build.sbatch
stack.sh build  [--rebuild] [--allow-dirty] STACK PKG [VERSION]
                                           # inside the job only (needs SLURM_JOB_ID), offline
stack.sh check  [STACK [PKG [VERSION]]]    # read-only
```

- `VERSION` defaults to the recipe's `VERSION`. `<DEP>_VERSION` in the
  environment (`OPENBLAS_VERSION=0.3.35`) picks a dependency's version.
- `fetch` is shared by every software stack and rerunnable.
- `submit` names the job `stack-<stack>-<pkg>-<version>`, uses the recipe's
  `CPUS` and `TIME`, QOS `$STACK_QOS` (default `gp_ehpc`) and
  `--hint=nomultithread`, writes the log to
  `$STACKS_ROOT/logs/<stack>-<pkg>-<version>-<jobid>.out`, and passes the
  arguments after `--` straight to `sbatch` (later ones win:
  `-- --qos=gp_debug --time=01:00:00`). It refuses when a job of that name is
  pending or running (`squeue --name`, no lock files), does nothing for a prefix
  that is complete with this recipe and toolchain, and chains on a dependency's
  pending build with `--dependency=afterok`.
- `check` with no argument checks every software stack under `$STACKS_ROOT`;
  with `STACK` it also prints the plan (state, version, dependencies,
  resources) of each package of that stack. It fails when the complete
  prefixes of one software stack name different toolchain hashes, or when
  their `mumps` blocks break the rules below.
- `STACK_PYTHON` names the Python that runs `provenance.py` (default
  `python3`, 3.9 or newer, standard library only).

The scripts run on the cluster: nothing here uses ssh, rsync or the tunnel.
The local harness ships `tools/stacks/` at a committed SHA to
`$STACKS_ROOT/recipes/<sha>/` (`git archive`, with the commit in a `COMMIT`
file or `$STACKS_RECIPE_COMMIT`) and runs `stack.sh` there.

## Layout

```
tools/stacks/
  stack.sh                  the one entry point
  build.sbatch              the Slurm job: runs `stack.sh build`
  provenance.py             writes, reads and compares provenance files
  toolchains/<stack>.sh     one per native software stack: the only place a toolchain lives
  packages/<pkg>.sh         one recipe per package

$STACKS_ROOT/
  src/{mirrors,tarballs,patches,pip,python}/   source mirror, from fetch
  src/tarballs/SHA256SUMS                      sha256 of unpinned tarballs (first fetch)
  recipes/<sha>/                               tools/stacks/ as shipped
  logs/<stack>-<pkg>-<version>-<jobid>.out
  <stack>/
    toolchain.sh                               frozen copy of toolchains/<stack>.sh
    <pkg>-<version>/                           bin/ lib/ include/ share/ ...
      provenance.json                          written last: the prefix is complete
      activate.sh
      share/stack/{build.log, recipe.tar, ...}
    .failed/<pkg>-<version>-<jobid>/           logs of a failed build
    .history/<pkg>-<version>-<utc>.json        provenance of a replaced build
```

## Build rules

- A prefix is complete iff `provenance.json` exists. `build` on a complete
  prefix with the same recipe hash and toolchain hash is a no-op. The recipe
  hash is per package: `stack.sh`, `build.sbatch`, `provenance.py` and
  `packages/<pkg>.sh`, by name and content. The README, other packages'
  recipes and `COMMIT` are not in it, so filling in one recipe leaves the
  other packages' prefixes complete. The toolchain has its own hash. A prefix without
  `provenance.json` is a killed job's leftovers and is wiped. A complete prefix
  built from another recipe or toolchain is refused unless `--rebuild`, which
  moves its provenance file to `.history/` and wipes the prefix (installs are
  not relocatable).
- A recipe tree in a git checkout with uncommitted changes under
  `tools/stacks/` is refused unless `--allow-dirty`, which is then recorded.
- The first build of a software stack freezes `toolchains/<stack>.sh` as
  `$STACKS_ROOT/<stack>/toolchain.sh`. A later build whose toolchain file
  differs is refused: a changed toolchain is a new software stack.
- The build tree is `$TMPDIR/stack-<stack>-<pkg>-<version>` (node-local in a
  job, never GPFS, never `/tmp`); the install goes straight into the prefix.
- The order of a build: `module purge` and the toolchain (the frozen file),
  each dependency's `activate.sh`, `build_<pkg>`, `activate.sh` written and
  sourced, `smoke_<pkg>`, then `provenance.json`. A failure anywhere copies
  the logs to `.failed/` and leaves the prefix incomplete.

## Toolchain files

`toolchains/<stack>.sh` is sourced (not run) by `build` and by every
`activate.sh` of the stack. It does `module purge`, then loads its modules one
per line, and exports:

| Variable | Meaning |
|---|---|
| `STACK_NAME` | the software stack |
| `CC`, `CXX`, `FC`, `F77` | GCC 12.3 compilers |
| `MPICC`, `MPICXX`, `MPIFC` | the stack MPI's wrappers |
| `CFLAGS`, `CXXFLAGS`, `FFLAGS`, `FCFLAGS` | `-O3 -march=sapphirerapids` |
| `LDFLAGS` | rpath to GCC 12.3's runtime (`STACK_GCC_LIBDIR`) |
| `STACK_FORTRAN_COMPAT_FLAGS` | `-fallow-argument-mismatch`, for recipes that need it |
| `STACK_BLAS_VENDOR`, `STACK_BLAS_VERSION`, `STACK_BLAS_LINK`, `STACK_BLAS_THREADING` | BLAS/LAPACK |
| `STACK_BLAS_PACKAGE`, `STACK_BLAS_LIB_PTHREADS`, `STACK_BLAS_LIB_OPENMP` | native-oss: the `openblas` package and its two libraries |
| `MKL_INTERFACE_LAYER`, `MKL_THREADING_LAYER` | native-intel: `GNU,LP64` and `GNU` |
| `STACK_SCALAPACK` | reference 2.2.3, built by each code |
| `STACK_MPI_VENDOR`, `STACK_MPI_VERSION` and runtime variables | `OMPI_MCA_pml`, `UCX_TLS`; `I_MPI_FABRICS`, `FI_PROVIDER` |

It sets no thread counts (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NGS_NUM_THREADS`): those belong to whoever runs. A new
native software stack is one new toolchain file.

## Recipe interface

`packages/<pkg>.sh` is sourced by `stack.sh` with `STACK_NAME` set to the
software stack (empty for `fetch`). It holds plain assignments at the top,
one per line (the local harness reads them with grep), and three functions:

```bash
VERSION=0.19.0              # default version, the pin
CPUS=56                     # sbatch --cpus-per-task default
TIME=03:00:00               # sbatch --time default
STACKS=(native-oss native-intel)        # software stacks the package is built on
case ${STACK_NAME:-} in                 # dependencies, per software stack
native-oss) DEPENDS=(openblas) ;;
*) DEPENDS=() ;;
esac

fetch_palace() {            # VERSION
  ...                       # login node, network only, into $STACK_SRC/...
}
build_palace() {            # VERSION PREFIX BUILD_DIR
  ...                       # in the job, offline; install into PREFIX;
                            # write $STACK_META/source.json and configure.json
}
smoke_palace() {            # VERSION PREFIX
  stack_smoke palace --version          # runs with the prefix's activate.sh sourced
}
activate_palace() {         # PREFIX (optional): extra lines for activate.sh
  echo "export PALACE_DIR=\"$1\""
}
```

- A function name is `<verb>_<pkg>`. A package that is not yet written keeps
  stubs that call `stack_die`.
- Dependency pins of a superbuild (Palace's externals, NGSolve's submodules,
  PETSc's `--download` pins) are read from the fetched tag, not repeated here.
- `build_<pkg>` output goes to `build.log`; it may also copy CMake caches or
  configure logs into `PREFIX/share/stack/`.
- What a recipe can use from `stack.sh`:

| Name | Meaning |
|---|---|
| `$STACKS_ROOT`, `$STACK_SRC` | install root, source mirror (`$STACKS_ROOT/src`) |
| `$STACK_HOME` | the recipe tree being run |
| `$STACK_NAME` | the software stack (build) |
| `$STACK_META` | directory of provenance fields, one `<field>.json` each (build) |
| `$STACK_JOBS` | the job's CPUs, for `make -j` (build) |
| `stack_meta FIELD key=string key:=json ...` | write the provenance field `FIELD` as a JSON object |
| `stack_json FILE key=string key:=json ...` | write any JSON object |
| `stack_smoke COMMAND...` | run the smoke command; record it and its first output line |
| `stack_tofu FILE` | sha256 of an unpinned tarball: recorded on first fetch, checked after |
| `stack_die MESSAGE` | fail |

## Provenance file (schema 1)

`<prefix>/provenance.json`, one JSON object. `stack.sh build` assembles it from
`$STACK_META/<field>.json` with `provenance.py write`; `provenance.py` refuses
a record that lacks a field.

| Field | Written by | Content |
|---|---|---|
| `schema`, `prefix`, `complete_utc` | `provenance.py` | `1`, the prefix, the completion time |
| `package`, `version`, `stack` | `stack.sh` | |
| `source` | recipe | upstream URL, tag, commit; each bundled dependency with version and commit or tarball sha256 |
| `configure` | recipe | the CMake options or configure line passed |
| `smoke` | recipe (`stack_smoke`) | command and first output line, plus any recipe checks |
| `mumps` | Palace and PETSc recipes | see below |
| `toolchain` | `stack.sh` (`provenance.py toolchain`) | `sha256` of the frozen toolchain file, `module -t list`, compilers with `--version`, flags, BLAS, ScaLAPACK, MPI with `mpicc -show`, `cmake --version` |
| `recipe` | `stack.sh` | gfem commit, the package's recipe hash, `dirty` |
| `build` | `stack.sh` | job id, host, QOS, CPUs, start, end, elapsed, CPUTimeRAW so far |
| `depends` | `stack.sh` | each dependency's prefix and the sha256 of its provenance file |

The `mumps` block (`version`, `tarball_sha256`, `intsize`, `gemmt`,
`blas.{vendor,version,threading}`, `scalapack`, `compiler`, `fflags`,
`arithmetics`, `openmp`, `orderings`, `metis`) is compared across the
complete prefixes of a software stack by `check`:

- must match: `version`, `tarball_sha256`, `intsize`, `scalapack`,
  `compiler`, `fflags`, `blas.vendor`, `blas.version`;
- must hold in each build: `intsize` 32, `gemmt` false (no `?gemmt_` import),
  `metis` among the `orderings`, no `parmetis` in PETSc's;
- allowed differences, printed every time: `arithmetics`, `openmp`,
  `orderings`, `metis` (patch level), `blas.threading`.

## Using a stack

```bash
. $STACKS_ROOT/native-oss/petsc-3.26.0/activate.sh
```

`activate.sh` sources the frozen toolchain file (once per shell) and each
dependency's `activate.sh`, prepends the prefix's `bin`, `lib`, `lib64` and
`pkgconfig` directories to `PATH`, `LD_LIBRARY_PATH`, `PKG_CONFIG_PATH` and
`CMAKE_PREFIX_PATH`, and exports `STACK_NAME` and
`STACK_<PKG>_PROVENANCE=<prefix>/provenance.json`, plus what the recipe's
`activate_<pkg>` adds. It sets no thread counts and no `PYTHONPATH`, and
activates no venv.
