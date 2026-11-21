#!/usr/bin/env bash
# Build a PETSc stack that gfem's PETSc and SLEPc solvers use: PETSc and SLEPc
# with complex scalars, with their Python bindings petsc4py and slepc4py, as a
# wheelhouse (a folder of four wheels) to install into the environment that
# runs gfem. The default stack has no MPI and serves one process, or copies of
# one process per rank. With --mpi it is built on the machine's Open MPI and
# also serves distributed solves. --with-parmetis adds ParMETIS to the MPI stack,
# for users its licence allows (see parmetis_terms): such a stack is for its
# builder alone and is never shared.
#
# Usage:
#   bash petsc_wheelhouse.sh [--mpi [--with-parmetis]] [--sources DIR] OUTPUT_DIR
#                                   build the four wheels into OUTPUT_DIR (a new
#                                   or empty folder)
#   bash petsc_wheelhouse.sh [--mpi [--with-parmetis]] [--sources DIR] --check
#                                   check the prerequisites only, and print the
#                                   configure options
#   bash petsc_wheelhouse.sh [--with-parmetis] --fetch-sources DIR
#                                   on a machine with internet: download into DIR
#                                   what --sources reads on one without it
# then, in the environment that runs gfem:
#   pip install --no-index --find-links OUTPUT_DIR petsc petsc4py slepc slepc4py
#
# Prerequisites: Linux (macOS is not tested), uv, curl (not with --sources), make,
# cmake, and the system's C, C++ and Fortran compilers gcc, g++ and gfortran (the
# script takes an active conda environment off PATH). With --mpi also Open MPI's
# mpicc, mpicxx and mpifort on PATH, bison 3 or newer, flex, and cmake 3.26 or
# newer. The script downloads the four source distributions from PyPI, and
# PETSc's configure downloads METIS and OpenBLAS, with --mpi also ScaLAPACK;
# the script fetches MUMPS, which it patches (see patch_mumps), and Scotch
# itself. All of them are compiled from source, and --sources reads them from
# DIR instead. It is a long build, done
# once per Python version. A stack runs under the Open MPI it was built on.
#
# Target Python: the bindings are compiled for one CPython version (3.12 or
# newer, not built by conda), so build with the Python of the environment
# that runs gfem. The script uses the Python that `uv python find` gives:
# the active virtual environment's, or the .venv of the project in the
# current folder, if any.
# Set PYTHON to choose another, as a path or a version, for example
# PYTHON=/path/to/venv/bin/python or PYTHON=3.12.
#
# MAKE_NP sets the number of parallel compile jobs (default: PETSc's own
# choice from the number of cores).
set -euo pipefail
# make runs each recipe through /bin/sh, which is bash on many Linux systems,
# and every such bash runs the file BASH_ENV names: one that runs `conda init`
# puts conda's compilers back on PATH inside the build, and conda's linker
# cannot link what the system's gcc compiled. No child shell reads one.
unset BASH_ENV ENV

# The release gfem's PETSc tests run on: gfem needs this minor or newer.
PETSC_VERSION=3.26.0
SLEPC_VERSION=3.26.0
# sha256 of each source distribution on PyPI
SHA256_petsc=f5230023e6e22ee607802a13c82bc25f3d81e71654ad386a5b9bdff17fed93df
SHA256_petsc4py=0a34bd083e24bee1a6759c4639a45d10c0117855eb9f0466834450245f75d7e5
SHA256_slepc=a2f4cc2af76d55c078c30ad8bc66b44736dac921a7912266eb44136fc1b6029d
SHA256_slepc4py=07bf970c861736b9b977dfecd4ad46415d6c4a6f26b0ed128ecffae59613d6ff

# Each tarball PETSc's configure reads, with its package: name, URL, sha256. The
# stack holds what these build and nothing else, the set the licence page names.
TARBALLS=(
  "MUMPS_5.9.1.tar.gz https://mumps-solver.org/MUMPS_5.9.1.tar.gz 659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46"
  "scalapack-v2.2.3.tar.gz https://github.com/Reference-ScaLAPACK/scalapack/archive/v2.2.3.tar.gz 5d93701eca663925e98010dd8d0f45fd79b2191d74e5afa5711d587370a8b9dd"
  "scotch-v7.0.18.tar.gz https://gitlab.inria.fr/scotch/scotch/-/archive/v7.0.18/scotch-v7.0.18.tar.gz 2cbdcf6da5a08ca31437d7c967b243b3c96d621409ecb710768de2aa2e09802d"
  "petsc-pkg-metis-v5.1.0-p12.tar.gz https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz 2f16abe87394d4fd1430d66a8995076c352af40f7a4d1c5300b3b1cc9d545663"
  "OpenBLAS-0.3.34.tar.gz https://github.com/xianyi/OpenBLAS/archive/v0.3.34.tar.gz cd7e129868320cc2d033afa920e31202dfe0b8066a5b66661900ccc0f197dfed"
)
# MUMPS (CeCILL-C), its ordering METIS (Apache-2.0), BLAS and LAPACK (BSD-3-Clause)
NO_MPI_PACKAGES=(mumps metis openblas)
# Also MUMPS's root ScaLAPACK (BSD-3-Clause) and its parallel ordering PT-Scotch
# (CeCILL-C). ParMETIS only with --with-parmetis: it is free only for non-profit
# research and education, so a stack with it is for its builder alone and is
# never shared.
MPI_PACKAGES=(mumps scalapack ptscotch metis openblas)
# ParMETIS's tarball, outside TARBALLS (the licence page's set): name, URL, sha256
PARMETIS_TARBALL="petsc-pkg-parmetis-v4.0.3-p9.tar.gz https://bitbucket.org/petsc/pkg-parmetis/get/v4.0.3-p9.tar.gz 612717e85992c984f09b0f5670be421bbb90a4c04145ab5b9a3358b92765d891"
# PETSc 3.26.0 pins Scotch 7.0.15, whose hdgraphFold2 crashes MUMPS's parallel
# analysis (fixed upstream in 7.0.18), so the MPI stack always takes this tarball.
SCOTCH_TARBALL=scotch-v7.0.18.tar.gz
# Both stacks build MUMPS from this tarball, after patch_mumps caps its ordering.
MUMPS_TARBALL=MUMPS_5.9.1.tar.gz

usage() {
  sed -n '/^# Usage:/,/^# then,/p' "${BASH_SOURCE[0]}" | sed '$d' | cut -c3- >&2
  exit 2
}

fail() {
  echo "petsc_wheelhouse.sh: $*" >&2
  exit 1
}

# ParMETIS's terms (its LICENSE.txt), on stderr. No prompt: builds run in batch
# jobs, and the flag is the explicit choice.
parmetis_terms() {
  cat >&2 << 'EOF'
warning: --with-parmetis adds ParMETIS, copyrighted by the Regents of the University of Minnesota. It can be freely used for educational and research purposes by non-profit institutions and US government agencies only; other organizations may use it for evaluation only, and any further use needs the University's prior approval. It may not be sold or redistributed without that approval. The wheels this build makes hold ParMETIS: they are for your own use and must not be shared or uploaded anywhere. Building with this flag is your acceptance of ParMETIS's terms.
EOF
}

# An active conda environment puts its own gcc, g++ and gfortran first on
# PATH. They search conda's include and lib folders by default, so the
# petsc4py and slepc4py builds compile and link against conda's PETSc,
# OpenBLAS and MPI, and give bindings that do not import. Take every folder
# of the environment, and of the conda installation, off PATH, so that the
# build uses the system's compilers and tools, and unset every other
# exported variable that names one of them: the activation's tools (AR, LD,
# NM, ...) and search paths (CMAKE_PREFIX_PATH, CMAKE_ARGS, ...) would pull
# conda in too. PYTHON and VIRTUAL_ENV stay: they choose the target Python,
# which the check then refuses if it is conda's. CONDA_ROOTS keeps those two
# folders for that check. The installation is the folder above CONDA_EXE's
# bin, if it holds conda-meta (not /usr for a system conda).
leave_conda() {
  CONDA_ROOTS=()
  [ -n "${CONDA_PREFIX:-}" ] || return 0
  CONDA_ROOTS=("${CONDA_PREFIX%/}")
  local base IFS=: entries entry root kept=() name
  base=$(dirname "$(dirname "${CONDA_EXE:-/}")")
  [ ! -d "$base/conda-meta" ] || CONDA_ROOTS+=("${base%/}")
  read -ra entries <<< "$PATH"
  for entry in "${entries[@]}"; do
    for root in "${CONDA_ROOTS[@]}"; do
      case $entry/ in "$root"/*) continue 2 ;; esac
    done
    kept+=("$entry")
  done
  PATH="${kept[*]-}"
  hash -r
  echo "warning: conda environment $CONDA_PREFIX is active: the build takes its folders off PATH, unsets the variables that name them, ignores its compilers and flags (CC, CFLAGS, LDFLAGS, ...) and uses the system's gcc, g++ and gfortran" >&2
  while IFS= read -r name; do
    case $name in PATH | PYTHON | VIRTUAL_ENV | [0-9]* | *[!A-Za-z0-9_]*) continue ;; esac
    for root in "${CONDA_ROOTS[@]}"; do
      case ${!name-} in *"$root"*)
        unset "$name" 2> /dev/null || true # readonly: left as is
        continue 2
        ;;
      esac
    done
  done < <(compgen -e)
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL
}

# What --mpi needs on top of the rest: refuse what is known not to work.
check_mpi_prerequisites() {
  local version
  version=$(mpicc --showme:version 2>&1) || true
  case $version in
    *"Open MPI"*) echo "mpicc: $(command -v mpicc) ($version)" ;;
    *) fail "mpicc is not Open MPI's ('mpicc --showme:version' says: $version): only Open MPI gives right numbers with NGSolve's wheel. Load Open MPI and run again" ;;
  esac
  mpicc --version | head -1 | grep -qi -E 'gcc|gnu' ||
    echo "warning: mpicc does not wrap gcc: OpenBLAS did not build under Intel's wrappers on MareNostrum 5, for a cause not known" >&2
  version=$(bison --version | head -1)
  case ${version##* } in
    [3-9].* | [1-9][0-9].*) ;;
    *) fail "PT-Scotch needs bison 3 or newer, not: $version" ;;
  esac
  version=$(cmake --version | head -1)
  version=${version##* }
  "$PY" -c 'import sys; sys.exit(tuple(map(int, sys.argv[1].split(".")[:2])) < (3, 26))' "$version" ||
    fail "ScaLAPACK needs cmake 3.26 or newer, not $version"
}

# Check every prerequisite, report all that are missing, and set PY to the
# target Python.
check_prerequisites() {
  local missing=() tool
  local tools=(uv curl)
  if [ -z "$FETCH_SOURCES" ]; then
    tools=(uv make cmake gcc g++ gfortran)
    [ -n "$SOURCES" ] || tools+=(curl)
  fi
  ((MPI == 0)) || tools+=(mpicc mpicxx mpifort bison flex)
  for tool in "${tools[@]}"; do
    command -v "$tool" > /dev/null || missing+=("$tool")
  done
  [ "$(uname -s)" = Linux ] || echo "warning: only Linux is tested, not $(uname -s)" >&2
  ((${#missing[@]} == 0)) || fail "missing prerequisites: ${missing[*]} (install them and run again)"
  PY=$(UV_NO_CONFIG=1 uv python find ${PYTHON:+"$PYTHON"}) ||
    fail "no Python found for PYTHON='${PYTHON:-}' (set PYTHON to the Python of the environment that runs gfem)"
  "$PY" -c 'import sys; sys.exit(sys.implementation.name != "cpython" or sys.version_info < (3, 12))' ||
    fail "$PY is not CPython 3.12 or newer, which gfem needs"
  # A conda Python, or a virtual environment made from one, builds the
  # bindings with its own compiler flags, which point at conda's headers and
  # libraries: the same failure as conda's compilers.
  "$PY" - ${CONDA_ROOTS[@]+"${CONDA_ROOTS[@]}"} << 'EOF' || fail "$PY is a conda Python: a conda-built Python's compiler flags point at conda's headers and libraries, and the bindings built with them do not import. Use a Python not built by conda, for example: uv python install 3.13, then uv venv --python-preference only-managed --python 3.13 and run gfem from that environment"
import os, sys
roots = [os.path.realpath(root) for root in sys.argv[1:]]
for prefix in {os.path.realpath(sys.prefix), os.path.realpath(sys.base_prefix)}:
    if os.path.isdir(os.path.join(prefix, "conda-meta")) or any(
        prefix == root or prefix.startswith(root + os.sep) for root in roots
    ):
        sys.exit(f"{prefix} is a conda environment")
EOF
  if [ -z "$FETCH_SOURCES" ]; then
    for tool in gcc g++ gfortran; do
      echo "$tool: $(command -v "$tool") ($("$tool" -dumpfullversion -dumpversion))"
    done
  fi
  echo "target Python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"
  ((MPI == 0)) || check_mpi_prerequisites
  [ -z "$SOURCES" ] || check_sources
}

# Raise unless a file's sha256 is the expected one.
check_sha256() {
  "$PY" - "$1" "$2" << 'EOF'
import hashlib, sys
path, expected = sys.argv[1:]
with open(path, "rb") as f:
    digest = hashlib.sha256(f.read()).hexdigest()
sys.exit(f"{path}: sha256 {digest}, expected {expected}" if digest != expected else 0)
EOF
}

# Every file a build from SOURCES reads, with its hash.
check_sources() {
  local name version sha256 line url
  [ -d "$SOURCES/wheels" ] || fail "$SOURCES/wheels is missing: --fetch-sources writes it"
  for line in "petsc $PETSC_VERSION $SHA256_petsc" "petsc4py $PETSC_VERSION $SHA256_petsc4py" \
    "slepc $SLEPC_VERSION $SHA256_slepc" "slepc4py $SLEPC_VERSION $SHA256_slepc4py"; do
    read -r name version sha256 <<< "$line"
    [ -f "$SOURCES/$name-$version.tar.gz" ] || fail "$SOURCES/$name-$version.tar.gz is missing"
    check_sha256 "$SOURCES/$name-$version.tar.gz" "$sha256"
  done
  for line in "${TARBALLS[@]}" ${WITH_PARMETIS:+"$PARMETIS_TARBALL"}; do
    read -r name url sha256 <<< "$line"
    case $name in
      scalapack* | scotch*) ((MPI == 1)) || continue ;;
    esac
    [ -f "$SOURCES/$name" ] || fail "$SOURCES/$name is missing"
    check_sha256 "$SOURCES/$name" "$sha256"
  done
  echo "sources: $SOURCES"
}

# The name of a package's tarball.
tarball() {
  local line
  for line in "${TARBALLS[@]}" "$PARMETIS_TARBALL"; do
    case $1:$line in
      mumps:MUMPS* | scalapack:scalapack* | ptscotch:scotch* | metis:petsc-pkg-metis* | parmetis:petsc-pkg-parmetis* | openblas:OpenBLAS*)
        echo "${line%% *}"
        return
        ;;
    esac
  done
}

# Set CONFIGURE_OPTIONS. A package comes from SOURCES when given, and MUMPS and
# PT-Scotch always from the tarballs the script checked (in ``$1``), MUMPS's
# patched.
set_configure_options() {
  local packages=("${NO_MPI_PACKAGES[@]}") package
  CONFIGURE_OPTIONS=(--with-scalar-type=complex) # gfem's operators are complex
  if ((MPI == 1)); then
    packages=("${MPI_PACKAGES[@]}")
    # 64-bit: the graph passes 2^31 near 54 M DOF at order 2
    CONFIGURE_OPTIONS+=(--with-64-bit-indices)
    if [ -n "$WITH_PARMETIS" ]; then
      packages+=(parmetis)
    else
      CONFIGURE_OPTIONS+=(--with-parmetis=0)
    fi
  else
    CONFIGURE_OPTIONS+=(--with-mpi=0) # one process: MUMPS on its serial libseq
  fi
  for package in "${packages[@]}"; do
    if [ "$package" = mumps ]; then
      CONFIGURE_OPTIONS+=("--download-mumps=$1/${MUMPS_TARBALL%.tar.gz}-gfem.tar.gz")
    elif [ -n "$SOURCES" ]; then
      CONFIGURE_OPTIONS+=("--download-$package=$SOURCES/$(tarball "$package")")
    elif [ "$package" = ptscotch ]; then
      CONFIGURE_OPTIONS+=("--download-ptscotch=$1/$SCOTCH_TARBALL")
    else
      CONFIGURE_OPTIONS+=("--download-$package")
    fi
  done
  CONFIGURE_OPTIONS+=(--download-openblas-use-pthreads) # OPENBLAS_NUM_THREADS sets the threads
  CONFIGURE_OPTIONS+=(--with-x=0)                       # no X11 graphics
  ((MPI == 0)) || CONFIGURE_OPTIONS+=(COPTFLAGS=-O2 CXXOPTFLAGS=-O2 FOPTFLAGS=-O2)
}

# Download one file and check its sha256.
download() {
  curl -fsSL --retry 3 -o "$3" "$1"
  check_sha256 "$3" "$2" ||
    fail "$1 changed: a host may rebuild its archives, so compare it by hand before trusting it"
}

# The sdist's URL on PyPI.
pypi_url() {
  echo "https://files.pythonhosted.org/packages/source/${1:0:1}/$1/$1-$2.tar.gz"
}

# Write into DIR what a build reads with --sources DIR.
fetch_sources() {
  local dir=$1 name url sha256 line version
  mkdir -p "$dir/wheels"
  dir=$(cd "$dir" && pwd)
  download "$(pypi_url petsc "$PETSC_VERSION")" "$SHA256_petsc" "$dir/petsc-$PETSC_VERSION.tar.gz"
  download "$(pypi_url petsc4py "$PETSC_VERSION")" "$SHA256_petsc4py" "$dir/petsc4py-$PETSC_VERSION.tar.gz"
  download "$(pypi_url slepc "$SLEPC_VERSION")" "$SHA256_slepc" "$dir/slepc-$SLEPC_VERSION.tar.gz"
  download "$(pypi_url slepc4py "$SLEPC_VERSION")" "$SHA256_slepc4py" "$dir/slepc4py-$SLEPC_VERSION.tar.gz"
  for line in "${TARBALLS[@]}" ${WITH_PARMETIS:+"$PARMETIS_TARBALL"}; do
    read -r name url sha256 <<< "$line"
    download "$url" "$sha256" "$dir/$name"
  done
  version=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  # the build tools of the four builds, for this Python on any glibc
  uvx pip download -d "$dir/wheels" --python-version "$version" --implementation cp \
    --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
    --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    --platform manylinux_2_5_x86_64 --platform manylinux1_x86_64 \
    setuptools wheel patchelf cython packaging numpy uv
  echo "sources: $dir"
}

# One source distribution, from PyPI or SOURCES, with its sha256 checked.
fetch() {
  local name=$1 version=$2 sha256=$3
  local sdist=$WORK/$name-$version.tar.gz
  if [ -n "$SOURCES" ]; then
    cp "$SOURCES/$name-$version.tar.gz" "$sdist"
    check_sha256 "$sdist" "$sha256"
  else
    download "$(pypi_url "$name" "$version")" "$sha256" "$sdist"
  fi
}

# Build one wheel into the wheelhouse. The wheelhouse itself provides the
# petsc and slepc a binding builds against; no cache, so a wheel built
# with other configure options is never reused.
build() {
  local name=$1 version=$2 start=$SECONDS
  echo "=== building $name $version"
  local offline=()
  [ -z "$SOURCES" ] || offline=(--no-index --find-links "$SOURCES/wheels")
  uv build --no-cache --wheel --python "$PY" --out-dir "$OUT" --find-links "$OUT" \
    ${offline[@]+"${offline[@]}"} \
    --build-constraints "$WORK/constraints.txt" "$WORK/$name-$version.tar.gz"
  echo "=== $name built in $((SECONDS - start)) s"
}

# PETSc's configure adds -DGEMMT_AVAILABLE to MUMPS when the BLAS has GEMMT, and
# MUMPS then sends every symmetric update through ZGEMMT. OpenBLAS's ZGEMMT, up to
# at least 0.3.34, computes each column's offset in a 32-bit int, which wraps on a
# complex front of 32,768 or more: wrong factors with no error, or a SEGV. Which
# OpenBLAS runs is not this build's to choose: libpetsc binds to NGSolve's by
# SONAME. Without the flag MUMPS uses its own blocked ZGEMM loop, on 64-bit
# offsets. configure has no option for this, so edit the sdist's MUMPS.py.
patch_petsc_sdist() {
  "$PY" - "$WORK/petsc-$PETSC_VERSION.tar.gz" << 'EOF'
import io, os, sys, tarfile

sdist = sys.argv[1]
line = b"if self.blasLapack.checkForRoutine('dgemmt'):"
hits = 0
with tarfile.open(sdist) as old, tarfile.open(sdist + ".patched", "w:gz") as new:
    for member in old:
        data = old.extractfile(member).read() if member.isfile() else None
        if member.name.endswith("config/BuildSystem/config/packages/MUMPS.py"):
            hits += data.count(line)
            data = data.replace(line, b"if False:  # gfem: OpenBLAS's ZGEMMT overflows")
            member.size = len(data)
        new.addfile(member, io.BytesIO(data) if data is not None else None)
if hits != 1:
    os.remove(sdist + ".patched")
    sys.exit(f"{sdist}: MUMPS.py's GEMMT check found {hits} times, expected once")
os.replace(sdist + ".patched", sdist)
EOF
}

# MUMPS orders with PT-Scotch on the largest power of two of processes up to
# min(NPROCS, N/16), with no control to cap it, and the factor's fill grows with
# that count: on a 30 M unknown coarse problem 8.9e9 entries on 32 processes,
# 1.1e10 on 64 and about 3e10 on 128, against 44-47 s of analysis at 32 for any
# number of ranks. So the stack caps PT-Scotch's ordering processes at 32 (not
# ParMETIS's, whose fill does not grow with them). MUMPS_PARAORD_MAX overrides
# the cap at run time, for measurement: 0 restores MUMPS's own count, N of 2
# and more caps at N, and 1 caps at 2 (MUMPS orders on every process when its
# count is 1); a value that is not a whole number of 0 and more keeps 32. Rank 0 writes a "PARAORD cap C nprocs P paraord W" line to MUMPS's
# output stream ICNTL(3) when ICNTL(4) is 2 or more. Edit src/tools_common.F
# (MUMPS_BUILD_COMM_PARA_ANA) at three lines, each found exactly once, into a
# repacked tarball beside the pristine one. A MUMPS bump fails the sha256 or
# that count; the edit is then derived again from the new source, or dropped if
# MUMPS has gained a control of its own. The name MUMPS_PARAORD_MAX is the
# marker check_paraord_cap reads in libpetsc: a new meaning takes a new name.
patch_mumps() {
  "$PY" - "$WORK/$MUMPS_TARBALL" "$WORK/${MUMPS_TARBALL%.tar.gz}-gfem.tar.gz" << 'EOF'
import io, os, sys, tarfile

pristine, patched = sys.argv[1:]
# Each anchor, and the lines that go after the line holding it.
edits = [
    (b"ROOT=0", b"""\
C     gfem: cap on PT-Scotch parallel-ordering processes, default 32;
C     MUMPS_PARAORD_MAX overrides it, 0 for no cap, 1 means 2
      CHARACTER(LEN=16) :: CAPSTR
      INTEGER :: CAPLEN, CAPST, CAPVAL, CAPIN
      CAPVAL = 32
      CALL GET_ENVIRONMENT_VARIABLE('MUMPS_PARAORD_MAX', CAPSTR,
     &     CAPLEN, CAPST)
      IF (CAPST .EQ. 0 .AND. CAPLEN .GT. 0) THEN
        READ(CAPSTR(1:CAPLEN), *, IOSTAT=CAPST) CAPIN
        IF (CAPST .EQ. 0 .AND. CAPIN .GE. 0) CAPVAL = CAPIN
      END IF
C     MUMPS orders on all processes when the count is 1, so 1 means 2
      IF (CAPVAL .EQ. 1) CAPVAL = 2
"""),
    (b"WORKERS = min(NPROCS,N/16)", b"""\
        IF (KEEP(245).EQ.1 .AND. CAPVAL.GE.1)
     &     WORKERS = min(WORKERS, CAPVAL)
"""),
    (b" 500  CONTINUE", b"""\
      IF (MYID.EQ.0 .AND. ICNTL(3).GT.0 .AND. ICNTL(4).GE.2)
     &  WRITE(ICNTL(3),'(A,I8,A,I8,A,I8)') ' PARAORD cap', CAPVAL,
     &  ' nprocs', NPROCS, ' paraord', NPROCS_PARAORD
"""),
]
hits = dict.fromkeys((anchor for anchor, _ in edits), 0)
with tarfile.open(pristine) as old, tarfile.open(patched, "w:gz") as new:
    for member in old:
        data = old.extractfile(member).read() if member.isfile() else None
        if member.name.endswith("/src/tools_common.F"):
            for anchor, lines in edits:
                hits[anchor] += data.count(anchor)
                if data.count(anchor) == 1:
                    end = data.index(b"\n", data.index(anchor)) + 1
                    data = data[:end] + lines + data[end:]
            member.size = len(data)
        new.addfile(member, io.BytesIO(data) if data is not None else None)
wrong = [f"'{a.decode()}' found {n} times" for a, n in hits.items() if n != 1]
if wrong:
    os.remove(patched)
    sys.exit(f"{pristine}: src/tools_common.F: {', '.join(wrong)}, expected once each")
EOF
}

# The built libpetsc must not import ZGEMMT: MUMPS is folded into it, so the name in
# its dynamic symbols is what the flag leaves behind. gfem refuses such a stack.
check_no_gemmt() {
  "$PY" - "$OUT"/petsc-"$PETSC_VERSION"-*.whl << 'EOF'
import os, sys, zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = [n for n in wheel.namelist() if os.path.basename(n).startswith("libpetsc.so")]
    if not names:
        sys.exit(f"{sys.argv[1]}: no libpetsc to check")
    for name in names:
        if b"\0zgemmt_\0" in wheel.read(name):
            sys.exit(f"{name} still imports zgemmt_: MUMPS was built with -DGEMMT_AVAILABLE")
        print(f"{name}: no zgemmt_")
EOF
}

# Every libpetsc must hold the ordering cap of patch_mumps: MUMPS is folded into
# it, and the cap's reading of MUMPS_PARAORD_MAX leaves that name in it. The
# wheel's libpetsc.so is a linker script naming the library, so not one itself.
check_paraord_cap() {
  "$PY" - "$OUT"/petsc-"$PETSC_VERSION"-*.whl << 'EOF'
import os, sys, zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    names = [n for n in wheel.namelist() if os.path.basename(n).startswith("libpetsc.so")]
    images = {n: wheel.read(n) for n in names}
    for name in [n for n, data in images.items() if data.startswith(b"INPUT(")]:
        print(f"{name}: linker script, skipped")
        del images[name]
    if not images:
        sys.exit(f"{sys.argv[1]}: no libpetsc to check")
    for name, data in images.items():
        if b"MUMPS_PARAORD_MAX" not in data:
            sys.exit(f"{name} lacks MUMPS_PARAORD_MAX: MUMPS was built without the ordering cap")
        print(f"{name}: MUMPS_PARAORD_MAX")
EOF
}

# The petsc wheel folds the static MUMPS, PORD and libseq libraries into
# libpetsc.so and then deletes them, but its petscvariables, petsc.pc and
# petscmachineinfo.h still list them, so SLEPc fails to link against it
# ("Unable to link with PETSc"). Remove those flags from the wheel's text
# files (never from a binary) and update their hashes in its RECORD.
patch_petsc_wheel() {
  "$PY" - "$OUT"/petsc-"$PETSC_VERSION"-*.whl << 'EOF'
import base64, csv, hashlib, io, os, re, sys, zipfile

wheel = sys.argv[1]
flags = re.compile(rb" ?-l(?:smumps|dmumps|cmumps|zmumps|mumps_common|pord|mpiseq)\b")
with zipfile.ZipFile(wheel) as old:
    (record,) = [n for n in old.namelist() if n.endswith(".dist-info/RECORD")]
    changed = {}
    for name in old.namelist():
        data = old.read(name)
        if name != record and b"\0" not in data and flags.search(data):
            changed[name] = flags.sub(b"", data)
            print(f"patched {name}")
    rows = list(csv.reader(io.StringIO(old.read(record).decode())))
    for row in rows:
        if row[0] in changed:
            digest = base64.urlsafe_b64encode(hashlib.sha256(changed[row[0]]).digest())
            row[1:] = ["sha256=" + digest.rstrip(b"=").decode(), str(len(changed[row[0]]))]
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    changed[record] = text.getvalue().encode()
    patched = wheel + ".patched"
    with zipfile.ZipFile(patched, "w") as new:
        for info in old.infolist():  # same names, order, permissions, compression
            data = changed[info.filename] if info.filename in changed else old.read(info)
            new.writestr(info, data)
os.replace(patched, wheel)
EOF
}

MPI=0 CHECK=0 SOURCES="" FETCH_SOURCES="" OUTPUT="" WITH_PARMETIS=""
while (($#)); do
  case $1 in
    --mpi) MPI=1 ;;
    --with-parmetis) WITH_PARMETIS=1 ;;
    --check) CHECK=1 ;;
    --sources)
      (($# >= 2)) || usage
      SOURCES=$2
      shift
      ;;
    --fetch-sources)
      (($# >= 2)) || usage
      FETCH_SOURCES=$2
      shift
      ;;
    -*) usage ;;
    *)
      [ -z "$OUTPUT" ] || usage
      OUTPUT=$1
      ;;
  esac
  shift
done
# exactly one of the three forms
FORMS=0
((CHECK == 0)) || FORMS=$((FORMS + 1))
[ -z "$OUTPUT" ] || FORMS=$((FORMS + 1))
[ -z "$FETCH_SOURCES" ] || FORMS=$((FORMS + 1))
((FORMS == 1)) || usage
if [ -n "$WITH_PARMETIS" ]; then
  # --fetch-sources fetches what either stack reads, so it takes no --mpi
  ((MPI == 1)) || [ -n "$FETCH_SOURCES" ] ||
    fail "ParMETIS orders only across ranks: --with-parmetis needs --mpi"
  parmetis_terms # before anything is downloaded
fi
if [ -n "$SOURCES" ]; then
  [ -d "$SOURCES" ] || fail "$SOURCES is not a folder"
  SOURCES=$(cd "$SOURCES" && pwd)
fi

leave_conda
if [ -n "$FETCH_SOURCES" ]; then
  check_prerequisites
  fetch_sources "$FETCH_SOURCES"
  exit 0
fi
check_prerequisites
if ((CHECK == 1)); then
  set_configure_options WORK
  echo "configure options: ${CONFIGURE_OPTIONS[*]}"
  echo "all prerequisites found"
  exit 0
fi

mkdir -p "$OUTPUT"
OUT=$(cd "$OUTPUT" && pwd)
[ -z "$(ls -A "$OUT")" ] || fail "$OUT is not empty: give a new or empty folder"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# A PETSC_DIR / PETSC_ARCH of another PETSc would leak into the build.
unset PETSC_DIR PETSC_ARCH SLEPC_DIR
# So would the compilers and flags an active conda environment exports:
# PETSc's configure ignores them, but the petsc4py and slepc4py builds use
# them, compile against conda's own PETSc headers, link conda's runtime
# libraries, and give bindings that do not import. Build with the gcc, g++
# and gfortran on PATH only.
unset CC CXX FC F77 F90 CPP CFLAGS CPPFLAGS CXXFLAGS FFLAGS FCFLAGS LDFLAGS
# MUMPS's PORD makefile reads this one from the environment, and stops on it.
unset OPTFLAGS
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export UV_NO_CONFIG=1
if ((MPI == 1)) && [ -z "$SOURCES" ]; then
  read -r _ url sha256 <<< "$(printf '%s\n' "${TARBALLS[@]}" | grep "^$SCOTCH_TARBALL ")"
  download "$url" "$sha256" "$WORK/$SCOTCH_TARBALL"
fi
read -r _ url sha256 <<< "$(printf '%s\n' "${TARBALLS[@]}" | grep "^$MUMPS_TARBALL ")"
if [ -n "$SOURCES" ]; then
  cp "$SOURCES/$MUMPS_TARBALL" "$WORK/$MUMPS_TARBALL"
  check_sha256 "$WORK/$MUMPS_TARBALL" "$sha256"
else
  download "$url" "$sha256" "$WORK/$MUMPS_TARBALL"
fi
patch_mumps
set_configure_options "$WORK"
export PETSC_CONFIGURE_OPTIONS="${CONFIGURE_OPTIONS[*]}${MAKE_NP:+ --with-make-np=$MAKE_NP}"
printf '%s\n' "petsc==$PETSC_VERSION" "petsc4py==$PETSC_VERSION" \
  "slepc==$SLEPC_VERSION" "slepc4py==$SLEPC_VERSION" > "$WORK/constraints.txt"

fetch petsc "$PETSC_VERSION" "$SHA256_petsc"
fetch petsc4py "$PETSC_VERSION" "$SHA256_petsc4py"
fetch slepc "$SLEPC_VERSION" "$SHA256_slepc"
fetch slepc4py "$SLEPC_VERSION" "$SHA256_slepc4py"
patch_petsc_sdist

build petsc "$PETSC_VERSION"
patch_petsc_wheel
check_no_gemmt
check_paraord_cap
build petsc4py "$PETSC_VERSION"
build slepc "$SLEPC_VERSION"
build slepc4py "$SLEPC_VERSION"

echo "wheelhouse: $OUT"
ls "$OUT"
echo "install: pip install --no-index --find-links $OUT petsc petsc4py slepc slepc4py"
