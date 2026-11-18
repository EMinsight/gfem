# Recipe of package petsc: PETSc and SLEPc with petsc4py and slepc4py, gfem's ADR 0067 contract (ADR 0209).
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=3.26.0
CPUS=32
TIME=02:00:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(openblas) ;;
native-intel) DEPENDS=() ;;
*) DEPENDS=() ;;
esac

# The --download tarballs of gfem's PETSc contract, as tools/petsc_wheelhouse.sh
# TARBALLS pins them (ADR 0067, ADR 0209): "NAME URL [SHA256]". PETSc's
# configure cannot download through the tunnel (urllib), so the build passes
# --download-<pkg>=$STACK_SRC/tarballs/NAME. ParMETIS has no pin anywhere:
# trusted on first fetch (kept for ticket 05, gfem's stack builds without it).
PETSC_TARBALLS=(
  "MUMPS_5.9.1.tar.gz https://mumps-solver.org/MUMPS_5.9.1.tar.gz 659c9b57646b5a003ac618baa1faf9dd2044e46c732b3daaccbc7158003e1b46"
  "scalapack-v2.2.3.tar.gz https://github.com/Reference-ScaLAPACK/scalapack/archive/v2.2.3.tar.gz 5d93701eca663925e98010dd8d0f45fd79b2191d74e5afa5711d587370a8b9dd"
  "scotch-v7.0.18.tar.gz https://gitlab.inria.fr/scotch/scotch/-/archive/v7.0.18/scotch-v7.0.18.tar.gz 2cbdcf6da5a08ca31437d7c967b243b3c96d621409ecb710768de2aa2e09802d"
  "petsc-pkg-metis-v5.1.0-p12.tar.gz https://bitbucket.org/petsc/pkg-metis/get/v5.1.0-p12.tar.gz 2f16abe87394d4fd1430d66a8995076c352af40f7a4d1c5300b3b1cc9d545663"
  "petsc-pkg-parmetis-v4.0.3-p9.tar.gz https://bitbucket.org/petsc/pkg-parmetis/get/v4.0.3-p9.tar.gz"
)

# petsc_pin NAME VERSION: the sha256 tools/petsc_wheelhouse.sh pins for a
# release tarball (petsc, slepc, petsc4py, slepc4py), empty if none.
petsc_pin() {
  case $1-$2 in
  petsc-3.26.0) echo f5230023e6e22ee607802a13c82bc25f3d81e71654ad386a5b9bdff17fed93df ;;
  petsc4py-3.26.0) echo 0a34bd083e24bee1a6759c4639a45d10c0117855eb9f0466834450245f75d7e5 ;;
  slepc-3.26.0) echo a2f4cc2af76d55c078c30ad8bc66b44736dac921a7912266eb44136fc1b6029d ;;
  slepc4py-3.26.0) echo 07bf970c861736b9b977dfecd4ad46415d6c4a6f26b0ed128ecffae59613d6ff ;;
  esac
}

# fetch_petsc VERSION: login node, network only, into $STACK_SRC.
# PETSc and SLEPc release tarballs (the PyPI sdists, same bytes as the
# release snapshots), petsc4py/slepc4py sdists, the --download tarballs.
fetch_petsc() {
  local version=$1 name dir line url pin
  for name in petsc slepc petsc4py slepc4py; do
    dir=tarballs
    [[ $name != *4py ]] || dir=pip
    stack_tarball "$dir" "$name-$version.tar.gz" \
      "https://files.pythonhosted.org/packages/source/${name:0:1}/$name/$name-$version.tar.gz" \
      $(petsc_pin "$name" "$version")
  done
  for line in "${PETSC_TARBALLS[@]}"; do
    read -r name url pin <<<"$line"
    stack_tarball tarballs "$name" "$url" $pin
  done
}

# The configure options of gfem's PETSc contract (tools/petsc_wheelhouse.sh
# --mpi, ADR 0067) plus the native parts of ADR 0209: the stack's MPI
# compilers, BLAS and flags, no OpenMP, an install into the prefix with
# petsc4py. The --download tarballs and --with-blaslapack-lib are added by
# build_petsc.
PETSC_OPTIONS=(
  --with-scalar-type=complex
  --with-64-bit-indices
  --with-parmetis=0
  --with-openmp=0
  --with-debugging=0
  --with-shared-libraries=1
  --with-x=0
  --with-petsc4py
)
# --download-<package> of each PETSC_TARBALLS entry (ParMETIS: none, ADR 0012).
PETSC_DOWNLOADS=(
  "mumps MUMPS_"
  "scalapack scalapack-"
  "ptscotch scotch-"
  "metis petsc-pkg-metis-"
)
# The build tools of petsc4py and slepc4py, from the stack's pip mirror.
PETSC_PIP_BUILD=(setuptools wheel cython numpy packaging)
PETSC_ARCH_NAME=arch-stack

# petsc_python: the stack interpreter (ADR 0207), the pyenv package's version.
petsc_python() {
  local python
  python=$STACK_SRC/python/cpython-$(dep_version pyenv)-linux-x86_64-gnu/bin/python3
  [[ -x $python ]] || stack_die "petsc: no $python: run stack.sh fetch pyenv"
  echo "$python"
}

# petsc_patch_gemmt PETSC_SRC: never let PETSc give MUMPS -DGEMMT_AVAILABLE
# (ADR 0205): the replace tools/petsc_wheelhouse.sh makes on the sdist, with
# the same "found once" guard, on the unpacked tree.
petsc_patch_gemmt() {
  "${STACK_PYTHON:-python3}" - "$1/config/BuildSystem/config/packages/MUMPS.py" <<'PY'
import sys

path = sys.argv[1]
line = "if self.blasLapack.checkForRoutine('dgemmt'):"
with open(path) as handle:
    text = handle.read()
hits = text.count(line)
if hits != 1:
    sys.exit(f"{path}: MUMPS.py's GEMMT check found {hits} times, expected once")
with open(path, "w") as handle:
    handle.write(text.replace(line, "if False:  # gfem: OpenBLAS's ZGEMMT overflows"))
PY
}

# petsc_blas: --with-blaslapack-lib for the stack's BLAS: the pthreads
# OpenBLAS of package openblas (native-oss), mkl_rt (native-intel).
petsc_blas() {
  case ${STACK_BLAS_VENDOR:-} in
  openblas)
    [[ -n ${STACK_OPENBLAS_PROVENANCE:-} ]] || stack_die "petsc: package openblas is not activated"
    echo "${STACK_OPENBLAS_PROVENANCE%/provenance.json}/lib/lib$STACK_BLAS_LIB_PTHREADS.so"
    ;;
  mkl) echo "$MKLROOT/lib/libmkl_rt.so" ;;
  *) stack_die "petsc: unknown STACK_BLAS_VENDOR '${STACK_BLAS_VENDOR:-}'" ;;
  esac
}

# petsc_clean COMMAND...: run COMMAND without the compiler variables PETSc's
# configure warns about and the petsc4py build would pick up, and without
# MUMPS PORD's OPTFLAGS (tools/petsc_wheelhouse.sh); flags go on the
# configure line instead.
petsc_clean() {
  env -u CC -u CXX -u FC -u F77 -u F90 -u CPP -u CFLAGS -u CPPFLAGS -u CXXFLAGS \
    -u FFLAGS -u FCFLAGS -u LDFLAGS -u OPTFLAGS -u CPATH -u C_INCLUDE_PATH \
    -u CPLUS_INCLUDE_PATH -u LIBRARY_PATH -u PETSC_DIR -u PETSC_ARCH -u SLEPC_DIR "$@"
}

# build_petsc VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
# PETSc then SLEPc from the release tarballs, unpacked in BUILD_DIR, built
# with a venv of the stack interpreter holding petsc4py's build tools;
# petsc4py and slepc4py land in PREFIX/lib (PETSc's PETSC4PY_INSTALL_PATH).
# PETSc's and SLEPc's own checks run against the installed prefix.
# install-lib, not install: the example trees (tens of thousands of files)
# copy at about 2 files/s onto GPFS (job 49169988); the checks run from the
# source trees.
build_petsc() {
  local version=$1 prefix=$2 dir=$3 python venv src slepc blas line package stem tarball entry
  local tarballs=$STACK_SRC/tarballs options=() downloads=()
  python=$(petsc_python)
  venv=$dir/venv
  "$python" -m venv "$venv"
  "$venv/bin/pip" install -q --disable-pip-version-check --no-index \
    --find-links "$STACK_SRC/pip/wheels" "${PETSC_PIP_BUILD[@]}"

  for line in "${PETSC_DOWNLOADS[@]}"; do
    read -r package stem <<<"$line"
    tarball=
    for entry in "${PETSC_TARBALLS[@]}"; do
      [[ $entry == "$stem"* ]] && tarball=$tarballs/${entry%% *}
    done
    [[ -f $tarball ]] || stack_die "petsc: no $stem* tarball: run stack.sh fetch petsc $version"
    downloads+=("--download-$package=$tarball")
  done
  blas=$(petsc_blas)
  [[ -f $blas ]] || stack_die "petsc: no BLAS library $blas"
  options=("--prefix=$prefix" "PETSC_ARCH=$PETSC_ARCH_NAME"
    "--with-cc=$MPICC" "--with-cxx=$MPICXX" "--with-fc=$MPIFC"
    "COPTFLAGS=$CFLAGS" "CXXOPTFLAGS=$CXXFLAGS" "FOPTFLAGS=$FFLAGS" "LDFLAGS=$LDFLAGS"
    "${PETSC_OPTIONS[@]}" "--with-blaslapack-lib=$blas" "${downloads[@]}"
    "--with-make-np=$STACK_JOBS")

  src=$dir/petsc-$version
  tar -xzf "$tarballs/petsc-$version.tar.gz" -C "$dir"
  petsc_patch_gemmt "$src"
  echo "petsc: configure ${options[*]}"
  (cd "$src" && petsc_clean "$venv/bin/python" ./configure "${options[@]}")
  petsc_clean make -C "$src" PETSC_DIR="$src" PETSC_ARCH="$PETSC_ARCH_NAME" all
  petsc_clean make -C "$src" PETSC_DIR="$src" PETSC_ARCH="$PETSC_ARCH_NAME" install-lib
  mkdir -p "$prefix/share/stack"
  cp "$src/$PETSC_ARCH_NAME/lib/petsc/conf/configure.log" "$prefix/share/stack/petsc-configure.log"
  find "$src/$PETSC_ARCH_NAME/externalpackages" -path '*MUMPS*' -name Makefile.inc \
    -exec cp {} "$prefix/share/stack/mumps-Makefile.inc" \; -quit
  [[ -f $prefix/share/stack/mumps-Makefile.inc ]] || stack_die "petsc: no MUMPS Makefile.inc"
  if ! (cd "$src" && petsc_clean make PETSC_DIR="$prefix" PETSC_ARCH= check) 2>&1 |
    tee "$prefix/share/stack/petsc-check.log"; then
    stack_die "petsc: make check against $prefix failed"
  fi

  # SLEPc's make, as its configure prints it: no PETSC_ARCH on the line (an
  # empty one overrides the arch SLEPc derives for a prefix build, job 49171925).
  # Its check against the installed prefix needs the empty one: without it,
  # it looks for the in-tree arch, runs no example and exits 0 (job 49172842).
  slepc=$dir/slepc-$version
  tar -xzf "$tarballs/slepc-$version.tar.gz" -C "$dir"
  (cd "$slepc" && petsc_clean env PETSC_DIR="$prefix" PYTHONPATH="$prefix/lib" \
    "$venv/bin/python" ./configure "--prefix=$prefix" --with-slepc4py)
  petsc_clean env PYTHONPATH="$prefix/lib" make -C "$slepc" SLEPC_DIR="$slepc" PETSC_DIR="$prefix"
  petsc_clean env PYTHONPATH="$prefix/lib" make -C "$slepc" SLEPC_DIR="$slepc" PETSC_DIR="$prefix" install-lib
  if ! (cd "$slepc" && petsc_clean make SLEPC_DIR="$prefix" PETSC_DIR="$prefix" PETSC_ARCH= check) 2>&1 |
    tee "$prefix/share/stack/slepc-check.log"; then
    stack_die "slepc: make check against $prefix failed"
  fi

  local sums=() name
  for name in "petsc-$version.tar.gz" "slepc-$version.tar.gz"; do
    sums+=("$name=$(sha256sum "$tarballs/$name" | cut -d' ' -f1)")
  done
  for line in "${downloads[@]}"; do
    sums+=("${line##*/}=$(sha256sum "${line#*=}" | cut -d' ' -f1)")
  done
  stack_json "$dir/tarballs.json" "${sums[@]}"
  stack_meta source "url=https://files.pythonhosted.org/packages/source/p/petsc/petsc-$version.tar.gz" \
    "tag=v$version" "tarballs:=$(<"$dir/tarballs.json")" \
    "gemmt_patch=config/BuildSystem/config/packages/MUMPS.py: dgemmt check -> if False (tools/petsc_wheelhouse.sh)" \
    "python=$("$python" -VV | head -n 1) ($python)"
  stack_json "$dir/options.json" "petsc:=$(printf '%s\n' "${options[@]}" | "$python" -c 'import json, sys; print(json.dumps(sys.stdin.read().splitlines()))')" \
    "slepc:=[\"--prefix=$prefix\", \"--with-slepc4py\"]"
  stack_meta configure "options:=$(<"$dir/options.json")" \
    "python_path=$prefix/lib" "build_venv=$("$venv/bin/pip" freeze | paste -sd' ')"
}

# petsc_mkl_zdotc DIR: ADR 0205's MKL build check: a gfortran program calling
# zdotc through the stack's mkl_rt returns the right complex value under the
# toolchain's MKL_INTERFACE_LAYER (mkl_rt's default Intel layer returns
# complex results the gfortran way only by accident). Prints the result line.
petsc_mkl_zdotc() {
  cat >"$1/zdotc.f90" <<'F90'
program zdotc_check
  implicit none
  complex(kind=8) :: x(3), y(3), r
  complex(kind=8), external :: zdotc
  x = [(1d0, 2d0), (3d0, -1d0), (0d0, 1d0)]
  y = [(2d0, 0d0), (1d0, 1d0), (-1d0, 4d0)]
  r = zdotc(3, x, 1, y, 1)
  ! conj(x) . y = (2-4i) + (2+4i) + (4+i) = 8+i
  if (abs(r - (8d0, 1d0)) > 1d-12) then
    print '(a, 2f10.4)', 'zdotc wrong:', r
    stop 1
  end if
  print '(a, 2f6.1)', 'zdotc ok:', r
end program
F90
  gfortran -o "$1/zdotc" "$1/zdotc.f90" -L"$MKLROOT/lib" -lmkl_rt -Wl,-rpath,"$MKLROOT/lib" &&
    "$1/zdotc"
}

# smoke_petsc VERSION PREFIX: checks the install; records it with stack_meta smoke.
# PETSc's and SLEPc's checks passed (with MUMPS and 2 MPI processes); the
# mumps block and ADR 0067's gates from the installed headers, nm -D and ldd
# of libpetsc (provenance.py petsc-mumps); petsc4py and slepc4py as a
# singleton: complex, 64-bit PetscInt, PETSc from this prefix, a complex MUMPS
# LU solve and a SLEPc eigenvalue.
smoke_petsc() {
  local version=$1 prefix=$2 dir lib out
  dir=$(mktemp -d "$TMPDIR/petsc-smoke.XXXXXX")
  grep -q "run successfully with 2 MPI processes" "$prefix/share/stack/petsc-check.log" ||
    stack_die "petsc smoke: make check did not run ex19 on 2 MPI processes"
  grep -q "run successfully with MUMPS" "$prefix/share/stack/petsc-check.log" ||
    stack_die "petsc smoke: make check did not run ex19 with MUMPS"
  grep -q "run successfully with 2 MPI processes" "$prefix/share/stack/slepc-check.log" ||
    stack_die "slepc smoke: make check did not run on 2 MPI processes"

  lib=$prefix/lib/libpetsc.so
  [[ -f $lib ]] || stack_die "petsc smoke: no $lib"
  nm -D "$lib" "$prefix"/lib/lib*mumps*.so >"$dir/dynsym.txt" 2>/dev/null || nm -D "$lib" >"$dir/dynsym.txt"
  ldd "$lib" >"$dir/needed.txt"
  cp "$prefix/share/stack/mumps-Makefile.inc" "$dir/Makefile.inc"
  echo "$(basename "$FC") $($FC -dumpfullversion)" >"$dir/compiler.txt"
  local entry
  for entry in "${PETSC_TARBALLS[@]}"; do
    case $entry in MUMPS_* | scalapack-* | scotch-* | petsc-pkg-metis-*)
      sha256sum "$STACK_SRC/tarballs/${entry%% *}" ;;
    esac
  done >"$dir/tarballs.txt"
  prov petsc-mumps "$prefix" "$dir" >"$STACK_META/mumps.json" ||
    stack_die "petsc smoke: the mumps block breaks ADR 0067/0209 (above)"
  grep -E '\b[sdcz]gemmt_\b' "$dir/dynsym.txt" && stack_die "petsc smoke: libpetsc imports ?gemmt_"
  local iomp zdotc=
  iomp=$(for lib in "$prefix"/lib/lib*.so; do ldd "$lib"; done | grep -E 'libiomp5|libmkl_intel_thread' | sort -u | xargs || true)
  [[ -z $iomp ]] || stack_die "petsc smoke: a library of $prefix links Intel OpenMP: $iomp"
  if [[ ${STACK_BLAS_VENDOR:-} == mkl ]]; then
    zdotc=$(petsc_mkl_zdotc "$dir" 2>&1) ||
      stack_die "petsc smoke: gfortran zdotc through mkl_rt failed (MKL_INTERFACE_LAYER=${MKL_INTERFACE_LAYER:-unset}): $zdotc"
    echo "petsc smoke: $zdotc (MKL_INTERFACE_LAYER=$MKL_INTERFACE_LAYER MKL_THREADING_LAYER=$MKL_THREADING_LAYER)"
  fi

  out=$(cd "$dir" && "$BUILD_DIR/venv/bin/python" -I - "$prefix" 2>&1 <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/lib")
import numpy as np
import petsc4py
petsc4py.init()
from petsc4py import PETSc
import slepc4py
slepc4py.init()
from slepc4py import SLEPc

assert PETSc.ScalarType == np.complex128, PETSc.ScalarType
assert PETSc.IntType == np.int64, PETSc.IntType
config = petsc4py.get_config()["PETSC_DIR"]
assert config == sys.argv[1], config
n = 400
A = PETSc.Mat().createAIJ([n, n], nnz=3, comm=PETSc.COMM_SELF)
for i in range(n):
    A[i, i] = 4 + 1j
    if i:
        A[i, i - 1] = -1
    if i + 1 < n:
        A[i, i + 1] = -1 + 0.5j
A.assemble()
x_true = A.createVecRight()
x_true.setArray(np.exp(1j * np.arange(n) / 7.0))
b = A.createVecLeft()
A.mult(x_true, b)
ksp = PETSc.KSP().create(PETSc.COMM_SELF)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
x = A.createVecRight()
ksp.solve(b, x)
assert pc.getFactorSolverType() == "mumps"
error = (x - x_true).norm() / x_true.norm()
assert error < 1e-12, error
eps = SLEPc.EPS().create(PETSc.COMM_SELF)
D = PETSc.Mat().createAIJ([50, 50], nnz=1, comm=PETSc.COMM_SELF)
for i in range(50):
    D[i, i] = i + 1
D.assemble()
eps.setOperators(D)
eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)
eps.solve()
top = eps.getEigenvalue(0)
assert abs(top - 50) < 1e-8, top
print(
    f"petsc4py {petsc4py.__version__} slepc4py {slepc4py.__version__}: complex, "
    f"int64, MUMPS LU error {error:.1e}, SLEPc largest eigenvalue {top.real:g}"
)
PY
) || stack_die "petsc smoke: petsc4py/slepc4py failed: $out"
  echo "petsc smoke: $out"
  stack_meta smoke "command=petsc4py+slepc4py singleton: MUMPS complex LU, SLEPc EPS (PYTHONPATH=$prefix/lib)" \
    "output=$(tail -n 1 <<<"$out")" \
    "make_check=$(grep 'run successfully' "$prefix/share/stack/petsc-check.log" | paste -sd';')" \
    "slepc_check=$(grep 'run successfully' "$prefix/share/stack/slepc-check.log" | paste -sd';')" \
    "blas=$(grep -E 'libopenblas|libmkl_rt' "$dir/needed.txt" | xargs)" \
    "mpi=$(grep -E 'libmpi' "$dir/needed.txt" | head -n 1 | xargs)" \
    "gemmt_imports=none" "iomp5=none" \
    "mkl_zdotc=${zdotc:-not an MKL stack}"
  rm -rf "$dir"
}
