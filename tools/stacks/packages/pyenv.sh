# Recipe of package pyenv: the stack venv template (python-build-standalone) naming NGSolve and PETSc (ADR 0207).
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=3.12.12
CPUS=16
TIME=01:00:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(ngsolve petsc) ;;
native-intel) DEPENDS=(ngsolve petsc) ;;
*) DEPENDS=() ;;
esac

# The python-build-standalone release of the interpreter (ticket 07 Q2).
PBS_RELEASE=20260211
# gfem's runtime dependencies at uv.lock's pins, without the PyPI NGSolve,
# netgen and their OpenBLAS (ticket 07 Q3). tests/unit/test_stacks.py fails
# when uv.lock moves away from this list.
PIP_RUNTIME=(
  jax==0.12.0
  jaxlib==0.12.0
  markdown-it-py==4.2.0
  mdurl==0.1.2
  meshio==5.3.5
  ml-dtypes==0.6.0
  netgen-occt==7.8.1
  numpy==2.5.4
  opt-einsum==3.4.0
  pygments==2.21.0
  rich==15.0.0
  scipy==1.18.1
)
# Build tools of the stack venv and the NGSolve/PETSc builds (offline in the job).
PIP_BUILD=(
  setuptools==84.2.1
  wheel==0.48.0
  packaging==26.3
  cython==3.3.1
  scikit-build-core==1.1.2
  pybind11-stubgen==3.0.0
)
# Built from source against the stack MPI, at uv.lock's pin (ticket 07 Q3).
PIP_SDISTS=(mpi4py==4.1.2)
# The pip twin of native-intel's module MKL 2025.2, installed --no-deps
# (ticket 07 Q4); the build checks it is byte-identical to $MKLROOT/lib.
PIP_MKL=mkl==2025.2.0

# fetch_pyenv VERSION: login node, network only, into $STACK_SRC.
# The interpreter (tarball pinned by its release's SHA256SUMS, unpacked once
# into python/cpython-VERSION-linux-x86_64-gnu/: slow on GPFS, see ADR 0212)
# and the wheels and sdists of the stack venv.
fetch_pyenv() {
  local version=$1 rel=$PBS_RELEASE
  local file=cpython-$version+$rel-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz
  local base=https://github.com/astral-sh/python-build-standalone/releases/download/$rel
  local sums=SHA256SUMS-$rel pin
  stack_download python "$sums" "$base/SHA256SUMS"
  pin=$(awk -v f="$file" '$2 == f { print $1 }' "$STACK_SRC/python/$sums")
  [[ -n $pin ]] || stack_die "pyenv: $file is not in python-build-standalone $rel's SHA256SUMS"
  stack_tarball python "$file" "$base/${file//+/%2B}" "$pin"

  local dir=$STACK_SRC/python/cpython-$version-linux-x86_64-gnu
  if [[ -x $dir/bin/python3 ]]; then
    fetch_report present "python/${dir##*/}"
  else
    rm -rf "$dir.part"
    mkdir -p "$dir.part"
    tar -xzf "$STACK_SRC/python/$file" -C "$dir.part"
    mv "$dir.part/python" "$dir"
    rmdir "$dir.part"
    fetch_report new "python/${dir##*/}"
  fi
  "$dir/bin/python3" -c 'import sys; print(sys.version.split()[0])' | grep -qx "$version" ||
    stack_die "pyenv: $dir/bin/python3 is not Python $version"

  stack_pip wheels "${PIP_RUNTIME[@]}" "${PIP_BUILD[@]}"
  stack_pip nodeps "$PIP_MKL"
  stack_pip sdists "${PIP_SDISTS[@]}"
}

# build_pyenv VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
build_pyenv() {
  stack_die "pyenv $1: build not written yet (native-builds ticket 06, 10)"
}

# smoke_pyenv VERSION PREFIX: checks the install; records it with stack_smoke.
smoke_pyenv() {
  stack_die "pyenv $1: smoke not written yet (native-builds ticket 06, 10)"
}
