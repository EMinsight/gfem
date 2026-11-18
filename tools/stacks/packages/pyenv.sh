# Recipe of package pyenv: the stack venv template (python-build-standalone) naming NGSolve and PETSc (ADR 0207).
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

# I/O bound: some 15 000 small files on GPFS, whose file creation ran at
# 0.6 s/file at times (jobs 49177238, 49177839); mpi4py is the only compile.
VERSION=3.12.12
CPUS=8
TIME=02:00:00
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


# pyenv_python VERSION: the fetched interpreter (ADR 0207).
pyenv_python() {
  local python=$STACK_SRC/python/cpython-$1-linux-x86_64-gnu/bin/python3
  [[ -x $python ]] || stack_die "pyenv: no $python: run stack.sh fetch pyenv"
  echo "$python"
}

# pyenv_pip PYTHON ARGS...: pip install from the stack's mirror only.
pyenv_pip() {
  local python=$1
  shift
  "$python" -m pip install -q --disable-pip-version-check --no-warn-script-location \
    --no-index --find-links "$STACK_SRC/pip/wheels" "$@"
}

# pyenv_dep_python_path PKG: configure.python_path of a dependency's provenance
# file (where its Python packages are installed).
pyenv_dep_python_path() {
  local var path
  var="STACK_$(tr '[:lower:]-' '[:upper:]_' <<<"$1")_PROVENANCE"
  [[ -f ${!var:-} ]] || stack_die "pyenv: no provenance file of dependency $1 (\$$var)"
  path=$("$STACK_PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1]))["configure"].get("python_path", ""))' "${!var}")
  [[ -n $path && -d $path ]] || stack_die "pyenv: dependency $1 records no python_path directory ($path)"
  echo "$path"
}

# pyenv_mkl_release: the toolchain's MKL release (INTEL_MKL_VERSION 20250200 -> 2025.2.0).
pyenv_mkl_release() {
  local header=$MKLROOT/include/mkl_version.h number
  number=$(awk '$1 == "#define" && $2 == "INTEL_MKL_VERSION" { print $3 }' "$header" 2>/dev/null)
  [[ $number =~ ^[0-9]{8}$ ]] || stack_die "pyenv: no INTEL_MKL_VERSION in $header"
  echo "${number:0:4}.$((10#${number:4:2})).$((10#${number:6:2}))"
}

# The MKL libraries a gfem process loads, which must be the toolchain's bytes.
PYENV_MKL_LIBS=(libmkl_rt.so.2 libmkl_core.so.2 libmkl_gnu_thread.so.2)

# pyenv_mkl_identical LIBDIR: each of PYENV_MKL_LIBS in LIBDIR has the sha256 of
# its $MKLROOT/lib copy: prints "name=sha256" lines, or sets PYENV_MKL_REASON
# and fails.
pyenv_mkl_identical() {
  local lib ours theirs
  for lib in "${PYENV_MKL_LIBS[@]}"; do
    [[ -f $1/$lib && -f $MKLROOT/lib/$lib ]] ||
      stack_die "pyenv: $lib missing in $1 or $MKLROOT/lib"
    ours=$(prov sha256 "$1/$lib")
    theirs=$(prov sha256 "$MKLROOT/lib/$lib")
    if [[ $ours != "$theirs" ]]; then
      PYENV_MKL_REASON="pip $PIP_MKL's $lib ($ours) is not the toolchain's $MKLROOT/lib/$lib ($theirs)"
      return 1
    fi
    echo "$lib=$ours"
  done
}

# pyenv_mkl_twin PYTHON PROBE: whether pip $PIP_MKL is a byte-identical twin
# of the toolchain's MKL (ADR 0207). Installs it --no-deps under the scratch
# prefix PROBE with PYTHON and compares: prints pyenv_mkl_identical's lines,
# or sets PYENV_MKL_REASON and fails. A stack without a twin gets no pip mkl,
# so no PARDISO, rather than two MKL releases in one process.
pyenv_mkl_twin() {
  local python=$1 probe=$2 release
  release=$(pyenv_mkl_release) || stack_die "pyenv: cannot read the toolchain's MKL release"
  if [[ $PIP_MKL != "mkl==$release" ]]; then
    PYENV_MKL_REASON="the toolchain's MKL is $release, the recipe pins $PIP_MKL"
    return 1
  fi
  pyenv_pip "$python" --no-deps --prefix "$probe" "$PIP_MKL" ||
    stack_die "pyenv: pip install of $PIP_MKL failed: run stack.sh fetch pyenv"
  pyenv_mkl_identical "$probe/lib"
}

# pyenv_mkl_install SITE: in an MKL stack venv, puts _stack_mkl.py and
# stack-mkl.pth (which imports it at interpreter start) into SITE.
# NGSolve built with USE_MKL (MKL_LINK=sdl) never binds its Python MKL
# PARDISO wrapper (ngsolve.solvers.mkl_pardiso runs _find_mkl only when
# not USE_MKL), and its own preload of the pip mkl_rt is skipped without
# intel-openmp (never installed: ADR 0207), so gfem would see no PARDISO
# (job 49183876). The module loads the venv's pip mkl_rt global first, so any
# later library that links mkl_rt (NGSolve, PETSc) maps that one copy, and
# runs the wrapper's _find_mkl when ngsolve imports it unbound.
pyenv_mkl_install() {
  local site=$1
  cat >"$site/_stack_mkl.py" <<'PY'
"""MKL stack venv: one pip mkl_rt per process, NGSolve's PARDISO wrapper bound.

Imported at interpreter start by stack-mkl.pth (tools/stacks/packages/pyenv.sh).
"""

import ctypes
import os
import sys
from importlib.machinery import PathFinder

_MKL_RT = os.path.join(sys.prefix, "lib", "libmkl_rt.so.2")
_WRAPPER = "ngsolve.solvers.mkl_pardiso"


class _BindPardiso:
    """Runs the wrapper's own MKL lookup after its module, if it left it unbound."""

    @staticmethod
    def find_spec(name, path=None, target=None):
        if name != _WRAPPER:
            return None
        spec = PathFinder.find_spec(name, path, target)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_bind(module):
            exec_module(module)
            if getattr(module, "_pardiso", None) is None and hasattr(module, "_find_mkl"):
                module._find_mkl()

        spec.loader.exec_module = exec_and_bind
        return spec


if os.path.isfile(_MKL_RT):
    ctypes.CDLL(_MKL_RT, mode=ctypes.RTLD_GLOBAL)
sys.meta_path.insert(0, _BindPardiso)
PY
  echo "import _stack_mkl" >"$site/stack-mkl.pth"
}

# build_pyenv VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
# PREFIX becomes a venv of the fetched interpreter holding gfem's runtime
# dependencies at uv.lock's pins (pip freeze must equal PIP_RUNTIME plus
# mpi4py, plus pip mkl in an MKL stack whose MKL has a byte-identical pip
# twin), mpi4py built from its sdist with the
# toolchain's MPICC in a separate build venv, stack.pth naming each
# dependency's python_path, byte-compiled so copies need not write.
build_pyenv() {
  local version=$1 prefix=$2 dir=$3 python site dep path mkl= no_twin= expected freeze
  python=$(pyenv_python "$version")
  "$python" -m venv "$prefix"
  site=$("$prefix/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
  [[ $site == "$prefix"/* ]] || stack_die "pyenv: venv site-packages $site is not under $prefix"
  pyenv_pip "$prefix/bin/python" --only-binary=:all: --report "$dir/runtime.json" "${PIP_RUNTIME[@]}"

  "$python" -m venv "$dir/build-venv"
  pyenv_pip "$dir/build-venv/bin/python" --only-binary=:all: setuptools wheel cython
  local sdist req
  for req in "${PIP_SDISTS[@]}"; do
    sdist=$(ls "$STACK_SRC"/pip/sdists/"${req%%==*}"-"${req#*==}".tar.gz)
    [[ -f $sdist ]] || stack_die "pyenv: no sdist of $req in $STACK_SRC/pip/sdists"
    echo "pyenv: building $req with MPICC=$MPICC ($(command -v "$MPICC"))"
    MPICC=$MPICC "$dir/build-venv/bin/python" -m pip wheel --disable-pip-version-check --no-cache-dir \
      --no-index --no-deps --no-build-isolation -w "$dir/wheels" "$sdist"
  done
  # by name, not by path: a path install is a direct URL in pip freeze
  pyenv_pip "$prefix/bin/python" --no-deps --find-links "$dir/wheels" "${PIP_SDISTS[@]}"

  if [[ ${STACK_BLAS_VENDOR:-} == mkl ]]; then
    [[ -n ${MKLROOT:-} ]] || stack_die "pyenv: STACK_BLAS_VENDOR=mkl but no MKLROOT"
    if pyenv_mkl_twin "$prefix/bin/python" "$dir/mkl-probe" >/dev/null; then
      pyenv_pip "$prefix/bin/python" --no-deps "$PIP_MKL"
      pyenv_mkl_identical "$prefix/lib" >"$dir/mkl.txt" ||
        stack_die "pyenv: $PYENV_MKL_REASON"
      pyenv_mkl_install "$site"
      mkl=$PIP_MKL
    else
      no_twin=$PYENV_MKL_REASON
      echo "pyenv: no byte-identical pip mkl ($no_twin): the venv gets none, so no PARDISO (ADR 0207)"
    fi
  fi

  for dep in "${DEPENDS[@]}"; do
    path=$(pyenv_dep_python_path "$dep")
    echo "$path"
  done >"$site/stack.pth"

  expected=$(printf '%s\n' "${PIP_RUNTIME[@]}" "${PIP_SDISTS[@]}" $mkl)
  freeze=$("$prefix/bin/python" -m pip freeze --path "$site" --disable-pip-version-check)
  [[ $(pyenv_normalize <<<"$freeze") == $(pyenv_normalize <<<"$expected") ]] ||
    stack_die "pyenv: the venv is not uv.lock's closure:"$'\n'"$(diff <(pyenv_normalize <<<"$expected") <(pyenv_normalize <<<"$freeze"))"
  "$prefix/bin/python" -m compileall -q -j "$STACK_JOBS" "$site" >/dev/null ||
    stack_die "pyenv: compileall failed"

  local tarball=cpython-$version+$PBS_RELEASE-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz files=()
  files+=("$tarball=$(prov sha256 "$STACK_SRC/python/$tarball")")
  while read -r path; do
    files+=("${path##*/}=$(prov sha256 "$path")")
  done < <("$STACK_PYTHON" -c '
import json, sys, urllib.parse, urllib.request
for item in json.load(open(sys.argv[1]))["install"]:
    print(urllib.request.url2pathname(urllib.parse.urlparse(item["download_info"]["url"]).path))
' "$dir/runtime.json")
  for path in "${PIP_SDISTS[@]}"; do
    path=$(ls "$STACK_SRC"/pip/sdists/"${path%%==*}"-"${path#*==}".tar.gz)
    files+=("${path##*/}=$(prov sha256 "$path")")
  done
  if [[ -n $mkl ]]; then
    for path in "$STACK_SRC"/pip/wheels/mkl-"${mkl#*==}"-*.whl; do
      files+=("${path##*/}=$(prov sha256 "$path")")
    done
  fi
  for path in "$dir"/wheels/*.whl; do
    files+=("built ${path##*/}=$(prov sha256 "$path")")
  done
  stack_json "$dir/files.json" "${files[@]}"
  stack_meta source "python=$("$python" -VV | head -n 1)" "interpreter=$python" \
    "python_build_standalone=$PBS_RELEASE" "files:=$(<"$dir/files.json")"
  local mkl_json=null
  if [[ -n $no_twin ]]; then
    mkl_json=$(stack_json - "pip=none" "reason=no byte-identical pip mkl: $no_twin")
  elif [[ -n $mkl ]]; then
    local pairs=()
    mapfile -t pairs <"$dir/mkl.txt"
    stack_json "$dir/mkl.json" "release=${mkl#*==}" "mklroot=$MKLROOT" "${pairs[@]}" \
      "hook=stack-mkl.pth imports _stack_mkl: preloads <venv>/lib/libmkl_rt.so.2 RTLD_GLOBAL, runs ngsolve.solvers.mkl_pardiso._find_mkl if unbound"
    mkl_json=$(<"$dir/mkl.json")
  fi
  stack_json "$dir/pth.json" "lines:=$("$STACK_PYTHON" -c 'import json, sys; print(json.dumps(open(sys.argv[1]).read().split()))' "$site/stack.pth")"
  stack_meta configure "venv=$python -m venv $prefix" \
    "runtime=pip install --no-index --only-binary=:all: ${PIP_RUNTIME[*]}" \
    "mpi4py=pip wheel --no-build-isolation --no-deps ${PIP_SDISTS[*]} (MPICC=$MPICC: $($MPICC -show 2>/dev/null | head -n 1))" \
    "stack_pth:=$(<"$dir/pth.json")" "mkl:=$mkl_json" \
    "freeze=$(paste -sd' ' <<<"$freeze")" "python_path=$site"
}

# pyenv_normalize: NAME==VERSION lines, names normalized (PEP 503), sorted.
pyenv_normalize() {
  awk -F== 'NF { name = tolower($1); gsub(/[-_.]+/, "-", name); print name "==" $2 }' | sort
}

# pyenv_gfem_wheel: the gfem wheel the smoke installs into a copy:
# $PYENV_GFEM_WHEEL, else the one wheel the verdict runner pinned under
# $STACKS_ROOT/verdict/gfem/<sha>/.
pyenv_gfem_wheel() {
  local wheels=()
  if [[ -n ${PYENV_GFEM_WHEEL:-} ]]; then
    wheels=("$PYENV_GFEM_WHEEL")
  else
    shopt -s nullglob
    wheels=("$STACKS_ROOT"/verdict/gfem/*/gfem-*.whl)
    shopt -u nullglob
  fi
  [[ ${#wheels[@]} -eq 1 && -f ${wheels[0]} ]] ||
    stack_die "pyenv smoke: need exactly one gfem wheel (PYENV_GFEM_WHEEL, else $STACKS_ROOT/verdict/gfem/*/gfem-*.whl): found ${wheels[*]:-none}"
  echo "${wheels[0]}"
}

# smoke_pyenv VERSION PREFIX: checks the install; records it with stack_meta smoke.
# In the stack venv (python -I: no PYTHONPATH, no user site): ngsolve, netgen,
# petsc4py from their prefixes through stack.pth, ngsolve at its version, no
# ngsolve_openblas; the BLAS the process maps is the stack's (one pthreads
# OpenBLAS, or one mkl_rt with no libiomp5); pip mkl and NGSolve's PARDISO
# iff the venv holds the pip twin of the toolchain's MKL (else the smoke
# records pardiso unavailable); mpi4py on the stack's MPI, its vendor
# recorded with whether gfem's ranked solves accept it (ADR 0057). Then a
# consumer's run: a cp -al copy (beside the prefix, same file system) with the
# gfem wheel installed --no-index --no-deps, pip check clean for gfem, and a
# tiny electrostatic solve against its exact capacitance. The copy is removed.
smoke_pyenv() {
  local version=$1 prefix=$2 dir ngsolve petsc blas mkl=false out wheel copy check pardiso=
  dir=$(mktemp -d "$TMPDIR/pyenv-smoke.XXXXXX")
  ngsolve=${STACK_NGSOLVE_PROVENANCE%/provenance.json}
  petsc=${STACK_PETSC_PROVENANCE%/provenance.json}
  case ${STACK_BLAS_VENDOR:-} in
  openblas) blas=$(readlink -f "${STACK_OPENBLAS_PROVENANCE%/provenance.json}/lib/lib$STACK_BLAS_LIB_PTHREADS.so") ;;
  mkl)
    # With pip mkl (the toolchain's twin) the process maps the venv's copy;
    # without, the toolchain's, and there is no PARDISO.
    if [[ -f $prefix/lib/libmkl_rt.so.2 ]]; then
      blas=$prefix/lib/libmkl_rt.so.2
      mkl=true
    else
      blas=$MKLROOT/lib/libmkl_rt.so.2
      pardiso="unavailable (no byte-identical pip mkl)"
    fi
    ;;
  *) stack_die "pyenv smoke: unknown STACK_BLAS_VENDOR '${STACK_BLAS_VENDOR:-}'" ;;
  esac
  [[ -f $blas ]] || stack_die "pyenv smoke: no BLAS library $blas"
  out=$(cd "$dir" && stack_singleton "$prefix/bin/python" -I - "$prefix" "$ngsolve" "$petsc" \
    "$(dep_version ngsolve)" "$(readlink -f "$blas")" "$STACK_MPI_VENDOR" "$mkl" "$dir/template.json" \
    "$STACK_BLAS_VENDOR" 2>&1 <<'PY'
import importlib.metadata
import importlib.util
import json
import os
import sys

prefix, ngsolve_prefix, petsc_prefix, version, blas, vendor, mkl, out, blas_vendor = sys.argv[1:10]
mkl = mkl == "true"
assert not os.environ.get("PYTHONPATH"), os.environ["PYTHONPATH"]
import netgen
import ngsolve
import numpy
import scipy.linalg  # noqa: F401
import petsc4py

petsc4py.init()
from petsc4py import PETSc
from mpi4py import MPI
from ngsolve.solvers import mkl_pardiso

for module, home in ((ngsolve, ngsolve_prefix), (netgen, ngsolve_prefix), (petsc4py, petsc_prefix)):
    assert module.__file__.startswith(home + "/"), (module.__name__, module.__file__)
import mpi4py

assert mpi4py.__file__.startswith(prefix + "/"), mpi4py.__file__
assert ngsolve.__version__.startswith(version), ngsolve.__version__
assert petsc4py.get_config()["PETSC_DIR"] == petsc_prefix, petsc4py.get_config()
assert importlib.util.find_spec("ngsolve_openblas") is None, "ngsolve_openblas is importable"
try:
    import ngsolve_openblas  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError("ngsolve_openblas imported")
names = {d.metadata["Name"].lower() for d in importlib.metadata.distributions()}
assert "ngsolve-openblas" not in names
assert ("mkl" in names) == mkl, sorted(names)
assert "intel-openmp" not in names
pardiso = mkl_pardiso._pardiso is not None
assert pardiso == mkl, f"NGSolve PARDISO {pardiso}, MKL stack {mkl}"

maps = set()
with open("/proc/self/maps") as f:
    for line in f:
        path = line.split()[-1]
        if path.startswith("/"):
            maps.add(os.path.realpath(path))
loaded = sorted(p for p in maps if any(k in os.path.basename(p) for k in ("openblas", "mkl", "iomp", "gomp")))
stack_blas = [p for p in loaded if "scipy_openblas" not in p and "numpy" not in p and "scipy" not in p]
assert blas in maps, f"the stack BLAS {blas} is not mapped: {loaded}"
assert not any("iomp5" in p for p in loaded), loaded
if blas_vendor == "mkl":
    assert [os.path.basename(p) for p in stack_blas if "mkl_rt" in p] == ["libmkl_rt.so.2"], loaded
else:
    assert not any("mkl" in p for p in loaded), loaded
    others = [p for p in stack_blas if "openblas" in p and p != blas]
    assert not others, f"another OpenBLAS beside {blas}: {others}"
got = MPI.get_vendor()
assert got[0] == vendor, got
# Open MPI's string ends in a NUL byte: never record it.
library = MPI.Get_library_version().replace("\x00", "").splitlines()[0].strip()
facts = {
    "ngsolve": f"{ngsolve.__version__} {ngsolve.__file__}",
    "petsc4py": f"{petsc4py.__version__} PETSc {'.'.join(map(str, PETSc.Sys.getVersion()))} {petsc4py.get_config()['PETSC_DIR']}",
    "mpi4py": f"{mpi4py.__version__} {mpi4py.__file__}",
    "numpy": numpy.__version__,
    "ngsolve_openblas": "not importable",
    "blas_mapped": stack_blas,
    "blas_bundled": [p for p in loaded if p not in stack_blas],
    "mkl": "pip mkl, NGSolve PARDISO" if mkl else "absent (no pip mkl, no NGSolve PARDISO)",
    "mpi_vendor": f"{got[0]} {'.'.join(map(str, got[1]))}",
    "mpi_library": library,
    "distributed": got[0] == "Open MPI",
}
with open(out, "w") as f:
    json.dump(facts, f)
print(
    f"ngsolve {ngsolve.__version__} from {ngsolve_prefix}, petsc4py {petsc4py.__version__}, "
    f"ngsolve_openblas not importable, BLAS {blas}, mkl {'present' if mkl else 'absent'}, "
    f"MPI {facts['mpi_vendor']} (distributed gfem: {facts['distributed']})"
)
PY
) || stack_die "pyenv smoke: the stack venv failed: $out"
  echo "pyenv smoke: $out"

  wheel=$(pyenv_gfem_wheel)
  copy=$(dirname "$prefix")/.smoke/pyenv-$version-${SLURM_JOB_ID:-$$}
  rm -rf "$copy"
  mkdir -p "$(dirname "$copy")"
  cp -al "$prefix" "$copy" || stack_die "pyenv smoke: cp -al $prefix $copy failed"
  "$copy/bin/python" -m pip install -q --disable-pip-version-check --no-index --no-deps "$wheel"
  [[ -z $("$prefix/bin/python" -m pip freeze --disable-pip-version-check --path "$("$prefix/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" | grep -i '^gfem') ]] ||
    stack_die "pyenv smoke: installing gfem into the copy changed the template"
  check=$("$copy/bin/python" -I -m pip check --disable-pip-version-check 2>&1 || true)
  if grep -qi '^gfem ' <<<"$check"; then
    stack_die "pyenv smoke: gfem's dependencies are not met in the stack venv: $check"
  fi
  out=$(cd "$dir" && stack_singleton "$copy/bin/python" -I - "$copy" "$mkl" 2>&1 <<'PY'
import itertools
import sys

import gfem
from gfem._linalg.pardiso import pardiso_available

copy, mkl = sys.argv[1], sys.argv[2] == "true"
assert gfem.__file__.startswith(copy + "/"), gfem.__file__
assert pardiso_available() == mkl, (pardiso_available(), mkl)
# A 1 mm cube of eps_r 2 between two plates (bottom 0 V, top terminal): the
# linear potential is exact on any tetrahedral mesh, so the capacitance is
# eps0 eps_r A / d = 2 eps0 1e-3 F.
n = 3
index, nodes = {}, []
for k, j, i in itertools.product(range(n + 1), repeat=3):
    index[i, j, k] = len(nodes) + 1
    nodes.append((i / n, j / n, k / n))
tets, faces = [], []
for i, j, k in itertools.product(range(n), repeat=3):
    for order in itertools.permutations(range(3)):  # Kuhn: 6 tetrahedra per cell
        v = [i, j, k]
        path = [index[tuple(v)]]
        for axis in order:
            v[axis] += 1
            path.append(index[tuple(v)])
        tets.append(path)
for axis in range(3):
    for side in (0, n):
        name = {(2, 0): "bottom", (2, n): "top"}.get((axis, side), "side")
        others = [x for x in range(3) if x != axis]
        for a, b in itertools.product(range(n), repeat=2):
            corner = []
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                v = [0, 0, 0]
                v[axis], v[others[0]], v[others[1]] = side, a + da, b + db
                corner.append(index[tuple(v)])
            faces += [(name, corner[:3]), (name, [corner[0], corner[2], corner[3]])]
tags = {"bottom": 1, "top": 2, "side": 3, "box": 4}
out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat", "$PhysicalNames", "4"]
out += [f'2 {tags[s]} "{s}"' for s in ("bottom", "top", "side")] + ['3 4 "box"']
out += ["$EndPhysicalNames", "$Nodes", str(len(nodes))]
out += [f"{i + 1} {x:g} {y:g} {z:g}" for i, (x, y, z) in enumerate(nodes)]
out += ["$EndNodes", "$Elements", str(len(faces) + len(tets))]
for e, (name, tri) in enumerate(faces, 1):
    out.append(f"{e} 2 2 {tags[name]} {tags[name]} " + " ".join(map(str, tri)))
for e, tet in enumerate(tets, len(faces) + 1):
    out.append(f"{e} 4 2 4 4 " + " ".join(map(str, tet)))
out.append("$EndElements")
with open("cube.msh", "w") as f:
    f.write("\n".join(out) + "\n")
mesh = gfem.Mesh.from_gmsh("cube.msh", length_unit="mm")
model = gfem.Model(mesh, materials={"box": gfem.Material(eps_r=2.0)})
result = gfem.solve(model, gfem.Electrostatic(terminals=["top"], ground="bottom"))
c = float(result.capacitance[0][0])
exact = 2 * 8.8541878188e-12 * 1e-3
assert abs(c - exact) < 1e-6 * exact, (c, exact)
pardiso = "absent"
if mkl:
    # One PARDISO factorization through gfem's inverse type, against sparse
    # Cholesky; then the process maps one copy of each MKL library, MKL's GNU
    # threading layer, and no libiomp5 (ADR 0205, ADR 0207).
    import os

    import ngsolve as ngs
    from netgen.geom2d import unit_square

    from gfem._linalg.pardiso import pardiso_inverse

    name = pardiso_inverse("hermitian", is_complex=False)
    assert name is not None
    m = ngs.Mesh(unit_square.GenerateMesh(maxh=0.05))
    fes = ngs.H1(m, order=3, dirichlet=".*")
    u, v = fes.TnT()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()
    f = ngs.LinearForm(1 * v * ngs.dx).Assemble()
    peaks = {}
    for inverse in (name, "sparsecholesky"):
        gfu = ngs.GridFunction(fes)
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse=inverse) * f.vec
        peaks[inverse] = gfu(m(0.5, 0.5))
    assert abs(peaks[name] - peaks["sparsecholesky"]) < 1e-12, peaks
    maps = set()
    with open("/proc/self/maps") as handle:
        for line in handle:
            path = line.split()[-1]
            if path.startswith("/"):
                maps.add(os.path.realpath(path))
    libs = sorted(p for p in maps if "mkl" in os.path.basename(p) or "iomp" in p)
    assert not any("iomp5" in p for p in libs), libs
    names = [os.path.basename(p) for p in libs]
    assert len(names) == len(set(names)), f"two copies of one MKL library: {libs}"
    assert "libmkl_gnu_thread.so.2" in names, libs
    assert "libmkl_intel_thread.so.2" not in names, libs
    pardiso = f"{name}: peak {peaks[name]:.10f} = sparsecholesky; mapped {' '.join(libs)}"
print(
    f"gfem {gfem.__version__} (--no-deps into a cp -al copy): cube capacitor "
    f"C = {c:.6e} F (exact {exact:.6e}), PARDISO {pardiso_available()}"
)
print(f"pardiso: {pardiso}")
PY
) || stack_die "pyenv smoke: gfem in a copy of the stack venv failed: $out"
  echo "pyenv smoke: $out"
  rm -rf "$copy"
  rmdir "$(dirname "$copy")" 2>/dev/null || true
  stack_meta smoke "command=python -I in the stack venv: ngsolve, petsc4py, mpi4py, /proc/self/maps; cp -al copy + pip install --no-index --no-deps gfem, electrostatic cube" \
    "output=$(grep -v '^pardiso: ' <<<"$out" | tail -n 1)" "pardiso=${pardiso:-$(sed -n 's/^pardiso: //p' <<<"$out")}" \
    "template:=$(<"$dir/template.json")" \
    "gfem_wheel=$wheel" "gfem_wheel_sha256=$(prov sha256 "$wheel")" \
    "pip_check=$(paste -sd';' <<<"$check")" \
    "ldd_libngla=$(ldd "$(find "${STACK_NGSOLVE_PROVENANCE%/provenance.json}" -name 'libngla.so*' | head -n 1)" | grep -E 'libopenblas|libmkl' | xargs)" \
    "ldd_libpetsc=$(ldd "${STACK_PETSC_PROVENANCE%/provenance.json}/lib/libpetsc.so" | grep -E 'libopenblas|libmkl' | xargs)"
  rm -rf "$dir"
}

# activate_pyenv PREFIX: STACK_PYENV names the venv template. The template is
# copied, not activated (ADR 0207): its bin and lib, which every activate.sh
# prepends, are taken off PATH and LD_LIBRARY_PATH again.
activate_pyenv() {
  cat <<EOF
export STACK_PYENV="$1"
PATH=":\$PATH:"; PATH=\${PATH//":$1/bin:"/:}; PATH=\${PATH#:}; PATH=\${PATH%:}; export PATH
for _d in lib lib64; do
  LD_LIBRARY_PATH=":\${LD_LIBRARY_PATH:-}:"; LD_LIBRARY_PATH=\${LD_LIBRARY_PATH//":$1/\$_d:"/:}
  LD_LIBRARY_PATH=\${LD_LIBRARY_PATH#:}; LD_LIBRARY_PATH=\${LD_LIBRARY_PATH%:}
done
unset _d; export LD_LIBRARY_PATH
EOF
}
