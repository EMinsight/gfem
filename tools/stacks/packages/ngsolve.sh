# Recipe of package ngsolve: NGSolve with netgen, plain CMake install.
# Sourced by stack.sh with STACK_NAME set to the software stack (empty for
# fetch). Interface: README.md, "Recipe interface".

VERSION=6.2.2608
CPUS=32
TIME=01:30:00
STACKS=(native-oss native-intel)
case ${STACK_NAME:-} in
native-oss) DEPENDS=(openblas) ;;
native-intel) DEPENDS=() ;;
*) DEPENDS=() ;;
esac

NGSOLVE_URL=https://github.com/NGSolve/ngsolve.git
NETGEN_URL=https://github.com/NGSolve/netgen.git

# fetch_ngsolve VERSION: login node, network only, into $STACK_SRC.
# Read from tag vVERSION: mirrors of NGSolve and its submodules at the commits
# the tag records (netgen, then netgen's pybind11), the UMFPACK source of
# cmake/SuperBuild.cmake (BUILD_UMFPACK), and the OCCT wheels its
# pyproject.toml pins (netgen-occt, netgen-occt-devel: ticket 07 Q1).
fetch_ngsolve() {
  local tag=v$1 line url ref occt umfpack=
  local lines=()
  stack_mirror "$NGSOLVE_URL" "$tag"
  stack_mirror_submodules "$NGSOLVE_URL" "$tag"
  local repos
  repos=$(stack_show "$NGSOLVE_URL" "$tag" cmake/SuperBuild.cmake | stack_sources git-repositories)
  mapfile -t lines <<<"$repos"
  for line in "${lines[@]}"; do
    read -r url ref <<<"$line"
    [[ $url == */suitesparse-metis-for-windows.git ]] || continue
    stack_mirror "$url" "$ref"
    umfpack=$url
  done
  [[ -n $umfpack ]] || stack_die "ngsolve $tag: no suitesparse repository in cmake/SuperBuild.cmake"
  occt=$(stack_show "$NGSOLVE_URL" "$tag" pyproject.toml |
    grep -oE 'netgen-occt-devel==[0-9][0-9.]*' | head -n 1) ||
    stack_die "ngsolve $tag: pyproject.toml pins no netgen-occt-devel"
  stack_pip wheels "netgen-occt==${occt#*==}" "$occt"
  # netgen's USE_MPI superbuild downloads METIS by URL at build time: its
  # tarball, checked against the URL_MD5 the netgen commit pins.
  local netgen name md5
  netgen=$(git --git-dir="$(stack_mirror_dir "$NGSOLVE_URL")" rev-parse "$tag:external_dependencies/netgen")
  while read -r name url md5; do
    ngsolve_md5_tarball "$name" "$url" "$md5"
  done < <(stack_show "$NETGEN_URL" "$netgen" cmake/external_projects/metis.cmake |
    stack_sources url-tarballs)
}

# ngsolve_md5_tarball NAME URL MD5: $STACK_SRC/tarballs/NAME, downloaded once
# and checked against the MD5 upstream pins (ExternalProject URL_MD5).
ngsolve_md5_tarball() {
  local file=$STACK_SRC/tarballs/$1 state=present
  if [[ ! -f $file ]]; then
    curl -fsSL --retry 3 -o "$file.part" "$2" || {
      rm -f "$file.part"
      stack_die "download of $2 failed"
    }
    mv "$file.part" "$file"
    state=new
  fi
  if [[ $(md5sum "$file" | cut -d' ' -f1) != "$3" ]]; then
    [[ $state == present ]] || rm -f "$file"
    stack_die "$1: md5 differs from the URL_MD5 pin $3 ($2)"
  fi
  fetch_report "$state" "tarballs/$1 (md5 pinned)"
}

# The CMake options of the native build (ticket 07 Q1, research/upstream-recipes.md
# §3.2): no -march=native but the toolchain's target through NG_COMPILE_FLAGS
# (PUBLIC on ngcore), MPI linked directly (no run-time wrapper; netgen then
# needs METIS; NG_COMPILE_FLAGS comes after CMAKE_CXX_FLAGS on the command
# line, and GCC keeps the toolchain's explicit -mno-avx512fp16 over a later
# -march), no NGSolve MUMPS (gfem's MUMPS is PETSc's), UMFPACK built
# against the stack's LAPACK, OCC from pip netgen-occt, no GUI, no stubs.
# build_ngsolve adds the compilers, flags, Python, BLAS and OCC paths.
NGSOLVE_OPTIONS=(
  -DCMAKE_BUILD_TYPE=Release
  -DUSE_NATIVE_ARCH=OFF
  -DNG_COMPILE_FLAGS=-march=sapphirerapids
  -DUSE_MPI=ON
  -DUSE_MPI_WRAPPER=OFF
  -DUSE_GUI=OFF
  -DUSE_MUMPS=OFF
  -DUSE_UMFPACK=ON
  -DBUILD_UMFPACK=ON
  -DUSE_OCC=ON
  -DBUILD_OCC=OFF
  -DBUILD_ZLIB=OFF
  -DUSE_LAPACK=ON
  -DUSE_HYPRE=OFF
  -DUSE_CUDA=OFF
  -DUSE_CCACHE=OFF
  -DBUILD_STUB_FILES=OFF
  -DENABLE_UNIT_TESTS=OFF
)

# ngsolve_python: the stack interpreter (ADR 0207), the pyenv package's version.
ngsolve_python() {
  local python
  python=$STACK_SRC/python/cpython-$(dep_version pyenv)-linux-x86_64-gnu/bin/python3
  [[ -x $python ]] || stack_die "ngsolve: no $python: run stack.sh fetch pyenv"
  echo "$python"
}

# ngsolve_gitconfig FILE: the job-local git config of ADR 0212: every
# upstream URL read from the source mirror, file:// submodules allowed.
ngsolve_gitconfig() {
  local host
  for host in github.com gitlab.com bitbucket.org; do
    printf '[url "file://%s/mirrors/%s/"]\n\tinsteadOf = https://%s/\n' "$STACK_SRC" "$host" "$host"
  done >"$1"
  printf '[protocol "file"]\n\tallow = always\n[advice]\n\tdetachedHead = false\n' >>"$1"
}

# ngsolve_check_python LOG ROOT: every Python CMake found in LOG is ROOT's.
ngsolve_check_python() {
  local found
  found=$(grep -a 'Found Python3:' "$1" | grep -v -F "$2/" || true)
  [[ -z $found ]] || stack_die "ngsolve: CMake found another Python than $2: $found"
}

# build_ngsolve VERSION PREFIX BUILD_DIR: in the job, offline; installs into PREFIX
# and writes $STACK_META/{source,configure}.json.
# A recursive clone of the tag from the mirror (netgen and pybind11 at the
# commits the tag records), the OCCT wheels into the prefix's site-packages
# (so the OCCT rpaths point there), netgen's METIS tarball pre-placed where
# its superbuild downloads it, then NGSolve's superbuild into PREFIX: the
# Python packages land in PREFIX/lib/python3.X/site-packages.
build_ngsolve() {
  local version=$1 prefix=$2 dir=$3 python src occt occdir site name url md5 blas
  local options=()
  python=$(ngsolve_python)
  ngsolve_gitconfig "$dir/gitconfig"
  export GIT_CONFIG_GLOBAL=$dir/gitconfig GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0
  src=$dir/ngsolve
  git clone -q --recursive -b "v$version" "$NGSOLVE_URL" "$src"

  occt=$(grep -oE 'netgen-occt-devel==[0-9][0-9.]*' "$src/pyproject.toml" | head -n 1)
  [[ -n $occt ]] || stack_die "ngsolve: pyproject.toml pins no netgen-occt-devel"
  "$python" -m pip install -q --disable-pip-version-check --no-warn-script-location \
    --no-index --find-links "$STACK_SRC/pip/wheels" --prefix "$prefix" \
    "netgen-occt==${occt#*==}" "$occt"
  site=$prefix/$("$python" -c 'import os, sysconfig; print(os.path.relpath(sysconfig.get_path("platlib"), sysconfig.get_path("data")))')
  # netgen-occt-devel's CMake config is wheel data: PREFIX/lib/cmake/opencascade.
  occdir=$(dirname "$(find "$prefix" -name OpenCASCADEConfig.cmake | head -n 1)")
  [[ -f $occdir/OpenCASCADEConfig.cmake ]] || stack_die "ngsolve: no OpenCASCADEConfig.cmake in $prefix"

  local netgen=$src/external_dependencies/netgen
  mkdir -p "$netgen/external_dependencies"
  while read -r name url md5; do
    [[ $(md5sum "$STACK_SRC/tarballs/$name" | cut -d' ' -f1) == "$md5" ]] ||
      stack_die "ngsolve: $name is not the METIS netgen pins (md5 $md5): run stack.sh fetch ngsolve"
    cp "$STACK_SRC/tarballs/$name" "$netgen/external_dependencies/${url##*/}"
  done < <(stack_sources url-tarballs <"$netgen/cmake/external_projects/metis.cmake")

  # netgen's and NGSolve's first find_package(Python3 COMPONENTS
  # Development.Module) ignores Python3_EXECUTABLE and took the system's
  # Python 3.9 headers (job 49170308): name the interpreter's root, headers
  # and library too.
  local pyroot pyinclude pylibrary
  pyroot=$(dirname "$(dirname "$python")")
  pyinclude=$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY"))')
  pylibrary=$("$python" -c 'import os, sysconfig as s; print(os.path.join(s.get_config_var("LIBDIR"), s.get_config_var("LDLIBRARY")))')
  [[ -f $pyinclude/Python.h && -f $pylibrary ]] ||
    stack_die "ngsolve: no Python.h in $pyinclude or no $pylibrary"
  case ${STACK_BLAS_VENDOR:-} in
  openblas)
    blas=${STACK_OPENBLAS_PROVENANCE%/provenance.json}/lib/lib$STACK_BLAS_LIB_PTHREADS.so
    [[ -f $blas ]] || stack_die "ngsolve: no BLAS library $blas"
    options+=(-DUSE_MKL=OFF "-DLAPACK_LIBRARIES=$blas")
    ;;
  mkl) options+=(-DUSE_MKL=ON -DMKL_LINK=sdl "-DMKL_ROOT=$MKLROOT") ;;
  *) stack_die "ngsolve: unknown STACK_BLAS_VENDOR '${STACK_BLAS_VENDOR:-}'" ;;
  esac
  options=("-DCMAKE_INSTALL_PREFIX=$prefix" "${NGSOLVE_OPTIONS[@]}" "${options[@]}"
    "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX"
    "-DCMAKE_C_FLAGS=$CFLAGS" "-DCMAKE_CXX_FLAGS=$CXXFLAGS"
    "-DCMAKE_SHARED_LINKER_FLAGS=$LDFLAGS" "-DCMAKE_MODULE_LINKER_FLAGS=$LDFLAGS"
    "-DCMAKE_EXE_LINKER_FLAGS=$LDFLAGS"
    "-DPython3_EXECUTABLE=$python" "-DPython3_FIND_STRATEGY=LOCATION"
    "-DPython3_ROOT_DIR=$pyroot" "-DPython3_INCLUDE_DIR=$pyinclude" "-DPython3_LIBRARY=$pylibrary"
    "-DOpenCascade_DIR=$occdir")
  echo "ngsolve: cmake ${options[*]}"
  cmake -S "$src" -B "$dir/build" -G "Unix Makefiles" "${options[@]}" 2>&1 | tee "$dir/configure.log"
  ngsolve_check_python "$dir/configure.log" "$pyroot"
  make -C "$dir/build" -j "$STACK_JOBS" 2>&1 | tee "$dir/make.log"
  ngsolve_check_python "$dir/make.log" "$pyroot"
  make -C "$dir/build" install
  mkdir -p "$prefix/share/stack"
  cp "$dir/build/CMakeCache.txt" "$prefix/share/stack/ngsolve-CMakeCache.txt"
  [[ ! -f $dir/build/netgen/CMakeCache.txt ]] ||
    cp "$dir/build/netgen/CMakeCache.txt" "$prefix/share/stack/netgen-CMakeCache.txt"

  local commits=() path wheels=()
  commits+=("ngsolve=$(git -C "$src" rev-parse HEAD)")
  while read -r _ path _; do
    commits+=("$path=$(git -C "$src/$path" rev-parse HEAD)")
  done < <(git -C "$src" submodule status --recursive)
  commits+=("suitesparse-metis-for-windows=$(grep -oE 'GIT_TAG [0-9a-f]+' "$src/cmake/SuperBuild.cmake" | head -n 1 | cut -d' ' -f2)")
  stack_json "$dir/commits.json" "${commits[@]}"
  for name in "$STACK_SRC"/pip/wheels/netgen_occt*-"${occt#*==}"-*.whl; do
    wheels+=("${name##*/}=$(sha256sum "$name" | cut -d' ' -f1)")
  done
  while read -r name url md5; do
    wheels+=("$name=md5 $md5, sha256 $(sha256sum "$STACK_SRC/tarballs/$name" | cut -d' ' -f1)")
  done < <(stack_sources url-tarballs <"$netgen/cmake/external_projects/metis.cmake")
  stack_json "$dir/files.json" "${wheels[@]}"
  stack_meta source "url=$NGSOLVE_URL" "tag=v$version" "commits:=$(<"$dir/commits.json")" \
    "files:=$(<"$dir/files.json")" "python=$("$python" -VV | head -n 1) ($python)"
  printf '%s\n' "${options[@]}" >"$dir/options.txt"
  stack_meta configure "options:=$("$python" -c 'import json, sys; print(json.dumps(open(sys.argv[1]).read().splitlines()))' "$dir/options.txt")" \
    "python_path=$site"
}

# smoke_ngsolve VERSION PREFIX: checks the install; records it with stack_meta smoke.
# With the stack interpreter in a venv holding numpy and PREFIX's
# site-packages on sys.path: ngsolve and netgen from PREFIX at VERSION, not a
# pip package (so ngsolve_openblas is never loaded), MPI linked; a Poisson
# problem solved by sparsecholesky and umfpack (gfem's two NGSolve inverses)
# against each other and the known peak; an OCC box meshed. Then libngla's
# BLAS is the stack's (ldd).
smoke_ngsolve() {
  local version=$1 prefix=$2 dir python site out lib blas mpi
  dir=$(mktemp -d "$TMPDIR/ngsolve-smoke.XXXXXX")
  python=$(ngsolve_python)
  "$python" -m venv "$dir/venv"
  "$dir/venv/bin/pip" install -q --disable-pip-version-check --no-index \
    --find-links "$STACK_SRC/pip/wheels" numpy
  site=$(sed -n 's/.*"python_path": "\(.*\)".*/\1/p' "$STACK_META/configure.json")
  [[ -d $site ]] || stack_die "ngsolve smoke: no site-packages $site"
  out=$(cd "$dir" && "$dir/venv/bin/python" -I - "$site" "$prefix" "$version" 2>&1 <<'PY'
import sys

site, prefix, version = sys.argv[1:4]
sys.path.insert(0, site)
import netgen
import ngsolve
from ngsolve import H1, BilinearForm, GridFunction, LinearForm, Mesh, dx, grad
from netgen.geom2d import unit_square
from netgen.occ import Box, OCCGeometry, Pnt

for module in (ngsolve, netgen):
    assert module.__file__.startswith(prefix), module.__file__
assert ngsolve.__version__.startswith(version), ngsolve.__version__
assert not ngsolve.config.is_python_package
assert not netgen.config.is_python_package
assert netgen.config.USE_MPI and not netgen.config.USE_NATIVE_ARCH
assert "-march=sapphirerapids" in netgen.config.NG_COMPILE_FLAGS, netgen.config.NG_COMPILE_FLAGS
try:
    import ngsolve_openblas  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError("ngsolve_openblas is importable")
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
fes = H1(mesh, order=3, dirichlet=".*")
u, v = fes.TnT()
a = BilinearForm(grad(u) * grad(v) * dx).Assemble()
f = LinearForm(1 * v * dx).Assemble()
peaks = {}
for inverse in ("sparsecholesky", "umfpack"):
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse=inverse) * f.vec
    peaks[inverse] = gfu(mesh(0.5, 0.5))
assert abs(peaks["sparsecholesky"] - peaks["umfpack"]) < 1e-12, peaks
assert abs(peaks["umfpack"] - 0.0736713532) < 1e-6, peaks
box = OCCGeometry(Box(Pnt(0, 0, 0), Pnt(1, 1, 1)))
elements = Mesh(box.GenerateMesh(maxh=0.5)).ne
assert elements > 0
print(
    f"ngsolve {ngsolve.__version__}: Poisson peak {peaks['umfpack']:.10f} "
    f"(sparsecholesky = umfpack), OCC box {elements} elements, "
    f"NG_COMPILE_FLAGS {netgen.config.NG_COMPILE_FLAGS}, USE_MPI"
)
PY
) || stack_die "ngsolve smoke failed: $out"
  echo "ngsolve smoke: $out"
  lib=$(find "$prefix" -name 'libngla.so*' | head -n 1)
  [[ -n $lib ]] || stack_die "ngsolve smoke: no libngla"
  blas=$(ldd "$lib" | grep -E 'libopenblas|libmkl_rt' | xargs)
  case ${STACK_BLAS_VENDOR:-} in
  openblas)
    [[ $blas == *"/openblas-$STACK_BLAS_VERSION/lib/lib$STACK_BLAS_LIB_PTHREADS.so"* &&
      $blas != *libopenblas_omp* ]] ||
      stack_die "ngsolve smoke: libngla's BLAS is not the stack's pthreads OpenBLAS: ${blas:-none}"
    ;;
  mkl) [[ $blas == *libmkl_rt* ]] || stack_die "ngsolve smoke: libngla links no mkl_rt" ;;
  esac
  mpi=$(find "$prefix" -name 'libngcore.so*' -exec ldd {} \; | grep -E 'libmpi\.' | head -n 1 | xargs)
  [[ -n $mpi ]] || stack_die "ngsolve smoke: libngcore links no MPI"
  stack_meta smoke "command=python -I (stack interpreter, numpy, sys.path += $site): import ngsolve; Poisson sparsecholesky/umfpack; OCC box" \
    "output=$(tail -n 1 <<<"$out")" "blas=$blas" "mpi=$mpi" "python_path=$site"
  rm -rf "$dir"
}
